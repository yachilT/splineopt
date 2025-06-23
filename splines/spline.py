from typing import Optional
import numpy as np
import torch
from torch import nn
from .curve import Bezier, Curve


class Spline(nn.Module):
    def __init__(self, num_dim: int, num_intervals: int, num_curves: int = 1, curve: Curve = Bezier(degree=3), joint_points: Optional[torch.Tensor] = None, control_points: Optional[torch.Tensor] = None):
        """
        Initialize the spline object with num_curves, control points, joint points, and curve object.
        
        Parameters:
            num_dim (int): Number of dimensions for the spline (e.g., 2 for 2D, 3 for 3D).
            num_intervals (int): Number of intervals in the spline.
            num_curve (int): Number of curves (points that are moving across time). default is 1.
            joint_points (torch.Tensor): Tensor of joint points, shape (num_curves, num_intervals + 1, num_dim).
            control_points (torch.Tensor): Tensor of control points, shape (num_curves, num_intervals * curve.degree - 1, num_dim).
            curve (Curve): Curve object for evaluating the spline.
        """
        super(Spline, self).__init__()
        self.num_dim = num_dim
        self.num_curves = num_curves
        self.num_intervals = num_intervals
        self.curve = curve
        self.__ctrl_pts_per_interval = self.curve.degree - 1


        if (control_points is None) != (joint_points is None):
            raise ValueError("Both control_points and joint_points must be provided or both must be None")
        
        if joint_points is None:
            joint_points = torch.zeros((num_curves, num_intervals + 1, num_dim), dtype=torch.float32)
        else:
            joint_points = joint_points

        if control_points is None:
            control_points = torch.zeros((num_curves, num_intervals * self.__ctrl_pts_per_interval, num_dim), dtype=torch.float32)
        else:
            control_points = control_points


        if joint_points.shape != (num_curves, num_intervals + 1, num_dim):
            raise ValueError(f"joint_points must have shape (num_curves, num_intervals + 1, num_dim), got {self.joint_points.shape}")

        if control_points.shape != (num_curves, num_intervals * self.__ctrl_pts_per_interval, num_dim):
            raise ValueError(f"control_points must have shape (num_curves, num_intervals * (degree - 1), num_dim), got {self.control_points.shape}")

        self.control_points = nn.Parameter(control_points.clone().detach())
        self.joint_points = nn.Parameter(joint_points.clone().detach())

        self._control_points_grd = torch.zeros_like(self.control_points, requires_grad=False)
        self._joint_points_grd = torch.zeros_like(self.joint_points, requires_grad=False)


    def create_from_pcd(self, pcd: torch.Tensor):
        """
        Initializes the spline parameters (joint and control points) from a given point cloud.

        This method sets up the initial spline configuration such that the spline curves exactly interpolate 
        the input point cloud with no deformation, i.e., the spline will always evaluate to the original 
        point regardless of the input parameter `t ∈ [0, 1]`.

        Specifically:
        - `joint_points` are initialized by repeating each point across all joint positions.
        - `control_points` are also initialized by repeating each point across all control point slots 
        to ensure a stationary spline (i.e., no movement during interpolation).

        Parameters:
            pcd (torch.Tensor): A tensor of shape (num_curves, num_dim), where each row represents a 
                                curve's target position in `num_dim`-dimensional space.

        Raises:
            ValueError: If the dimensionality of the input `pcd` does not match the expected `self.num_dim`.
        """

        num_curves, num_dim = pcd.shape

        if num_dim != self.num_dim:
            raise ValueError(f"Point cloud data must be {self.num_dim}D, got {num_dim}D")
        
        self.num_curves = num_curves



        self.joint_points = nn.Parameter(pcd.clone().detach.unsqueeze(1).repeat(1, self.num_intervals + 1, 1))  # Shape: (num_curves, num_intervals + 1, num_dim)
        self.control_points = nn.Parameter(pcd.clone().detach.unsqueeze(1).repeat(1, self.num_intervals * self.__ctrl_pts_per_interval, 1))  # Shape: (num_curves, num_intervals * (degree - 1), num_dim)



    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the spline at given parameter values `t`.

        Parameters:
            t (torch.Tensor): Tensor of parameter values in the range [0, 1].

        Returns:
            torch.Tensor: Tensor of points on the spline corresponding to `t`.
        """
        epsilon = 1e-10
        t = torch.clamp(t, max=1-epsilon)

        # Create the tensor of powers of `t` values grouped by segments
        t_powers, valid_locations = self.create_t_powers_tensor(t)
        t_powers = t_powers.unsqueeze(0).repeat(self.num_curves, 1, 1, 1)  # Shape: (num_curves, num_intervals, max_pts_in_interval, degree + 1)

        # Prepare the control points for each segment
        control_points = self.control_points.view(self.num_curves, self.num_intervals, self.__ctrl_pts_per_interval, self.num_dim)
        points = torch.cat(
            [
                self.joint_points[:, :-1].unsqueeze(2),  # Start joint points
                control_points,                      # Control points
                self.joint_points[:, 1:].unsqueeze(2),  # End joint points
            ],
            dim=2,
        )  # Shape: (batch_size, num_intervals, degree + 1, dim)

        # Call the curve's evaluate method
        result = self.curve.evaluate(t_powers, points) # Shape: (batch_size, num_intervals, max_pts_in_interval, dim)

        max_pts_in_interval = result.shape[2]

         # Reshape the result to a 3D tensor:
        result = result.view(self.num_curves, self.num_intervals * max_pts_in_interval,  self.num_dim)

        valid_indices = valid_locations[:, 0] * max_pts_in_interval + valid_locations[:, 1] # (N,)
        valid_indices = valid_indices.unsqueeze(0).expand(self.num_curves, -1) # (batch_size, N)
        valid_indices = valid_indices.unsqueeze(2).expand(-1, -1, self.num_dim) # (batch_size, N, dim)



        return torch.gather(result, dim=1, index=valid_indices).squeeze(1) # (batch_size, N, dim)
    
    def get_lines(self, batch: int) -> list[np.ndarray]:

        create_line = lambda joint_index, control_index : np.array([self.joint_points[batch, joint_index, :].detach().numpy(), self.control_points[batch, control_index, :].detach().numpy()])

        lines = []

        for i in range(self.joint_points.shape[1]):
            if i > 0:
                lines.append(create_line(i, self.__ctrl_pts_per_interval * i - 1))
            
            if i < self.joint_points.shape[1] - 1:
                lines.append(create_line(i, self.__ctrl_pts_per_interval * i))
        
        return lines

    def create_t_powers_tensor(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a tensor where each slice corresponds to an interval, and each slice contains
        a matrix where each row is the powers of a `t` value from the same interval.

        Parameters:
            t (torch.Tensor): A 1D sorted tensor of parameter values in the range [0, 1].
            num_intervals (int): The number of spline intervals.
            degree (int): The degree of the curve.

        Returns:
            torch.Tensor: A tensor of shape (num_intervals, max_t_in_segment, degree + 1).
            torch.Tensor: A tensor of valid locations (interval index, row index) for each `t` value.
        """
        # Compute u and split into integer and fractional parts
        u = self.num_intervals * t
        point_index = torch.floor(u).long()  # Integer part of `u`
        polynomial_t = u - point_index       # Fractional part of `u`

        last_point_indices = torch.where(point_index == self.num_intervals)
        point_index[last_point_indices] = self.num_intervals - 1
        polynomial_t[last_point_indices] = 1.0

        # Count how many `t` values belong to each segment
        segment_counts = torch.bincount(point_index, minlength=self.num_intervals)
        max_samples_in_interval = int(segment_counts.max().item())  # Maximum number of `t` values in any segment

        # Compute offsets for each segment
        segment_offsets = torch.cumsum(segment_counts, dim=0) - segment_counts

        # Compute scatter indices for placing `polynomial_t` into the correct rows
        scatter_indices = torch.arange(len(t), device=t.device) - segment_offsets[point_index]
        # print(f"segment offsets:{segment_offsets}")
        # print(f"scatter indices: {scatter_indices}")

        # Create a tensor to hold the powers of `t`
        max_degree = self.curve.degree + 1
        t_powers = torch.zeros((self.num_intervals, max_samples_in_interval, max_degree), device=t.device)

        # Compute powers of `polynomial_t`
        powers = polynomial_t.unsqueeze(1).pow(torch.arange(max_degree, device=t.device).float())

        # Scatter the powers into the correct positions in `t_powers`
        t_powers[point_index, scatter_indices] = powers
         # Create a tensor of valid locations (segment index, row index)
        valid_locations = torch.stack([point_index, scatter_indices], dim=1)

        return t_powers, valid_locations
    

    def zero_gradient_cache(self):
        self._control_points_grd.zero_()
        self._joint_points_grd.zero_()

    def cache_gradient(self):
        if self.control_points.grad is None or self.joint_points.grad is None:
            raise ValueError("Gradient cache has not been initialized. Call zero_gradient_cache() first.")

        self._control_points_grd += self.control_points.grad.clone()
        self._joint_points_grd += self.joint_points.grad.clone()

    def set_batch_gradient(self, cnt):
        ratio = 1 / cnt
        self._control_points.grad = self._control_points_grd * ratio
        self._joint_points.grad = self._joint_points_grd * ratio




