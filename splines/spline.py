import numpy as np
import torch
from torch import nn
from .curve import Curve


class Spline(nn.Module):
    def __init__(self, batch_size: int, control_points: torch.Tensor, joint_points: torch.Tensor, curve: Curve):
        """
        Initialize the spline object with control points, joint points, and curve object.
        
        Parameters:
            control_points (torch.Tensor): Tensor of control points, shape (n, d).
            joint_points (torch.Tensor): Tensor of joint points, shape (m, d).
            curve (Curve): Curve object for evaluating the spline.
        """
        super(Spline, self).__init__()
        if (len(joint_points) - 1) * (curve.degree - 1) != len(control_points):
            raise ValueError("Number of control points doesn't fit the bezier degree")

        self.batch_size = batch_size
        self.control_points = nn.Parameter(control_points.clone().detach())
        self.joint_points = nn.Parameter(joint_points.clone().detach())
        self.curve = curve
        self.__ctrl_pts_per_section = self.curve.degree - 1

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
        num_intervals = self.joint_points.shape[1] - 1
        dim = self.joint_points.shape[2]

        # Create the tensor of powers of `t` values grouped by segments
        t_powers, valid_locations = self.create_t_powers_tensor(t, num_intervals, self.curve.degree)
        t_powers = t_powers.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)  # Shape: (batch_size, num_intervals, max_pts_in_interval, degree + 1)

        # Prepare the control points for each segment
        control_points = self.control_points.view(self.batch_size, num_intervals, self.__ctrl_pts_per_section, dim)
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
        result = result.view(self.batch_size, num_intervals * max_pts_in_interval,  dim)

        valid_indices = valid_locations[:, 0] * max_pts_in_interval + valid_locations[:, 1] # (N,)
        valid_indices = valid_indices.unsqueeze(0).expand(self.batch_size, -1) # (batch_size, N)
        valid_indices = valid_indices.unsqueeze(2).expand(-1, -1, dim) # (batch_size, N, dim)



        return torch.gather(result, dim=1, index=valid_indices).squeeze(1) # (batch_size, N, dim)
    
    def get_lines(self, batch: int) -> list[np.ndarray]:

        create_line = lambda joint_index, control_index : np.array([self.joint_points[batch, joint_index, :].detach().numpy(), self.control_points[batch, control_index, :].detach().numpy()])

        lines = []

        for i in range(self.joint_points.shape[1]):
            if i > 0:
                lines.append(create_line(i, self.__ctrl_pts_per_section * i - 1))
            
            if i < self.joint_points.shape[1] - 1:
                lines.append(create_line(i, self.__ctrl_pts_per_section * i))
        
        return lines

    def create_t_powers_tensor(self, t: torch.Tensor, num_intervals: int, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        u = num_intervals * t
        point_index = torch.floor(u).long()  # Integer part of `u`
        polynomial_t = u - point_index       # Fractional part of `u`

        last_point_indices = torch.where(point_index == num_intervals)
        point_index[last_point_indices] = num_intervals - 1
        polynomial_t[last_point_indices] = 1.0

        # Count how many `t` values belong to each segment
        segment_counts = torch.bincount(point_index, minlength=num_intervals)
        max_samples_in_interval = int(segment_counts.max().item())  # Maximum number of `t` values in any segment

        # Compute offsets for each segment
        segment_offsets = torch.cumsum(segment_counts, dim=0) - segment_counts

        # Compute scatter indices for placing `polynomial_t` into the correct rows
        scatter_indices = torch.arange(len(t), device=t.device) - segment_offsets[point_index]
        # print(f"segment offsets:{segment_offsets}")
        # print(f"scatter indices: {scatter_indices}")

        # Create a tensor to hold the powers of `t`
        max_degree = degree + 1
        t_powers = torch.zeros((num_intervals, max_samples_in_interval, max_degree), device=t.device)

        # Compute powers of `polynomial_t`
        powers = polynomial_t.unsqueeze(1).pow(torch.arange(max_degree, device=t.device).float())

        # Scatter the powers into the correct positions in `t_powers`
        t_powers[point_index, scatter_indices] = powers
         # Create a tensor of valid locations (segment index, row index)
        valid_locations = torch.stack([point_index, scatter_indices], dim=1)

        return t_powers, valid_locations




