from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from .curve import Bezier, Curve


class Spline(nn.Module):
    def __init__(self, num_dim: int, num_intervals: Union[int, torch.Tensor], num_curves: int = 1, curve: Curve = Bezier(degree=3), joint_points: Optional[torch.Tensor] = None, control_points: Optional[torch.Tensor] = None, name: str = ""):
        """
        Initialize the spline object with num_curves, control points, joint points, and curve object.

        Parameters:
            num_dim (int): Number of dimensions for the spline (e.g., 2 for 2D, 3 for 3D).
            num_intervals (int or torch.Tensor): Number of intervals per curve. If int, all curves
                share the same interval count. If tensor of shape (num_curves,), each curve can have
                a different interval count. Tensors are padded to max_intervals internally.
            num_curves (int): Number of curves (points that are moving across time). default is 1.
            joint_points (torch.Tensor): Tensor of joint points, shape (num_curves, max_intervals + 1, num_dim).
            control_points (torch.Tensor): Tensor of control points, shape (num_curves, max_intervals * (degree - 1), num_dim).
            curve (Curve): Curve object for evaluating the spline.
            name (str): Optional name prefix for parameter groups.
        """
        super(Spline, self).__init__()
        self.num_dim = num_dim
        self.num_curves = num_curves
        self.curve = curve
        self.__ctrl_pts_per_interval = self.curve.degree - 1
        self.name = name

        # Handle num_intervals: int (uniform) or tensor (per-curve)
        if isinstance(num_intervals, int):
            self.intervals_per_curve = torch.full((num_curves,), num_intervals, dtype=torch.long)
        else:
            self.intervals_per_curve = num_intervals.long()
            if self.intervals_per_curve.shape != (num_curves,):
                raise ValueError(f"intervals_per_curve must have shape ({num_curves},), got {self.intervals_per_curve.shape}")

        self.max_intervals = int(self.intervals_per_curve.max().item())

        if (control_points is None) != (joint_points is None):
            raise ValueError("Both control_points and joint_points must be provided or both must be None")

        if joint_points is None:
            joint_points = torch.zeros((num_curves, self.max_intervals + 1, num_dim), dtype=torch.float32)

        if control_points is None:
            control_points = torch.zeros((num_curves, self.max_intervals * self.__ctrl_pts_per_interval, num_dim), dtype=torch.float32)

        if joint_points.shape != (num_curves, self.max_intervals + 1, num_dim):
            raise ValueError(f"joint_points must have shape ({num_curves}, {self.max_intervals + 1}, {num_dim}), got {joint_points.shape}")

        if control_points.shape != (num_curves, self.max_intervals * self.__ctrl_pts_per_interval, num_dim):
            raise ValueError(f"control_points must have shape ({num_curves}, {self.max_intervals * self.__ctrl_pts_per_interval}, {num_dim}), got {control_points.shape}")

        self.control_points = nn.Parameter(control_points.clone().detach())
        self.joint_points = nn.Parameter(joint_points.clone().detach())

        self._control_points_grd = torch.zeros_like(self.control_points, requires_grad=False)
        self._joint_points_grd = torch.zeros_like(self.joint_points, requires_grad=False)

        # Precompute exponent range for t_powers
        self.register_buffer('_exponents', torch.arange(self.curve.degree + 1, dtype=torch.float32))

    # Backward-compatible property
    @property
    def num_intervals(self) -> int:
        """Returns max_intervals for backward compatibility."""
        return self.max_intervals

    def get_initial_points(self) -> torch.Tensor:
        """
        Returns the initial joint points of the spline, which are the positions at t=0.

        Returns:
            torch.Tensor: Tensor of shape (num_curves, num_dim) representing the initial joint points.
        """
        return self.joint_points[:, 0, :].clone().detach()

    def create_from_pcd(self, pcd: torch.Tensor):
        """
        Initializes the spline parameters (joint and control points) from a given point cloud.

        Sets up a stationary spline where every joint and control point equals the input position,
        so the spline evaluates to the same point for all t in [0, 1].

        Parameters:
            pcd (torch.Tensor): A tensor of shape (num_curves, num_dim).
        """
        num_curves, num_dim = pcd.shape

        if num_dim != self.num_dim:
            raise ValueError(f"Point cloud data must be {self.num_dim}D, got {num_dim}D")

        self.num_curves = num_curves

        self.joint_points = nn.Parameter(pcd.clone().detach().unsqueeze(1).repeat(1, self.max_intervals + 1, 1))
        self.control_points = nn.Parameter(pcd.clone().detach().unsqueeze(1).repeat(1, self.max_intervals * self.__ctrl_pts_per_interval, 1))

    def _assemble_interval_points(self) -> torch.Tensor:
        """
        Assembles the full per-interval control point tensor from joint_points and control_points.

        Returns:
            torch.Tensor: Shape (C, max_intervals, degree+1, dim)
        """
        control_points = self.control_points.view(self.num_curves, self.max_intervals, self.__ctrl_pts_per_interval, self.num_dim)
        points = torch.cat(
            [
                self.joint_points[:, :-1].unsqueeze(2),   # Start joint points
                control_points,                            # Control points
                self.joint_points[:, 1:].unsqueeze(2),    # End joint points
            ],
            dim=2,
        )  # Shape: (C, max_intervals, degree + 1, dim)
        return points

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the spline at given parameter values `t`.

        Uses direct per-(curve, query_point) evaluation:
        1. Per-curve interval mapping using intervals_per_curve
        2. Compute t_powers and apply Bernstein matrix
        3. Gather relevant control points per query
        4. Evaluate via einsum

        Parameters:
            t (torch.Tensor): Tensor of parameter values in [0, 1], shape (N,) or (N, 1).

        Returns:
            torch.Tensor: Shape (C, N, dim) — evaluated points on each curve.
        """
        if t.dim() == 2 and t.size(1) == 1:
            t = t.view(-1)
        elif t.dim() != 1:
            raise ValueError(f"Expected 1D or 2D tensor with shape (n,1) for `t`, got {t.shape}")

        N = t.shape[0]
        device = t.device
        intervals_f = self.intervals_per_curve.float().to(device)  # (C,)

        # Step 1: Per-curve interval mapping
        # u[c, n] = intervals_per_curve[c] * t[n]
        u = intervals_f.unsqueeze(1) * t.unsqueeze(0)  # (C, N)
        interval_idx = torch.floor(u).long()            # (C, N)
        local_t = u - interval_idx.float()              # (C, N)

        # Step 2: Clamp endpoint — when t == 1.0, interval_idx == num_intervals (out of bounds)
        max_idx = (self.intervals_per_curve.to(device) - 1).unsqueeze(1)  # (C, 1)
        clamped = interval_idx > max_idx
        interval_idx = torch.clamp(interval_idx, max=max_idx.expand_as(interval_idx))
        local_t = torch.where(clamped, torch.ones_like(local_t), local_t)

        # Step 3: Compute t_powers: (C, N, degree+1)
        exponents = self._exponents.to(device)  # (degree+1,)
        t_powers = local_t.unsqueeze(-1).pow(exponents)  # (C, N, degree+1)

        # Step 4: Apply characteristic (Bernstein) matrix: (C, N, degree+1)
        char_mat = self.curve.char_mat.to(device)  # (degree+1, degree+1)
        t_transformed = t_powers @ char_mat         # (C, N, degree+1)

        # Step 5: Assemble interval control points: (C, max_intervals, degree+1, dim)
        points = self._assemble_interval_points()

        # Step 6: Gather relevant points per (curve, query_point)
        # interval_idx: (C, N) -> expand to index into points dim=1
        idx = interval_idx.unsqueeze(-1).unsqueeze(-1)  # (C, N, 1, 1)
        idx = idx.expand(-1, -1, self.curve.degree + 1, self.num_dim)  # (C, N, degree+1, dim)
        gathered_points = torch.gather(points, dim=1, index=idx)  # (C, N, degree+1, dim)

        # Step 7: Evaluate via einsum
        result = torch.einsum('cnk,cnkd->cnd', t_transformed, gathered_points)  # (C, N, dim)

        return result

    def get_lines(self, batch: int) -> List[np.ndarray]:
        num_intervals = int(self.intervals_per_curve[batch].item())

        create_line = lambda joint_index, control_index : np.array([self.joint_points[batch, joint_index, :].detach().numpy(), self.control_points[batch, control_index, :].detach().numpy()])

        lines = []

        for i in range(num_intervals + 1):
            if i > 0:
                lines.append(create_line(i, self.__ctrl_pts_per_interval * i - 1))

            if i < num_intervals:
                lines.append(create_line(i, self.__ctrl_pts_per_interval * i))

        return lines

    def zero_gradient_cache(self):
        self._control_points_grd = torch.zeros_like(self.control_points, requires_grad=False)
        self._joint_points_grd = torch.zeros_like(self.joint_points, requires_grad=False)

    def cache_gradient(self):
        if self.control_points.grad is None or self.joint_points.grad is None:
            raise ValueError("Gradient cache has not been initialized. Call zero_gradient_cache() first.")

        self._control_points_grd += self.control_points.grad.clone()
        self._joint_points_grd += self.joint_points.grad.clone()

    def set_batch_gradient(self, cnt):
        ratio = 1 / cnt
        self.control_points.grad = self._control_points_grd * ratio
        self.joint_points.grad = self._joint_points_grd * ratio


    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """
        Flattens the spline's control points and joint points into a dictionary
        of named 1D NumPy arrays, suitable for structured export (e.g., to PLY format).

        Uses max_intervals for the padded tensor dimensions.

        Returns:
            dict[str, np.ndarray]: A dictionary where keys are attribute names and values are
                                NumPy arrays of shape (N,), where N is the number of spline curves.
        """
        joint = self.joint_points.detach().cpu().numpy()   # shape: (N, max_intervals+1, dim)
        ctrl = self.control_points.detach().cpu().numpy()   # shape: (N, max_intervals*K, dim)

        num_joint_pts = joint.shape[1]
        num_ctrl_pts = ctrl.shape[1]

        axis_names = ['x', 'y', 'z'][:self.num_dim]
        result = {}

        for i in range(num_joint_pts):
            for d in range(self.num_dim):
                key = f"{self.name}_joint_pt_{axis_names[d]}_{i}"
                result[key] = joint[:, i, d]

        for i in range(num_ctrl_pts):
            for d in range(self.num_dim):
                key = f"{self.name}_ctrl_pt_{axis_names[d]}_{i}"
                result[key] = ctrl[:, i, d]

        return result


    def get_optimizable_groups(self, base_lr : float) -> List[dict]:
        """
        Returns a list of dictionaries representing optimizable parameter groups for the spline.

        Parameters:
            base_lr (float): Base learning rate for the optimizer.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'params' and 'lr' keys.
        """
        return [
            { "params": [self.joint_points], "lr": base_lr, "name": f"{self.name}_joint_points" },
            { "params": [self.control_points], "lr": base_lr, "name": f"{self.name}_control_points" }
        ]


    def update_params_(self, params: dict):
        """
        Updates the spline parameters based on the provided dictionary of parameters.

        Parameters:
            params (dict): A dictionary containing the parameters for the spline.
        """
        self.joint_points = params[f"{self.name}_joint_points"]
        self.control_points = params[f"{self.name}_control_points"]
        self.num_curves = self.joint_points.shape[0]


    def clone_and_modify(self, fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]]) -> dict:
        """
        Clones the spline parameters and applies functions to each parameter.

        Args:
            fns (dict): Dictionary mapping parameter names ('joint_points', 'control_points') to functions.

        Returns:
            dict: Updated parameters using the same naming keys as `get_optimizable_groups()`.
        """
        updated = {}

        if "joint_points" in fns:
            updated[f"{self.name}_joint_points"] = fns["joint_points"](self.joint_points)
        else:
            updated[f"{self.name}_joint_points"] = self.joint_points.clone()

        if "control_points" in fns:
            updated[f"{self.name}_control_points"] = fns["control_points"](self.control_points)
        else:
            updated[f"{self.name}_control_points"] = self.control_points.clone()

        return updated

    def create_new_curves(self, initial_points: torch.Tensor, num_intervals: Optional[int] = None) -> dict:
        """
        Prepares new spline curves initialized with constant values across time steps.
        New curves are padded to max_intervals.

        Parameters:
            initial_points (torch.Tensor): Shape (N, D), where N is the number of new curves.
            num_intervals (int, optional): Interval count for the new curves. Defaults to max_intervals.

        Returns:
            dict: New parameter tensors for joint_points and control_points.
        """
        N, D = initial_points.shape
        intervals = num_intervals if num_intervals is not None else self.max_intervals

        num_joint = self.max_intervals + 1
        num_ctrl = self.max_intervals * self.__ctrl_pts_per_interval

        new_joint = initial_points.unsqueeze(1).expand(N, num_joint, D).clone()
        new_ctrl = initial_points.unsqueeze(1).expand(N, num_ctrl, D).clone()

        return {
            f"{self.name}_joint_points": new_joint,
            f"{self.name}_control_points": new_ctrl
        }


    def mask_curves_bounding_box(self, threshold: float):
        all_points = torch.cat([self.joint_points, self.control_points], dim=1)  # (N, K, dim)
        min_xyz = all_points.min(dim=1).values
        max_xyz = all_points.max(dim=1).values
        movement_extent = torch.norm(max_xyz - min_xyz, dim=1)  # (N,)

        return movement_extent > threshold
