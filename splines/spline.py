from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from .curve import Bezier, Curve


class Spline(nn.Module):
    def __init__(self, num_dim: int, num_intervals: Union[int, torch.Tensor], num_curves: int = 1, curve: Curve = Bezier(degree=3), joint_points: Optional[torch.Tensor] = None, control_points: Optional[torch.Tensor] = None, c1_mask: Optional[torch.Tensor] = None, name: str = ""):
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
            c1_mask (torch.Tensor, optional): Boolean tensor of shape (num_curves, max_intervals + 1).
                c1_mask[c, k] = True enforces C1 continuity at junction k of curve c, deriving
                its first control point as the reflection through the joint. Only internal junctions
                (1 <= k <= intervals_per_curve[c] - 1) are meaningful. Defaults to all False.
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

        # C1 continuity mask: c1_mask[c, k] = True enforces C1 at junction k of curve c.
        # Only internal junctions (1 <= k <= intervals_per_curve[c] - 1) are meaningful.
        if c1_mask is None:
            c1_mask = torch.zeros(num_curves, self.max_intervals + 1, dtype=torch.bool)
        if c1_mask.shape != (num_curves, self.max_intervals + 1):
            raise ValueError(f"c1_mask must have shape ({num_curves}, {self.max_intervals + 1}), got {c1_mask.shape}")
        self.register_buffer('c1_mask', c1_mask)

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

    def set_c1(self, curve_idx: int, joint_idx: int, enabled: bool = True):
        """
        Mark a joint as smooth (C1) or release it.

        A control point C_{k,0} is derived only when BOTH joint k and joint k+1 are
        marked smooth. Marking a single joint does not derive any control point until
        its neighbour is also marked.

        Parameters:
            curve_idx (int): Index of the curve.
            joint_idx (int): Index of the joint (0 to intervals_per_curve[c]).
            enabled (bool): True to mark the joint smooth, False to release it.
        """
        n = int(self.intervals_per_curve[curve_idx].item())
        if not (0 <= joint_idx <= n):
            raise ValueError(f"joint_idx must be in [0, {n}], got {joint_idx}")
        self.c1_mask[curve_idx, joint_idx] = enabled

    def get_effective_control_points(self) -> torch.Tensor:
        """
        Returns control points with C1 continuity constraints applied.

        At each junction k where c1_mask[c, k] is True, the first control point of
        interval k is derived as the reflection of the previous interval's last control
        point through the joint:
            C_{k,0} = 2 * J_k - C_{k-1, last}

        The remaining control points of the interval are unchanged (free).

        Returns:
            torch.Tensor: Shape (num_curves, max_intervals, degree-1, num_dim).
        """
        k_per = self.__ctrl_pts_per_interval
        C = self.num_curves

        # (C, max_intervals, k_per, dim)
        raw_cp = self.control_points.view(C, self.max_intervals, k_per, self.num_dim)

        if not self.c1_mask.any():
            return raw_cp

        result_curves = []
        for c in range(C):
            n = int(self.intervals_per_curve[c].item())
            intervals = []
            for k in range(self.max_intervals):
                if k >= 1 and k < n and self.c1_mask[c, k] and self.c1_mask[c, k + 1]:
                    J_k = self.joint_points[c, k]        # (dim,)
                    last_prev = raw_cp[c, k - 1, k_per - 1]  # (dim,)
                    c_k0 = 2.0 * J_k - last_prev         # (dim,) — C1 reflection
                    if k_per > 1:
                        derived = torch.cat([c_k0.unsqueeze(0), raw_cp[c, k, 1:]], dim=0)
                    else:
                        derived = c_k0.unsqueeze(0)
                    intervals.append(derived)
                else:
                    intervals.append(raw_cp[c, k])
            result_curves.append(torch.stack(intervals, dim=0))
        return torch.stack(result_curves, dim=0)

    def _assemble_interval_points(self) -> torch.Tensor:
        """
        Assembles the full per-interval control point tensor from joint_points and control_points.

        Returns:
            torch.Tensor: Shape (C, max_intervals, degree+1, dim)
        """
        effective_cp = self.get_effective_control_points()  # (C, max_intervals, k_per, dim)
        points = torch.cat(
            [
                self.joint_points[:, :-1].unsqueeze(2),   # Start joint points
                effective_cp,                              # Control points (C1-derived where applicable)
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
        k_per = self.__ctrl_pts_per_interval

        eff_cp = self.get_effective_control_points()  # (C, max_intervals, k_per, dim)

        lines = []
        for i in range(num_intervals + 1):
            if i > 0:
                jp = self.joint_points[batch, i, :].detach().numpy()
                cp = eff_cp[batch, i - 1, k_per - 1].detach().numpy()
                lines.append(np.array([jp, cp]))
            if i < num_intervals:
                jp = self.joint_points[batch, i, :].detach().numpy()
                cp = eff_cp[batch, i, 0].detach().numpy()
                lines.append(np.array([jp, cp]))

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

        # Export effective control points so derived (C1-constrained) slots are correct
        eff_cp = self.get_effective_control_points()        # (N, max_intervals, k_per, dim)
        ctrl = eff_cp.reshape(self.num_curves, self.max_intervals * self.__ctrl_pts_per_interval, self.num_dim)
        ctrl = ctrl.detach().cpu().numpy()

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

        # Export c1_mask as uint8 (0/1) so it survives round-trips through PLY / numpy
        c1 = self.c1_mask.cpu().numpy().astype(np.uint8)  # (N, max_intervals+1)
        for k in range(c1.shape[1]):
            result[f"{self.name}_c1_joint_{k}"] = c1[:, k]

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


    def split_intervals(self, split_mask: torch.Tensor, interval_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split specified intervals of specified curves using De Casteljau bisection at t=0.5.

        For each curve c where split_mask[c] is True, splits the interval at interval_indices[c]
        into two sub-intervals. The split is geometrically exact using the De Casteljau algorithm,
        preserving the curve shape while doubling the resolution at the chosen interval.

        Parameters:
            split_mask (torch.Tensor): bool (C,) — which curves to split
            interval_indices (torch.Tensor): long (C,) — which interval index to split per curve
                                             (values ignored where split_mask[c] is False)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - new_joint_points:        (C, new_max_intervals + 1, dim)
                - new_control_points:      (C, new_max_intervals * (degree-1), dim)
                - new_intervals_per_curve: (C,) long tensor

            Side effect: self.c1_mask is updated in-place to reflect the inserted junction.
            The new junction inherits C1 only if both its neighbours were already C1.
        """
        C = self.num_curves
        dim = self.num_dim
        degree = self.curve.degree
        k_per = self.__ctrl_pts_per_interval  # = degree - 1

        new_intervals_per_curve = self.intervals_per_curve.clone()
        new_intervals_per_curve[split_mask] += 1
        new_max = int(new_intervals_per_curve.max().item())

        jp = self.joint_points.detach()    # (C, old_max+1, dim)
        cp = self.control_points.detach()  # (C, old_max*k_per, dim)

        new_jp = torch.zeros(C, new_max + 1, dim, dtype=jp.dtype, device=jp.device)
        new_cp = torch.zeros(C, new_max * k_per, dim, dtype=cp.dtype, device=cp.device)

        for c in range(C):
            n = int(self.intervals_per_curve[c].item())  # current interval count

            if not split_mask[c]:
                # Copy active data; padding positions remain zero
                new_jp[c, :n + 1] = jp[c, :n + 1]
                new_cp[c, :n * k_per] = cp[c, :n * k_per]
            else:
                k = int(interval_indices[c].item())
                assert 0 <= k < n, f"interval_indices[{c}]={k} out of range [0, {n})"

                # Assemble the degree+1 control polygon points for interval k:
                #   P[0]       = joint_points[k]       (start joint)
                #   P[1..d-1]  = control_points[k*k_per .. (k+1)*k_per - 1]
                #   P[d]       = joint_points[k+1]     (end joint)
                pts = [jp[c, k]]
                for i in range(k_per):
                    pts.append(cp[c, k * k_per + i])
                pts.append(jp[c, k + 1])

                # De Casteljau pyramid at t=0.5: each level midpoints the previous
                pyramid = [pts]
                for r in range(1, degree + 1):
                    prev = pyramid[-1]
                    level = [(prev[i] + prev[i + 1]) * 0.5 for i in range(degree + 1 - r)]
                    pyramid.append(level)

                # Left sub-curve:  pyramid[r][0] for r = 0..degree
                # Right sub-curve: pyramid[degree-r][r] for r = 0..degree
                # Shared split joint: pyramid[degree][0]
                S = pyramid[degree][0]
                left_ctrl  = [pyramid[r][0] for r in range(1, degree)]      # k_per points
                right_ctrl = [pyramid[degree - r][r] for r in range(1, degree)]  # k_per points

                # Build new joint points: insert S between positions k and k+1
                new_jp[c, :k + 1]      = jp[c, :k + 1]
                new_jp[c, k + 1]       = S
                new_jp[c, k + 2:n + 2] = jp[c, k + 1:n + 1]

                # Build new control points:
                #   0 .. k*k_per-1        : unchanged (intervals before k)
                #   k*k_per .. (k+1)*k_per-1  : left sub-interval ctrl pts
                #   (k+1)*k_per .. (k+2)*k_per-1: right sub-interval ctrl pts
                #   (k+2)*k_per ..           : old intervals k+1..n-1 shifted right by one
                new_cp[c, :k * k_per] = cp[c, :k * k_per]
                new_cp[c, k * k_per:(k + 1) * k_per] = torch.stack(left_ctrl)
                new_cp[c, (k + 1) * k_per:(k + 2) * k_per] = torch.stack(right_ctrl)

                old_after_start = (k + 1) * k_per
                old_after_end   = n * k_per
                new_after_start = (k + 2) * k_per
                remaining = old_after_end - old_after_start
                if remaining > 0:
                    new_cp[c, new_after_start:new_after_start + remaining] = cp[c, old_after_start:old_after_end]

        # Build shifted c1_mask. The new junction inherits C1 only if both its
        # neighbours (the original left and right joints of the split interval) are C1.
        new_c1_mask = torch.zeros(C, new_max + 1, dtype=torch.bool, device=self.c1_mask.device)
        for c in range(C):
            n = int(self.intervals_per_curve[c].item())
            if not split_mask[c]:
                new_c1_mask[c, :n + 1] = self.c1_mask[c, :n + 1]
            else:
                k = int(interval_indices[c].item())
                # Junctions 0..k are unchanged
                new_c1_mask[c, :k + 1] = self.c1_mask[c, :k + 1]
                # New junction k+1: C1 only if both original neighbours were C1
                new_c1_mask[c, k + 1] = self.c1_mask[c, k] and self.c1_mask[c, k + 1]
                # Old junctions k+1..n shift right by one to k+2..n+1
                new_c1_mask[c, k + 2:n + 2] = self.c1_mask[c, k + 1:n + 1]

        self.c1_mask = new_c1_mask
        return new_jp, new_cp, new_intervals_per_curve

    def mask_curves_bounding_box(self, threshold: float):
        all_points = torch.cat([self.joint_points, self.control_points], dim=1)  # (N, K, dim)
        min_xyz = all_points.min(dim=1).values
        max_xyz = all_points.max(dim=1).values
        movement_extent = torch.norm(max_xyz - min_xyz, dim=1)  # (N,)

        return movement_extent > threshold
