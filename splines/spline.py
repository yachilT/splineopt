from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from .curve import Bezier, Curve


class Spline(nn.Module):
    def __init__(self, num_dim: int, num_intervals: Union[int, torch.Tensor], num_curves: int = 1, curve: Curve = Bezier(degree=3), joint_points: Optional[torch.Tensor] = None, control_points: Optional[torch.Tensor] = None, c1_mask: Optional[torch.Tensor] = None, interval_widths: Optional[torch.Tensor] = None, name: str = ""):
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
            interval_widths (torch.Tensor, optional): Float tensor of shape (num_curves, max_intervals).
                interval_widths[c, k] is the fraction of [0, 1] owned by interval k of curve c.
                Valid slots for each curve must sum to 1.0. Padding slots should be 0.
                Defaults to uniform widths (1 / intervals_per_curve[c] for each valid slot).
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

        # Interval widths: fraction of [0, 1] owned by each interval.
        # Defaults to uniform (1/n for each valid slot, 0 for padding).
        if interval_widths is None:
            interval_widths = torch.zeros(num_curves, self.max_intervals, dtype=torch.float32)
            for c in range(num_curves):
                n = int(self.intervals_per_curve[c].item())
                interval_widths[c, :n] = 1.0 / n
        if interval_widths.shape != (num_curves, self.max_intervals):
            raise ValueError(f"interval_widths must have shape ({num_curves}, {self.max_intervals}), got {interval_widths.shape}")
        self.register_buffer('interval_widths', interval_widths)

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
        K = self.max_intervals
        device = self.control_points.device

        # (C, K, k_per, dim)
        raw_cp = self.control_points.view(C, K, k_per, self.num_dim)

        if not self.c1_mask.any():
            return raw_cp

        # Vectorized derive mask: derive_mask[c, k] = True when interval k of curve c
        # has a C1-derived first control point.
        # Conditions: k >= 1, k < intervals_per_curve[c], c1_mask[c,k], c1_mask[c,k+1]
        k_range = torch.arange(K, device=device)          # (K,)
        ipc = self.intervals_per_curve.to(device)          # (C,)
        interval_valid = k_range.unsqueeze(0) < ipc.unsqueeze(1)   # (C, K)
        internal       = k_range.unsqueeze(0) >= 1                  # (1, K) → broadcasts

        c1 = self.c1_mask.to(device)              # (C, K+1)
        left_smooth  = c1[:, :K]                  # (C, K) — c1_mask[c, k]
        right_smooth = c1[:, 1:]                   # (C, K) — c1_mask[c, k+1]

        derive_mask = internal & interval_valid & left_smooth & right_smooth  # (C, K)

        if not derive_mask.any():
            return raw_cp

        # Derived first CP: 2 * J_k - C_{k-1, last}
        # J_k = joint_points[:, k, :]  →  joint_points[:, :K, :]  (C, K, dim)
        J = self.joint_points[:, :K, :]  # (C, K, dim)

        # C_{k-1, last}: raw_cp[:, k-1, k_per-1, :] for k=1..K-1; dummy zeros for k=0
        prev_last = torch.cat([
            torch.zeros(C, 1, self.num_dim, device=device),  # dummy for k=0 (never used)
            raw_cp[:, :-1, k_per - 1, :],                    # k=1..K-1
        ], dim=1)  # (C, K, dim)

        c1_first = 2.0 * J - prev_last  # (C, K, dim)

        # Replace raw_cp[:, :, 0, :] where derive_mask is True
        new_first = torch.where(
            derive_mask.unsqueeze(-1),  # (C, K, 1) → broadcasts to (C, K, dim)
            c1_first,
            raw_cp[:, :, 0, :],
        )  # (C, K, dim)

        if k_per == 1:
            return new_first.unsqueeze(2)  # (C, K, 1, dim)

        return torch.cat([new_first.unsqueeze(2), raw_cp[:, :, 1:, :]], dim=2)  # (C, K, k_per, dim)

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

    def map_to_local(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps global parameter t in [0, 1] to per-curve (interval_idx, local_t, width).

        Uses cumulative sum of interval_widths to support non-uniform interval spacing.

        Parameters:
            t (torch.Tensor): Shape (N,), values in [0, 1].

        Returns:
            Tuple of:
                - interval_idx (torch.Tensor): Shape (C, N) long — which interval each query falls in.
                - local_t (torch.Tensor): Shape (C, N) float — local parameter within the interval [0, 1].
                - widths (torch.Tensor): Shape (C, N) float — width of the interval each query falls in.
        """
        device = t.device
        C = self.num_curves
        N = t.shape[0]

        w = self.interval_widths.to(device)              # (C, max_intervals)
        cum_w = torch.cumsum(w, dim=1)                   # (C, max_intervals) — right edges

        # searchsorted: for each curve, find the first interval whose right edge > t
        # t_expanded: (C, N)
        t_expanded = t.unsqueeze(0).expand(C, N)
        interval_idx = torch.searchsorted(cum_w, t_expanded.contiguous())  # (C, N)

        # Clamp to valid range [0, intervals_per_curve[c] - 1]
        max_idx = (self.intervals_per_curve.to(device) - 1).unsqueeze(1)  # (C, 1)
        interval_idx = torch.clamp(interval_idx, max=max_idx.expand_as(interval_idx))

        # Left edge of each interval: cum_w shifted right by 1 (0 for k=0)
        cum_w_padded = torch.cat([torch.zeros(C, 1, device=device), cum_w], dim=1)  # (C, max_intervals+1)
        left_edges = torch.gather(cum_w_padded, 1, interval_idx)     # (C, N)
        widths = torch.gather(w, 1, interval_idx)                     # (C, N)

        # Local t: (global_t - left_edge) / width, clamped to [0, 1]
        local_t = (t_expanded - left_edges) / widths.clamp(min=1e-12)
        local_t = local_t.clamp(0.0, 1.0)

        return interval_idx, local_t, widths

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

        # Step 1: Per-curve interval mapping (non-uniform widths)
        interval_idx, local_t, _widths = self.map_to_local(t)

        # Step 2: Compute t_powers: (C, N, degree+1)
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

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the first derivative (velocity) of the spline w.r.t. the global
        parameter t in [0, 1].

        Uses the same interval mapping and characteristic matrix as forward(), but
        replaces t_powers with d_t_powers:
            d/dt [t^0, t^1, ..., t^D] = [0, 1, 2t, ..., D*t^(D-1)]
        which is curve-type-agnostic — char_mat handles all curve-specific math.
        Multiplies by 1/width (chain rule) to convert local to global.

        Parameters:
            t (torch.Tensor): Shape (N,) or (N, 1), values in [0, 1].

        Returns:
            torch.Tensor: Shape (C, N, dim) — velocity on each curve at each query point.
        """
        if t.dim() == 2 and t.size(1) == 1:
            t = t.view(-1)
        elif t.dim() != 1:
            raise ValueError(f"Expected 1D or 2D tensor with shape (n,1) for `t`, got {t.shape}")

        N = t.shape[0]
        device = t.device
        D = self.curve.degree

        # Same interval mapping as forward()
        interval_idx, local_t, widths = self.map_to_local(t)

        # Derivative t-powers: d/dt[t^i] = i * t^(i-1), with d/dt[t^0] = 0
        # Computed as: coeffs[i] * local_t^shifted_exps[i]
        #   coeffs        = [0, 1, 2, ..., D]
        #   shifted_exps  = [0, 0, 1, ..., D-1]  (index 0 is killed by coeff=0)
        coeffs = torch.arange(D + 1, dtype=torch.float32, device=device)  # (D+1,)
        shifted_exps = torch.cat([
            torch.zeros(1, dtype=torch.float32, device=device),
            torch.arange(D, dtype=torch.float32, device=device),
        ])  # (D+1,)
        d_t_powers = coeffs * local_t.unsqueeze(-1).pow(shifted_exps)  # (C, N, D+1)

        # Apply characteristic matrix (curve-type-specific, same as forward)
        char_mat = self.curve.char_mat.to(device)
        t_transformed = d_t_powers @ char_mat  # (C, N, D+1)

        # Gather interval control points (same as forward)
        points = self._assemble_interval_points()  # (C, max_intervals, D+1, dim)
        idx = interval_idx.unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, -1, D + 1, self.num_dim)
        gathered_points = torch.gather(points, dim=1, index=idx)  # (C, N, D+1, dim)

        result = torch.einsum('cnk,cnkd->cnd', t_transformed, gathered_points)  # (C, N, dim)

        # Chain rule: d/dt_global = (1 / width) * d/dt_local
        result = result * (1.0 / widths.clamp(min=1e-12)).unsqueeze(-1)

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

        # Export interval_widths
        iw = self.interval_widths.cpu().numpy()  # (N, max_intervals)
        for k in range(iw.shape[1]):
            result[f"{self.name}_interval_width_{k}"] = iw[:, k]

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
        eff_cp = self.get_effective_control_points().detach()  # (C, old_max, k_per, dim)

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

                # Assemble the degree+1 control polygon using EFFECTIVE control points
                # (C1-derived where applicable) so De Casteljau operates on the true curve.
                pts = [jp[c, k]]
                for i in range(k_per):
                    pts.append(eff_cp[c, k, i])
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

                # --- C1-compatible tangent adjustment (3/4 rule) ---
                # De Casteljau bisection halves the tangent at external junctions,
                # but C1 (reflection formula) expects the original tangent length.
                # We meet in the middle: scale both sides to 3/4 of the original
                # tangent so that C1 is satisfied at both external junctions.
                #
                # At junction k (start): if C1 was deriving interval k's first CP,
                #   adjust the previous interval's last CP so tangent = 3/4 * V.
                #   C1 then derives the left sub's first CP to match.
                c1_at_k = (
                    k >= 1
                    and bool(self.c1_mask[c, k].item())
                    and bool(self.c1_mask[c, k + 1].item())
                )
                if c1_at_k:
                    prev_last = cp[c, (k - 1) * k_per + k_per - 1]
                    J_k = jp[c, k]
                    new_cp[c, (k - 1) * k_per + k_per - 1] = (J_k + 3.0 * prev_last) / 4.0

                # At junction k+1 (end): if C1 was deriving old interval k+1's
                #   first CP, adjust the right sub's last CP so tangent = 3/4 * W.
                #   C1 then derives the next interval's first CP to match.
                c1_at_k1 = (
                    k + 1 >= 1 and k + 1 < n
                    and bool(self.c1_mask[c, k + 1].item())
                    and bool(self.c1_mask[c, k + 2].item())
                )
                if c1_at_k1:
                    eff_last_k = eff_cp[c, k, k_per - 1]
                    J_k1 = jp[c, k + 1]
                    new_cp[c, (k + 1) * k_per + k_per - 1] = (J_k1 + 3.0 * eff_last_k) / 4.0

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
                # New junction k+1: inherit C1 from meaningful neighbours.
                # Boundary junctions (0 and n) are not meaningful for C1, so:
                #   - First interval (k=0): inherit from inner neighbour (k+1)
                #   - Last interval (k=n-1): inherit from inner neighbour (k)
                #   - Middle interval: AND of both neighbours
                #   - Single interval (n=1): both are boundaries → False
                if n == 1:
                    new_c1_mask[c, k + 1] = False
                elif k == 0:
                    new_c1_mask[c, k + 1] = bool(self.c1_mask[c, k + 1].item())
                elif k == n - 1:
                    new_c1_mask[c, k + 1] = bool(self.c1_mask[c, k].item())
                else:
                    new_c1_mask[c, k + 1] = bool(self.c1_mask[c, k].item()) and bool(self.c1_mask[c, k + 1].item())
                # Old junctions k+1..n shift right by one to k+2..n+1
                new_c1_mask[c, k + 2:n + 2] = self.c1_mask[c, k + 1:n + 1]

        self.c1_mask = new_c1_mask

        # Build shifted interval_widths: split interval k's width is halved into two.
        old_w = self.interval_widths  # (C, old_max)
        new_w = torch.zeros(C, new_max, dtype=old_w.dtype, device=old_w.device)
        for c in range(C):
            n = int(self.intervals_per_curve[c].item())
            if not split_mask[c]:
                new_w[c, :n] = old_w[c, :n]
            else:
                k = int(interval_indices[c].item())
                # Intervals 0..k-1 are unchanged
                new_w[c, :k] = old_w[c, :k]
                # Split interval k into two halves
                half_w = old_w[c, k] * 0.5
                new_w[c, k] = half_w
                new_w[c, k + 1] = half_w
                # Old intervals k+1..n-1 shift right by one
                if k + 1 < n:
                    new_w[c, k + 2:n + 1] = old_w[c, k + 1:n]

        self.interval_widths = new_w
        return new_jp, new_cp, new_intervals_per_curve

    def compute_control_polygon_lengths(self) -> torch.Tensor:
        """
        Computes the control polygon length for each interval of each curve as a
        proxy for arc length.

        Uses effective control points (C1 continuity enforced via
        get_effective_control_points) so that derived control points at smooth
        junctions are reflected correctly in the length estimate.

        The control polygon for a degree-d Bezier interval consists of d+1 points:
            P0 = joint_point[i],  P1..P_{d-1} = control_points,  P_d = joint_point[i+1]
        Its total chord length  sum_k ||P_{k+1} - P_k||  is an upper bound on the
        true arc length and is a reliable proxy for how far the Gaussian travels
        within that interval.

        Returns:
            torch.Tensor: Shape (num_curves, max_intervals) on the same device as
                          joint_points.  Padding slots (i >= intervals_per_curve[c])
                          are set to zero.
        """
        # (C, max_intervals, degree+1, dim) — already uses get_effective_control_points
        points = self._assemble_interval_points()

        # Segment differences along the control polygon: (C, max_intervals, degree, dim)
        seg_diffs = points[:, :, 1:, :] - points[:, :, :-1, :]

        # Euclidean length of each segment: (C, max_intervals, degree)
        seg_lengths = seg_diffs.norm(dim=-1)

        # Sum over the degree segments to get total polygon length per interval: (C, max_intervals)
        arc_lengths = seg_lengths.sum(dim=-1)

        # Zero out padding slots (intervals that don't exist for a given curve)
        device = arc_lengths.device
        k_range = torch.arange(self.max_intervals, device=device)  # (max_intervals,)
        ipc = self.intervals_per_curve.to(device)                   # (C,)
        valid_mask = k_range.unsqueeze(0) < ipc.unsqueeze(1)        # (C, max_intervals)
        arc_lengths = arc_lengths * valid_mask.float()

        return arc_lengths

    def mask_curves_bounding_box(self, threshold: float):
        all_points = torch.cat([self.joint_points, self.control_points], dim=1)  # (N, K, dim)
        min_xyz = all_points.min(dim=1).values
        max_xyz = all_points.max(dim=1).values
        movement_extent = torch.norm(max_xyz - min_xyz, dim=1)  # (N,)

        return movement_extent > threshold
