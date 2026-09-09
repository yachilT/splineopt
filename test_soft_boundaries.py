"""
Tests for the soft-boundary (trapezoidal POU) blending in Spline.

Hard path (soft_boundaries=False) is unchanged and bit-identical to before.
Soft path adds a 3-candidate stencil with a trapezoidal mask of relative
half-width alpha*min(w_k, w_{k+1}) around each joint.
"""
import torch
from splines.spline import Spline
from splines.curve import Bezier


def _rand_spline(num_intervals=4, num_curves=1, dim=1, seed=0,
                 trainable_widths=False, soft_boundaries=False, alpha=0.1):
    torch.manual_seed(seed)
    s = Spline(
        num_dim=dim, num_intervals=num_intervals, num_curves=num_curves,
        curve=Bezier(degree=3),
        trainable_widths=trainable_widths,
        soft_boundaries=soft_boundaries,
        soft_boundary_alpha=alpha,
    )
    with torch.no_grad():
        s.joint_points.add_(torch.randn_like(s.joint_points) * 0.5)
        s.control_points.add_(torch.randn_like(s.control_points) * 0.5)
    return s


def test_hard_path_unchanged():
    s_hard = _rand_spline(num_intervals=5, seed=1)
    s_soft = _rand_spline(num_intervals=5, seed=1, soft_boundaries=True, alpha=0.1)
    # Same construction so all parameters match; only the flag differs.
    assert torch.allclose(s_hard.joint_points, s_soft.joint_points)
    assert torch.allclose(s_hard.control_points, s_soft.control_points)

    t = torch.linspace(0.0, 1.0, 100)
    y_hard = s_hard(t)
    y_soft = s_soft(t)
    # The soft path differs from the hard path only inside ε-bands. We compute
    # the joint t-values and the half-band width, then check identity outside.
    joints_t = s_hard.joint_t_values()[0, 1:-1]  # internal joints only
    widths = s_hard.interval_widths[0]
    alpha = s_soft.soft_boundary_alpha
    half_eps = alpha * torch.minimum(widths[:-1], widths[1:])  # at joint between k, k+1
    # A query t is "outside any band" if it is > half_eps away from every internal joint.
    far_mask = torch.ones_like(t, dtype=torch.bool)
    for j_t, he in zip(joints_t.tolist(), half_eps.tolist()):
        far_mask &= (t - j_t).abs() > he + 1e-6
    assert torch.allclose(y_hard[:, far_mask, :], y_soft[:, far_mask, :], atol=1e-6), \
        "Soft path must equal hard path outside ε-bands"
    print(f"PASS: hard path unchanged outside ε-bands ({int(far_mask.sum())}/{len(t)} queries checked)")


def test_partition_of_unity():
    s = _rand_spline(num_intervals=5, seed=2, soft_boundaries=True, alpha=0.15)
    t = torch.linspace(0.001, 0.999, 200)
    _, _, mask, _ = s._map_to_local_soft(t)            # (C, N, 3)
    mask_sum = mask.sum(dim=2)                          # (C, N)
    assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-5), \
        f"Mask must sum to 1; got min {mask_sum.min().item()}, max {mask_sum.max().item()}"
    print(f"PASS: partition of unity (mask sum in [{mask_sum.min().item():.6f}, {mask_sum.max().item():.6f}])")


def test_alpha_zero_matches_hard():
    # At very small alpha, the ε-band shrinks toward zero, so soft path → hard path
    # everywhere except a vanishing measure.
    s_hard = _rand_spline(num_intervals=4, seed=3)
    s_soft = _rand_spline(num_intervals=4, seed=3, soft_boundaries=True, alpha=1e-5)
    # Pick t values that intentionally avoid the tiny band.
    t = torch.linspace(0.01, 0.99, 50)
    y_hard = s_hard(t)
    y_soft = s_soft(t)
    assert torch.allclose(y_hard, y_soft, atol=1e-4), \
        f"max diff {(y_hard - y_soft).abs().max().item()}"
    print("PASS: soft path at α→0 matches hard path")


def test_endpoints():
    s = _rand_spline(num_intervals=4, seed=4, soft_boundaries=True, alpha=0.1)
    t = torch.tensor([0.0, 1.0 - 1e-6])
    y = s(t)
    # t=0: should hit first joint exactly (only candidate m=0 has mask 1; left
    # ramp is hard step).
    assert torch.allclose(y[:, 0, :], s.joint_points[:, 0, :], atol=1e-5), \
        f"t=0 output {y[:, 0, :]} vs joint {s.joint_points[:, 0, :]}"
    # t≈1: should hit last valid joint.
    last = int(s.intervals_per_curve[0].item())
    assert torch.allclose(y[:, 1, :], s.joint_points[:, last, :], atol=1e-3), \
        f"t≈1 output {y[:, 1, :]} vs last joint {s.joint_points[:, last, :]}"
    print("PASS: endpoints reach correct joints")


def test_selection_gradient_exists_and_differs():
    # With trainable widths and the soft path, the gradient on _width_logits
    # should differ from the hard path — because the soft path injects an
    # additional "selection" component that the hard path lacks.
    s_hard = _rand_spline(num_intervals=6, seed=5, trainable_widths=True)
    s_soft = _rand_spline(num_intervals=6, seed=5, trainable_widths=True,
                          soft_boundaries=True, alpha=0.15)
    # Make params (including _width_logits) identical.
    with torch.no_grad():
        s_soft._width_logits.copy_(s_hard._width_logits)

    t = torch.linspace(0.0, 1.0, 200)
    # Pick a target with a sharp feature that prefers redistribution.
    target = torch.sin(8.0 * t).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

    y_hard = s_hard(t)
    loss_hard = ((y_hard - target) ** 2).mean()
    loss_hard.backward()
    g_hard = s_hard._width_logits.grad.detach().clone()

    y_soft = s_soft(t)
    loss_soft = ((y_soft - target) ** 2).mean()
    loss_soft.backward()
    g_soft = s_soft._width_logits.grad.detach().clone()

    diff = (g_soft - g_hard).abs().max().item()
    g_hard_norm = g_hard.norm().item()
    g_soft_norm = g_soft.norm().item()
    print(f"  hard widths grad norm: {g_hard_norm:.6f}")
    print(f"  soft widths grad norm: {g_soft_norm:.6f}")
    print(f"  max |soft - hard| component: {diff:.6f}")
    assert torch.isfinite(g_soft).all(), "Soft widths gradient must be finite"
    assert g_soft_norm > 0, "Soft widths gradient should be non-zero"
    assert diff > 1e-6, "Soft path should change the widths gradient (selection signal)"
    print("PASS: soft path produces a non-trivially different widths gradient")


def test_no_nan_with_varying_intervals():
    # Mix curves with different interval counts — exercises the padding NaN trap.
    intervals = torch.tensor([1, 3, 4])
    max_int = 4
    C = 3
    s = Spline(
        num_dim=2, num_intervals=intervals, num_curves=C,
        curve=Bezier(degree=3),
        trainable_widths=True,
        soft_boundaries=True,
        soft_boundary_alpha=0.12,
    )
    with torch.no_grad():
        s.joint_points.add_(torch.randn_like(s.joint_points) * 0.5)
        s.control_points.add_(torch.randn_like(s.control_points) * 0.5)
    t = torch.linspace(0.0, 1.0, 64)
    y = s(t)
    assert y.shape == (C, 64, 2)
    assert torch.isfinite(y).all(), "Output must not contain NaN/Inf with padding"
    # Backward through soft path must also stay finite on widths.
    y.pow(2).mean().backward()
    assert torch.isfinite(s._width_logits.grad).all()
    print("PASS: varying intervals + padding produces finite outputs and gradients")


def test_k0_short_circuit():
    s = Spline(num_dim=2, num_intervals=0, num_curves=1, curve=Bezier(degree=3),
               soft_boundaries=True, soft_boundary_alpha=0.1)
    t = torch.linspace(0.0, 1.0, 10)
    y = s(t)
    assert y.shape == (1, 10, 2)
    # All outputs should be the single joint point.
    assert torch.allclose(y - y[:, :1, :], torch.zeros_like(y), atol=1e-7)
    print("PASS: K=0 short-circuit still works with soft_boundaries=True")


def test_derivative_finite():
    s = _rand_spline(num_intervals=4, seed=7, soft_boundaries=True, alpha=0.1)
    t = torch.linspace(0.01, 0.99, 50)
    d = s.derivative(t)
    assert torch.isfinite(d).all()
    print("PASS: soft-path derivative finite")


if __name__ == "__main__":
    test_hard_path_unchanged()
    test_partition_of_unity()
    test_alpha_zero_matches_hard()
    test_endpoints()
    test_selection_gradient_exists_and_differs()
    test_no_nan_with_varying_intervals()
    test_k0_short_circuit()
    test_derivative_finite()
    print("\nAll soft-boundary tests passed.")
