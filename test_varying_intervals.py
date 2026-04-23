"""
Test varying interval support for the Spline module.

Tests:
1. Backward compatibility: uniform intervals via int produce same results
2. Varying intervals: curves with different interval counts evaluate correctly
3. Gradient flow: gradients propagate through the new forward pass
"""
import torch
from splines.Spline import Spline
from splines.curve import Bezier


def test_uniform_intervals_backward_compat():
    """Passing num_intervals as int should work exactly as before."""
    dims = 2
    num_curves = 2
    num_intervals = 2

    joint_points = torch.tensor([
        [[100.0, 200.0], [200.0, 300.0], [300.0, 100.0]],
        [[50.0, 150.0], [150.0, 250.0], [250.0, 150.0]]
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[-100.0, 50.0], [-130.0, 350.0], [400.0, 200.0], [200.0, 200.0]],
        [[40.0, 160.0], [130.0, 260.0], [240.0, 140.0], [260.0, 130.0]]
    ], dtype=torch.float32)

    spline = Spline(dims, num_intervals, num_curves, Bezier(3), joint_points, control_points)

    # Check attributes
    assert spline.max_intervals == 2
    assert spline.num_intervals == 2  # backward compat property
    assert torch.all(spline.intervals_per_curve == 2)

    # Evaluate
    t = torch.linspace(0, 1, 20)
    result = spline(t)
    assert result.shape == (2, 20, 2), f"Expected (2, 20, 2), got {result.shape}"

    # Check endpoints: t=0 should give first joint point, t≈1 should give last
    t_endpoints = torch.tensor([0.0, 0.999999])
    endpoints = spline(t_endpoints)
    assert torch.allclose(endpoints[:, 0, :], joint_points[:, 0, :], atol=1e-3), "t=0 should give first joint point"
    assert torch.allclose(endpoints[:, 1, :], joint_points[:, -1, :], atol=1e-2), "t≈1 should give last joint point"

    print("PASS: uniform intervals backward compatibility")


def test_uniform_via_tensor():
    """Passing num_intervals as a uniform tensor should give same results as int."""
    dims = 2
    num_curves = 2

    joint_points = torch.tensor([
        [[100.0, 200.0], [200.0, 300.0], [300.0, 100.0]],
        [[50.0, 150.0], [150.0, 250.0], [250.0, 150.0]]
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[-100.0, 50.0], [-130.0, 350.0], [400.0, 200.0], [200.0, 200.0]],
        [[40.0, 160.0], [130.0, 260.0], [240.0, 140.0], [260.0, 130.0]]
    ], dtype=torch.float32)

    spline_int = Spline(dims, 2, num_curves, Bezier(3), joint_points, control_points)
    spline_tensor = Spline(dims, torch.tensor([2, 2]), num_curves, Bezier(3), joint_points.clone(), control_points.clone())

    t = torch.linspace(0, 1, 50)
    result_int = spline_int(t)
    result_tensor = spline_tensor(t)

    assert torch.allclose(result_int, result_tensor, atol=1e-5), "Tensor and int num_intervals should give same results"
    print("PASS: uniform tensor gives same results as int")


def test_varying_intervals():
    """Two curves with different interval counts."""
    dims = 2
    num_curves = 2
    # Curve 0: 1 interval, Curve 1: 2 intervals
    intervals = torch.tensor([1, 2])
    max_intervals = 2

    # Joint points padded to max_intervals+1 = 3
    # Curve 0 (1 interval): uses joints [0] and [1], joint [2] is padding
    # Curve 1 (2 intervals): uses all 3 joints
    joint_points = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],   # curve 0: only first 2 matter
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],    # curve 1: all 3 matter
    ], dtype=torch.float32)

    # Control points padded to max_intervals * K = 2 * 2 = 4
    # Curve 0 (1 interval): uses ctrl [0] and [1], rest is padding
    # Curve 1 (2 intervals): uses all 4
    control_points = torch.tensor([
        [[0.33, 0.5], [0.66, 0.5], [0.0, 0.0], [0.0, 0.0]],  # curve 0
        [[0.15, 0.5], [0.35, 0.8], [0.65, 0.8], [0.85, 0.5]], # curve 1
    ], dtype=torch.float32)

    spline = Spline(dims, intervals, num_curves, Bezier(3), joint_points, control_points)

    assert spline.max_intervals == 2
    assert spline.intervals_per_curve[0] == 1
    assert spline.intervals_per_curve[1] == 2

    t = torch.linspace(0, 1, 30)
    result = spline(t)
    assert result.shape == (2, 30, 2), f"Expected (2, 30, 2), got {result.shape}"

    # Verify endpoints
    t_endpoints = torch.tensor([0.0, 0.999999])
    endpoints = spline(t_endpoints)

    # Curve 0: t=0 -> joint[0]=(0,0), t≈1 -> joint[1]=(1,1)
    assert torch.allclose(endpoints[0, 0], joint_points[0, 0], atol=1e-3)
    assert torch.allclose(endpoints[0, 1], joint_points[0, 1], atol=1e-2)

    # Curve 1: t=0 -> joint[0]=(0,0), t≈1 -> joint[2]=(1,0)
    assert torch.allclose(endpoints[1, 0], joint_points[1, 0], atol=1e-3)
    assert torch.allclose(endpoints[1, 1], joint_points[1, 2], atol=1e-2)

    print("PASS: varying intervals")


def test_gradient_flow():
    """Verify gradients flow through the new forward pass."""
    dims = 2
    num_curves = 2
    intervals = torch.tensor([1, 2])

    joint_points = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[0.33, 0.5], [0.66, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[0.15, 0.5], [0.35, 0.8], [0.65, 0.8], [0.85, 0.5]],
    ], dtype=torch.float32)

    spline = Spline(dims, intervals, num_curves, Bezier(3), joint_points, control_points)

    t = torch.linspace(0, 1, 20)
    result = spline(t)

    # Create a simple loss and backprop
    target = torch.randn_like(result)
    loss = ((result - target) ** 2).mean()
    loss.backward()

    assert spline.joint_points.grad is not None, "joint_points should have gradients"
    assert spline.control_points.grad is not None, "control_points should have gradients"
    assert spline.joint_points.grad.abs().sum() > 0, "joint_points gradients should be non-zero"
    assert spline.control_points.grad.abs().sum() > 0, "control_points gradients should be non-zero"

    print("PASS: gradient flow")


def test_single_curve_single_interval():
    """Edge case: 1 curve, 1 interval."""
    dims = 2
    joint_points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    control_points = torch.tensor([[[0.3, 0.8], [0.7, 0.2]]], dtype=torch.float32)

    spline = Spline(dims, 1, 1, Bezier(3), joint_points, control_points)

    t = torch.linspace(0, 1, 10)
    result = spline(t)
    assert result.shape == (1, 10, 2)

    print("PASS: single curve single interval")


def test_split_preserves_shape():
    """Splitting an interval should not change the evaluated curve shape."""
    dims = 2
    num_curves = 2
    intervals = torch.tensor([1, 2])

    joint_points = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[0.33, 0.5], [0.66, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[0.15, 0.5], [0.35, 0.8], [0.65, 0.8], [0.85, 0.5]],
    ], dtype=torch.float32)

    spline = Spline(dims, intervals, num_curves, Bezier(3), joint_points, control_points)

    t = torch.linspace(0, 1, 200)
    before = spline(t).detach().clone()

    # Split interval 0 of curve 1
    split_mask = torch.tensor([False, True])
    interval_indices = torch.tensor([0, 0])
    new_jp, new_cp, new_ipc = spline.split_intervals(split_mask, interval_indices)

    # Rebuild spline with new geometry
    spline2 = Spline(dims, new_ipc, num_curves, Bezier(3), new_jp, new_cp,
                      c1_mask=spline.c1_mask, interval_widths=spline.interval_widths)

    after = spline2(t).detach()

    assert torch.allclose(before, after, atol=1e-4), \
        f"Split changed curve shape! Max diff: {(before - after).abs().max().item()}"
    print("PASS: split preserves shape")


def test_split_preserves_c1_with_tangent_adjustment():
    """Splitting with C1 active should maintain C1 continuity and have bounded shape deviation."""
    dims = 2
    num_curves = 1
    intervals = torch.tensor([3])

    joint_points = torch.tensor([
        [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]],
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[0.3, 1.0], [0.7, 1.5], [1.3, 2.5], [1.7, 1.5], [2.3, 1.0], [2.7, 2.5]],
    ], dtype=torch.float32)

    c1_mask = torch.zeros(1, 4, dtype=torch.bool)
    c1_mask[0, 1:] = True  # C1 at all internal junctions

    spline = Spline(dims, intervals, num_curves, Bezier(3), joint_points, control_points, c1_mask=c1_mask)

    t = torch.linspace(0, 1, 200)
    before = spline(t).detach().clone()

    split_mask = torch.tensor([True])
    interval_indices = torch.tensor([1])  # split the middle interval
    new_jp, new_cp, new_ipc = spline.split_intervals(split_mask, interval_indices)

    spline2 = Spline(dims, new_ipc, num_curves, Bezier(3), new_jp, new_cp,
                      c1_mask=spline.c1_mask, interval_widths=spline.interval_widths)

    after = spline2(t).detach()

    # Shape deviation is bounded (3/4 tangent adjustment causes small changes)
    max_diff = (before - after).abs().max().item()
    assert max_diff < 0.1, f"Shape deviation too large: {max_diff}"

    # Verify C1 continuity is maintained at all internal junctions:
    # At each junction, the tangent direction from left and right should match.
    eff_cp2 = spline2.get_effective_control_points().detach()
    n2 = int(new_ipc[0].item())
    k_per = spline2.curve.degree - 1
    for j in range(1, n2):
        J = spline2.joint_points[0, j].detach()
        left_tangent = J - eff_cp2[0, j - 1, k_per - 1]    # from prev interval
        right_tangent = eff_cp2[0, j, 0] - J                 # from next interval
        # Directions should match (cross product ≈ 0 for 2D)
        cross = left_tangent[0] * right_tangent[1] - left_tangent[1] * right_tangent[0]
        assert abs(cross.item()) < 1e-4, \
            f"C1 broken at junction {j}: cross product = {cross.item()}"

    print(f"PASS: split preserves C1 with tangent adjustment (max shape diff: {max_diff:.4f})")


def test_split_preserves_derivative():
    """Splitting should not change the derivative (velocity) either."""
    dims = 2
    num_curves = 1
    intervals = torch.tensor([2])

    joint_points = torch.tensor([
        [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]],
    ], dtype=torch.float32)

    control_points = torch.tensor([
        [[0.3, 1.0], [0.7, 1.5], [1.5, 2.5], [2.5, 1.5]],
    ], dtype=torch.float32)

    spline = Spline(dims, intervals, num_curves, Bezier(3), joint_points, control_points)

    t = torch.linspace(0.01, 0.99, 100)  # avoid exact endpoints
    deriv_before = spline.derivative(t).detach().clone()

    split_mask = torch.tensor([True])
    interval_indices = torch.tensor([1])
    new_jp, new_cp, new_ipc = spline.split_intervals(split_mask, interval_indices)

    spline2 = Spline(dims, new_ipc, num_curves, Bezier(3), new_jp, new_cp,
                      c1_mask=spline.c1_mask, interval_widths=spline.interval_widths)

    deriv_after = spline2.derivative(t).detach()

    assert torch.allclose(deriv_before, deriv_after, atol=1e-3), \
        f"Split changed derivative! Max diff: {(deriv_before - deriv_after).abs().max().item()}"
    print("PASS: split preserves derivative")


if __name__ == '__main__':
    test_uniform_intervals_backward_compat()
    test_uniform_via_tensor()
    test_varying_intervals()
    test_gradient_flow()
    test_single_curve_single_interval()
    test_split_preserves_shape()
    test_split_preserves_c1_with_tangent_adjustment()
    test_split_preserves_derivative()
    print("\nAll tests passed!")
