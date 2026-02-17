"""
Test varying interval support for the Spline module.

Tests:
1. Backward compatibility: uniform intervals via int produce same results
2. Varying intervals: curves with different interval counts evaluate correctly
3. Gradient flow: gradients propagate through the new forward pass
"""
import torch
from splines.spline import Spline
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


if __name__ == '__main__':
    test_uniform_intervals_backward_compat()
    test_uniform_via_tensor()
    test_varying_intervals()
    test_gradient_flow()
    test_single_curve_single_interval()
    print("\nAll tests passed!")
