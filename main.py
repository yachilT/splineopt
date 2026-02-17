from PyQt5 import QtWidgets # type: ignore
from GUI.editor import SplineEditor
import sys
import torch

from splines.spline import Spline
from splines.curve import Bezier
from train.train import Trainer

def main():
    # Initialize the Qt application
    app = QtWidgets.QApplication(sys.argv)
    dims = 2
    num_curves = 2

    # Varying intervals: curve 0 has 1 interval, curve 1 has 3 intervals
    intervals_per_curve = torch.tensor([1, 3])
    max_intervals = 3  # padded dimension

    # Joint points: (2, max_intervals+1=4, 2)
    # Curve 0 (1 interval): a tall arch from bottom-left to bottom-right
    # Curve 1 (3 intervals): an S-curve spanning the full canvas
    joint_points = torch.tensor([
        [  # Curve 0: single-segment arch
            [50.0, 50.0],
            [350.0, 50.0],
            [0.0, 0.0],    # padding
            [0.0, 0.0]     # padding
        ],
        [  # Curve 1: S-curve with 3 segments
            [50.0, 350.0],
            [150.0, 150.0],
            [250.0, 350.0],
            [350.0, 150.0]
        ]
    ], dtype=torch.float32)

    # Control points: (2, max_intervals * 2 = 6, 2)
    # Curve 0: control points pull the arch upward into a tall hump
    # Curve 1: control points create pronounced S-shaped wiggles
    control_points = torch.tensor([
        [  # Curve 0: 2 control points — tall arch
            [100.0, 400.0],
            [300.0, 400.0],
            [0.0, 0.0],    # padding
            [0.0, 0.0],    # padding
            [0.0, 0.0],    # padding
            [0.0, 0.0]     # padding
        ],
        [  # Curve 1: 6 control points — deep S-curve
            [70.0, 350.0],
            [90.0, 100.0],
            [180.0, 100.0],
            [220.0, 400.0],
            [310.0, 400.0],
            [330.0, 150.0]
        ]
    ], dtype=torch.float32)

    # Create target spline with varying intervals
    sample_spline = Spline(dims, intervals_per_curve, num_curves, Bezier(degree=3), joint_points, control_points)

    # Generate sample points from the target spline
    sample_points = Trainer.generate_sample_points(sample_spline, num_points=50, add_noise=True, noise_std=5.0)

    # Create a random initial spline to optimize (also with varying intervals)
    rand_joints = torch.rand((2, max_intervals + 1, 2)) * 300 + 50
    rand_controls = torch.rand((2, max_intervals * 2, 2)) * 300 + 50

    spline = Spline(dims, intervals_per_curve, num_curves, Bezier(3), rand_joints, rand_controls)

    # Create the spline editor
    editor = SplineEditor(spline, sample_points)
    editor.show()

    # Optimize the spline
    def optimize():
        trainer = Trainer(spline, sample_points, num_iterations=5000, learning_rate=2, editor=editor)
        trainer.optimize()
        editor.update_spline()

    optimize()

    # Run the Qt application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
