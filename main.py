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

    batch_size = 2

    # Joint points: (batch_size, 3, 2)
    joint_points = torch.tensor([
        [  # First spline
            [100.0, 200.0],
            [200.0, 300.0],
            [300.0, 100.0]
        ],
        [  # Second spline
            [50.0, 150.0],
            [150.0, 250.0],
            [250.0, 150.0]
        ]
    ], dtype=torch.float32)

    # Control points: (batch_size, 4, 2)
    control_points = torch.tensor([
        [  # First spline's control points
            [-100.0, 50.0],
            [-130.0, 350.0],
            [400.0, 200.0],
            [200.0, 200.0]
        ],
        [  # Second spline's control points
            [40.0, 160.0],
            [130.0, 260.0],
            [240.0, 140.0],
            [260.0, 130.0]
        ]
    ], dtype=torch.float32)


    # Create a spline object
    sample_spline = Spline(batch_size, control_points, joint_points, Bezier(degree=3))

    # Generate sample points from the spline
    sample_points = Trainer.generate_sample_points(sample_spline, num_points=50, add_noise=True, noise_std=5.0)

    rand_joints = torch.rand((2,3,2)) * 500
    rand_controls = torch.rand((2,4,2)) * 500




    spline = Spline(batch_size, rand_controls,
                    rand_joints,
                    Bezier(3))
    

    # Create the spline editor
    editor = SplineEditor(spline, sample_points)
    editor.show()

    # Optimize the spline after 2 seconds
    def optimize():
        trainer = Trainer(spline, sample_points, num_iterations=5000, learning_rate=2,editor=editor)
        trainer.optimize()
        editor.update_spline()  # Update the editor after optimization

    # # Use a timer to delay optimization (e.g., 2 seconds after the editor is shown)
    # QtCore.QTimer.singleShot(2000, optimize)
    optimize()

    # Run the Qt application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()