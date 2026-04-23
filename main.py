import argparse
import sys

import torch
from PyQt5 import QtWidgets  # type: ignore

from GUI.editor import SplineEditor
from splines.curve import Bezier
from splines.Spline import Spline
from train.train import Trainer


def _make_demo_target() -> Spline:
    # A 3-interval S-curve — complex enough that 1 interval can't fit it well
    joint_points = torch.tensor([
        [[50.0, 300.0], [183.0, 100.0], [316.0, 400.0], [450.0, 200.0]]
    ], dtype=torch.float32)
    control_points = torch.tensor([
        [[100.0, 100.0], [150.0, 100.0],
         [233.0, 400.0], [266.0, 400.0],
         [366.0, 100.0], [400.0, 100.0]]
    ], dtype=torch.float32)
    return Spline(2, 3, 1, Bezier(degree=3), joint_points, control_points)


def run_demo():
    reply = QtWidgets.QMessageBox.question(
        None, "Demo Setup",
        "Load a target spline from file?\n\nNo = use the built-in S-curve target.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )

    if reply == QtWidgets.QMessageBox.Yes:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Load Target Spline", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        target_spline = Spline.load(path)
    else:
        target_spline = _make_demo_target()

    # Sample points from the first curve of the target (shape: (1, N, 2))
    sample_pts = Trainer.generate_sample_points(
        target_spline, num_points=100, add_noise=True, noise_std=5.0
    )
    # Both optimization curves fit the same points (shape: (2, N, 2))
    sample_points = sample_pts[:1].expand(2, -1, -1).contiguous()

    # Both curves start as random 2-interval C1 splines.
    # Split one (or both) during the demo to show how splitting affects fit quality.
    max_intervals = 2
    rand_joints   = torch.rand(2, max_intervals + 1, 2) * 300 + 50
    rand_controls = torch.rand(2, max_intervals * 2, 2) * 300 + 50
    c1_mask = torch.zeros(2, max_intervals + 1, dtype=torch.bool)
    c1_mask[:, 1] = True  # internal junction — both sides marked so C1 activates
    c1_mask[:, 2] = True
    spline = Spline(
        num_dim=2,
        num_intervals=2,
        num_curves=2,
        curve=Bezier(degree=3),
        joint_points=rand_joints,
        control_points=rand_controls,
        c1_mask=c1_mask,
    )

    editor = SplineEditor(spline, sample_pts[0])
    editor.show()

    def optimize():
        trainer = Trainer(spline, sample_points, num_iterations=5000,
                          learning_rate=2, editor=editor)
        trainer.optimize()
        editor.update_spline()

    optimize()


def run_editor():
    joint_points   = torch.tensor([[[100.0, 200.0], [200.0, 200.0], [300.0, 200.0]]], dtype=torch.float32)
    control_points = torch.tensor([[[130.0, 280.0], [170.0, 120.0], [230.0, 280.0], [270.0, 120.0]]], dtype=torch.float32)
    c1_mask = torch.zeros(1, 3, dtype=torch.bool)
    c1_mask[0, 1] = True
    c1_mask[0, 2] = True
    spline = Spline(num_dim=2, num_intervals=2, num_curves=1, curve=Bezier(degree=3),
                    joint_points=joint_points, control_points=control_points, c1_mask=c1_mask)
    editor = SplineEditor(spline)
    editor.show()


def main():
    parser = argparse.ArgumentParser(description="Spline editor / optimizer")
    parser.add_argument("--mode", choices=["demo", "editor"], default="editor",
                        help="demo: fit red (1-interval) and green (2-interval) splines to the same target; "
                             "editor: open a blank spline for manual editing")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    if args.mode == "demo":
        run_demo()
    else:
        run_editor()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
