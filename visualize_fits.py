"""Side-by-side comparison of two trained splines fit to the same samples.

Each panel shows one trained curve, the shared sample points, and a "whisker"
from every sample to the closest point on that curve — short whiskers = good fit.
"""
import argparse
import os
import sys

import numpy as np
import pyqtgraph as pg
import torch
from PyQt5 import QtGui, QtWidgets

from splines.spline import Spline
from train.train import Trainer

TRAINED_PATH = "data/non-uniform-trained.json"
TARGET_PATH = "data/non-uniformv3.json"
NUM_SAMPLES = 100
NOISE_STD = 2.0
NUM_CURVE_PTS = 500
COLORS = ["#15616d", "#ff7d00"]
TITLES = ["Uniform Splitting [1/3]", "Non-Uniform Splitting"]
DEFAULT_OUT = "figures/fit_comparison.png"
EXPORT_SIZE = (1600, 800)

# CSS stack for the HTML-rendered plot title — picks the first available LaTeX-style serif.
LATEX_TITLE_FONT = "'CMU Serif', 'Latin Modern Roman', 'Times New Roman', serif"
# QFont fallback for tick labels (CSS doesn't apply there).
LATEX_TICK_FONT_FAMILY = "Times New Roman"
LATEX_TICK_FONT_SIZE = 16
TITLE_PT = 24
AXIS_LINE_WIDTH = 2


def closest_curve_distances(curve_xy: np.ndarray, samples_xy: np.ndarray):
    """For each sample, return (closest_point_on_curve, distance)."""
    diff = samples_xy[:, None, :] - curve_xy[None, :, :]  # (S, C, 2)
    d2 = (diff * diff).sum(axis=-1)                        # (S, C)
    idx = d2.argmin(axis=1)                                # (S,)
    closest = curve_xy[idx]
    dists = np.sqrt(d2[np.arange(len(idx)), idx])
    return closest, dists


def build_panel(
    title: str,
    color: str,
    curve_xy: np.ndarray,
    samples_xy: np.ndarray,
    joint_xy: np.ndarray,
    control_xy: np.ndarray,
    polygon_lines: list,
) -> pg.PlotWidget:
    closest, dists = closest_curve_distances(curve_xy, samples_xy)
    rmse = float(np.sqrt(np.mean(dists ** 2)))

    pw = pg.PlotWidget()
    pw.setBackground("white")
    pw.setTitle(
        f'<span style="font-family: {LATEX_TITLE_FONT}; font-size: {TITLE_PT}pt; color: #222;">'
        f'{title} &nbsp; RMSE={rmse:.2f}</span>'
    )
    pw.setAspectLocked(True)

    tick_font = QtGui.QFont(LATEX_TICK_FONT_FAMILY, LATEX_TICK_FONT_SIZE)
    axis_pen = pg.mkPen(color="#222", width=AXIS_LINE_WIDTH)
    for axis_name in ("bottom", "left"):
        axis = pw.getAxis(axis_name)
        axis.setStyle(tickFont=tick_font, tickLength=-8)
        axis.setTextPen("#222")
        axis.setPen(axis_pen)

    # Whiskers: one PlotDataItem with NaN-separated segments (cheap, single draw).
    seg_x = np.empty(samples_xy.shape[0] * 3, dtype=np.float32)
    seg_y = np.empty(samples_xy.shape[0] * 3, dtype=np.float32)
    seg_x[0::3] = samples_xy[:, 0]
    seg_y[0::3] = samples_xy[:, 1]
    seg_x[1::3] = closest[:, 0]
    seg_y[1::3] = closest[:, 1]
    seg_x[2::3] = np.nan
    seg_y[2::3] = np.nan
    pw.plot(seg_x, seg_y, pen=pg.mkPen(color="#888", width=1), connect="finite")

    # Samples (red ×) — drawn before the spline so the curve renders on top.
    pw.plot(
        samples_xy[:, 0], samples_xy[:, 1],
        pen=None, symbol="x", symbolSize=12,
        symbolPen=pg.mkPen("#78290F", width=2), symbolBrush=pg.mkBrush("#78290F"),
    )

    # Control polygon — gray segments from each joint to its adjacent control point.
    for line in polygon_lines:
        pw.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color="#A0A0A0", width=2))

    # Trained curve.
    pw.plot(curve_xy[:, 0], curve_xy[:, 1], pen=pg.mkPen(color=color, width=5))

    # Control points (red dots).
    pw.plot(
        control_xy[:, 0], control_xy[:, 1],
        pen=None, symbol="o", symbolSize=11,
        symbolPen=pg.mkPen("#B00020"), symbolBrush=pg.mkBrush("#B00020"),
    )

    # Joint points (green circles).
    pw.plot(
        joint_xy[:, 0], joint_xy[:, 1],
        pen=None, symbol="o", symbolSize=14,
        symbolPen=pg.mkPen("#1B7F3A"), symbolBrush=pg.mkBrush("#1B7F3A"),
    )

    return pw


def main():
    parser = argparse.ArgumentParser(description="Render the trained-spline fit comparison")
    parser.add_argument("--out", default=DEFAULT_OUT, help="PNG output path (set to empty string to skip export)")
    parser.add_argument("--no-show", action="store_true", help="Render and export without showing the window")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    trained = Spline.load(TRAINED_PATH)
    target = Spline.load(TARGET_PATH)

    torch.manual_seed(0)
    sample_pts = Trainer.generate_sample_points(
        target, num_points=NUM_SAMPLES, add_noise=True, noise_std=NOISE_STD,
    )
    samples_xy = sample_pts[0].detach().cpu().numpy()  # shared across panels

    t = torch.linspace(0.0, 1.0, steps=NUM_CURVE_PTS)
    curves = trained(t).detach().cpu().numpy()  # (num_curves, NUM_CURVE_PTS, 2)

    ctrl_per_interval = trained.curve.degree - 1
    eff_cp = trained.get_effective_control_points()  # (C, max_intervals, k_per, dim)

    win = QtWidgets.QWidget()
    win.setWindowTitle("Trained spline fit comparison")
    win.setStyleSheet("background-color: white;")
    layout = QtWidgets.QHBoxLayout(win)

    panels = []
    for i in range(trained.num_curves):
        n_int = int(trained.intervals_per_curve[i].item())
        joint_xy = trained.joint_points[i, : n_int + 1, :].detach().cpu().numpy()
        control_xy = eff_cp[i, :n_int].reshape(n_int * ctrl_per_interval, trained.num_dim).detach().cpu().numpy()
        polygon_lines = trained.get_lines(i)

        panel = build_panel(
            TITLES[i], COLORS[i], curves[i], samples_xy,
            joint_xy, control_xy, polygon_lines,
        )
        layout.addWidget(panel)
        panels.append(panel)

    # Link axes so panels share the same view — easier visual comparison.
    if len(panels) > 1:
        for p in panels[1:]:
            p.setXLink(panels[0])
            p.setYLink(panels[0])

    win.resize(*EXPORT_SIZE)

    if args.out:
        # Render off-screen so the grab captures a fully laid-out widget.
        win.show()
        app.processEvents()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if not win.grab().save(args.out):
            raise RuntimeError(f"Failed to save PNG to {args.out}")
        print(f"Saved {args.out}")
        if args.no_show:
            return

    if not args.no_show:
        win.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
