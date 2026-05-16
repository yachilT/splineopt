"""
1D cosine reconstruction experiment.

Fits a Spline (num_dim=1) to A * cos(2*pi*f*t) on t in [0, 1] and shows
training progress live in a PyQtGraph window. Toggle trainable interval
widths via --trainable-widths to see them slide toward high-curvature
regions of the target.

Run:
    python -m experiments.cosine_1d --freq 2 --intervals 8 --iters 2000
    python -m experiments.cosine_1d --freq 4 --intervals 8 --iters 4000 --trainable-widths
"""

import argparse
import math
import signal
import sys

import numpy as np
import torch
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets  # type: ignore

from splines.spline import Spline
from splines.curve import Bezier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--freq", type=float, default=2.0)
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--intervals", type=int, default=8)
    p.add_argument("--trainable-widths", action="store_true")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--width-lr-scale", type=float, default=0.1)
    p.add_argument("--n-eval", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--update-every", type=int, default=5)
    return p.parse_args()


def build_spline(args: argparse.Namespace) -> Spline:
    torch.manual_seed(args.seed)
    spline = Spline(
        num_dim=1,
        num_intervals=args.intervals,
        num_curves=1,
        curve=Bezier(degree=3),
        trainable_widths=args.trainable_widths,
    )
    with torch.no_grad():
        spline.joint_points.add_(torch.randn_like(spline.joint_points) * 0.05)
        spline.control_points.add_(torch.randn_like(spline.control_points) * 0.05)

    # Enforce G1 (tangent-direction continuity, learnable magnitude ratio
    # g1_scale[c, k]) at every internal joint. Endpoints (0 and K) are skipped
    # because they have no neighbour on one side.
    for k in range(1, args.intervals):
        spline.set_g1(curve_idx=0, joint_idx=k, scale=1.0, enabled=True)

    return spline


def make_target(t_eval: torch.Tensor, amplitude: float, freq: float) -> torch.Tensor:
    return amplitude * torch.cos(2.0 * math.pi * freq * t_eval)


def _dash_line_style():
    # pyqtgraph may be bound to PyQt5 (Qt.DashLine) or PyQt6 (Qt.PenStyle.DashLine).
    Qt = pg.QtCore.Qt
    return Qt.PenStyle.DashLine if hasattr(Qt, "PenStyle") else Qt.DashLine


def build_window(args: argparse.Namespace, t_eval_np: np.ndarray, target_np: np.ndarray):
    pg.setConfigOptions(antialias=True)
    win = pg.GraphicsLayoutWidget(title="Spline fit: A*cos(2*pi*f*t)")
    win.resize(1000, 700)

    p_fit = win.addPlot(row=0, col=0, title="Fit")
    p_fit.showGrid(x=True, y=True, alpha=0.2)
    p_fit.setLabel("bottom", "t")
    p_fit.setLabel("left", "value")
    p_fit.addLegend(offset=(10, 10))

    p_fit.plot(t_eval_np, target_np, pen=pg.mkPen((0, 0, 0), width=2), name="target")
    fit_curve = p_fit.plot([], [], pen=pg.mkPen((220, 40, 40), width=2), name="fit")

    target_samples = pg.ScatterPlotItem(
        t_eval_np, target_np,
        size=5, brush=pg.mkBrush(0, 0, 0, 160), pen=None, name="target samples",
    )
    p_fit.addItem(target_samples)
    pred_samples = pg.ScatterPlotItem(
        size=5, brush=pg.mkBrush(220, 40, 40, 160), pen=None, name="fit samples",
    )
    p_fit.addItem(pred_samples)

    joint_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(40, 80, 220))
    p_fit.addItem(joint_scatter)

    dash = _dash_line_style()
    v_lines = [
        pg.InfiniteLine(angle=90, pen=pg.mkPen((180, 180, 180), style=dash))
        for _ in range(args.intervals + 1)
    ]
    for vl in v_lines:
        p_fit.addItem(vl)

    win.nextRow()
    p_loss = win.addPlot(row=1, col=0, title="Loss (log)")
    p_loss.showGrid(x=True, y=True, alpha=0.2)
    p_loss.setLogMode(x=False, y=True)
    p_loss.setLabel("bottom", "iteration")
    p_loss.setLabel("left", "MSE")
    loss_curve = p_loss.plot([], [], pen=pg.mkPen((20, 120, 20), width=2))

    return win, fit_curve, pred_samples, joint_scatter, v_lines, loss_curve


def main():
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)
    # Restore default Ctrl-C handler so SIGINT terminates the process even
    # while Qt's C event loop is running (PyQt5 otherwise swallows signals).
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    spline = build_spline(args)
    t_eval = torch.linspace(0.0, 1.0, args.n_eval)
    target = make_target(t_eval, args.amplitude, args.freq)

    t_eval_np = t_eval.numpy()
    target_np = target.numpy()

    win, fit_curve, pred_samples, joint_scatter, v_lines, loss_curve = build_window(args, t_eval_np, target_np)
    win.show()

    optimizer = torch.optim.Adam(spline.get_optimizable_groups(args.lr, args.width_lr_scale))

    state = {"it": 0, "loss_hist": []}

    def refresh_plot():
        with torch.no_grad():
            fit_y = spline(t_eval).squeeze(0).squeeze(-1).cpu().numpy()
            joint_t = spline.joint_t_values()[0].cpu().numpy()
            joint_y = spline.joint_points[0, :, 0].cpu().numpy()

        fit_curve.setData(t_eval_np, fit_y)
        pred_samples.setData(t_eval_np, fit_y)
        joint_scatter.setData(joint_t, joint_y)
        for i, vl in enumerate(v_lines):
            vl.setValue(float(joint_t[i]))

        loss_hist = state["loss_hist"]
        if loss_hist:
            loss_curve.setData(np.arange(len(loss_hist)), np.array(loss_hist))
            win.setWindowTitle(
                f"iter {state['it']}/{args.iters} | loss {loss_hist[-1]:.4e}"
            )

    refresh_plot()

    def tick():
        for _ in range(args.update_every):
            if state["it"] >= args.iters:
                timer.stop()
                break
            optimizer.zero_grad()
            pred = spline(t_eval).squeeze(0).squeeze(-1)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            state["loss_hist"].append(loss.item())
            state["it"] += 1
        refresh_plot()

    # Stop the timer explicitly on quit so no tick fires against torn-down
    # Qt objects. (Can't parent QTimer to `win` — pyqtgraph and PyQt5 are
    # bound to different Qt namespaces, so the parent type check fails.)
    timer = QtCore.QTimer()
    timer.setInterval(0)
    timer.timeout.connect(tick)
    app.aboutToQuit.connect(timer.stop)
    timer.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
