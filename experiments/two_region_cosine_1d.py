"""
1D two-region cosine reconstruction experiment.

Target on t in [0, 1] is piecewise cosine, continuous at the transition point:
    t <= transition:  cos(pi * t / transition)                  (half cycle, slow)
    t >  transition:  -cos(2 * pi * (t - transition) / (1 - transition))  (full cycle, fast)

With the default transition=0.65, the slow region owns 65% of [0, 1] and the fast
region 35%, so a uniform-width spline allocates *fewer* intervals to the fast
region where they're needed most. With --trainable-widths, widths should
redistribute toward the fast region. With --soft-boundaries on top, the selection
gradient at interval joints adds a signal that the hard path lacks.

Run:
    python -m experiments.two_region_cosine_1d --intervals 8 --iters 3000 --trainable-widths
    python -m experiments.two_region_cosine_1d --intervals 8 --iters 3000 --trainable-widths --soft-boundaries
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
    p.add_argument("--transition", type=float, default=0.65,
                   help="t at which the slow→fast transition occurs (0 < transition < 1)")
    p.add_argument("--intervals", type=int, default=8)
    p.add_argument("--trainable-widths", action="store_true")
    p.add_argument("--soft-boundaries", action="store_true",
                   help="Enable trapezoidal POU blending across interval joints.")
    p.add_argument("--soft-alpha", type=float, default=0.2,
                   help="Relative half-width of the ε-band at each joint.")
    p.add_argument("--relative-cp", action="store_true",
                   help="Store control points as offsets from anchor joints instead of absolute positions.")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--width-lr-scale", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=48,
                   help="Number of training samples used in the loss (shown as scatter).")
    p.add_argument("--n-eval", type=int, default=512,
                   help="Number of points used to render the smooth target/fit curves.")
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
        relative_control_points=args.relative_cp,
        soft_boundaries=args.soft_boundaries,
        soft_boundary_alpha=args.soft_alpha,
    )
    with torch.no_grad():
        spline.joint_points.add_(torch.randn_like(spline.joint_points) * 0.05)
        spline.control_points.add_(torch.randn_like(spline.control_points) * 0.05)

    # G1 at every internal joint (tangent-direction continuity, magnitude free).
    for k in range(1, args.intervals):
        spline.set_g1(curve_idx=0, joint_idx=k, scale=1.0, enabled=True)

    return spline


def make_target(t_eval: torch.Tensor, transition: float) -> torch.Tensor:
    """Piecewise cosine: half cycle on [0, transition], full cycle on [transition, 1].
    Continuous at t = transition (both sides equal -1)."""
    if not (0.0 < transition < 1.0):
        raise ValueError(f"transition must be in (0, 1), got {transition}")
    out = torch.empty_like(t_eval)
    slow = t_eval <= transition
    fast = ~slow
    out[slow] = torch.cos(math.pi * t_eval[slow] / transition)
    out[fast] = -torch.cos(2.0 * math.pi * (t_eval[fast] - transition) / (1.0 - transition))
    return out


def _dash_line_style():
    Qt = pg.QtCore.Qt
    return Qt.PenStyle.DashLine if hasattr(Qt, "PenStyle") else Qt.DashLine


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class SciLogAxis(pg.AxisItem):
    """Log-axis that labels powers of 10 as 10ⁿ instead of pyqtgraph's
    default decimal/SI-prefix formatting."""
    def logTickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            iv = int(round(v))
            if abs(v - iv) < 1e-6:
                out.append(f"10{str(iv).translate(_SUPERSCRIPT)}")
            else:
                out.append("")  # suppress intermediate (non-integer-power) ticks
        return out


def build_window(args: argparse.Namespace, t_eval_np: np.ndarray, target_np: np.ndarray,
                 t_train_np: np.ndarray, target_train_np: np.ndarray):
    pg.setConfigOptions(antialias=True)
    mode = "soft" if args.soft_boundaries else "hard"
    win = pg.GraphicsLayoutWidget(title=f"Two-region cosine fit [{mode}]")
    win.resize(1000, 700)

    p_fit = win.addPlot(row=0, col=0, title=f"Fit (transition={args.transition})")
    p_fit.showGrid(x=True, y=True, alpha=0.2)
    p_fit.setLabel("bottom", "t")
    p_fit.setLabel("left", "value")
    p_fit.addLegend(offset=(10, 10))
    # Zero the viewbox default padding before any item is added so the auto-fit
    # pyqtgraph performs on the first plot() call doesn't expand the range.
    p_fit.getViewBox().setDefaultPadding(0.0)
    p_fit.setMouseEnabled(x=False, y=False)
    p_fit.hideButtons()

    p_fit.plot(t_eval_np, target_np, pen=pg.mkPen((100, 100, 100), width=2), name="target")
    fit_curve = p_fit.plot([], [], pen=pg.mkPen((220, 40, 40), width=2), name="fit")

    # Sparse training samples — the actual points the loss is computed on.
    train_samples = pg.ScatterPlotItem(
        t_train_np, target_train_np,
        symbol="x", size=6,
        brush=pg.mkBrush(255, 255, 255),
        pen=pg.mkPen((255, 255, 255)),
        name="training samples",
    )
    p_fit.addItem(train_samples)

    joint_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(40, 80, 220), name="joints")
    p_fit.addItem(joint_scatter)

    dash = _dash_line_style()
    v_lines = [
        pg.InfiniteLine(angle=90, pen=pg.mkPen((180, 180, 180), style=dash))
        for _ in range(args.intervals + 1)
    ]
    for vl in v_lines:
        p_fit.addItem(vl)

    # Transition reference line (orange).
    transition_line = pg.InfiniteLine(
        pos=args.transition, angle=90,
        pen=pg.mkPen((240, 140, 0), width=2, style=dash),
        label=f"transition={args.transition}",
        labelOpts={"position": 0.95, "color": (240, 140, 0)},
    )
    p_fit.addItem(transition_line)

    # Lock the view AFTER all items are added — each addItem can re-trigger an
    # auto-range, so set the range once at the end and clamp the viewbox via
    # setLimits so the data domain can't be exceeded.
    p_fit.setLimits(xMin=0.0, xMax=1.0, yMin=-1.4, yMax=1.4)
    vb_fit = p_fit.getViewBox()
    vb_fit.disableAutoRange()
    vb_fit.setRange(xRange=(0.0, 1.0), yRange=(-1.4, 1.4), padding=0)

    win.nextRow()
    p_loss = win.addPlot(
        row=1, col=0, title="Loss (log)",
        axisItems={"left": SciLogAxis(orientation="left")},
    )
    p_loss.showGrid(x=True, y=True, alpha=0.2)
    p_loss.setLogMode(x=False, y=True)
    p_loss.setLabel("bottom", "iteration")
    p_loss.setLabel("left", "MSE")
    # Disable the SI-prefix multiplier (pyqtgraph would otherwise factor out
    # small values as "MSE (x0.001)"). SciLogAxis handles the 10ⁿ rendering.
    p_loss.getAxis("left").enableAutoSIPrefix(False)
    # Zero auto-range padding so the data reaches the right edge instead of leaving
    # a gap between the last tick and where the curve ends.
    p_loss.getViewBox().setDefaultPadding(0.0)
    p_loss.setLimits(xMin=0.0)
    loss_curve = p_loss.plot([], [], pen=pg.mkPen((20, 120, 20), width=2))

    return win, fit_curve, joint_scatter, v_lines, loss_curve


def main():
    args = parse_args()

    # High-DPI opt-in must be set BEFORE QApplication is constructed, otherwise
    # the viewbox layout drifts on monitors with display scaling != 100% (e.g.
    # external 4K monitors): widgets stretch but the data range doesn't.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    spline = build_spline(args)
    # Dense grid for rendering smooth target/fit curves.
    t_eval = torch.linspace(0.0, 1.0, args.n_eval)
    target = make_target(t_eval, args.transition)
    # Sparse grid used in the actual loss — these are the "scatter" the spline fits.
    t_train = torch.linspace(0.0, 1.0, args.n_train)
    target_train = make_target(t_train, args.transition)

    t_eval_np = t_eval.numpy()
    target_np = target.numpy()
    t_train_np = t_train.numpy()
    target_train_np = target_train.numpy()

    win, fit_curve, joint_scatter, v_lines, loss_curve = build_window(
        args, t_eval_np, target_np, t_train_np, target_train_np
    )

    # Wrap the plot widget in a container with a start/pause button at the top.
    container = QtWidgets.QWidget()
    container.setWindowTitle(win.windowTitle())
    container.resize(1000, 740)
    play_btn = QtWidgets.QPushButton("Start")
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.addWidget(play_btn)
    layout.addWidget(win)
    container.show()

    optimizer = torch.optim.Adam(spline.get_optimizable_groups(args.lr, args.width_lr_scale))

    state = {"it": 0, "loss_hist": []}

    def refresh_plot():
        with torch.no_grad():
            fit_y = spline(t_eval).squeeze(0).squeeze(-1).cpu().numpy()
            joint_t = spline.joint_t_values()[0].cpu().numpy()
            joint_y = spline.joint_points[0, :, 0].cpu().numpy()

        fit_curve.setData(t_eval_np, fit_y)
        joint_scatter.setData(joint_t, joint_y)
        for i, vl in enumerate(v_lines):
            vl.setValue(float(joint_t[i]))

        loss_hist = state["loss_hist"]
        if loss_hist:
            loss_curve.setData(np.arange(len(loss_hist)), np.array(loss_hist))
            mode = "soft" if args.soft_boundaries else "hard"
            container.setWindowTitle(
                f"[{mode}] iter {state['it']}/{args.iters} | loss {loss_hist[-1]:.4e}"
            )

    refresh_plot()

    def tick():
        for _ in range(args.update_every):
            if state["it"] >= args.iters:
                timer.stop()
                break
            optimizer.zero_grad()
            pred = spline(t_train).squeeze(0).squeeze(-1)
            loss = torch.mean((pred - target_train) ** 2)
            loss.backward()
            optimizer.step()
            state["loss_hist"].append(loss.item())
            state["it"] += 1
        refresh_plot()

    timer = QtCore.QTimer()
    timer.setInterval(0)
    timer.timeout.connect(tick)
    app.aboutToQuit.connect(timer.stop)

    def toggle_training():
        if timer.isActive():
            timer.stop()
            play_btn.setText("Start")
        else:
            # Don't restart past the iteration budget.
            if state["it"] < args.iters:
                timer.start()
                play_btn.setText("Pause")
    play_btn.clicked.connect(toggle_training)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
