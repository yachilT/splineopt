"""Export fit-only PNGs from two_region_cosine_1d at a fixed iteration.

Trains the spline headlessly for one or more configs (default: uniform-width
and trainable-width) and renders the fit panel via matplotlib (no loss
subplot, no transition reference line) to figures/.

Run:
    python -m experiments.export_two_region_fit --iters 3000 --intervals 3
"""

import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless backend — no Qt event loop needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from experiments.two_region_cosine_1d import build_spline, make_target


DEFAULT_OUT_DIR = "figures"
DEFAULT_CACHE_DIR = "data/two_region_fit_cache"
FIGSIZE = (3.3, 1.65)  # one column wide at PG two-column layout, 2:1 aspect

COLOR_TARGET = "#5a5a5a"
COLOR_FIT = "#c81e1e"
COLOR_JOINT = "#1941af"
COLOR_SAMPLE = "#222222"
COLOR_JOINT_DASH = "#bbbbbb"

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_sci(x: float, digits: int = 2) -> str:
    """Format x as 'm × 10ⁿ' using Unicode superscripts (e.g. 1.66 × 10⁻²)."""
    mantissa, exp = f"{x:.{digits}e}".split("e")
    exp_int = int(exp)
    return f"{mantissa} × 10{str(exp_int).translate(_SUPERSCRIPT)}"


def _configure_matplotlib():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222",
        "axes.labelcolor": "#222",
        "xtick.color": "#222",
        "ytick.color": "#222",
        "text.color": "#222",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--transition", type=float, default=0.65)
    p.add_argument("--intervals", type=int, default=3)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--width-lr-scale", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--n-eval", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--soft-boundaries", action="store_true")
    p.add_argument("--soft-alpha", type=float, default=0.2)
    p.add_argument("--relative-cp", action="store_true")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--force-retrain", action="store_true",
                   help="Ignore cached fit data and retrain.")
    return p.parse_args()


def make_spline_args(base: argparse.Namespace, trainable_widths: bool) -> SimpleNamespace:
    """Translate the export args into the SimpleNamespace shape build_spline expects."""
    return SimpleNamespace(
        transition=base.transition,
        intervals=base.intervals,
        trainable_widths=trainable_widths,
        soft_boundaries=base.soft_boundaries,
        soft_alpha=base.soft_alpha,
        relative_cp=base.relative_cp,
        seed=base.seed,
    )


def cache_key(args, trainable_widths: bool) -> dict:
    """Args that change the trained fit — anything not listed here is render-only."""
    return {
        "transition": args.transition,
        "intervals": args.intervals,
        "iters": args.iters,
        "lr": args.lr,
        "width_lr_scale": args.width_lr_scale,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "soft_boundaries": args.soft_boundaries,
        "soft_alpha": args.soft_alpha,
        "relative_cp": args.relative_cp,
        "trainable_widths": trainable_widths,
    }


def cache_path(cache_dir: str, label: str, key: dict) -> str:
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{label}_{digest}.npz")


def train_headless(spline, t_train, target_train, iters, lr, width_lr_scale, desc=None):
    optimizer = torch.optim.Adam(spline.get_optimizable_groups(lr, width_lr_scale))
    pbar = tqdm(range(iters), desc=desc or "train", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        pred = spline(t_train).squeeze(0).squeeze(-1)
        loss = torch.mean((pred - target_train) ** 2)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(mse=f"{loss.item():.2e}")
    return loss.item()


def render_panel(
    out_path: str,
    t_eval_np, target_np,
    t_train_np, target_train_np,
    fit_y, joint_t, joint_y,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Joint markers on the x-axis (no transition reference line).
    for jt in joint_t:
        ax.axvline(jt, color=COLOR_JOINT_DASH, linestyle="--", linewidth=0.6, zorder=1)

    ax.plot(t_eval_np, target_np, color=COLOR_TARGET, linewidth=1.7, zorder=2)
    ax.plot(t_eval_np, fit_y, color=COLOR_FIT, linewidth=1.9, zorder=4)
    ax.scatter(t_train_np, target_train_np, marker="x", s=15,
               color=COLOR_SAMPLE, linewidths=0.9, zorder=3)
    ax.scatter(joint_t, joint_y, s=48, color=COLOR_JOINT,
               edgecolors="#222", linewidths=0.8, zorder=5)

    x_margin = 0.03
    ax.set_xlim(-x_margin, 1.0 + x_margin)
    ax.set_ylim(-1.4, 1.4)
    ax.tick_params(width=0.6, length=3)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def export_one(args, trainable_widths: bool):
    label = "trainable-widths" if trainable_widths else "uniform-widths"

    t_eval = torch.linspace(0.0, 1.0, args.n_eval)
    target = make_target(t_eval, args.transition)
    t_train = torch.linspace(0.0, 1.0, args.n_train)
    target_train = make_target(t_train, args.transition)

    key = cache_key(args, trainable_widths)
    cpath = cache_path(args.cache_dir, label, key)

    if not args.force_retrain and os.path.exists(cpath):
        cached = np.load(cpath)
        fit_y = cached["fit_y"]
        joint_t = cached["joint_t"]
        joint_y = cached["joint_y"]
        final_loss = float(cached["final_loss"])
        print(f"Loaded cache {cpath}  (MSE = {format_sci(final_loss)})")
    else:
        spline_args = make_spline_args(args, trainable_widths)
        spline = build_spline(spline_args)
        final_loss = train_headless(
            spline, t_train, target_train, args.iters, args.lr, args.width_lr_scale,
            desc=label,
        )
        with torch.no_grad():
            fit_y = spline(t_eval).squeeze(0).squeeze(-1).cpu().numpy()
            joint_t = spline.joint_t_values()[0].cpu().numpy()
            joint_y = spline.joint_points[0, :, 0].cpu().numpy()
        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez(
            cpath,
            fit_y=fit_y, joint_t=joint_t, joint_y=joint_y,
            final_loss=np.array(final_loss),
        )
        print(f"Saved cache {cpath}  (MSE = {format_sci(final_loss)})")

    out_path = os.path.join(
        args.out_dir,
        f"two_region_cosine_{label}_k{args.intervals}_iter{args.iters}.pdf",
    )
    render_panel(
        out_path,
        t_eval.numpy(), target.numpy(),
        t_train.numpy(), target_train.numpy(),
        fit_y, joint_t, joint_y,
    )
    print(f"Saved {out_path}  (MSE = {format_sci(final_loss)})")


def main():
    args = parse_args()
    _configure_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)
    for trainable in (False, True):
        export_one(args, trainable)


if __name__ == "__main__":
    main()
