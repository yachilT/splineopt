from typing import Optional
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui # type: ignore

from splines.curve import Bezier
from splines.Spline import Spline
import torch

COLORS = [
    "#15616d",  # dark red
    "#ff7d00",  # dark green
    "#78290f",  # dark blue
    "#001524",  # dark amber
    '#00a0a0',  # dark cyan
    '#a000a0',  # dark magenta
    '#282828',  # near-black
]

# Pixel distance threshold for picking a point
PICK_RADIUS = 15

class SplineEditor(QtWidgets.QWidget):
    def __init__(self, spline: Spline, sample_points: Optional[torch.Tensor] = None, freeze: bool = False):
        super().__init__()
        self.setWindowTitle("Spline Editor")

        self.spline = spline
        self.sample_points = sample_points
        self.num_steps = 100  # Default number of steps
        self.freeze = freeze

        # Drag state
        self._drag_curve_idx = None
        self._drag_points_type = None  # "joint_points" or "control_points"
        self._drag_point_idx = None

        # Optional callback invoked after split_intervals replaces the parameter tensors
        self.on_params_changed = None

        # Layout and plot widget
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#F4F4F4")
        self.plot_layout.addWidget(self.plot_widget)

        ctrl_per_interval = spline.curve.degree - 1
        eff_cp = spline.get_effective_control_points()

        self.draggable_splines = []
        for i in range(spline.num_curves):
            num_intervals_i = int(spline.intervals_per_curve[i].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval
            eff_cp_flat = eff_cp[i, :num_intervals_i].reshape(num_valid_ctrls, spline.num_dim)

            self.draggable_splines.append(DraggableSpline(
                self,
                curve_index=i,
                joint_points=spline.joint_points[i, :num_valid_joints, :],
                control_points=eff_cp_flat,
                lines=spline.get_lines(i),
                color=COLORS[i % len(COLORS)]
            ))


        if self.sample_points is not None:
            self.sample_scatter = pg.ScatterPlotItem(
                pos=self.sample_points.detach().reshape(-1, 2).numpy(),
                brush=pg.mkBrush("#78290F"),  # Dark amber
                size=8,
                symbol='x'
            )
            self.plot_widget.addItem(self.sample_scatter)


        # Slider for controlling the number of steps
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(2)  # Minimum number of steps
        self.slider.setMaximum(200)  # Maximum number of steps
        self.slider.setValue(self.num_steps)  # Set default value
        self.slider.valueChanged.connect(self.update_num_steps)  # Connect slider to update method
        self.plot_layout.addWidget(self.slider)

        io_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Spline")
        self.load_button = QtWidgets.QPushButton("Load Spline")
        io_layout.addWidget(self.save_button)
        io_layout.addWidget(self.load_button)
        self.plot_layout.addLayout(io_layout)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)

        self.paused = True
        self.pause_button = QtWidgets.QPushButton("Start")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.plot_layout.addWidget(self.pause_button)

        # Optimization status label
        self.status_label = QtWidgets.QLabel("Iteration: 0 | Loss: 0.0000")
        self.plot_layout.addWidget(self.status_label)

        # Disable pyqtgraph's built-in right-click context menu so our event filter can handle it
        self.plot_widget.getPlotItem().getViewBox().setMenuEnabled(False)

        # Install event filter on the viewport for drag handling
        self.plot_widget.viewport().installEventFilter(self)

        self.update_spline()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def update_status(self, iteration: int, loss: float):
        self.status_label.setText(f"Iteration: {iteration} | Loss: {loss:.4f}")

    def update_num_steps(self, value):
        """Update the number of steps based on the slider value."""
        self.num_steps = value
        self.update_spline()

    def update_spline(self):
        spline_pts = self.spline(torch.linspace(0.0, 1.0, steps=self.num_steps))
        ctrl_per_interval = self.spline.curve.degree - 1
        eff_cp = self.spline.get_effective_control_points()

        for i in range(self.spline.num_curves):
            num_intervals_i = int(self.spline.intervals_per_curve[i].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval
            eff_cp_flat = eff_cp[i, :num_intervals_i].reshape(num_valid_ctrls, self.spline.num_dim)

            self.draggable_splines[i].update_visuals(
                joint_points=self.spline.joint_points[i, :num_valid_joints, :],
                control_points=eff_cp_flat,
                lines=self.spline.get_lines(i),
                curve=spline_pts[i, :, :]
            )

    def _find_nearest_point(self, pixel_pos: QtCore.QPointF):
        """
        Find the nearest joint or control point to a pixel position.
        Returns (curve_idx, points_type, point_idx) or (None, None, None).
        """
        vb = self.plot_widget.getPlotItem().getViewBox()
        ctrl_per_interval = self.spline.curve.degree - 1
        best_dist = PICK_RADIUS
        best = (None, None, None)

        for curve_idx in range(self.spline.num_curves):
            num_intervals_i = int(self.spline.intervals_per_curve[curve_idx].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval

            # Check joint points
            for j in range(num_valid_joints):
                pt = self.spline.joint_points[curve_idx, j].detach()
                scene_pt = vb.mapViewToScene(pg.Point(pt[0].item(), pt[1].item()))
                widget_pt = self.plot_widget.mapFromScene(scene_pt)
                dx = widget_pt.x() - pixel_pos.x()
                dy = widget_pt.y() - pixel_pos.y()
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = (curve_idx, "joint_points", j)

            # Check control points (use effective positions so derived CPs are hit-testable)
            eff_cp = self.spline.get_effective_control_points()
            k_per = self.spline.curve.degree - 1
            for j in range(num_valid_ctrls):
                k = j // k_per
                pos_in_interval = j % k_per
                pt = eff_cp[curve_idx, k, pos_in_interval].detach()
                scene_pt = vb.mapViewToScene(pg.Point(pt[0].item(), pt[1].item()))
                widget_pt = self.plot_widget.mapFromScene(scene_pt)
                dx = widget_pt.x() - pixel_pos.x()
                dy = widget_pt.y() - pixel_pos.y()
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = (curve_idx, "control_points", j)

        return best

    def _find_nearest_interval(self, pixel_pos: QtCore.QPointF, threshold_px: float = 15.0):
        """
        Find the nearest spline interval to a pixel position.
        Uses segment-based distance (point-to-segment) for reliable detection.
        Returns (curve_idx, interval_idx) or (None, None).
        """
        vb = self.plot_widget.getPlotItem().getViewBox()
        best_dist = threshold_px
        best = (None, None)
        samples_per_interval = 50
        px, py = pixel_pos.x(), pixel_pos.y()

        def pt_to_widget(pt):
            sp = vb.mapViewToScene(pg.Point(pt[0].item(), pt[1].item()))
            wp = self.plot_widget.mapFromScene(sp)
            return wp.x(), wp.y()

        def seg_dist(ax, ay, bx, by, px, py):
            """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
            dx, dy = bx - ax, by - ay
            len_sq = dx * dx + dy * dy
            if len_sq == 0:
                return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
            return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5

        for curve_idx in range(self.spline.num_curves):
            num_intervals_i = int(self.spline.intervals_per_curve[curve_idx].item())

            widths = self.spline.interval_widths[curve_idx]  # (max_intervals,)
            cum_w = torch.cumsum(widths, dim=0)
            for k in range(num_intervals_i):
                t_start = 0.0 if k == 0 else cum_w[k - 1].item()
                t_end = cum_w[k].item()
                t_samples = torch.linspace(t_start, t_end - 1e-6, samples_per_interval)
                pts = self.spline(t_samples)[curve_idx]  # (samples, dim)

                # Convert all samples to widget pixel coords
                widget_pts = [pt_to_widget(pts[i]) for i in range(len(pts))]

                # Check distance to each segment between consecutive samples
                for i in range(len(widget_pts) - 1):
                    ax, ay = widget_pts[i]
                    bx, by = widget_pts[i + 1]
                    dist = seg_dist(ax, ay, bx, by, px, py)
                    if dist < best_dist:
                        best_dist = dist
                        best = (curve_idx, k)

        return best

    def _split_interval(self, curve_idx: int, interval_idx: int):
        """Call split_intervals for one curve/interval and rebuild the spline and visuals."""
        from torch import nn
        C = self.spline.num_curves
        split_mask = torch.zeros(C, dtype=torch.bool)
        split_mask[curve_idx] = True
        interval_indices = torch.zeros(C, dtype=torch.long)
        interval_indices[curve_idx] = interval_idx

        new_jp, new_cp, new_ipc = self.spline.split_intervals(split_mask, interval_indices)

        # Capture old params before replacement so optimizer can transfer state
        old_jp_param = self.spline.joint_points
        old_cp_param = self.spline.control_points
        old_max_intervals = self.spline.max_intervals

        # Update spline parameters (shapes change so replace the Parameters)
        self.spline.joint_points = nn.Parameter(new_jp)
        self.spline.control_points = nn.Parameter(new_cp)
        self.spline.intervals_per_curve = new_ipc
        self.spline.max_intervals = int(new_ipc.max().item())

        self._rebuild_draggable_splines()
        if self.on_params_changed is not None:
            self.on_params_changed(old_jp_param, old_cp_param, split_mask, old_max_intervals)
        self.update_spline()

    def _on_save_clicked(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Spline", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.spline.save(path)

    def _on_load_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Spline", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.spline = Spline.load(path)
            self._rebuild_draggable_splines()
            self.update_spline()

    def _rebuild_draggable_splines(self):
        """Remove all spline visuals from the plot and recreate them."""
        for ds in self.draggable_splines:
            ds.remove_from_plot(self.plot_widget)
        self.draggable_splines.clear()

        ctrl_per_interval = self.spline.curve.degree - 1
        eff_cp = self.spline.get_effective_control_points()
        for i in range(self.spline.num_curves):
            num_intervals_i = int(self.spline.intervals_per_curve[i].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval
            eff_cp_flat = eff_cp[i, :num_intervals_i].reshape(num_valid_ctrls, self.spline.num_dim)

            self.draggable_splines.append(DraggableSpline(
                self,
                curve_index=i,
                joint_points=self.spline.joint_points[i, :num_valid_joints, :],
                control_points=eff_cp_flat,
                lines=self.spline.get_lines(i),
                color=COLORS[i % len(COLORS)]
            ))

    def eventFilter(self, obj, event):
        if not self.paused or self.freeze:
            return False

        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            curve_idx, pts_type, pt_idx = self._find_nearest_point(event.pos())
            if curve_idx is not None:
                self._drag_curve_idx = curve_idx
                self._drag_points_type = pts_type
                self._drag_point_idx = pt_idx
                # Disable ViewBox panning while dragging a point
                self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(False, False)
                return True

        elif event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
            curve_idx, interval_idx = self._find_nearest_interval(event.pos())
            if curve_idx is not None:
                self._split_interval(curve_idx, interval_idx)
                return True

        elif event.type() == QtCore.QEvent.MouseMove and self._drag_curve_idx is not None:
            # Map widget pixel position to data coordinates
            vb = self.plot_widget.getPlotItem().getViewBox()
            scene_pos = self.plot_widget.mapToScene(event.pos())
            data_pos = vb.mapSceneToView(scene_pos)
            new_coords = torch.tensor([data_pos.x(), data_pos.y()], dtype=torch.float32)

            if self._drag_points_type == "joint_points":
                self.spline.joint_points.data[self._drag_curve_idx, self._drag_point_idx] = new_coords
            else:
                c = self._drag_curve_idx
                j = self._drag_point_idx
                k_per = self.spline.curve.degree - 1
                k = j // k_per
                # If this is a C1-derived control point, redirect to its free driver
                if (j % k_per == 0 and k >= 1
                        and self.spline.c1_mask[c, k]
                        and self.spline.c1_mask[c, k + 1]):
                    J_k = self.spline.joint_points.data[c, k]
                    free_idx = (k - 1) * k_per + (k_per - 1)
                    self.spline.control_points.data[c, free_idx] = 2.0 * J_k - new_coords
                else:
                    self.spline.control_points.data[c, j] = new_coords

            self.update_spline()
            return True

        elif event.type() == QtCore.QEvent.MouseButtonRelease and self._drag_curve_idx is not None:
            self._drag_curve_idx = None
            self._drag_points_type = None
            self._drag_point_idx = None
            # Re-enable ViewBox panning
            self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(True, True)
            return True

        return False


class DraggableSpline:
    def __init__(self, parent_plot: SplineEditor, curve_index: int, joint_points: torch.Tensor, control_points: torch.Tensor, lines: list[np.ndarray], color: str):
        self.editor = parent_plot
        self.curve_index = curve_index

        self.joint_points_scatter = pg.ScatterPlotItem(
            spots=[{'pos': pt, 'data': i, 'brush': 'g', 'symbol': 's', 'size': 18}
                   for i, pt in enumerate(joint_points.detach().numpy())]
        )
        self.control_points_scatter = pg.ScatterPlotItem(
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 14}
                   for i, pt in enumerate(control_points.detach().numpy())]
        )

        parent_plot.plot_widget.addItem(self.joint_points_scatter)
        parent_plot.plot_widget.addItem(self.control_points_scatter)

        self.show_labels = False
        self.labels = []
        self._update_labels(joint_points.detach().numpy(), control_points.detach().numpy())

        self.lines = [parent_plot.plot_widget.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color='gray', width=2))
                      for line in lines]

        self.spline_curve = parent_plot.plot_widget.plot([], pen=pg.mkPen(color=color, width=4))

    def _update_labels(self, joint_data, control_data):
        vb = self.joint_points_scatter.getViewBox()
        if vb is not None:
            for label in self.labels:
                vb.removeItem(label)
        self.labels.clear()

        if not self.show_labels:
            return

        vb = self.joint_points_scatter.getViewBox()
        if vb is None:
            return

        for pos in joint_data:
            label = pg.TextItem(f"({pos[0]:.0f}, {pos[1]:.0f})", anchor=(0.5, -0.5), color=(120, 120, 120))
            label.setPos(pos[0], pos[1])
            self.labels.append(label)
            vb.addItem(label)

        for pos in control_data:
            label = pg.TextItem(f"({pos[0]:.0f}, {pos[1]:.0f})", anchor=(0.5, -0.5), color=(120, 120, 120))
            label.setPos(pos[0], pos[1])
            self.labels.append(label)
            vb.addItem(label)

    def remove_from_plot(self, plot_widget: pg.PlotWidget):
        """Remove all visual items belonging to this spline from the plot."""
        # Get ViewBox before removing scatter items, as getViewBox() returns None afterwards
        vb = self.joint_points_scatter.getViewBox()
        if vb is not None:
            for label in self.labels:
                vb.removeItem(label)
        self.labels.clear()

        plot_widget.removeItem(self.joint_points_scatter)
        plot_widget.removeItem(self.control_points_scatter)
        plot_widget.removeItem(self.spline_curve)
        for line in self.lines:
            plot_widget.removeItem(line)

    def update_visuals(self, joint_points: torch.Tensor, control_points: torch.Tensor, lines: list[np.ndarray], curve: torch.Tensor):
        """Redraw the spline and its lines."""
        self.joint_points_scatter.setData(pos=joint_points.detach().numpy())
        self.control_points_scatter.setData(pos=control_points.detach().numpy())
        self._update_labels(joint_points.detach().numpy(), control_points.detach().numpy())

        # Update the lines
        for i, line in enumerate(lines):
            self.lines[i].setData(line[:, 0], line[:, 1])

        # Update the spline curve
        self.spline_curve.setData(curve[:, 0].detach(), curve[:, 1].detach())
