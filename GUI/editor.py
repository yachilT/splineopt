from typing import Optional
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui # type: ignore

from splines.curve import Bezier
from splines.spline import Spline
import torch

COLORS = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

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

        # Layout and plot widget
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot_widget)

        ctrl_per_interval = spline.curve.degree - 1

        self.draggable_splines = []
        for i in range(spline.num_curves):
            num_intervals_i = int(spline.intervals_per_curve[i].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval

            self.draggable_splines.append(DraggableSpline(
                self,
                curve_index=i,
                joint_points=spline.joint_points[i, :num_valid_joints, :],
                control_points=spline.control_points[i, :num_valid_ctrls, :],
                lines=spline.get_lines(i),
                color=COLORS[i % len(COLORS)]
            ))


        if self.sample_points is not None:
            self.sample_scatter = pg.ScatterPlotItem(
                pos=self.sample_points.detach().reshape(-1, 2).numpy(),
                brush=pg.mkBrush(255, 255, 0, 120),  # Yellow-ish, semi-transparent
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

        self.paused = True
        self.pause_button = QtWidgets.QPushButton("Start")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.plot_layout.addWidget(self.pause_button)

        # Optimization status label
        self.status_label = QtWidgets.QLabel("Iteration: 0 | Loss: 0.0000")
        self.plot_layout.addWidget(self.status_label)

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

        for i in range(self.spline.num_curves):
            num_intervals_i = int(self.spline.intervals_per_curve[i].item())
            num_valid_joints = num_intervals_i + 1
            num_valid_ctrls = num_intervals_i * ctrl_per_interval

            self.draggable_splines[i].update_visuals(
                joint_points=self.spline.joint_points[i, :num_valid_joints, :],
                control_points=self.spline.control_points[i, :num_valid_ctrls, :],
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

            # Check control points
            for j in range(num_valid_ctrls):
                pt = self.spline.control_points[curve_idx, j].detach()
                scene_pt = vb.mapViewToScene(pg.Point(pt[0].item(), pt[1].item()))
                widget_pt = self.plot_widget.mapFromScene(scene_pt)
                dx = widget_pt.x() - pixel_pos.x()
                dy = widget_pt.y() - pixel_pos.y()
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = (curve_idx, "control_points", j)

        return best

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

        elif event.type() == QtCore.QEvent.MouseMove and self._drag_curve_idx is not None:
            # Map widget pixel position to data coordinates
            vb = self.plot_widget.getPlotItem().getViewBox()
            scene_pos = self.plot_widget.mapToScene(event.pos())
            data_pos = vb.mapSceneToView(scene_pos)
            new_coords = torch.tensor([data_pos.x(), data_pos.y()], dtype=torch.float32)

            if self._drag_points_type == "joint_points":
                self.spline.joint_points.data[self._drag_curve_idx, self._drag_point_idx] = new_coords
            else:
                self.spline.control_points.data[self._drag_curve_idx, self._drag_point_idx] = new_coords

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


class DraggableSpline(pg.ScatterPlotItem):
    def __init__(self, parent_plot: SplineEditor, curve_index: int, joint_points: torch.Tensor, control_points: torch.Tensor, lines: list[np.ndarray], color: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.editor = parent_plot
        self.curve_index = curve_index

        self.joint_points_scatter = pg.ScatterPlotItem(
            spots=[{'pos': pt, 'data': i, 'brush': 'g', 'symbol': 's', 'size': 12}
                   for i, pt in enumerate(joint_points.detach().numpy())]
        )
        self.control_points_scatter = pg.ScatterPlotItem(
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
                   for i, pt in enumerate(control_points.detach().numpy())]
        )

        parent_plot.plot_widget.addItem(self.joint_points_scatter)
        parent_plot.plot_widget.addItem(self.control_points_scatter)

        # labels
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

        vb = self.joint_points_scatter.getViewBox()
        if vb is None:
            return

        for pos in joint_data:
            label = pg.TextItem(f"({pos[0]:.2f}, {pos[1]:.2f})", anchor=(0.5, -0.5), color='w')
            label.setPos(pos[0], pos[1])
            self.labels.append(label)
            vb.addItem(label)

        for pos in control_data:
            label = pg.TextItem(f"({pos[0]:.2f}, {pos[1]:.2f})", anchor=(0.5, -0.5), color='w')
            label.setPos(pos[0], pos[1])
            self.labels.append(label)
            vb.addItem(label)

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
