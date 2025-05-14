import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from splines.curve import Bezier
from splines.spline import Spline
import torch

COLORS = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

class SplineEditor(QtWidgets.QWidget):
    def __init__(self, spline: Spline, sample_points: Spline = None, freeze: bool = False):
        super().__init__()
        self.setWindowTitle("Spline Editor")

        self.spline = spline
        self.sample_points = sample_points
        self.num_steps = 100  # Default number of steps

        # Layout and plot widget
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot_widget)

        self.draggable_splines = [DraggableSpline(self, spline.joint_points[i, :, :], spline.control_points[i, :, :], spline.get_lines(i), freeze, COLORS[i]) for i in range(spline.batch_size)]

        
        if self.sample_points is not None:
            self.sample_scatter = pg.ScatterPlotItem(
                pos=sample_points.detach().reshape(-1, 2).numpy(),
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

        for i in range(self.spline.batch_size):
            self.draggable_splines[i].update_visuals(
                joint_points=self.spline.joint_points[i, :, :],
                control_points=self.spline.control_points[i, :, :],
                lines=self.spline.get_lines(i),
                curve=spline_pts[i, :, :]
            )






class DraggableSpline(pg.ScatterPlotItem):
    def __init__(self, parent_plot: SplineEditor, joint_points: torch.Tensor, control_points: torch.Tensor, lines: list[np.ndarray], freeze: bool, color: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_points_scatter : DraggableScatter = DraggableScatter(
            parent_editor=self,
            points_type="joint_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'g', 'symbol': 's', 'size': 12}
                   for i, pt in enumerate(joint_points.detach().numpy())]
        )
        self.control_points_scatter: DraggableScatter = DraggableScatter(
            parent_editor=self,
            points_type="control_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
                   for i, pt in enumerate(control_points.detach().numpy())]
        )

        parent_plot.plot_widget.addItem(self.joint_points_scatter)
        parent_plot.plot_widget.addItem(self.control_points_scatter)

        # labels
        self.joint_points_scatter.update_labels(joint_points.detach().numpy())
        self.control_points_scatter.update_labels(control_points.detach().numpy())

        self.lines = [parent_plot.plot_widget.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color='gray', width=2))
                      for line in lines]
        
        self.spline_curve = parent_plot.plot_widget.plot([], pen=pg.mkPen(color=color, width=4))
    
    def update_visuals(self, joint_points: torch.Tensor, control_points: torch.Tensor, lines: list[np.ndarray], curve: torch.Tensor):
        """Redraw the spline and its lines."""
        self.joint_points_scatter.setData(pos=joint_points.detach().numpy())
        self.joint_points_scatter.update_labels(joint_points.detach().numpy())

        self.control_points_scatter.setData(pos=control_points.detach().numpy())
        self.control_points_scatter.update_labels(control_points.detach().numpy())

        # Update the lines
        for i, line in enumerate(lines):
            self.lines[i].setData(line[:, 0], line[:, 1])

        # Update the spline curve
        self.spline_curve.setData(curve[:, 0].detach(), curve[:, 1].detach())

class DraggableScatter(pg.ScatterPlotItem):
    def __init__(self, parent_editor, points_type, freeze: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_editor = parent_editor
        self.points_type = points_type  # Either "control_points" or "joint_points"
        self.dragged_index = None
        self.freeze = freeze
        self.labels = []


    def update_labels(self, points_data):
        # Remove old labels from the view
        vb = self.getViewBox()
        if vb is not None:
            for label in self.labels:
                vb.removeItem(label)

        self.labels.clear()

        # Add new labels
        for pos in points_data:
            label = pg.TextItem(f"({pos[0]:.2f}, {pos[1]:.2f})", anchor=(0.5, -0.5), color='w')
            label.setPos(pos[0], pos[1])
            self.labels.append(label)
            vb.addItem(label)



    def mousePressEvent(self, event):
        points = self.pointsAt(event.pos())
        if points.size > 0:
            self.dragged_index = points[0].data()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragged_index is None or self.freeze:
            return
        # TODO update code to support multiple splines
        # # Update the corresponding points (control_points or joint_points)
        # if self.points_type == "control_points":
        #     self.parent_editor.spline.control_points.data[self.dragged_index] = torch.tensor(
        #         [event.pos().x(), event.pos().y()], dtype=torch.float32
        #     )
        # elif self.points_type == "joint_points":
        #     self.parent_editor.spline.joint_points.data[self.dragged_index] = torch.tensor(
        #         [event.pos().x(), event.pos().y()], dtype=torch.float32
        #     )

        # # Update points visually
        # self.parent_editor.update_spline()
        # event.accept()

    def mouseReleaseEvent(self, event):
        self.dragged_index = None
        event.accept()
