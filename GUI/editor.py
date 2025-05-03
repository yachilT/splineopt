import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from splines.curve import Bezier
from splines.spline import Spline
import torch


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

        # Scatter plot for control points
        self.control_scatter = DraggableScatter(
            parent_editor=self,
            points_type="control_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
                for i, pt in enumerate(self.spline.control_points.detach().numpy())]
        )
        self.plot_widget.addItem(self.control_scatter)

        # Scatter plot for joint points
        self.joint_scatter = DraggableScatter(
            parent_editor=self,
            points_type="joint_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'g', 'symbol': 's', 'size': 12}
                for i, pt in enumerate(self.spline.joint_points.detach().numpy())]
        )
        self.plot_widget.addItem(self.joint_scatter)

        if self.sample_points is not None:
            self.sample_scatter = pg.ScatterPlotItem(
                pos=sample_points.detach().numpy(),
                brush=pg.mkBrush(255, 255, 0, 120),  # Yellow-ish, semi-transparent
                size=8,
                symbol='x'
            )
            self.plot_widget.addItem(self.sample_scatter)

        self.lines = [self.plot_widget.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color='gray', width=2)) for line in self.spline.get_lines()]
        # Spline curve plot
        self.spline_curve = self.plot_widget.plot([], pen=pg.mkPen(color='b', width=10))

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
        self.spline_curve.setData(spline_pts[:, 0].detach(), spline_pts[:, 1].detach())
        
        [self.lines[i].setData(line[:, 0], line[:, 1]) for i, line in enumerate(self.spline.get_lines())]

        # Update scatter plots
        self.control_scatter.setData(pos=self.spline.control_points.detach().numpy())
        self.joint_scatter.setData(pos=self.spline.joint_points.detach().numpy())


class SplinePlot:
    def __init__(self, plot_widget: pg.PlotWidget, spline: Spline, freeze: bool = False):
        self.plot_widget = plot_widget
        self.spline = spline
        self.freeze = freeze
        self.num_steps = 100

        # Scatter for control points (red circles)
        self.control_scatter = DraggableScatter(
            parent_plot=self,
            points_type="control_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
                   for i, pt in enumerate(self.spline.control_points.detach().numpy())]
        )
        self.plot_widget.addItem(self.control_scatter)

        # Scatter for joint points (green squares)
        self.joint_scatter = DraggableScatter(
            parent_plot=self,
            points_type="joint_points",
            freeze=freeze,
            spots=[{'pos': pt, 'data': i, 'brush': 'g', 'symbol': 's', 'size': 12}
                   for i, pt in enumerate(self.spline.joint_points.detach().numpy())]
        )
        self.plot_widget.addItem(self.joint_scatter)

        # Lines connecting control points (grey lines)
        self.lines = [self.plot_widget.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color='gray', width=2))
                      for line in self.spline.get_lines()]

        # The actual spline curve (blue line)
        self.curve = self.plot_widget.plot([], pen=pg.mkPen(color='b', width=4))
        self.update()

    def update(self):
        """Redraw the spline and its lines."""
        spline_pts = self.spline(torch.linspace(0.0, 1.0, steps=self.num_steps))
        self.curve.setData(spline_pts[:, 0].detach(), spline_pts[:, 1].detach())

        for i, line in enumerate(self.spline.get_lines()):
            self.lines[i].setData(line[:, 0], line[:, 1])

        self.control_scatter.setData(pos=self.spline.control_points.detach().numpy())
        self.joint_scatter.setData(pos=self.spline.joint_points.detach().numpy())

    def set_steps(self, steps: int):
        """Update the resolution of the curve."""
        self.num_steps = steps
        self.update()



class DraggableScatter(pg.ScatterPlotItem):
    def __init__(self, parent_editor, points_type, freeze: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_editor = parent_editor
        self.points_type = points_type  # Either "control_points" or "joint_points"
        self.dragged_index = None
        self.freeze = freeze

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

        # Update the corresponding points (control_points or joint_points)
        if self.points_type == "control_points":
            self.parent_editor.spline.control_points.data[self.dragged_index] = torch.tensor(
                [event.pos().x(), event.pos().y()], dtype=torch.float32
            )
        elif self.points_type == "joint_points":
            self.parent_editor.spline.joint_points.data[self.dragged_index] = torch.tensor(
                [event.pos().x(), event.pos().y()], dtype=torch.float32
            )

        # Update points visually
        self.parent_editor.update_spline()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.dragged_index = None
        event.accept()
