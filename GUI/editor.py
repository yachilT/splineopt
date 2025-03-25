import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from splines.curve import Bezier
from splines.spline import Spline


class SplineEditor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spline Editor")

        self.joint_points = [(100.0, 200.0), (200.0, 300.0), (300.0, 100.0)]
        self.control_points = [(50.0, 150.0), (100.0, 150.0), (400.0, 200.0), (200.0, 200.0)]

        self.spline = Spline(self.control_points, self.joint_points, Bezier(degree=3))


        # Layout and plot widget
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot_widget)

        # Scatter plot for control points
        self.scatter = DraggableScatter(
            parent_editor=self,
            spots=[{'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
                for i, pt in enumerate(self.control_points)]
        )
        self.plot_widget.addItem(self.scatter)

        self.lines = [self.plot_widget.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color='gray', width=2)) for line in self.spline.get_lines()]
        # Spline curve plot
        self.spline_curve = self.plot_widget.plot([], pen=pg.mkPen(color='b', width=10))

        self.update_spline()

    def update_spline(self):
        self.spline.control_points = self.control_points
        spline_pts = np.vstack([self.spline.evaluate(t) for t in np.linspace(0, 1, 200)])
        self.spline_curve.setData(spline_pts[:, 0], spline_pts[:, 1])
        
        [self.lines[i].setData(line[:, 0], line[:, 1]) for i, line in enumerate(self.spline.get_lines())]


class DraggableScatter(pg.ScatterPlotItem):
    def __init__(self, parent_editor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_editor = parent_editor
        self.dragged_index = None
        self.mouse_offset = 0

    def mousePressEvent(self, event):
        points = self.pointsAt(event.pos())
        if points.size > 0:
            self.dragged_index = points[0].data()

            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        print(f"mouse pos: {event.pos()}")
        if self.dragged_index is None:
            return

        # Update the control point with the new position
        self.parent_editor.control_points[self.dragged_index] = (event.pos().x(), event.pos().y())

        # Update points visually
        self.setData([
            {'pos': pt, 'data': i, 'brush': 'r', 'symbol': 'o', 'size': 10}
            for i, pt in enumerate(self.parent_editor.control_points)
        ])
        self.parent_editor.update_spline()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.dragged_index = None
        event.accept()
