import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from splines.curve import Bezier 
from splines.spline import Spline  


class SplineEditor(QtWidgets.QWidget):
    def __init__(self):
        print("init strating")
        super().__init__()

        self.joint_points = [(100, 200), (200, 300), (300, 100)] 
        self.control_points = [(50, 150), (400, 200)]  
                


        self.spline = Spline(self.control_points, self.joint_points, Bezier(degree=2))

        # Set up the PyQtGraph plot
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot_widget)

        # Add control points as draggable points (ROIs)
        self.draggables = []
        for pt in self.control_points:
            roi = pg.CircleROI(pt, radius=10)
            roi.setPen(pg.mkPen(color='r', width=2,))
            roi.setSize((10, 10))  # Fixed size for the circle (radius)

            roi.sigRegionChanged.connect(self.update_spline)
            self.plot_widget.addItem(roi)
            self.draggables.append(roi)

        # Initialize the spline curve on the plot
        self.spline_curve = self.plot_widget.plot([], pen='b')
        self.update_spline()

    def update_spline(self):
        # Get current positions from draggable points
        new_control_points = [tuple(roi.pos()) for roi in self.draggables]

        # Update the control points of the existing spline
        self.spline.control_points = new_control_points

        # Recompute the spline points
        spline_pts = np.vstack([self.spline.evaluate(t) for t in np.linspace(0, 1, 200)])

        # Update the plot with the new spline curve
        self.spline_curve.setData(spline_pts[:, 0], spline_pts[:, 1])

