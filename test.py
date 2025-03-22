import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)

    plot_widget = pg.PlotWidget()
    layout.addWidget(plot_widget)
    window.setWindowTitle("PyQtGraph Test")
    window.show()

    app.exec_()
