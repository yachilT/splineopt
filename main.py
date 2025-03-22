
from PyQt5 import QtWidgets
from GUI.editor import SplineEditor
import sys

print("creating app")
app = QtWidgets.QApplication(sys.argv)
print("app created")

print("creating widget")
# Create and show the main widget
main_window = SplineEditor()

print("widget created")
main_window.show()

# Run the Qt application
app.exec_()  # Starts the event loop