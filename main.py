from PyQt5 import QtWidgets
from GUI.editor import SplineEditor
import sys
import torch
from splines.spline import Spline

def optimize_spline(spline: Spline, sample_points: torch.Tensor, num_iterations: int = 1000, learning_rate: float = 0.01):
    """
    Optimize the spline control points to fit the given sample points.

    Parameters:
        spline (Spline): The spline object to optimize.
        sample_points (torch.Tensor): Tensor of sample points, shape (k, 2).
        num_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.Tensor: Optimized control points.
    """
    # Make control points trainable
    control_points = spline.control_points
    control_points.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([control_points], lr=learning_rate)

    # Optimization loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Evaluate the spline at evenly spaced parameter values
        t = torch.linspace(0, 1, len(sample_points), requires_grad=False)
        spline_points = spline.evaluate(t)

        # Compute the loss (mean squared error between spline points and sample points)
        loss = torch.mean((spline_points - sample_points) ** 2)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

    return control_points.detach()


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