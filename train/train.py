import time
from typing import Optional
from splines.spline import Spline
import torch
from GUI.editor import SplineEditor
from PyQt5 import QtWidgets, QtCore # type: ignore

class Trainer:
    def __init__(self, spline: Spline, sample_points: torch.Tensor, num_iterations: int = 1000, learning_rate: float = 0.01, device = None, editor: Optional[SplineEditor]=None):
        """
        Initialize the Trainer.

        Parameters:
            spline (Spline): The spline object to optimize.
            sample_points (torch.Tensor): Tensor of sample points, shape (k, 2).
            num_iterations (int): Number of optimization iterations.
            learning_rate (float): Learning rate for the optimizer.
            device (str or torch.device): Device for model (e.g., 'cpu', 'cuda').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.spline = spline.to(self.device)  # Send spline model to device
        self.sample_points = sample_points.to(self.device)  # Send sample points to device
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.spline.parameters(), lr=self.learning_rate)
        self.editor = editor

    def optimize(self):
        """
        Optimize the spline control points to fit the sample points.
        """
        t = torch.linspace(0, 1, self.sample_points.shape[1], requires_grad=False).detach()

        for iteration in range(self.num_iterations):

            if self.editor:
                QtWidgets.QApplication.processEvents()
                if self.editor.paused:
                    time.sleep(0.1)
                    continue
        
            self.optimizer.zero_grad()

            # Evaluate the spline at evenly spaced parameter values

            spline_points = self.spline(t)

            # Compute the loss (mean squared error between spline points and sample points)
            loss = torch.mean((spline_points - self.sample_points) ** 2)

            # Backpropagation
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if self.editor:
                self.editor.update_spline()
                self.editor.update_status(iteration, loss.item())


            # print(f"Iteration {iteration}, Loss: {loss.item()}")

    @staticmethod
    def generate_sample_points(spline: Spline, num_points: int, add_noise: bool = False, noise_std: float = 1.0, device=None) -> torch.Tensor:
        """
        Generate sample points from a given spline.

        Parameters:
            spline (Spline): The spline object to sample from.
            num_points (int): The number of sample points to generate.
            batch_size (int): The number of batches to generate.
            add_noise (bool): Whether to add noise to the sample points.
            noise_std (float): Standard deviation of the noise to add.
            device (str or torch.device): Device for model (e.g., 'cpu', 'cuda').

        Returns:
            torch.Tensor: Tensor of sample points, shape (num_points, 2).
        """
        t = torch.linspace(0, 1, steps=num_points)
        sample_points = spline(t)

        if add_noise:
            noise = torch.normal(mean=0.0, std=noise_std, size=sample_points.shape, device=device)
            sample_points += noise

        return sample_points