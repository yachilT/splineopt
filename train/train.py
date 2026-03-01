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
        if editor is not None:
            editor.on_params_changed = lambda *args: self.reset_optimizer(*args)

    def reset_optimizer(self, old_jp=None, old_cp=None, split_mask=None, old_max_intervals=None):
        """Reinitialize the optimizer after a split, preserving Adam moments for unaffected curves."""
        old_state = self.optimizer.state
        self.optimizer = torch.optim.Adam(self.spline.parameters(), lr=self.learning_rate)

        # If we have no old state info, or the old param was never stepped, nothing to copy
        if old_jp is None or old_jp not in old_state:
            return
        old_jp_s = old_state[old_jp]
        old_cp_s = old_state.get(old_cp, {})
        if 'exp_avg' not in old_jp_s:
            return

        k_per = self.spline.curve.degree - 1
        old_jp_cols = old_max_intervals + 1
        old_cp_cols = old_max_intervals * k_per
        unaffected = ~split_mask

        new_jp = self.spline.joint_points
        new_cp = self.spline.control_points

        # Build moment tensors for joint_points, zeroed for split/new curves
        new_jp_avg = torch.zeros_like(new_jp)
        new_jp_avg_sq = torch.zeros_like(new_jp)
        new_jp_avg[unaffected, :old_jp_cols] = old_jp_s['exp_avg'][unaffected, :old_jp_cols]
        new_jp_avg_sq[unaffected, :old_jp_cols] = old_jp_s['exp_avg_sq'][unaffected, :old_jp_cols]

        # Build moment tensors for control_points
        new_cp_avg = torch.zeros_like(new_cp)
        new_cp_avg_sq = torch.zeros_like(new_cp)
        if 'exp_avg' in old_cp_s:
            new_cp_avg[unaffected, :old_cp_cols] = old_cp_s['exp_avg'][unaffected, :old_cp_cols]
            new_cp_avg_sq[unaffected, :old_cp_cols] = old_cp_s['exp_avg_sq'][unaffected, :old_cp_cols]

        self.optimizer.state[new_jp] = {
            'step': old_jp_s['step'],
            'exp_avg': new_jp_avg,
            'exp_avg_sq': new_jp_avg_sq,
        }
        self.optimizer.state[new_cp] = {
            'step': old_jp_s['step'],
            'exp_avg': new_cp_avg,
            'exp_avg_sq': new_cp_avg_sq,
        }

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