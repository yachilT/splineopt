from abc import ABC
try:
    from math import comb
except ImportError:
    from scipy.special import comb
import numpy as np
import torch
class Curve(ABC):
    def __init__(self, degree=3):
        self.degree: int = degree
        self.char_mat: torch.Tensor = torch.eye(self.degree + 1)
        if type(self) is Curve:
            raise TypeError("Cannot instansiate Curve. It is an abstract class")
    
    def evaluate(self, t_powers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the curve efficiently without per-curve expansion of t_powers or char_tensor.
        
        Args:
            t_powers: (num_intervals, max_pts, D+1)
            points: (num_curves, num_intervals, D+1, dim)
        
        Returns:
            result: (num_curves, num_intervals, max_pts, dim)
        """
        # Apply the characteristic matrix: (max_pts, D+1) @ (D+1, D+1) = (max_pts, D+1)
        char_mat = self.char_mat.to(points.device)  # (D+1, D+1)
        t_transformed = t_powers @ char_mat  # (num_intervals, max_pts, D+1)

        # Reshape to allow broadcasting:
        # (1, num_intervals, max_pts, D+1)
        t_transformed = t_transformed.unsqueeze(0)  # batch dim = 1 for num_curves

        # points: (num_curves, num_intervals, D+1, dim)
        # We want: (num_curves, num_intervals, max_pts, dim)
        result = torch.einsum('bipk,bikd->bipd', t_transformed, points)

        return result




class Bezier(Curve):
    def __init__(self, degree=3):
        super().__init__(degree)

        # Create a tensor where each slice is the characteristic matrix for the given degree
        self.char_mat: torch.Tensor = torch.tensor(
            [
                [self.bernstein_coeff(row, col) for col in range(degree + 1)]
                for row in range(degree + 1)
            ], dtype=torch.float32
        )

    
    def bernstein_coeff(self, t_degree, point_index):
        """
        Calculates the Bernstein coefficient for a given point index and a specific degree `t`.

        Parameters:
            t_degree (int): The degree of `t` used in the Bernstein polynomial.
            point_index (int): The index of the control point.

        Returns:
            float: The Bernstein coefficient for the given point and degree `t`.
        """
        return comb(self.degree, point_index) * comb(self.degree - point_index, self.degree - t_degree) * (-1) ** (t_degree - point_index)
