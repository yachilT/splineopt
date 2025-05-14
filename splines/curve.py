from abc import ABC
from math import comb
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
        Evaluates the curve at given parameter values `t_powers` and control points `points`.
        This method performs the matrix multiplication to compute the curve points.
        The characteristic matrix is used to transform the control points.
        Parameters:
            t_powers (torch.Tensor): Tensor of powers of parameter values, shape (batch_size, num_intervals, degree + 1).
            points (torch.Tensor): Tensor of control points, shape (batch_size, num_intervals, degree + 1, dim).
            Returns:
        torch.Tensor: Tensor of points on the curve, shape (batch_size, num_intervals, max_pts_in_interval, dim).
        """
        batch_size = t_powers.shape[0]
        num_intervals = t_powers.shape[1]

        # Dynamically create the characteristic matrix tensor by stacking the single slice
        char_tensor = self.char_mat.to(points.device).view(1, 1, self.degree+1, self.degree+1).repeat(batch_size, num_intervals, 1, 1)


        # Perform matrix multiplication
        result = t_powers @ char_tensor @ points
        return result # (batch_size, num_intervals, max_pts_in_interval, dim)



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
