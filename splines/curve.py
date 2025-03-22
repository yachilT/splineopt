from abc import ABC
from math import comb
import numpy as np
class Curve(ABC):
    def __init__(self, degree=3):
        self.degree = degree
        self.char_mat = np.eye(self.degree + 1)
        if type(self) is Curve:
            raise TypeError("Cannot instansiate Curve. It is an abstract class")
    
    def evaluate(self, t: float, points: np.ndarray):

        return np.power(t, np.arange(self.degree+1)).reshape(1, -1) @ self.char_mat @ points.reshape(-1, 2)



class Bezier(Curve):
    def __init__(self, degree=3):
        super().__init__(degree)

        self.char_mat: np.ndarray = np.array([
            [self.bernstein_coeff(row, col) for col in range(degree + 1)] 
            for row in range(degree + 1)
        ])

    

    def bernstein_coeff(self, t_degree, point_index):
        """### bernstein_coeff function

            This function calculates the Bernstein coefficient for a given point index and a specific degree `t`.

            #### Formula
            The Bernstein coefficient is calculated using the following formula:

            .. math::
                    {n \choose i} {n - i \choose k} (-1)^{n - i - k} t^{n - k}

            Where:
            - \( n \) is the degree of the curve.
            - \( i \) is the point index (control point).
            - \( k \) is calculated as \( n - t\_degree \).

            #### Parameters:
            - **t_degree**: The degree of `t` used in the Bernstein polynomial.
            - **point_index**: The index of the control point.

            #### Returns:
            - The Bernstein coefficient for the given point and degree `t`."""
        return comb(self.degree, point_index) * comb(self.degree - point_index, self.degree - t_degree) * (-1) ** (t_degree - point_index)
