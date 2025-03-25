import numpy as np
import torch
from .curve import Curve

class Spline:
    def __init__(self, control_points: torch.Tensor, joint_points: torch.Tensor, curve: Curve):
        """
        Initialize the spline object with control points, joint points, and curve object.
        
        Parameters:
            control_points (torch.Tensor): Tensor of control points, shape (n, 2).
            joint_points (torch.Tensor): Tensor of joint points, shape (m, 2).
            curve (Curve): Curve object for evaluating the spline.
        """
        if (len(joint_points) - 1) * (curve.degree - 1) != len(control_points):
            raise ValueError("Number of control points doesn't fit the bezier degree")

        self.control_points = control_points
        self.joint_points = joint_points
        self.curve = curve
        self.__ctrl_pts_per_section = self.curve.degree - 1

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the spline at given parameter values `t`.

        Parameters:
            t (torch.Tensor): Tensor of parameter values in the range [0, 1].

        Returns:
            torch.Tensor: Tensor of points on the spline corresponding to `t`.
        """
        epsilon = 1e-10
        t = torch.clamp(t, max=1.0 - epsilon)

        u = (len(self.joint_points) - 1) * t
        point_index = torch.floor(u).long()
        polynomial_t = u - point_index

        current_control = torch.stack([
            self.control_points[point_index[i] * self.__ctrl_pts_per_section : (point_index[i] + 1) * self.__ctrl_pts_per_section]
            for i in range(len(point_index))
        ])

        return self.curve.evaluate(polynomial_t, torch.cat([self.joint_points[point_index].unsqueeze(1), current_control, self.joint_points[point_index + 1].unsqueeze(1)], dim=1))
    
    def get_lines(self):
        lines = []

        for i in range(len(self.joint_points)):
            if i > 0:
                lines.append(np.array([self.joint_points[i], self.control_points[self.__ctrl_pts_per_section * i - 1]]))
            
            if i < len(self.joint_points) - 1:
                lines.append(np.array([self.joint_points[i], self.control_points[self.__ctrl_pts_per_section * i]]))
        
        return lines






