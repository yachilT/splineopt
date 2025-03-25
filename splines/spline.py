from typing import List, Tuple
import numpy as np
from .curve import Curve

class Spline:
    def __init__(self, control_points: List[Tuple[float, float]], joint_points: List[Tuple[float, float]], curve: Curve):
        """
        Initialize the spline object with control points, joint points, and degree of the spline.
        
        Parameters:
            control_points (np.ndarray): Array of control points, shape (n, 2).
            joint_points (np.ndarray): Array of joint points (sample points), shape (m, 2).
            degree (int): Degree of the spline, default is cubic B-splines (degree=3).
        """

        if (len(joint_points) - 1) * (curve.degree - 1) != len(control_points):
            raise ValueError("number of control points doesn't fit the bezier degree")
        
        

        self.control_points = control_points
        self.joint_points = joint_points
        self.curve = curve
        self.__ctrl_pts_per_section = self.curve.degree - 1

    
    def evaluate(self, t):
        """
            Evaluates the spline at a given parameter `t`.

            Parameters:
                t (float): The parameter value in the range [0, 1] for spline evaluation.

            Returns:
                np.ndarray: The point on the spline at the given parameter `t`.

            The method computes the appropriate spline segment using the joint and control points, 
            then evaluates the curve's position for the calculated segment.
        """
        epsilon = 1e-10
        t = min(t, 1.0 - epsilon)

        u = (len(self.joint_points) - 1) * t


        point_index, polynomial_t = divmod(u, 1)
        point_index = int(point_index)

        current_control = self.control_points[point_index * self.__ctrl_pts_per_section : (point_index + 1) * self.__ctrl_pts_per_section]

        return self.curve.evaluate(polynomial_t, np.array([self.joint_points[point_index]] + current_control + [self.joint_points[point_index + 1]]))
    
    def get_lines(self):
        lines = []

        for i in range(len(self.joint_points)):
            if i > 0:
                lines.append(np.array([self.joint_points[i], self.control_points[self.__ctrl_pts_per_section * i - 1]]))
            
            if i < len(self.joint_points) - 1:
                lines.append(np.array([self.joint_points[i], self.control_points[self.__ctrl_pts_per_section * i]]))
        
        return lines


    

        

