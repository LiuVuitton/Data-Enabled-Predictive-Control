import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt
from system.abstract import System
from system.discrete import DiscreteSystem
from system.helper import discretize_euler, discretize_rk4, discretize_exact

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override


class ContinuousSystem(System):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                 dim_x: int,
                 dim_u: int,
                 ) -> None:
        super().__init__(dynamics, dim_x, dim_u)

    @override
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'ContinuousSystem':
        A, B, C, D = self.get_jacobian(x_op, u_op)
        dynamics_lin = lambda x, u: A @ x + B @ u
        return ContinuousSystem(dynamics_lin, self.dim_x, self.dim_u)
    
    def discretize(self,
                   sample_time: float,
                   discretization_method: str
                   ) -> 'DiscreteSystem':
        match discretization_method:
            case 'euler':
                def dynamics_discrete(x, u):
                    return x + sample_time * self.dynamics(x, u)
                
            case 'rk4':
                def dynamics_discrete(x, u):
                    k1 = self.dynamics(x, u)
                    k2 = self.dynamics(x + 0.5 * sample_time * k1, u)
                    k3 = self.dynamics(x + 0.5 * sample_time * k2, u)
                    k4 = self.dynamics(x + sample_time * k3, u)
                    return x + (sample_time / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return DiscreteSystem(dynamics_discrete, self.dim_x, self.dim_u, sample_time)
    

class LinearContinuousSystem(ContinuousSystem):
    def __init__(self,
                 A: npt.NDArray,
                 B: npt.NDArray,
                 dim_x: int,
                 dim_u: int,
                 ) -> None:
        def dynamics(x, u):
            return A @ x + B @ u
        super().__init__(dynamics, dim_x, dim_u)
        self.A = A
        self.B = B

    # TODO change this to return a linear discrete system
    @override
    def discretize(self,
                   sample_time: float,
                   discretization_method: str
                   ) -> 'DiscreteSystem':
        match discretization_method:
            case 'euler':
                def dynamics_discrete(x, u):
                    return x + sample_time * self.dynamics(x, u)
                
            case 'rk4':
                def dynamics_discrete(x, u):
                    k1 = self.dynamics(x, u)
                    k2 = self.dynamics(x + 0.5 * sample_time * k1, u)
                    k3 = self.dynamics(x + 0.5 * sample_time * k2, u)
                    k4 = self.dynamics(x + sample_time * k3, u)
                    return x + (sample_time / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            case 'exact':
                def dynamics_discrete(x, u):
                    A_d = scipy.linalg.expm(self.A * sample_time)
                    B_d = np.linalg.pinv(A) @ (A_d - np.eye(self.A.shape[0])) @ self.B
                    return A_d @ x + B_d @ u

        return DiscreteSystem(dynamics_discrete, self.dim_x, self.dim_u, sample_time)