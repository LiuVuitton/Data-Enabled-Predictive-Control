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
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDarray],
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
                dynamics_discrete = discretize_euler(self.dynamics, self.x, self.u, sample_time)
                
            case 'rk4':
                dynamics_discrete = discretize_rk4(self.dynamics, self.x, self.u, sample_time)

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

    @override
    def discretize(self,
                   sample_time: float,
                   discretization_method: str
                   ) -> 'DiscreteSystem':
        match discretization_method:
            case 'euler':
                dynamics_discrete = discretize_euler(self.dynamics, self.x, self.u, sample_time)
                
            case 'rk4':
                dynamics_discrete = discretize_rk4(self.dynamics, self.x, self.u, sample_time)
            
            case 'exact':
                dynamics_discrete = discretize_exact(self.A, self.B, self.x, self.u, sample_time)

        return DiscreteSystem(dynamics_discrete, self.dim_x, self.dim_u, sample_time)