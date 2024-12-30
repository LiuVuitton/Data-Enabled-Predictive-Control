import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt
from system.abstract import System

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override


class DiscreteSystem(System):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDarray],
                 dim_x: int,
                 dim_u: int,
                 sample_time: Optional[int] = 1.0
                 ) -> None:
        super().__init__(dynamics, dim_x, dim_u)
        self.sample_time = sample_time

    def step(self,
             x: npt.NDArray,
             u: npt.NDArray
             ) -> npt.NDArray:
        return self.dynamics(x, u)
    
    @override
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'DiscreteSystem':
        A, B = self.get_jacobian(x_op, u_op)
        dynamics_lin = lambda x, u: A @ x + B @ u
        return DiscreteSystem(dynamics_lin, self.dim_x, self.dim_u, self.sample_time)