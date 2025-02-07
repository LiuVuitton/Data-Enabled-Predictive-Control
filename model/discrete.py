import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt
from model.abstract import Model

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override


class DiscreteModel(Model):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                 dim_x: int,
                 dim_u: int,
                 sample_time: Optional[int] = 1.0
                 ) -> None:
        super().__init__(dynamics, dim_x, dim_u)
        self.sample_time = sample_time
    
    @override
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'DiscreteModel':
        A, B = self.get_jacobian(x_op, u_op)
        dynamics_lin = lambda x, u: A @ x + B @ u
        return DiscreteModel(dynamics_lin, self.dim_x, self.dim_u, self.sample_time)
    

# TODO not relevant for now
class LinearDiscreteSystem(DiscreteModel):
    def __init__(self,
                 A: npt.NDArray,
                 B: npt.NDArray,
                 dim_x: int,
                 dim_u: int,
                 sample_time: float
                 ) -> None:
        def dynamics(x, u):
            return A @ x + B @ u
        super().__init__(dynamics, dim_x, dim_u)
        self.A = A
        self.B = B