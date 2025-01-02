import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override

class System(ABC):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                 dim_x: int,
                 dim_u: int,
                 ) -> None:
        self.dynamics = dynamics
        self.x = ca.MX.sym('x', dim_x)
        self.u = ca.MX.sym('u', dim_u)
        self.dim_x = dim_x
        self.dim_u = dim_u

    def get_jacobian(self,
                     x_op: npt.NDArray,
                     u_op: npt.NDArray
                     ) -> Tuple[npt.NDArray]:
        f = self.dynamics(self.x, self.u)
        A = ca.jacobian(f, x)(x_op, u_op)
        B = ca.jacobian(f, u)(x_op, u_op)
        return (A, B)
    
    @abstractmethod
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'System':
        pass

    def observe(self, x, u):
        return x
    