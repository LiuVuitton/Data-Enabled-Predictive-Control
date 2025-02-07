import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override

class Model(ABC):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                 dim_x: int,
                 dim_u: int,
                 observe: Optional[Callable[[npt.NDarray], npt.NDArray]] | None,
                 dim_y: Optional[int] | None
                 ) -> None:
        self.dynamics = dynamics
        self.dim_x = dim_x
        self.dim_u = dim_u
        if observe is None or dim_y is None:
            def observe(x, u=None):
                return x
            self.observe = observe
            self.dim_y = dim_x
        else:
            self.observe = observe
            self.dim_y = dim_y
        self.x = ca.SX.sym('x', dim_x)
        self.u = ca.SX.sym('u', dim_u)
        self.y = ca.SX.sym('y', dim_y)

    def get_jacobian(self,
                     x_op: npt.NDArray,
                     u_op: npt.NDArray
                     ) -> Tuple[npt.NDArray]:
        f = self.dynamics(self.x, self.u)
        A = ca.jacobian(f, self.x)(x_op, u_op)
        B = ca.jacobian(f, self.u)(x_op, u_op)
        return (A, B)
    
    @abstractmethod
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'Model':
        pass
