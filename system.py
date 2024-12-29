import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, override

class System(ABC):
    def __init__(self, 
                 dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDarray],
                 dim_x: int,
                 dim_u: int,
                 ) -> None:
        self.dynamics = dynamics
        self.dim_x = dim_x
        self.dim_u = dim_u

    def get_jacobian(self,
                     x_op: npt.NDArray,
                     u_op: npt.NDArray
                     ) -> Tuple[npt.NDArray]:
        x = ca.MX.sym('x', self.dim_x)
        u = ca.MX.sym('u', self.dim_u)
        f = self.dynamics(x, u)
        A = ca.jacobian(f, x)(x_op, u_op)
        B = ca.jacobian(f, u)(x_op, u_op)
        return (A, B)
    
    @abstractmethod
    def linearize(self, x_op: npt.NDArray, u_op: npt.NDArray) -> 'System':
        pass
        


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
                dynamics_discrete = _discretize_euler(self.dynamics, self.x, self.u, sample_time)
                
            case 'rk4':
                dynamics_discrete = _discretize_rk4(self.dynamics, self.x, self.u, sample_time)

        return DiscreteSystem(dynamics_discrete, self.dim_x, self.dim_u, sample_time)


def _discretize_euler(dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDarray],
                      x: npt.NDArray,
                      u: npt.NDArray,
                      sample_time: float
                      ) -> npt.NDArray:
    return x + sample_time * dynamics(x, u)

def _discretize_rk4(dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDarray],
                      x: npt.NDArray,
                      u: npt.NDArray,
                      sample_time: float
                      ) -> npt.NDArray:
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * sample_time * k1, u)
    k3 = dynamics(x + 0.5 * sample_time * k2, u)
    k4 = dynamics(x + sample_time * k3, u)
    return x + (sample_time / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    

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
                dynamics_discrete = _discretize_euler(self.dynamics, self.x, self.u, sample_time)
                
            case 'rk4':
                dynamics_discrete = _discretize_rk4(self.dynamics, self.x, self.u, sample_time)
            
            case 'exact':
                dynamics_discrete = _discretize_exact(self.A, self.B, self.x, self.u, sample_time)

        return DiscreteSystem(dynamics_discrete, self.dim_x, self.dim_u, sample_time)
    
def _discretize_exact(A: npt.NDArray,
                    B: npt.NDArray,
                    x: npt.NDArray,
                    u: npt.NDArray,
                    sample_time: float
                    ) -> npt.NDArray:
    A_d = scipy.linalg.expm(A * sample_time)
    B_d = np.linalg.pinv(A) @ (A_d - np.eye(A.shape[0])) @ B
    return A_d @ x + B_d @ u
    

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