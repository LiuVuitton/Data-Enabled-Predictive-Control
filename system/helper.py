import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import numpy.typing as npt

from typing import Optional, Callable, List, Tuple, override

def discretize_euler(dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                      x: npt.NDArray,
                      u: npt.NDArray,
                      sample_time: float
                      ) -> npt.NDArray:
    return x + sample_time * dynamics(x, u)

def discretize_rk4(dynamics: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                      x: npt.NDArray,
                      u: npt.NDArray,
                      sample_time: float
                      ) -> npt.NDArray:
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * sample_time * k1, u)
    k3 = dynamics(x + 0.5 * sample_time * k2, u)
    k4 = dynamics(x + sample_time * k3, u)
    return x + (sample_time / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def discretize_exact(A: npt.NDArray,
                    B: npt.NDArray,
                    x: npt.NDArray,
                    u: npt.NDArray,
                    sample_time: float
                    ) -> npt.NDArray:
    A_d = scipy.linalg.expm(A * sample_time)
    B_d = np.linalg.pinv(A) @ (A_d - np.eye(A.shape[0])) @ B
    return A_d @ x + B_d @ u