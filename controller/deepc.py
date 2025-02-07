from typing import override
import numpy as np
import casadi as ca
import numpy.typing as npt
from controller.abstract import Controller

class DeePC(Controller):
    def __init__(self,
                 u_d: npt.NDArray,
                 y_d: npt.NDArray,
                 Q: npt.NDArray,
                 R: npt.NDArray,
                 initial_length: int,
                 horizon_length: int
                 ) -> None:
        self.u_d = u_d
        self.y_d = y_d
        self.initial_length = initial_length
        self.horizon_length = horizon_length
        self._setup_hankel(u_d, y_d, initial_length, horizon_length)
        self._setup_DeePC()

    @override
    def control(self,
                y_ref: npt.NDArray,
                u_init: npt.NDArray,
                y_init: npt.NDArray,
                ) -> npt.NDArray:
        pass

    def _setup_hankel(self) -> None:
        U_d = hankel(self.u_d, self.initial_length + self.horizon_length)
        Y_d = hankel(self.y_d, self.initial_length + self.horizon_length)
        self.U_p = U_d[:self.initial_length]
        self.U_f = U_d[self.initial_length:]
        self.Y_p = Y_d[:self.initial_length]
        self.Y_f = Y_d[self.initial_length:]

    def _setup_DeePC(self):
        dim_u = self.u_d.shape[0]
        dim_y = self.y_d.shape[0]
        T_init = self.initial_length
        N = self.horizon_length
        Q = self.Q
        R = self.R
        U_p = self.U_p
        Y_p = self.Y_p
        U_f = self.U_f
        Y_f = self.Y_f

        opti = ca.Opti()
        # Optimization variables.
        opti = opti
        u = opti.variable(dim_u, N)
        y = opti.variable(dim_y, N)
        g = opti.variable(self.U_p.shape[1])

        # Parameters.
        y_ref = opti.parameter(dim_y, N)
        u_init = opti.parameter(dim_u, T_init)
        y_init = opti.parameter(dim_y, T_init)

        # Objective
        objective = 0
        for i in range(N):
            objective += (
                (u[:, i].T @ R @  u[:, i]) 
                + (y[:, i] - y_ref[:, i]).T @ Q @ (y[:, i] - y_ref[:, i]))
        opti.minimize(objective)

        # Constraints.
        opti.subject_to(ca.vertcat(U_p, Y_p, U_f, Y_f) @ g == ca.vertcat(u_init, y_init, u, y))
        # TODO finish implementing constraints.
        for i in range(N):
            pass

        # Solver.
        opti.solver('ipopt')
        self.opti = opti
        self.opti_vars = {'u': u, 'y': y, 'g': g}
        self.opti_params = {'y_ref': y_ref, 'u_init': u_init, 'y_init': y_init}


def hankel(signal_sequence: npt.NDArray, 
           traj_length: int
           ) -> npt.NDArray:
    '''TODO
    '''
    dim = signal_sequence.shape[0]
    sequence_length = signal_sequence.shape[1]
    hankel_matrix = np.zeros((dim * traj_length, sequence_length - traj_length + 1))
    for i in range(traj_length):
        hankel_matrix[i, :] = signal_sequence[:, i:i + sequence_length - traj_length]
    return hankel_matrix