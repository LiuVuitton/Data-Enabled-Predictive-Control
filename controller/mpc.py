import numpy as np
import casadi as ca
import scipy

from controller.abstract import Controller
from controller.tools import tracking_cost, slack_cost

class MPC(Controller):
    def __init__(self, model, mpc_opts, observer=None, solver_opts=None):
        opti = ca.Opti()

        dim_x = model.dim_x
        dim_u = model.dim_u
        dim_y = model.dim_y
        N = mpc_opts['prediction_horizon']

        self.x = opti.variable(dim_x, N+1)
        self.u = opti.variable(dim_u, N)
        self.y = opti.variable(dim_y, N)
        self.eps_y = opti.variable(dim_x, N)
        self.eps_u = opti.variable(dim_u, N)

        self.x_init = opti.parameter(dim_x)
        self.y_ref = opti.parameter(dim_x, N)

        self.opti = opti
        self.model = model
        self.mpc_opts = mpc_opts
        self.solver_opts = solver_opts
        self._setup_MPC()

    def control(self, x_init, y_traj):
        # TODO
        opti = self.opti
        opti.set_value(self.x_init, x_init)
        opti.set_value(self.y, y_traj)

        try:
            opti.solve()
        except:
            pass
        solution = {'x': opti.value(self.x),
                    'u': opti.value(self.u),
                    'y': opti.value(self.y),
                    'eps_y': opti.value(self.eps_y),
                    'eps_u': opti.value(self.eps_u),
                    'cost': opti.value(self.cost)}
        u0 = solution['u'][:, 0]

        return u0, solution

    def _setup_MPC(self):
        self._setup_objective()
        self._setup_constraints()
        self._setup_solver()

    def _setup_objective(self):
        # TODO the slack cost should be changed to account for exact penalization
        opti = self.opti
        N = self.mpc_opts['prediction_horizon']
        y = self.y
        u = self.u

        eps_y = self.eps_y
        eps_u = self.eps_u

        y_ref = self.y

        Q = self.mpc_opts['Q']
        R = self.mpc_opts['R']
        lam_y = self.mpc_opts['lam_y']
        lam_u = self.mpc_opts['lam_u']

        cost = 0
        cost_slack = 0
        for i in range(N):
            y_error = y_ref[:, i] - y[:, i]
            y_viol = eps_y[:, i]
            u_viol = eps_u[:, i]
            cost += (y_error.T @ Q @ y_error 
                          + u[:, i].T @ R @ u[:, i])
            cost_slack += y_viol.T @ lam_y @ y_viol + u_viol.T @ lam_u @ u_viol
        cost = cost + cost_slack
        opti.minimize(cost)
        self.cost = cost

    def _setup_constraints(self):
        opti = self.opti
        N = self.mpc_opts['prediction_horizon']
        f = self.model.dynamics
        h = self.model.observe
        x = self.x
        u = self.u
        y = self.y

        eps_y = self.eps_y
        eps_u = self.eps_u

        x_init = self.x_init
        y_ref = self.y_ref

        lby = self.mpc_opts['lby']
        uby = self.mpc_opts['uby']
        lbu = self.mpc_opts['lbu']
        ubu = self.mpc_opts['ubu']

        # Initial condition
        opti.subject_to(x[:, 0] == x_init)

        # Path constraints
        for i in range(N):
            x_next = f(x[:, i], u[:, i])
            y_obs = h(x_next)
            opti.subject_to(x[:, i+1] == x_next)
            opti.subject_to(y[:, i] == y_obs)
            
            slack_y = eps_y[:, i]
            slack_u = eps_u[:, i]
            opti.subject_to(lby - slack_y <= y[:, i])
            opti.subject_to(y[:, i] <= uby + slack_y)
            opti.subject_to(lbu - slack_u <= u[:, i])
            opti.subject_to(u[:, i] <= ubu + slack_u)

        opti.subject_to(eps_y >= 0)
        opti.subject_to(eps_u >= 0)

        # No terminal constraints

    def _setup_solver(self):
        opti = self.opti
        solver_name = self.solver_opts['name']
        solver_opts = self.solver_opts['opts']
        opti.solver(solver_name, solver_opts)