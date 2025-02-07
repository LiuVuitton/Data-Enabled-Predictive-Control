import numpy as np
import casadi as ca
from model.continuous import ContinuousModel

class Pendulum(ContinuousModel):
    def __init__(self, length, mass, damping, gravity=9.81):
        self.length = length
        self.mass = mass
        self.damping = damping
        self.gravity = gravity
        super().__init__(self.dynamics, dim_x=2, dim_u=1)

    # TODO finish this
    def dynamics(self, x, u):
        length = self.length
        mass = self.mass
        damping = self.damping
        gravity = self.gravity
        theta = x[0]
        omega = x[1]
        torque = u
        theta_dot = omega
        omega_dot = (- damping / (mass * length**2) * omega 
                     - gravity / length * ca.sin(theta)
                     + torque / (mass * length**2))
        return ca.vertcat(theta_dot, omega_dot)