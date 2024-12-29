from typing import override
from controller.abstract import AbstractController
import numpy as np
import numpy.typing as npt

class PID(AbstractController):
    def __init__(self,
                 Kp: npt.NDArray,
                 Ki: npt.NDArray,
                 Kd: npt.NDArray,
                 sample_time: float = 1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(Kp.shape[0])
        self.previous_error = np.zeros(Kp.shape[0])

    @override
    def control(self,
                y: npt.NDarray,
                ref: npt.NDArray
                ) -> npt.NDArray:
        error = ref - y
        dt = self.sample_time
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        control_signal = self.Kp @ error + self.Ki @ self.integral + self.Kd @ derivative
        return control_signal

    @override
    def reset(self):
        self.integral = np.zeros(self.Kp.shape[0])
        self.previous_error = np.zeros(self.Kp.shape[0])