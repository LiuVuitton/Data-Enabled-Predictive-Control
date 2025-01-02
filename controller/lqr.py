from controller.statefeedback import StateFeedbackController
import numpy as np
import numpy.typing as npt
import scipy
from typing import Optional


class LQR(StateFeedbackController):
    def __init__(self,
                 A: npt.NDArray,
                 B: npt.NDArray,
                 Q: npt.NDArray,
                 R: npt.NDArray,
                 horizon_length: Optional[int] = None
                 ) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.horizon_length = horizon_length
        self._setup_LQR()
        
    def _setup_LQR(self):
        P = np.zeros_like(self.Q)
        K = np.zeros((self.B.shape[1], self.A.shape[0]))
        if self.horizon_length is None:
            P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        else:
            P = self.Q
            for _ in range(self.horizon_length):
                P = (self.A.T @ P @ self.A 
                     - (self.A @ P @ self.B) 
                     @ np.linalg.pinv(self.R + self.B.T @ P @ self.B) 
                     @ (self.B.T @ P @ self.A) + self.Q)
        self.K = np.linalg.pinv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A

