from typing import override
from controller.abstract import AbstractController
import numpy.typing as npt

class StateFeedbackController(AbstractController):
    def __init__(self, K: npt.NDArray):
        self.K = K

    @override
    def control(self, x: npt.NDArray, x_ref: npt.NDArray) -> npt.NDArray:
        if x_ref is None:
            x_ref = np.zeros_like(x)
        return - self.K @ (x - x_ref)