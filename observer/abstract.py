from typing import Optional
import numpy as np
import casadi as ca
from abc import ABC

import numpy.typing as npt

class Observer():
    def __init__(self):
        pass

    def observe(self,
                x: npt.NDArray,
                u: Optional[npt.NDArray] | None,
                w: Optional[npt.NDArray] | None
                ) -> npt.NDArray:
        """
        """
        return x

class LinearObserver():
    def __init__(self):