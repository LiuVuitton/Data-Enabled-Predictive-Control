from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def control(self, x, ref):
        pass

    @abstractmethod
    def reset(self):
        pass