from .base import BaseArm
import numpy as np


class ExponentialArm(BaseArm):
    name = "Exponential"

    """
    f(x) = lambd * exp(-lambd * x)
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def pull(self):
        return np.random.exponential(1 / self.lambd)

    def get_expected_reward(self):
        return 1 / self.lambd

    def __repr__(self):
        return self._generate_repr(f"lambda={self.lambd}")
