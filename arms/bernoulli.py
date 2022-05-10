from .base import BaseArm
import numpy as np


class BernoulliArm(BaseArm):
    name = "Bernoulli"

    def __init__(self, p):
        self.p = p

    def pull(self):
        return float(np.random.rand() <= self.p)

    def get_expected_reward(self):
        return self.p

    def __repr__(self):
        return self._generate_repr(f"p={self.p}")
