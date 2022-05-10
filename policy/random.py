from .base import BasePolicy
import numpy as np


class RandomPolicy(BasePolicy):
    name = "Random"

    def __init__(self, K):
        self.K = K
        self.log = []

    def initialize(self):
        pass

    def select_arm(self):
        return np.random.randint(0, self.K)

    def update_state(self, k, r):
        pass

    def __repr__(self):
        return self._generate_repr(f"K={self.K}")
