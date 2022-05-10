from .base import BasePolicy
import numpy as np


class EpsGreedyPolicy(BasePolicy):
    name = "Epsilon_Greedy"

    def __init__(self, K, eps):
        self.K = K
        self.eps = eps
        self.N = np.zeros(self.K).astype(int)
        self.R = np.zeros(self.K).astype(float)
        self.log = []

    def initialize(self):
        self.N.fill(0)
        self.R.fill(0)
        self.log = []

    def select_arm(self):
        if np.random.random() < self.eps:
            self.log.append("None")
            return np.random.randint(0, self.K)
        self.log.append(self.R / self.N.clip(1e-12))
        return np.argmax(self.log[-1])

    def update_state(self, k, r):
        self.N[k] += 1
        self.R[k] += r

    def __repr__(self):
        return self._generate_repr(f"K={self.K}, eps={self.eps}")
