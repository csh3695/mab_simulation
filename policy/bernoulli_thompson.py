from .base import BasePolicy
import numpy as np


class BernoulliThompsonSamplingPolicy(BasePolicy):
    name = "Bernoulli_Thompson_Sampling"

    def __init__(self, K, a0=1, b0=1):
        self.K = K
        self.N = np.zeros(self.K).astype(int)
        self.R = np.zeros(self.K).astype(int)
        self.a0, self.b0 = a0, b0
        self.log = []

    def initialize(self):
        self.N.fill(0)
        self.R.fill(0)
        self.log = []

    def select_arm(self):
        self.log.append(np.random.beta(self.R + self.a0, self.N - self.R + self.b0))
        return self.log[-1].argmax()

    def update_state(self, k, r):
        self.N[k] += 1
        self.R[k] += int(r > 0)

    def __repr__(self):
        return self._generate_repr(f"K={self.K}, a0={self.a0}, b0={self.b0}")
