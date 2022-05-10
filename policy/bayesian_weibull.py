from .base import BasePolicy
import numpy as np


class BayesianWeibullSamplingPolicy(BasePolicy):
    name = "Bayesian_Weibull_Sampling"

    def __init__(self, K, k, a0=1e-12, b0=1e-12):
        self.K = K
        self.k = k
        self.N = np.zeros(self.K).astype(int)
        self.R = np.zeros(self.K).astype(float)
        self.a0, self.b0 = a0, b0
        self.log = []

    def initialize(self):
        self.N.fill(0)
        self.R.fill(0)
        self.log = []

    def select_arm(self):
        self.log.append(np.random.gamma(self.a0 + self.N, 1 / (self.b0 + self.R)))
        return self.log[-1].argmax()

    def update_state(self, k, r):
        self.N[k] += 1
        self.R[k] += r**self.k

    def __repr__(self):
        return self._generate_repr(f"K={self.K}, a0={self.a0}, b0={self.b0}")
