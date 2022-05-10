from .base import BaseArm
import numpy as np


class GaussianArm(BaseArm):
    name = "Gaussian"

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pull(self):
        return np.random.normal(self.mean, self.std)

    def get_expected_reward(self):
        return self.mean

    def __repr__(self):
        return self._generate_repr(f"mean={self.mean}, std={self.std}")
