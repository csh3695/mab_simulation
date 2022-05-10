from abc import ABCMeta, abstractmethod
from collections import Iterable


class BaseArm(metaclass=ABCMeta):
    name = ""

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def get_expected_reward(self):
        pass

    def _generate_repr(self, annot=None):
        return f"Arm(type={self.name}, {annot})"

    def __repr__(self):
        return self._generate_repr()

    def __lshift__(self, other):
        return CompositeArms([other, self])

    def __rshift__(self, other):
        return CompositeArms([self, other])


class CompositeArms(BaseArm, Iterable):
    name = "Composite"

    def __init__(self, arms: list):
        self.arms = self._flatten(arms)

    def __iter__(self):
        return self.arms.__iter__()

    def _flatten(self, xs: list):
        if not isinstance(xs, (Iterable)):
            return [xs]
        r = []
        for x in xs:
            r += self._flatten(x)
        return r

    def __getitem__(self, item):
        return self.arms.__getitem__(item)

    def pull(self):
        val = 1
        for arm in self.arms:
            val *= arm.pull()
        return val

    def get_expected_reward(self):
        val = 1
        for arm in self.arms:
            val *= arm.get_expected_reward()
        return val

    def __repr__(self):
        return self._generate_repr(self.arms.__repr__())
