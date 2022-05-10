from abc import ABCMeta, abstractmethod


class BasePolicy(metaclass=ABCMeta):
    name = ""

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update_state(self, k, r):
        pass

    @abstractmethod
    def initialize(self):
        pass

    def _generate_repr(self, annot=None):
        return f"Policy(type={self.name}, {annot})"

    def __repr__(self):
        return self._generate_repr()
