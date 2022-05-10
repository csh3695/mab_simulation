from typing import List
import numpy as np
from tqdm import tqdm

from arms.base import BaseArm
from policy.base import BasePolicy


class Simulator:
    def __init__(
        self, arms: List[BaseArm], policies: List[BasePolicy], num_iter: int = 10000
    ):
        self.arms = arms
        self.policies = policies
        self.num_iter = num_iter
        self.rewards = np.zeros([num_iter, len(self.policies)])
        self.max_rewards = []
        self.avg_rewards = np.mean([arm.get_expected_reward() for arm in self.arms])
        self.reward_logs = []
        self.selection_logs = []

    def pull_arms(self):
        return np.array([arm.pull() for arm in self.arms])

    def select_arms(self):
        return np.array([policy.select_arm() for policy in self.policies])

    def initialize(self):
        self.rewards = np.zeros([self.num_iter, len(self.policies)])
        self.reward_logs = []
        self.selection_logs = []
        self.max_rewards = []
        for policy in self.policies:
            policy.initialize()

    def update_policies(self, selected, rewards):
        for p, s, r in zip(self.policies, selected, rewards):
            p.update_state(s, r)

    def play(self):
        for i in tqdm(range(self.num_iter)):
            rewards = self.pull_arms()
            self.reward_logs.append(rewards)
            self.max_rewards.append(max(rewards))
            selected = self.select_arms()
            self.selection_logs.append(selected)
            self.rewards[i] = rewards[selected]
            self.update_policies(selected, rewards)
        return self.rewards
