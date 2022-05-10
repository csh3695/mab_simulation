import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arms import BernoulliArm
from policy import BernoulliThompsonSamplingPolicy, EpsGreedyPolicy, RandomPolicy
from simulator import Simulator

arms = [BernoulliArm(p) for p in np.random.rand(100) / 20]

policies = [
    RandomPolicy(len(arms)),
    EpsGreedyPolicy(len(arms), 0.0),
    EpsGreedyPolicy(len(arms), 0.01),
    EpsGreedyPolicy(len(arms), 0.1),
    BernoulliThompsonSamplingPolicy(len(arms), 0.1, 0.1),
    BernoulliThompsonSamplingPolicy(len(arms), 1, 1),
    BernoulliThompsonSamplingPolicy(len(arms), 10, 10),
]

sim = Simulator(arms, policies, num_iter=1000)

sim.play()

rewards = sim.rewards
cum_rewards = np.cumsum(rewards, 0)
plt.plot(cum_rewards)
plt.legend(range(len(rewards[0])))
plt.xlabel("time")
plt.ylabel("cumulative rewards")
plt.show()

sns.heatmap(cum_rewards)
plt.xlabel("policy")
plt.ylabel("time")
plt.show()
