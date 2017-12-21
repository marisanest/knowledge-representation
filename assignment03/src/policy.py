import numpy as np


class Policy(object):

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, state, *args, **kwargs):
        raise NotImplementedError

    def set(self, state, action):
        self.policy[state] = action


class StaticPolicy(Policy):

    def __init__(self, policy):
        super(StaticPolicy, self).__init__(policy)

    def __call__(self, state, *args, **kwargs):
        return self.policy[state]


class EpsilonGreedyPolicy(Policy):

    def __init__(self, policy, nb_actions, epsilon):
        super(EpsilonGreedyPolicy, self).__init__(policy)
        self.nb_actions = nb_actions
        self.epsilon = (epsilon * 100)

    def __call__(self, state, *args, **kwargs):
        if np.random.randint(100) < self.epsilon:
            return self.policy[state]
        else:
            return np.random.randint(len(self.nb_actions))
