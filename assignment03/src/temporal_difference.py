import numpy as np
from assignment03.src.control import Controller
from assignment03.src.value_function import V, Q
from assignment03.src.perfomance import PerformanceTester
from assignment03.src.policy import StaticPolicy


class TemporalDifference(object):

    def __init__(self, alpha):
        self.alpha = alpha


class TemporalDifferenceZero(TemporalDifference, Controller, V):

    def __init__(self, policy, nb_states, env, nb_episodes=100000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        V.__init__(self, nb_states)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            r_prime = self.v[episode[index + 1][0]] if index is not (len(episode) - 1) else 0
            self.v[episode[index][0]] = self.v[episode[index][0]] \
                                        + self.alpha * (episode[index][2]
                                                        + self.gamma * r_prime - self.v[episode[index][0]])

    def improve(self, episode):
        raise NotImplementedError

    def test_performance(self):
        return PerformanceTester.test(self.nb_episodes, self.env, StaticPolicy(self.policy.policy))


class SARSA(TemporalDifference, Controller, Q):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=100000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        Q.__init__(self, nb_states, nb_actions)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            r_prime = self.q[episode[index + 1][0], episode[index + 1][1]] if index is not (len(episode) - 1) else 0
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma * r_prime
                                                                           - self.q[episode[index][0],
                                                                                    episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=0))

    def test_performance(self):
        return PerformanceTester.test(self.nb_episodes, self.env, StaticPolicy(self.policy.policy))


class QLearning(TemporalDifference, Controller, Q):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=100000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        Q.__init__(self, nb_states, nb_actions)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            r_prime = np.amax(self.q[episode[index + 1][0], :], axis=0) if index is not (len(episode) - 1) else 0
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma * r_prime
                                                                           - self.q[episode[index][0],
                                                                                    episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=0))

    def test_performance(self):
        return PerformanceTester.test(self.nb_episodes, self.env, StaticPolicy(self.policy.policy))
