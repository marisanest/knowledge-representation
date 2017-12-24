import numpy as np
from assignment03.src.control import Controller
from assignment03.src.value_function import V, Q
from assignment03.src.perfomance import PerformanceTester
from assignment03.src.policy import StaticPolicy


class MonteCarlo(Controller, V, Q, PerformanceTester):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes=100000, gamma=.99):
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        V.__init__(self, nb_states)
        Q.__init__(self, nb_states, nb_actions)
        self.n = np.zeros((nb_states, nb_actions))

    def evaluate(self, episode):
        r = 0
        for index in reversed(range(len(episode))):
            r = episode[index][2] + self.gamma * r
            self.q[episode[index][0], episode[index][1]] = (self.q[episode[index][0], episode[index][1]] *
                                                            self.n[episode[index][0], episode[index][1]] + r) / \
                                                           (self.n[episode[index][0], episode[index][1]] + 1)
            self.n[episode[index][0], episode[index][1]] += 1

    def evaluate_v_with_q(self):
        self.v = np.sum(self.q * self.n, axis=1) / np.sum(self.n, axis=1)
        self.v[np.isnan(self.v)] = 0

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=0))

    def test_performance(self):
        return PerformanceTester.test(self.nb_episodes, self.env, StaticPolicy(self.policy.policy))



