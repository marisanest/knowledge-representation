import numpy as np
from assignment03.src.control import QController


class MonteCarlo(QController):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes=10000, gamma=.99):
        super().__init__(policy, env, nb_states, nb_actions, nb_episodes, gamma)
        self.n = np.zeros(nb_states, nb_actions)
        self.v = np.zeros(nb_states)

    def evaluate(self, episode):
        r = 0
        for index in reversed(range(len(episode))):
            r = episode[index][2] + self.gamma * r
            self.q[episode[index][0], episode[index][1]] = (self.q[episode[index][0], episode[index][1]] *
                                                            self.n[episode[index][0], episode[index][1]] + r) / \
                                                           (self.q[episode[index][0], episode[index][1]] + 1)
            self.n[episode[index][0], episode[index][1]] += 1

    def evaluate_v_with_q(self, episode):
        self.evaluate(episode)
        self.v = np.sum(self.q * self.n, 1) / np.sum(self.n, 1)
