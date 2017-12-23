import numpy as np
from assignment03.src.evaluation import QEvaluator
from assignment03.src.value_function import V


class MonteCarlo(QEvaluator, V):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes=10000, gamma=.99):
        QEvaluator.__init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma)
        V.__init__(self, nb_states)
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
