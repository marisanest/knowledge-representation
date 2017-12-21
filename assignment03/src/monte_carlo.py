import numpy as np
from assignment03.src.reinforcement_learning import ReinforcementLearningAlgorithm
from assignment03.src.evaluation import VEvaluator, QEvaluator


class MonteCarlo(ReinforcementLearningAlgorithm, VEvaluator, QEvaluator):

    def __init__(self, policy, epsilon, nb_states, nb_actions, env, nb_episodes=10000, gamma=.99):
        super().__init__(policy, epsilon, nb_actions, env, nb_episodes, gamma)
        super().__init__(nb_states)
        super().__init__(nb_states, nb_actions)
        self.n = np.zeros(nb_states, nb_actions)
        # self.g = self._init_returns()

    # def _init_returns(self):
    #    g = []
    #    for index in range(self.nb_states):
    #        g.append([])
    #        for _ in range(self.nb_actions):
    #            g[index].append([])
    #    return g

    def evaluate(self, episode):
        r = 0
        for index in reversed(range(len(episode))):
            r = episode[index][2] + self.gamma * r
            # self.g[episode[index][0]][episode[index][1]].append(r)
            self.q[episode[index][0], episode[index][1]] = (self.q[episode[index][0], episode[index][1]] *
                                                            self.n[episode[index][0], episode[index][1]] + r) / \
                                                           (self.q[episode[index][0], episode[index][1]] + 1)
            # np.average(self.g[episode[index][0]][episode[index][1]])
            self.n[episode[index][0], episode[index][1]] += 1

    def evaluate_n_episodes_v_with_q(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate_v_with_q(episode)

    def evaluate_v_with_q(self, episode):
        self.evaluate(episode)
        self.v = np.sum(self.q * self.n, 1) / np.sum(self.n, 1)
        # v_n = np.sum(self.n, 1)
        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         self.v[s] += self.q[s, a] * self.n[s, a]
        #     self.v[s] = self.v[s] / v_n[s]

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))