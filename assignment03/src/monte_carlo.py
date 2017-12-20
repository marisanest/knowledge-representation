import numpy as np
from typing import List, Tuple


class MonteCarloPrediction(object):

    def __init__(self, pi, nb_states, nb_actions, env):
        self.pi = pi
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.env = env
        self.v = np.zeros(self.nb_states)
        self.q = np.zeros(self.nb_states, self.nb_actions)
        self.n = np.zeros(self.nb_states, self.nb_actions)
        # self.g = self._init_returns()

    # def _init_returns(self):
    #    g = []
    #    for index in range(self.nb_states):
    #        g.append([])
    #        for _ in range(self.nb_actions):
    #            g[index].append([])
    #    return g

    def generate_episode(self) -> List[Tuple[int, int, int]]:

        state = self.env.reset()
        action = self.pi[state]
        state_prime, reward, done, info = self.env.step(action)
        episode = [(state, action, reward)]

        while not done:
            state = state_prime,
            action = self.pi[state]
            state_prime, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))

        return episode

    def every_visit_q(self, target_state=0, nb_episodes=10000, gamma=.99):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            r = 0
            for index in reversed(range(len(episode))):
                r = episode[index][2] + gamma * r
                # self.g[episode[index][0]][episode[index][1]].append(r)
                self.q[episode[index][0], episode[index][1]] = (self.q[episode[index][0], episode[index][1]] *
                                                                self.n[episode[index][0], episode[index][1]] + r) / \
                                                               (self.q[episode[index][0], episode[index][1]] + 1)
                # np.average(self.g[episode[index][0]][episode[index][1]])
                self.n[episode[index][0], episode[index][1]] += 1

    def every_visit_v(self, target_state=0, nb_episodes=10000, gamma=.99):
        self.every_visit_q(target_state, nb_episodes, gamma)
        self.v = np.sum(self.q * self.n, 1) / np.sum(self.n, 1)

        # v_n = np.sum(self.n, 1)
        # for s in range(self.nb_states):
        #     for a in range(self.nb_actions):
        #         self.v[s] += self.q[s, a] * self.n[s, a]
        #     self.v[s] = self.v[s] / v_n[s]


class MonteCarloControl(MonteCarloPrediction):

    def __init__(self, pi, nb_states, nb_actions, env):
        super(MonteCarloControl, self).__init__(pi, nb_states, nb_actions, env)

    def every_visit_q(self, target_state=0, nb_episodes=10000, gamma=.99):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            r = 0
            for index in reversed(range(len(episode))):
                r = episode[index][2] + gamma * r
                # self.g[episode[index][0]][episode[index][1]].append(r)
                self.q[episode[index][0], episode[index][1]] = (self.q[episode[index][0], episode[index][1]] *
                                                                self.n[episode[index][0], episode[index][1]] + r) / \
                                                               (self.q[episode[index][0], episode[index][1]] + 1)
                # np.average(self.g[episode[index][0]][episode[index][1]])
                self.n[episode[index][0], episode[index][1]] += 1


    def improve(self, epsilon=0.1):
        pass
