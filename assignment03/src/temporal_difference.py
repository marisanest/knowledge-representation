import numpy as np
from assignment03.src.control import VController, QController


class TemporalDifference(object):

    def __init__(self, alpha):
        self.alpha = alpha


class TemporalDifferenceZero(TemporalDifference, VController):

    def __init__(self, policy, nb_states, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        VController.__init__(self, policy, env, nb_states, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):

            self.v[episode[index][0]] = self.v[episode[index][0]] + self.alpha * (episode[index][2]
                                                                                  + self.gamma * self.v[
                                                                                      episode[index + 1][0]]
                                                                                  - self.v[episode[index][0]])

    def improve(self, episode):
        raise NotImplementedError


class SARSA(TemporalDifference, QController):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        QController.__init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma * self.q[
                                                                               episode[index + 1][0],
                                                                               episode[index + 1][1]]
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])


class QLearning(TemporalDifference, QController):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        TemporalDifference.__init__(self, alpha)
        QController.__init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma *
                                                                           np.argmax(
                                                                               self.q[episode[index + 1][0], :], axis=1)
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])
