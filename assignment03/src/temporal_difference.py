import numpy as np
from assignment03.src.control import VController, QController
from assignment03.src.perfomance import PerformanceTester


class TemporalDifference(object):

    def __init__(self, alpha):
        self.alpha = alpha


class TemporalDifferenceZero(TemporalDifference, VController):

    def __init__(self, policy, nb_states, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(alpha)
        super().__init__(policy, env, nb_states, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.v[episode[index][0]] = self.v[episode[index][0]] + self.alpha * (episode[index][2]
                                                                                  + self.gamma * self.v[
                                                                                      episode[index + 1][0]]
                                                                                  - self.v[episode[index][0]])

    def improve(self, episode):
        raise NotImplementedError


class SARSA(TemporalDifference, QController, PerformanceTester):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(alpha)
        super().__init__(policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma * self.q[
                                                                               episode[index + 1][0],
                                                                               episode[index + 1][1]]
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])

    def test_performance(self):
        sum_returns = 0
        for i in range(self.nb_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                state, reward, done, info = self.env.step(action)
                if done:
                    sum_returns += reward
        return sum_returns / self.nb_episodes


class QLearning(TemporalDifference, QController):

    def __init__(self, policy, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(alpha)
        super().__init__(policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma *
                                                                           np.argmax(
                                                                               self.q[episode[index + 1][0], :], axis=1)
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])
