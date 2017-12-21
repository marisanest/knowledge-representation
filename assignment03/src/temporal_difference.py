import numpy as np
from assignment03.src.reinforcement_learning import ReinforcementLearningAlgorithm
from assignment03.src.evaluation import VEvaluator, QEvaluator


class TemporalDifference(ReinforcementLearningAlgorithm):

    def __init__(self, policy, epsilon, nb_states, nb_actions, env, nb_episodes, gamma, alpha):
        super().__init__(policy, epsilon, nb_actions, env, nb_episodes, gamma)
        super().__init__(nb_states)
        self.alpha = alpha

    def evaluate(self, episode):
        raise NotImplementedError

    def improve(self, episode):
        raise NotImplementedError


class TemporalDifferenceZero(TemporalDifference, VEvaluator):

    def __init__(self, policy, epsilon, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(policy, epsilon, nb_states, nb_actions, env, nb_episodes, gamma, alpha)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.v[episode[index][0]] = self.v[episode[index][0]] + self.alpha * (episode[index][2]
                                                                                  + self.gamma * self.v[
                                                                                      episode[index + 1][0]]
                                                                                  - self.v[episode[index][0]])

    def improve(self, episode):
        raise NotImplementedError


class SARSA(TemporalDifference, QEvaluator):

    def __init__(self, policy, epsilon, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(policy, epsilon, nb_states, nb_actions, env, nb_episodes, gamma, alpha)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma * self.q[
                                                                               episode[index + 1][0],
                                                                               episode[index + 1][1]]
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))


class QLearning(TemporalDifference, QEvaluator):

    def __init__(self, policy, epsilon, nb_states, nb_actions, env, nb_episodes=1000000, gamma=.99, alpha=.05):
        super().__init__(policy, epsilon, nb_states, nb_actions, env, nb_episodes, gamma, alpha)

    def evaluate(self, episode):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + self.alpha * (episode[index][2]
                                                                           + self.gamma *
                                                                           np.argmax(
                                                                               self.q[episode[index + 1][0], :], axis=1)
                                                                           - self.q[
                                                                               episode[index][0], episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))
