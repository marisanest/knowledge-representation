import numpy as np
from assignment03.src.evaluation import VEvaluator, QEvaluator


class Improver(object):

    def improve(self, episode):
        raise NotImplementedError


class VImprover(Improver, VEvaluator):

    def __init__(self, policy, env, nb_states, nb_episodes, gamma):
        super().__init__(policy, env, nb_states, nb_episodes, gamma)

    def evaluate(self, episode):
        raise NotImplementedError

    def improve(self, episode):
        raise NotImplementedError


class QImprover(Improver, QEvaluator):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma):
        super().__init__(policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def evaluate(self, episode):
        raise NotImplementedError

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))
