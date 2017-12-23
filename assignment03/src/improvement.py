import numpy as np
from assignment03.src.value_function import V, Q
from assignment03.src.policy import PolicyHandler


class Improver(PolicyHandler):

    def __init__(self, policy):
        super().__init__(policy)

    def improve(self, episode):
        raise NotImplementedError


class VImprover(V, Improver):

    def __init__(self, nb_states, policy):
        V.__init__(self, nb_states)
        Improver.__init__(self, policy)

    def improve(self, episode):
        raise NotImplementedError


class QImprover(Q, Improver):

    def __init__(self, nb_states, nb_actions, policy):
        Q.__init__(self, nb_states, nb_actions)
        Improver.__init__(self, policy)

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))
