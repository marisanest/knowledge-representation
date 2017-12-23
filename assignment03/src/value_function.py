import numpy as np


class V(object):

    def __init__(self, nb_states):
        self.v = np.zeros(nb_states)


class Q(object):

    def __init__(self, nb_states, nb_actions):
        self.q = np.zeros((nb_states, nb_actions))
