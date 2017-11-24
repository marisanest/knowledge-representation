import numpy as np
from assignment02.src.jacks_car_rental.model import JacksCarRentalEnvironmentModel


class JacksCarRentalEnvironmentMDP(object):

    def __init__(self):
        self.model = JacksCarRentalEnvironmentModel()
        self.policy = self._init_policy()
        self.q = self._init_q()
        self.p, self.r = self._init_p_and_r()

    def _index_to_stats_map(self):

        nstates = self.model.MAX_CAPACITY + 1
        index_to_states = np.zeros(nstates * len(self.model.LOCATIONS))

        for index_a in range(0, nstates):
            for index_b in range(0, nstates):
                index_to_states[index_a * nstates + index_b] = (index_a, index_b)

        return index_to_states

    def _stats_to_index(self, stat_a, stat_b):
            return stat_a * self.model.MAX_CAPACITY + stat_b

    def _init_p_and_r(self):

        nstates = self.model.MAX_CAPACITY + 1
        nactions = len(self.model.ACTIONS)

        # s, a, s', r
        p = np.zeros((nstates, nstates, nactions, nstates, nstates))
        expected_rewards = np.zeros((nstates, nstates, nactions))

        for s_a in range(0, nstates):
            for s_b in range(0, nstates):
                for a in self.model.ACTIONS:

                    transition_probabilities, expected_rewards[s_a, s_b, a] = self.model.get_transition_probabilities_and_expected_reward((s_a, s_b), a)

                    for next_s_a, p_a in enumerate(transition_probabilities[0]):
                        for next_s_b, p_b in enumerate(transition_probabilities[1]):
                            p[s_a, s_b, a, next_s_a, next_s_b, 0] = p_a * p_b

        return p, expected_rewards

    def _init_policy(self):

        nstates = self.model.MAX_CAPACITY + 1
        policy = np.zeros((nstates, nstates))

        for s_a in range(0, nstates):
            for s_b in range(0, nstates):
                policy[s_a, s_b] = np.random.randint(min(self.model.ACTIONS), min(self.model.ACTIONS) + 1)

        return policy

    def _init_q(self):

        nstates = self.model.MAX_CAPACITY + 1
        nactions = len(self.model.ACTIONS)

        return np.zeros((nstates, nstates, nactions))

    def evaluate(self, theta=0.05, gamma=.9):

        nstates = self.model.MAX_CAPACITY + 1

        converged = False

        while not converged:

            delta = 0.

            for s_a in range(0, nstates):
                for s_b in range(0, nstates):
                    a = self.policy[s_a, s_b]

                    old_q = self.q[s_a, s_b, a]
                    new_q = 0

                    for next_s_a in range(0, nstates):
                        for next_s_b in range(0, nstates):
                                new_q += self.p[s_a, s_b, a, next_s_a, next_s_b, 0] * (self.r[s_a, s_b, a] + gamma * self.q[next_s_a, next_s_b, self.policy[next_s_a, next_s_b]])

                    self.q[s_a, s_b, a] = new_q
                    delta = max([delta, abs(old_q - new_q)])

            if delta >= theta:
                converged = True

    def improve(self):

        nstates = self.model.MAX_CAPACITY + 1

        policy_stable = False

        while not policy_stable:
            for s_a in range(0, nstates):
                for s_b in range(0, nstates):
                    old_a = self.policy[s_a, s_b]
                    new_action = self.q[s_a, s_b].max


