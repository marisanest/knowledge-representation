import numpy as np
import scipy
from scipy import stats
import logging
from assignment02.src.jacks_car_rental.environment import JacksCarRentalEnvironment


class JacksCarRentalEnvironmentModel(JacksCarRentalEnvironment):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        super(JacksCarRentalEnvironmentModel, self).__init__()

    def get_transition_probabilities_and_expected_reward(self, state, action):
        """
            Compute the $p(s', r\mid s,a)$
            Parameters
            ----------
            state: tuple of two ints
                the state (cars_at_A, cars_at_B)
            action: int
                nigthly movements of the cars as a int between -5 to 5, e.g.:
                action +3: move three cars from A to B.
                action -2: move two cars from B to A.

            Returns
            -------
            numpy array (2d - float): mapping from (new) states to probabilities
                index first dimension: cars at A
                index second dimension: cars at B
                value: probability
            float:  expected reward for the state-action pair
        """

        assert np.abs(action) <= 5

        state = self._nightly_moves(state, action)

        expected_reward = - self.TRANSFER_COST * np.abs(action)
        expected_reward += self._expected_reward_rent(state)

        transition_probabilities = self._rent_transition_probabilities(state)
        transition_probabilities = self._returns_transition_probabilities(transition_probabilities)

        return transition_probabilities, expected_reward

    def _nightly_moves(self, state, action):

        cars_at_a = state[0]
        cars_at_b = state[1]

        if action > 0:
            cars_moved = min(action, cars_at_a)
        else:
            cars_moved = max(action, -cars_at_b)

        cars_at_a = min(cars_at_a - cars_moved, self.MAX_CAPACITY)
        cars_at_b = min(cars_at_b + cars_moved, self.MAX_CAPACITY)

        return [cars_at_a, cars_at_b]

    def _expected_reward_rent(self, state):
        expected_reward_rent = 0.
        m = self.MAX_CAPACITY + 1

        for index, location in enumerate(self.LOCATIONS):
            cars_at_loc = state[index]
            rv = scipy.stats.poisson(self.REQUEST_RATE[location])
            rent_prob = (rv.pmf(range(m)))
            logging.debug(rent_prob)
            rent_prob[cars_at_loc] = rent_prob[cars_at_loc:].sum()
            rent_prob[cars_at_loc + 1:] = 0.
            logging.debug(rent_prob)
            expected_reward_rent += np.dot(np.arange(len(rent_prob)), rent_prob) * self.RENTAL_INCOME

        return expected_reward_rent

    def _rent_transition_probabilities(self, state):

        num_states_for_a_location = self.MAX_CAPACITY + 1
        m = 15
        n = num_states_for_a_location + 2 * m
        p_ = [np.zeros(n), np.zeros(n)]

        for index, location in enumerate(self.LOCATIONS):
            rv = scipy.stats.poisson(self.REQUEST_RATE[location])
            cars_at_loc = state[index]
            x = cars_at_loc + m + 1
            rent_prob = (rv.pmf(range(x)))

            assert state[index] - x + m + 1 == 0

            p_[index][0:cars_at_loc + m + 1] = rent_prob[::-1]
            p_[index][m] = p_[index][:m + 1].sum()
            p_[index] = p_[index][m:-m]

        return p_

    def _returns_transition_probabilities(self, state_probability):

        num_states_for_a_location = self.MAX_CAPACITY + 1
        m = 11
        n = num_states_for_a_location + 2 * m
        p_ = [np.zeros(num_states_for_a_location), np.zeros(num_states_for_a_location)]

        for index, location in enumerate(self.LOCATIONS):
            rv = scipy.stats.poisson(self.RETURN_RATE[location])
            logging.debug(len(state_probability[index]))

            for cars_at_loc in range(len(state_probability[index])):
                p = np.zeros(n)
                logging.debug(p.shape)
                x = num_states_for_a_location - cars_at_loc + m - 1
                return_prob = (rv.pmf(range(x)))
                logging.debug(p[cars_at_loc + m:-1].shape)
                p[cars_at_loc + m:-1] = return_prob
                logging.debug(return_prob)
                p[num_states_for_a_location + m - 1] = p[num_states_for_a_location + m - 1:].sum()
                p = p[m:-m]
                logging.debug(p)
                logging.debug(p.sum())
                p_[index] += p * state_probability[index][cars_at_loc]

        return p_
