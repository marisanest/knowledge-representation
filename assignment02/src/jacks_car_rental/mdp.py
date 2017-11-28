import numpy as np
from assignment02.src.jacks_car_rental.model import JacksCarRentalEnvironmentModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import logging


class JacksCarRentalEnvironmentMDP(object):

    def __init__(self):
        logging.basicConfig(level=logging.INFO)

        self.model = JacksCarRentalEnvironmentModel()

        self.nb_separate_states = self.model.MAX_CAPACITY + 1
        self.nb_states = self.nb_separate_states * self.nb_separate_states
        self.nb_actions = len(self.model.ACTIONS)

        self.index_to_stats = self._init_index_to_stats()

        self.p, self.r = self._init_p_and_r()
        self.p_s_prime = self.p.sum(axis=3)

        self.policy = self._init_policy()
        self.q = self._init_q()
        self.v = None
        self.state_to_action = None

    def _init_index_to_stats(self):

        index_to_states = {}

        for index_a in range(self.nb_separate_states):
            for index_b in range(self.nb_separate_states):
                index_to_states[index_a * self.nb_separate_states + index_b] = (index_a, index_b)

        return index_to_states

    def _stats_to_index(self, stat_a, stat_b):
        return stat_a * self.nb_separate_states + stat_b

    def _init_p_and_r(self):

        # s, a, s', r
        p = np.zeros((self.nb_states, self.nb_actions, self.nb_states, 1))
        r = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):
            for a in range(self.nb_actions):

                p_a_b, r[s, a] = self.model.get_transition_probabilities_and_expected_reward(self.index_to_stats[s],
                                                                                             self.model.ACTIONS[a])
                for next_s_a, p_a in enumerate(p_a_b[0]):
                    for next_s_b, p_b in enumerate(p_a_b[1]):
                        p[s, a, self._stats_to_index(next_s_a, next_s_b), 0] = p_a * p_b

        return p, r

    def _init_policy(self):
        return np.random.randint(self.nb_actions, size=self.nb_states)

    def _init_q(self):
        return np.zeros((self.nb_states, self.nb_actions))

    def _init_v(self):

        self.v = np.zeros(self.nb_states)

        for s in range(self.nb_states):
            self.v[s] = np.amax(self.q[s, :])

    def _init_policy_to_action(self):

        self.state_to_action = np.zeros(self.nb_states)

        for key, value in enumerate(self.policy):
            self.state_to_action[key] = self.model.ACTIONS[value]

    def reset(self):
        self.policy = self._init_policy()
        self.q = self._init_q()
        self.v = None
        self.state_to_action = None

    def evaluate(self, theta=1, gamma=.9):

        converged = False

        logging.info("start evaluation.")

        while not converged:

            logging.info("evaluate...")

            delta = .0

            for s in range(self.nb_states):
                for a in range(self.nb_actions):

                    old_q = self.q[s, a]

                    self.q[s, a] = sum([self.p_s_prime[s, a, next_s] * (self.r[s, a] + gamma * self.q[next_s, self.policy[next_s]]) for next_s in range(self.nb_states)])

                    delta = np.amax([delta, abs(old_q - self.q[s, a])])

            if delta < theta:
                converged = True

    def improve(self):

        logging.info("start improvement.")

        self.policy = np.argmax(self.q[:, :], axis=1)

    def iterate_policy(self, theta=1):

        policy_stable = False

        logging.info("start iteration")

        while not policy_stable:

            self.evaluate(theta=theta)

            old_policy = np.copy(self.policy)

            self.improve()

            if (old_policy == self.policy).all():
                policy_stable = True

    def iterate_values(self, theta=1, gamma=.9):

        logging.info("start iteration")

        converged = False

        logging.info("start evaluation.")

        while not converged:

            logging.info("evaluate...")

            delta = .0

            for s in range(self.nb_states):
                for a in range(self.nb_actions):

                    old_q = self.q[s, a]

                    self.q[s, a] = sum(
                        [self.p_s_prime[s, a, next_s] * (self.r[s, a] + gamma * self.q[next_s, np.argmax(self.q[next_s, :])]) for
                         next_s in range(self.nb_states)])

                    delta = np.amax([delta, abs(old_q - self.q[s, a])])

            if delta < theta:
                converged = True

        self.improve()

    def plot3d_over_states(self, z_label="", ):

        if self.v is None:
            self._init_v()

        a = np.arange(0, self.nb_separate_states)
        b = np.arange(0, self.nb_separate_states)

        # b, a !!!
        b, a = np.meshgrid(b, a)

        v = self.v.reshape(self.nb_separate_states, -1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(a, b, v, rstride=1, cstride=1, cmap=cm.coolwarm,
        #                   linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.scatter(a, b, v, c='b', marker='.')
        ax.set_xlabel("cars at A")
        ax.set_ylabel("cars at B")
        ax.set_zlabel(z_label)

        # ax.view_init(elev=10., azim=10)

        plt.show()

    def plot_policy(self):

        if self.state_to_action is None:
            self._init_policy_to_action()

        a = np.arange(0, self.nb_separate_states)
        b = np.arange(0, self.nb_separate_states)

        a, b = np.meshgrid(a, b)

        po = self.state_to_action.reshape(self.nb_separate_states, -1)
        levels = range(min(self.model.ACTIONS), max(self.model.ACTIONS) + 1, 1)
        plt.figure(figsize=(7, 6))

        cs = plt.contourf(a, b, po, levels)

        cbar = plt.colorbar(cs)

        cbar.ax.set_ylabel('actions')
        # plt.clabel(cs, inline=1, fontsize=10)

        plt.title('Policy')
        plt.xlabel("cars at B")
        plt.ylabel("cars at A")
