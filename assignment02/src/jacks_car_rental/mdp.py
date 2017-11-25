import numpy as np
from assignment02.src.jacks_car_rental.model import JacksCarRentalEnvironmentModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


class JacksCarRentalEnvironmentMDP(object):

    def __init__(self):
        self.model = JacksCarRentalEnvironmentModel()
        self.policy = self._init_policy()
        self.q = self._init_q()
        self.p, self.r = self._init_p_and_r()
        self.nb_states = self.model.MAX_CAPACITY + 1
        self.nb_actions = len(self.model.ACTIONS)

    def _index_to_stats_map(self):

        index_to_states = np.zeros(self.nb_states * len(self.model.LOCATIONS))

        for index_a in range(0, self.nb_states):
            for index_b in range(0, self.nb_states):
                index_to_states[index_a * self.nb_states + index_b] = (index_a, index_b)

        return index_to_states

    def _stats_to_index(self, stat_a, stat_b):
            return stat_a * self.model.MAX_CAPACITY + stat_b

    def _init_p_and_r(self):

        # s, a, s', r
        p = np.zeros((self.nb_states, self.nb_states, self.nb_actions, self.nb_states, self.nb_states))
        expected_rewards = np.zeros((self.nb_states, self.nb_states, self.nb_actions))

        for s_a in range(0, self.nb_states):
            for s_b in range(0, self.nb_states):
                for a in self.model.ACTIONS:

                    transition_probabilities, expected_rewards[s_a, s_b, a] = self.model.get_transition_probabilities_and_expected_reward((s_a, s_b), a)

                    for next_s_a, p_a in enumerate(transition_probabilities[0]):
                        for next_s_b, p_b in enumerate(transition_probabilities[1]):
                            p[s_a, s_b, a, next_s_a, next_s_b, 0] = p_a * p_b

        return p, expected_rewards

    def _init_policy(self):

        policy = np.zeros((self.nb_states, self.nb_states))

        for s_a in range(0, self.nb_states):
            for s_b in range(0, self.nb_states):
                policy[s_a, s_b] = np.random.randint(self.nb_actions)

        return policy

    def _init_q(self):
        return np.zeros((self.nb_states, self.nb_states, self.nb_actions))

    def evaluate(self, theta=.05, gamma=.9):

        converged = False

        while not converged:

            delta = .0

            for s_a in range(0, self.nb_states):
                for s_b in range(0, self.nb_states):
                    a = self.policy[s_a, s_b]

                    old_q = self.q[s_a, s_b, a]
                    new_q = 0

                    for next_s_a in range(0, self.nb_states):
                        for next_s_b in range(0, self.nb_states):
                                new_q += self.p.sum(axis=5)[s_a, s_b, a, next_s_a, next_s_b] * (self.r[s_a, s_b, a] + gamma * self.q[next_s_a, next_s_b, self.policy[next_s_a, next_s_b]])

                    self.q[s_a, s_b, a] = new_q
                    delta = np.amax([delta, abs(old_q - new_q)])

            if delta >= theta:
                converged = True

    def improve(self):

        for s_a in range(0, self.nb_states):
            for s_b in range(0, self.nb_states):

                max_a, max_value = None, None

                for a in range(0, self.nb_actions):

                    value = self.q[s_a, s_b, a]

                    if max_value is None or max_value < value:
                        max_value = value
                        max_a = a

                self.policy[s_a, s_b] = max_a

    def iterate_policy(self):

        policy_stable = False

        while not policy_stable:

            self.evaluate()

            old_policy = self.policy

            self.improve()

            if (old_policy == self.policy).all():
                policy_stable = True

    def iterate_values(self, theta=.05, gamma=.9):

        converged = False

        while not converged:

            delta = .0

            for s_a in range(0, self.nb_states):
                for s_b in range(0, self.nb_states):

                    old_max_q = np.amax(self.q[s_a, s_b, :])
                    old_a = np.argmax(self.q[s_a, s_b, :])

                    for a in range(0, self.nb_actions):

                        new_q = 0

                        for next_s_a in range(0, self.nb_states):
                            for next_s_b in range(0, self.nb_states):
                                    new_q += self.p.sum(axis=5)[s_a, s_b, a, next_s_a, next_s_b] * (self.r[s_a, s_b, a] + gamma * self.q[next_s_a, next_s_b, old_a])

                        self.q[s_a, s_b, a] = new_q

                    new_max_q = np.amax(self.q[s_a, s_b, :])

                    delta = np.amax([delta, abs(old_max_q - new_max_q)])

            if delta >= theta:
                converged = True

        self.improve()

    def plot3d_over_states(self, z_label="", ):

        a = np.arange(0, self.nb_states)
        b = np.arange(0, self.nb_states)

        # b, a !!!
        b, a = np.meshgrid(b, a)

        v = self.q.reshape(self.nb_states, -1)
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

        a = np.arange(0, self.nb_states)
        b = np.arange(0, self.nb_states)

        a, b = np.meshgrid(a, b)

        po = self.policy.reshape(self.nb_states, -1)
        levels = range(-5, 6, 1)
        plt.figure(figsize=(7, 6))

        cs = plt.contourf(a, b, po, levels)

        cbar = plt.colorbar(cs)

        cbar.ax.set_ylabel('actions')
        # plt.clabel(cs, inline=1, fontsize=10)

        plt.title('Policy')
        plt.xlabel("cars at B")
        plt.ylabel("cars at A")
