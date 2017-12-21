import numpy as np
from assignment03.src.policy import EpsilonGreedyPolicy, StaticPolicy


class TDZero(object):
    def __init__(self, policy, nb_states, nb_actions, env):
        self.policy = StaticPolicy(policy)
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.env = env
        self.v = np.zeros(self.nb_states)

    def generate_episode(self):

        state = self.env.reset()
        action = self.policy(state)
        state_prime, reward, done, info = self.env.step(action)
        episode = [(state, action, reward)]

        while not done:
            state = state_prime,
            action = self.policy(state)
            state_prime, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))

        return episode

    def evaluate_n_episodes(self, alpha=.05, nb_episodes=1000000, gamma=.99):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode, alpha, gamma)

    def evaluate(self, episode, alpha=.05, gamma=0.99):
        for index in reversed(range(len(episode))):
            self.v[episode[index][0]] = self.v[episode[index][0]] + alpha * (episode[index][2]
                                                                             + gamma * self.v[episode[index + 1][0]]
                                                                             - self.v[episode[index][0]])


class SARSA(object):

    def __init__(self, policy, nb_states, nb_actions, env, epsilon):
        self.policy = EpsilonGreedyPolicy(policy, epsilon, nb_actions)
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.env = env
        self.q = np.zeros(self.nb_states, self.nb_actions)

    def generate_episode(self):

        state = self.env.reset()
        action = self.policy(state)
        state_prime, reward, done, info = self.env.step(action)
        episode = [(state, action, reward)]

        while not done:
            state = state_prime,
            action = self.policy(state)
            state_prime, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))

        return episode

    def control(self, nb_episodes=10000, gamma=.99, v=False):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode, gamma, v)
            self.improve(episode)

    def evaluate_n_episodes(self, alpha=.05, nb_episodes=1000000, gamma=.99):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode, alpha, gamma)

    def evaluate(self, episode, alpha=.05, gamma=0.99):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + alpha * (episode[index][2]
                                                                      + gamma * self.q[
                                                                          episode[index + 1][0], episode[index + 1][1]]
                                                                      - self.q[episode[index][0], episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))

    def test_performance(self, nb_episodes=1000):
        sum_returns = 0
        for i in range(nb_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                state, reward, done, info = self.env.step(action)
                if done:
                    sum_returns += reward
        return sum_returns / nb_episodes


class QLearning(object):

    def __init__(self, policy, nb_states, nb_actions, env, epsilon):
        self.policy = EpsilonGreedyPolicy(policy, epsilon, nb_actions)
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.env = env
        self.q = np.zeros(self.nb_states, self.nb_actions)

    def generate_episode(self):

        state = self.env.reset()
        action = self.policy(state)
        state_prime, reward, done, info = self.env.step(action)
        episode = [(state, action, reward)]

        while not done:
            state = state_prime,
            action = self.policy(state)
            state_prime, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))

        return episode

    def control(self, nb_episodes=10000, gamma=.99, v=False):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode, gamma, v)
            self.improve(episode)

    def evaluate_n_episodes(self, alpha=.05, nb_episodes=1000000, gamma=.99):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode, alpha, gamma)

    def evaluate(self, episode, alpha=.05, gamma=0.99):
        for index in reversed(range(len(episode))):
            self.q[episode[index][0], episode[index][1]] = self.q[episode[index][0], episode[index][1]] \
                                                           + alpha * (episode[index][2]
                                                                      + gamma *
                                                                      np.argmax(
                                                                          self.q[episode[index + 1][0], :], axis=1)
                                                                      - self.q[episode[index][0], episode[index][1]])

    def improve(self, episode):
        for state in set([step[0] for step in episode]):
            self.policy.set(state, np.argmax(self.q[state, :], axis=1))
