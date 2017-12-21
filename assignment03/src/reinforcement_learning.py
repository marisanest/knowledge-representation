import numpy as np
from assignment03.src.policy import EpsilonGreedyPolicy


class ReinforcementLearningAlgorithm(object):

    def __init__(self, policy, epsilon, nb_actions, env, nb_episodes, gamma):
        self.policy = EpsilonGreedyPolicy(policy, epsilon, nb_actions)
        self.env = env
        self.nb_episodes = nb_episodes
        self.gamma = gamma

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

    def control(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)
            self.improve(episode)

    def evaluate_n_episodes(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)

    def evaluate(self, episode):
        raise NotImplementedError

    def improve(self, episode):
        raise NotImplementedError

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


