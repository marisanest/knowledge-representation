import gym


class TemporalDifferenceZero(object):

    def __init__(self, pi):
        self.pi = pi
        self.env = gym.make('FrozenLake-v0')

    def policy_evaluation(self, alpha=.05, nb_episodes=1000000, gamma=0.99):
        pass

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


class SARSA(object):

    def __init__(self, pi):
        self.pi = pi


class QLearning(object):

    def __init__(self, pi):
        self.pi = pi
