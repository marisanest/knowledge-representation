class EpisodeGenerator(object):

    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

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
