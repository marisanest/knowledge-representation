class EpisodeGenerator(object):

    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def generate_episode(self):

        state, reward, done, action, episode = self.env.reset(), None, False, None, []

        while not done:
            action = self.policy(state)
            state_prime, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = state_prime

        return episode
