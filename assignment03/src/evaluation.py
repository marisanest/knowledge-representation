from assignment03.src.generation import EpisodeGenerator


class Evaluator(EpisodeGenerator):

    def __init__(self, policy, env, nb_episodes, gamma):
        EpisodeGenerator.__init__(self, policy, env)
        self.nb_episodes = nb_episodes
        self.gamma = gamma

    def evaluate_n_episodes(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)

    def evaluate(self, episode):
        raise NotImplementedError
