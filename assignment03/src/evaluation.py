import numpy as np
from assignment03.src.generation import EpisodeGenerator


class Evaluator(EpisodeGenerator):

    def __init__(self, policy, env, nb_episodes, gamma):
        super().__init__(policy, env)
        self.nb_episodes = nb_episodes
        self.gamma = gamma

    def evaluate_n_episodes(self, nb_episodes):
        for _ in range(nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)

    def evaluate(self, episode):
        raise NotImplementedError


class VEvaluator(Evaluator):

    def __init__(self, policy, env, nb_states, nb_episodes, gamma):
        super().__init__(policy, env, nb_episodes, gamma)
        self.v = np.zeros(nb_states)

    def evaluate(self, episode):
        raise NotImplementedError


class QEvaluator(Evaluator):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma):
        super().__init__(policy, env, nb_episodes, gamma)
        self.q = np.zeros(nb_states, nb_actions)

    def evaluate(self, episode):
        raise NotImplementedError
