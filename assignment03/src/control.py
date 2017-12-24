from assignment03.src.improvement import Improver
from assignment03.src.evaluation import Evaluator


class Controller(Evaluator, Improver):

    def __init__(self, policy, env, nb_episodes, gamma):
        Evaluator.__init__(self, policy, env, nb_episodes, gamma)

    def control(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)
            self.improve(episode)

    def improve(self, episode):
        raise NotImplementedError

    def evaluate(self, episode):
        raise NotImplementedError
