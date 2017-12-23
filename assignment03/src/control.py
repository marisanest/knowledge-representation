from assignment03.src.improvement import VImprover, QImprover, Improver
from assignment03.src.evaluation import VEvaluator, QEvaluator, Evaluator


class Controller(Evaluator, Improver):

    def __init__(self, policy, env, nb_episodes, gamma):
        Evaluator.__init__(self, policy, env, nb_episodes, gamma)
        Improver.__init__(self, policy)

    def control(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)
            self.improve(episode)

    def improve(self, episode):
        raise NotImplementedError

    def evaluate(self, episode):
        raise NotImplementedError


class VController(Controller, VEvaluator, VImprover):

    def __init__(self, policy, env, nb_states, nb_episodes, gamma):
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        VEvaluator.__init__(self, policy, env, nb_states, nb_episodes, gamma)
        VImprover.__init__(self, nb_states, policy)

    def improve(self, episode):
        raise NotImplementedError

    def evaluate(self, episode):
        raise NotImplementedError


class QController(Controller, QEvaluator, QImprover):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma):
        Controller.__init__(self, policy, env, nb_episodes, gamma)
        QEvaluator.__init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma)
        QImprover.__init__(self, nb_states, nb_actions, policy)

    def evaluate(self, episode):
        raise NotImplementedError
