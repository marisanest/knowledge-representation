from assignment03.src.improvement import VImprover, QImprover


class Controller(object):

    def control(self):
        raise NotImplementedError


class VController(Controller, VImprover):

    def __init__(self, policy, env, nb_states, nb_episodes, gamma):
        super().__init__(policy, env, nb_states, nb_episodes, gamma)

    def control(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)
            self.improve(episode)

    def improve(self, episode):
        raise NotImplementedError

    def evaluate(self, episode):
        raise NotImplementedError


class QController(Controller, QImprover):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma):
        super().__init__(policy, env, nb_states, nb_actions, nb_episodes, gamma)

    def control(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)
            self.improve(episode)

    def evaluate(self, episode):
        raise NotImplementedError
