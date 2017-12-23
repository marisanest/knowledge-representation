from assignment03.src.generation import EpisodeGenerator
from assignment03.src.perfomance import PerformanceTester
from assignment03.src.policy import PolicyHandler
from assignment03.src.value_function import V, Q


class Evaluator(EpisodeGenerator, PerformanceTester, PolicyHandler):

    def __init__(self, policy, env, nb_episodes, gamma):
        EpisodeGenerator.__init__(self, policy, env)
        PolicyHandler.__init__(self, policy)
        self.nb_episodes = nb_episodes
        self.gamma = gamma

    def evaluate_n_episodes(self):
        for _ in range(self.nb_episodes):
            episode = self.generate_episode()
            self.evaluate(episode)

    def evaluate(self, episode):
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


class VEvaluator(V, Evaluator):

    def __init__(self, policy, env, nb_states, nb_episodes, gamma):
        V.__init__(self, nb_states)
        Evaluator.__init__(self, policy, env, nb_episodes, gamma)

    def evaluate(self, episode):
        raise NotImplementedError


class QEvaluator(Q, Evaluator):

    def __init__(self, policy, env, nb_states, nb_actions, nb_episodes, gamma):
        Q.__init__(self, nb_states, nb_actions)
        Evaluator.__init__(self, policy, env, nb_episodes, gamma)

    def evaluate(self, episode):
        raise NotImplementedError
