class PerformanceTester(object):

    @staticmethod
    def test(nb_episodes, env, policy):
        sum_returns = 0
        for i in range(nb_episodes):
            state = env.reset()
            done = False
            while not done:
                action = policy(state)
                state, reward, done, info = env.step(action)
                if done:
                    sum_returns += reward
        return sum_returns / nb_episodes
