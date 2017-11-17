from assignment02.src.core import Env
from assignment02.src.spaces.discrete import Discrete
from assignment02.src.spaces.tuple import Tuple


class JacksCarRentalEnvironment(Env):
    """
    Jack’s Car Rental: Jack manages two locations for a nationwide car rental company.
    Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he
    rents it out and is credited $10 by the national company. If he is out of cars at that location, then the
    business is lost. Cars become available for renting the day after they are returned. To help ensure that
    cars are available where they are needed, Jack can move them between the two locations overnight, at
    a cost of $2 per car moved. We assume that the number of cars requested and returned at each location
    are Poisson random variables, meaning that the probability that the number is n is (λ^/n!)*e^(−λ), where λ is
    the expected number. Suppose λ is 3 and 4 for rental requests at the first and second locations and 3 and
    2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each
    location (any additional cars are returned to the nationwide company, and thus disappear from the problem)
    and a maximum of five cars can be moved from one location to the other in one night. We take the discount
    rate to be γ = 0.9 and formulate this as a continuing finite MDP, where the time steps are days, the state is
    the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight. Figure 4.2 shows
    """

    MAX_CAPACITY = 20
    COST_PER_CAR_MOVE = 2
    CREDIT_PER_CAR_RENATL = 10
    λ = {'rental': {'A': 3, 'B': 4},
         'return': {'A': 3, 'B': 2}}
    γ = 0.9

    def __init__(self):
        self.action_space = Discrete(-5, 6)
        self.observation_space = Tuple((
            Discrete(0, 21),
            Discrete(0, 21)))
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def _step(self, action):
        assert self.action_space.contains(action)
        return self._get_obs()  # , reward, done, ''
        pass

    def _get_obs(self):
        return self.location_a, self.location_b

    def _reset(self, default=None):
        if default:
            if self.observation_space.contains(default):
                self.location_a, self.location_b = default
        else:
            self.location_a = self.np_random  # todo
            self.location_b = self.np_random  # todo

        return self._get_obs()
