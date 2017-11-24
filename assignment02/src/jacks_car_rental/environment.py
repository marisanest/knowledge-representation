from assignment02.src.core import Env
from assignment02.src.spaces.discrete import Discrete
from assignment02.src.spaces.tuple import Tuple
from numpy.random import poisson


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
    the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved
    between the two locations overnight. Figure 4.2 shows
    """

    MAX_CAPACITY = 20
    COST_PER_CAR_MOVE = 2
    CREDIT_PER_CAR_RENTAL = 10
    LAMBDA = {'a': {'rental': 3, 'return': 3},
              'b': {'rental': 4, 'return': 2}}
    GAMMA = 0.9

    def __init__(self):
        self.action_space = Discrete(-5, 6)
        self.observation_space = Tuple((
            Discrete(0, self.MAX_CAPACITY + 1),
            Discrete(0, self.MAX_CAPACITY + 1)))
        self.locations = {}
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)

        _from = 'a'
        _to = 'b'

        if action < 0:
            _from = 'b'
            _to = 'a'

        action = abs(action)

        if self.location[_from] < action:
            self.location[_to] = self.location[_to] + self.location[_from]
            reward = -(self.location[_from] * self.COST_PER_CAR_MOVE)
            self.location[_from] = 0
        else:
            self.location[_to] = self.location[_to] + action
            reward = (action * self.COST_PER_CAR_MOVE)
            self.location[_from] = self.location[_from] - action

        for key, value in self.locations.items():
            _rental = round(poisson(self.LAMBDA[key]['rental']))

            if value < _rental:
                reward += value * self.CREDIT_PER_CAR_RENTAL
                value = 0
            else:
                reward += _rental * self.CREDIT_PER_CAR_RENTAL
                value = value - _rental

            _return = round(poisson(self.LAMBDA[key]['return']))

            if value + _return > self.MAX_CAPACITY:
                self.locations[key] = self.MAX_CAPACITY
            else:
                self.locations[key] = value + _return

        # done = (self.location_a == 0 and self.location_b == 0)

        return self._get_obs(), reward, False, ''
        pass

    def _get_obs(self):
        return self.locations['a'], self.locations['b']

    def _reset(self, default=None):
        if default is not None:
            if self.observation_space.contains(default):
                self.locations['a'], self.locations['b'] = default
        else:
            self.locations['a'], self.locations['b'] = self.observation_space.sample()

        return self._get_obs()
