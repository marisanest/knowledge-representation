from assignment02.src.core import Env
from assignment02.src.spaces.discrete import Discrete
from assignment02.src.spaces.tuple import Tuple
from numpy.random import poisson
from typing import Tuple as _Tuple


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
    MAX_MOVE = 5
    TRANSFER_COST = 2
    CREDIT_PER_CAR_RENTAL = 10
    LAMBDA = {'a': {'rental': 3, 'return': 3},
              'b': {'rental': 4, 'return': 2}}
    GAMMA = 0.9

    def __init__(self):
        self.action_space = Discrete(-self.MAX_MOVE, self.MAX_MOVE + 1)
        self.observation_space = Tuple((
            Discrete(0, self.MAX_CAPACITY + 1),
            Discrete(0, self.MAX_CAPACITY + 1)))
        self.locations = {}
        self._reset()

    def _step(self, action: int) -> _Tuple[_Tuple[int, int], int, bool, str]:
        assert self.action_space.contains(action)

        reward = self._move_cars(action)

        for key in self.locations.keys():
            reward += self._rent_cars(key)
            self._return_cars(key)

        return self._get_obs(), reward, False, ''

    def _get_obs(self) -> _Tuple[int, int]:
        return self.locations['a'], self.locations['b']

    def _reset(self, default=None) -> _Tuple[int, int]:
        if default is not None:
            if self.observation_space.contains([default, default]):
                self.locations['a'], self.locations['b'] = default, default
        else:
            self.locations['a'], self.locations['b'] = self.observation_space.sample()

        return self._get_obs()

    def _move_cars(self, action: int) -> int:

        _from = 'a'
        _to = 'b'

        if action < 0:
            _from = 'b'
            _to = 'a'

        action = abs(action)

        if self.locations[_from] < action:
            self.locations[_to] = self.locations[_to] + self.locations[_from]
            self.locations[_from] = 0
        else:
            self.locations[_to] = self.locations[_to] + action
            self.locations[_from] = self.locations[_from] - action

        return -(action * self.TRANSFER_COST)

    def _rent_cars(self, location) -> int:

        rent = round(poisson(self.LAMBDA[location]['rental']))

        if self.locations[location] < rent:
            reward = self.locations[location] * self.CREDIT_PER_CAR_RENTAL
            self.locations[location] = 0
        else:
            reward = rent * self.CREDIT_PER_CAR_RENTAL
            self.locations[location] = self.locations[location] - rent

        return reward

    def _return_cars(self, location):

        _return = round(poisson(self.LAMBDA[location]['return']))

        if self.locations[location] + _return > self.MAX_CAPACITY:
            self.locations[location] = self.MAX_CAPACITY
        else:
            self.locations[location] = self.locations[location] + _return
