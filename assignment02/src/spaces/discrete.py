import numpy as np
from assignment02.src.core import Space


class Discrete(Space):
    """
    {low,...,high}
    Example usage:
    self.observation_space = spaces.Discrete(0, 11)
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.randint(self.low, self.high)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return self.low <= as_int < self.high

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n