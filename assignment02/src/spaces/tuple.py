from assignment02.src.core import Space


class Tuple(Space):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        self.shape = self._get_shape()

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def _get_shape(self):
        return tuple([space.shape for space in self.spaces])

    def __repr__(self):
        return "Tuple(" + ", ". join([str(s) for s in self.spaces]) + ")"