import numpy as np

from akro.space import Space


class Discrete(Space):
    """
    {0,1,...,n-1}
    """

    def __init__(self, n):
        self._n = n

    @property
    def n(self):
        return self._n

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def flatten(self, x):
        ret = np.zeros(self.n)
        ret[x] = 1
        return ret

    def unflatten(self, x):
        return np.nonzero(x)[0][0]

    def flatten_n(self, x):
        ret = np.zeros((len(x), self.n))
        ret[np.arange(len(x)), x] = 1
        return ret

    def unflatten_n(self, x):
        return np.nonzero(x)[1]

    @property
    def flat_dim(self):
        return self.n

    def weighted_sample(self, weights):
        # An array of the weights, cumulatively summed.
        cs = np.cumsum(weights)
        # Find the index of the first weight over a random value.
        idx = sum(cs < np.random.rand())
        return min(idx, self.n - 1)

    @property
    def default_value(self):
        return 0

    def __hash__(self):
        return hash(self.n)

    def new_tensor_variable(self, name, extra_dims):
        raise NotImplementedError
