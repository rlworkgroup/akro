"""A space representing a selection between a finite number of items."""
import numpy as np

from akro.space import Space


class Discrete(Space):
    """{0,1,...,n-1}."""

    def __init__(self, n):
        self._n = n

    @property
    def n(self):
        """Return the number of elements in the Discrete space."""
        return self._n

    def sample(self):
        """Uniformly randomly sample a random element of this space."""
        return np.random.randint(self.n)

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space."""
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        """Compute a representation of the space."""
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        """Compare two Discrete Spaces for equality."""
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def flatten(self, x):
        """
        Return a flattened observation x.

        Returns:
            x (flattened)

        """
        ret = np.zeros(self.n)
        ret[x] = 1
        return ret

    def unflatten(self, x):
        """
        Return an unflattened observation x.

        Returns:
            x (unflattened)

        """
        return np.nonzero(x)[0][0]

    def flatten_n(self, x):
        """
        Return flattened observations xs.

        Returns:
            xs (flattened)

        """
        ret = np.zeros((len(x), self.n))
        ret[np.arange(len(x)), x] = 1
        return ret

    def unflatten_n(self, x):
        """
        Return unflattened observations xs.

        Returns:
            xs (unflattened)

        """
        return np.nonzero(x)[1]

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return self.n

    def weighted_sample(self, weights):
        """Compute a weighted sample of the elements in the Discrete Space."""
        # An array of the weights, cumulatively summed.
        cs = np.cumsum(weights)
        # Find the index of the first weight over a random value.
        idx = sum(cs < np.random.rand())
        return min(idx, self.n - 1)

    @property
    def default_value(self):
        """Return the default value of the spaceself.

        This is always just 0.
        """
        return 0

    def __hash__(self):
        """Hash the Discrete Space."""
        return hash(self.n)

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable given the name and extra dimensions.

        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
