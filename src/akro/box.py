"""A Space representing a rectangular region of space."""
import numpy as np

from akro.space import Space


class Box(Space):
    """A box in R^n.

    Each coordinate is bounded above and below.
    """

    def __init__(self, low, high, shape=None):
        """
        Initialize the Box.

        Two kinds of bounds are supported, scalars and arrays:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is
            provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are
            arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

    def sample(self):
        """Uniformly randomly sample a random element of this space."""
        return np.random.uniform(
            low=self.low, high=self.high,
            size=self.low.shape).astype(np.float32)

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space."""
        return x.shape == self.shape and (x >= self.low).all() and (
            x <= self.high).all()

    @property
    def shape(self):
        """Return the shape of samples contained in this Space."""
        return self.low.shape

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return np.prod(self.low.shape)

    @property
    def bounds(self):
        """Return a 2-tuple containing the lower and upper bounds."""
        return self.low, self.high

    def flatten(self, x):
        """
        Return a flattened observation x.

        Returns:
            x (flattened)

        """
        return np.asarray(x).flatten()

    def unflatten(self, x):
        """
        Return an unflattened observation x.

        Returns:
            x (unflattened)

        """
        return np.asarray(x).reshape(self.shape)

    def flatten_n(self, xs):
        """
        Return flattened observations xs.

        Returns:
            xs (flattened)

        """
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_n(self, xs):
        """
        Return unflattened observations xs.

        Returns:
            xs (unflattened)

        """
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], ) + self.shape)

    def __repr__(self):
        """Compute a representation of the Box Space."""
        return "Box" + str(self.shape)

    def __eq__(self, other):
        """Compare two Box Spaces for approximate equality."""
        return isinstance(other, Box) \
            and np.allclose(self.low, other.low) \
            and np.allclose(self.high, other.high)

    def __hash__(self):
        """Hash the Box Space by value."""
        return hash((self.low, self.high))

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable given the name and extra dimensions.

        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
