"""Cartesian product of multiple Spaces (also known as a tuple of Spaces).

This Space produces samples which are Tuples, where the elments of those Tuples
are drawn from the components of this Space.
"""

import numpy as np

from akro.space import Space


class Tuple(Space):
    """A Tuple of Spaces which produces samples which are Tuples of samples."""

    def __init__(self, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]
        self._components = tuple(components)
        dtypes = [
            c.new_tensor_variable("tmp", extra_dims=0).dtype
            for c in components
        ]
        if dtypes and hasattr(dtypes[0], "as_numpy_dtype"):
            dtypes = [d.as_numpy_dtype for d in dtypes]
        self._common_dtype = np.core.numerictypes.find_common_type([], dtypes)

    def sample(self):
        """Uniformly randomly sample a random element of this space."""
        return tuple(x.sample() for x in self._components)

    @property
    def components(self):
        """Each of the spaces making up this Tuple space."""
        return self._components

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, tuple) and all(
            c.contains(xi) for c, xi in zip(self._components, x))

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return np.sum([c.flat_dim for c in self._components])

    def flatten(self, x):
        """
        Return a flattened observation x.

        Returns:
            x (flattened)

        """
        return np.concatenate(
            [c.flatten(xi) for c, xi in zip(self._components, x)])

    def flatten_n(self, xs):
        """
        Return flattened observations xs.

        Returns:
            xs (flattened)

        """
        xs_regrouped = [[x[i] for x in xs] for i in range(len(xs[0]))]
        flat_regrouped = [
            c.flatten_n(xi) for c, xi in zip(self.components, xs_regrouped)
        ]
        return np.concatenate(flat_regrouped, axis=-1)

    def unflatten(self, x):
        """
        Return an unflattened observation x.

        Returns:
            x (unflattened)

        """
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(
            c.unflatten(xi) for c, xi in zip(self._components, flat_xs))

    def unflatten_n(self, xs):
        """
        Return unflattened observations xs.

        Returns:
            xs (unflattened)

        """
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(xs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [
            c.unflatten_n(xi) for c, xi in zip(self.components, flat_xs)
        ]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped

    def __eq__(self, other):
        """Compute the equality between two Tuple spaces."""
        if not isinstance(other, Tuple):
            return False
        return tuple(self.components) == tuple(other.components)

    def __hash__(self):
        """Hash the Tuple space."""
        return hash(tuple(self.components))

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable given the name and extra dimensions.

        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
