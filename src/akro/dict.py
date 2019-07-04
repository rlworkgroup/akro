"""Cartesian product of multiple named Spaces (also known as a dict of Spaces).

This Space produces samples which are dicts, where the values of those dicts
are drawn from the values of this Space.
"""
import collections

import gym.spaces
import numpy as np

import akro
from akro.requires import requires_tf, requires_theano
from akro.space import Space


class Dict(gym.spaces.Dict, Space):
    """
    A dictionary of simpler spaces, e.g. Discrete, Box.

    Example usage:
        self.observation_space = spaces.Dict({"position": spaces.Discrete(2),
                                              "velocity": spaces.Discrete(3)})
    """

    def __init__(self, spaces=None, **kwargs):
        super().__init__(spaces, **kwargs)
        self.spaces = (collections.OrderedDict(
            [(k, akro.from_gym(s)) for k, s in self.spaces.items()]))

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return sum([space.flat_dim for _, space in self.spaces.items()])

    def flat_dim_with_keys(self, keys):
        """
        Return a flat dimension of the spaces specified by the keys.

        Returns:
            sum (int)

        """
        return sum([self.spaces[key].flat_dim for key in keys])

    def flatten(self, x):
        """Return an observation of x with collapsed values.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            Dict: A Dict where each value is collapsed into a single dimension.
                  Keys are unchanged.

        """
        return np.concatenate(
            [space.flatten(x[key]) for key, space in self.spaces.items()],
            axis=-1,
        )

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            collections.OrderedDict

        """
        dims = np.array([s.flat_dim for s in self.spaces.values()])
        flat_x = np.split(x, np.cumsum(dims)[:-1])
        return collections.OrderedDict(
            [(key, self.spaces[key].unflatten(xi))
             for key, xi in zip(self.spaces.keys(), flat_x)])

    def flatten_n(self, xs):
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """
        return np.array([self.flatten(x) for x in xs])

    def unflatten_n(self, xs):
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            List[OrderedDict]

        """
        return [self.unflatten(x) for x in xs]

    def flatten_with_keys(self, x, keys):
        """
        Return flattened obs of spaces specified by the keys using x.

        Returns:
            list

        """
        return np.concatenate(
            [
                space.flatten(x[key])
                for key, space in self.spaces.items() if key in keys
            ],
            axis=-1,
        )

    def unflatten_with_keys(self, x, keys):
        """
        Return an unflattened observation.

        This is the inverse of `flatten_with_keys`.

        Returns:
            collections.OrderedDict

        """
        dims = np.array([
            space.flat_dim for key, space in self.spaces.items() if key in keys
        ])
        flat_x = np.split(x, np.cumsum(dims)[:-1])
        return collections.OrderedDict(
            [(key, space.unflatten(xi))
             for (key, space), xi in zip(self.spaces.items(), flat_x)
             if key in keys])

    @requires_tf
    def to_tf_placeholder(self, name, batch_dims):
        """Create a tensor placeholder from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            tf.Tensor: Tensor object with the same properties as
                the Dict where the shape is modified by batch_dims.

        """
        newdict = Dict()
        for key, space in self.spaces.items():
            newdict.spaces[key] = space.to_tf_placeholder(name, batch_dims)
        return newdict

    @requires_theano
    def to_theano_tensor(self, name, batch_dims):
        """Create a theano tensor from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            theano.tensor.TensorVariable: Tensor object with the
                same properties as the Dict where the shape is
                modified by batch_dims.

        """
        newdict = Dict()
        for key, space in self.spaces.items():
            newdict.spaces[key] = space.to_theano_tensor(name, batch_dims)
        return newdict
