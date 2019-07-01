"""Cartesian product of multiple Spaces (also known as a tuple of Spaces).

This Space produces samples which are Tuples, where the elments of those Tuples
are drawn from the components of this Space.
"""

import gym.spaces
import numpy as np

import akro
from akro.requires import requires_tf, requires_theano
from akro.space import Space


class Tuple(gym.spaces.Tuple, Space):
    """A Tuple of Spaces which produces samples which are Tuples of samples."""

    def __init__(self, spaces):
        super().__init__([akro.from_gym(space) for space in spaces])

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return np.sum([c.flat_dim for c in self.spaces])

    def flatten(self, x):
        """Return a flattened observation x.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            np.ndarray: An array of x collapsed into one dimension.

        """
        return np.concatenate([c.flatten(xi) for c, xi in zip(self.spaces, x)])

    def flatten_n(self, obs):
        """Return flattened observations obs.

        Args:
            obs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of obs in a shape inferred by the size of
                its first element.

        """
        obs_regrouped = [[x[i] for x in obs] for i in range(len(obs[0]))]
        flat_regrouped = [
            c.flatten_n(xi) for c, xi in zip(self.spaces, obs_regrouped)
        ]
        return np.concatenate(flat_regrouped, axis=-1)

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            tuple: A tuple of x in the shape of self.shape.

        """
        dims = [c.flat_dim for c in self.spaces]
        flat_x = np.split(x, np.cumsum(dims)[:-1])
        return tuple(c.unflatten(xi) for c, xi in zip(self.spaces, flat_x))

    def unflatten_n(self, obs):
        """Return unflattened observations obs.

        Args:
            obs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of obs in a shape inferred by the size of
                its first element and self.shape.

        """
        dims = [c.flat_dim for c in self.spaces]
        flat_obs = np.split(obs, np.cumsum(dims)[:-1], axis=-1)
        unflat_obs = [
            c.unflatten_n(xi) for c, xi in zip(self.spaces, flat_obs)
        ]
        unflat_obs_grouped = list(zip(*unflat_obs))
        return unflat_obs_grouped

    def __hash__(self):
        """
        Hash the Tuple Space.

        Returns:
            int: A hash of the Tuple's components.

        """
        return hash(tuple(self.spaces))

    @requires_tf
    def to_tf_placeholder(self, name, batch_dims):
        """Create a tensor placeholder from the Space object.

        Args:
            name (str): name to append to the akro type when naming
                the tensor.  e.g. When name is 'tmp' - 'Box-tmp',
                'Discrete-tmp'.

            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of each object in self.spaces.

        Returns:
            tuple(tf.Tensor): A tuple of Tensor objects converted
                from each Space in self.spaces. Each Tensor's
                shape is modified by batch_dims.

        """
        return tuple(
            s.to_tf_placeholder(type(s).__name__ + '-' + name, batch_dims)
            for s in self.spaces)

    @requires_theano
    def to_theano_tensor(self, name, batch_dims):
        """Create a theano tensor from the Space object.

        Args:
            name (str): name to append to the akro type when naming
                the tensor.  e.g. When name is 'tmp' - 'Box-tmp',
                'Discrete-tmp'.

            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of each object in self.spaces.

        Returns:
            theano.tensor.TensorVariable: A tuple of Tensor objects converted
                from each Space in self.spaces. Each Tensor's shape is
                modified by batch_dims.

        """
        return tuple(
            s.to_theano_tensor(type(s).__name__ + '-' + name, batch_dims)
            for s in self.spaces)
