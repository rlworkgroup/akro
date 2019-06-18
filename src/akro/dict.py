"""Cartesian product of multiple named Spaces (also known as a dict of Spaces).

This Space produces samples which are dicts, where the values of those dicts
are drawn from the values of this Space.
"""
import gym.spaces

from akro.requires import requires_tf, requires_theano
from akro.space import Space


class Dict(gym.spaces.Dict, Space):
    """
    A dictionary of simpler spaces, e.g. Discrete, Box.

    Example usage:
        self.observation_space = spaces.Dict({"position": spaces.Discrete(2),
                                              "velocity": spaces.Discrete(3)})
    """

    @property  # pragma: no cover
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        raise NotImplementedError

    def flatten(self, x):  # pragma: no cover
        """Return an observation of x with collapsed values.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            Dict: A Dict where each value is collapsed into a single dimension.
                  Keys are unchanged.

        """
        raise NotImplementedError

    def unflatten(self, x):  # pragma: no cover
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            np.ndarray: An array of x in the shape of self.shape.

        """
        raise NotImplementedError

    def flatten_n(self, xs):  # pragma: no cover
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """
        raise NotImplementedError

    def unflatten_n(self, xs):  # pragma: no cover
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element and self.shape.

        """
        raise NotImplementedError

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
