"""A Space representing a rectangular region of space."""
import gym.spaces
import numpy as np

from akro import tf, theano
from akro.requires import requires_tf, requires_theano
from akro.space import Space


class Box(gym.spaces.Box, Space):
    """A box in R^n.

    Each coordinate is bounded above and below.
    """

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return np.prod(self.low.shape)

    @property
    def bounds(self):
        """Return a 2-tuple containing the lower and upper bounds."""
        return self.low, self.high

    def flatten(self, x):
        """Return a flattened observation x.

        Args:
            x (:obj:'Iterable`): The object to flatten.

        Returns:
            np.ndarray: An array of x collapsed into one dimension.

        """
        return np.asarray(x).flatten()

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            np.ndarray: An array of x in the shape of self.shape.

        """
        return np.asarray(x).reshape(self.shape)

    def flatten_n(self, obs):
        """Return flattened observations obs.

        Args:
            obs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of obs in a shape inferred by the size of
                its first element.

        """
        return np.asarray(obs).reshape((len(obs), -1))

    def unflatten_n(self, obs):
        """Return unflattened observation of obs.

        Args:
            obs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of obs in a shape inferred by the size of
                its first element and self.shape.

        """
        return np.asarray(obs).reshape((len(obs), ) + self.shape)

    def __hash__(self):
        """
        Hash the Box Space.

        Returns:
            int: A hash of the low, high, and shape of the Box.

        Only the first element of low and high are hashed because numpy
        ndarrays can't be hashed. When a Box is created the low and high
        bounds are duplicated across the shape of the arrays so any of the
        values will suffice for the hash. The shape of the Box is added
        for uniqueness.

        """
        return hash((self.low[0][0], self.high[0][0], self.shape))

    @requires_tf
    def to_tf_placeholder(self, name, batch_dims):
        """Create a tensor placeholder from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            tf.Tensor: Tensor object with the same properties as
                the Box where the shape is modified by batch_dims.

        """
        return tf.placeholder(
            dtype=self.dtype,
            shape=[None] * batch_dims + list(self.shape),
            name=name)

    @requires_theano
    def to_theano_tensor(self, name, batch_dims):
        """Create a theano tensor from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            theano.tensor.TensorVariable: Tensor object with the
                same properties as the Box where the shape is
                modified by batch_dims.

        """
        return theano.tensor.TensorType(self.dtype,
                                        (False, ) * (batch_dims + 1))(name)
