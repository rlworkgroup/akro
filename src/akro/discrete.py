"""A space representing a selection between a finite number of items."""

import gym.spaces
import numpy as np

from akro import tf, theano
from akro.requires import requires_tf, requires_theano
from akro.space import Space


class Discrete(gym.spaces.Discrete, Space):
    """{0,1,...,n-1}."""

    def flatten(self, x):
        """Return a flattened observation x.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            np.ndarray: An array of x collapsed into one dimension.

        """
        ret = np.zeros(self.n)
        ret[x] = 1
        return ret

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            np.ndarray: An array of x in the shape of self.shape.

        """
        return np.nonzero(x)[0][0]

    def flatten_n(self, xs):
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """
        ret = np.zeros((len(xs), self.n))
        ret[np.arange(len(xs)), xs] = 1
        return ret

    def unflatten_n(self, xs):
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element and self.shape.

        """
        return np.nonzero(xs)[1]

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return self.n

    def weighted_sample(self, weights):
        """
        Compute a weighted sample of the elements in the Discrete Space.

        Args:
            weights (:obj:`list`): Values to use in the sample.

        Returns:
            int or np.ndarray: A random sample of n based on
                probabilities in weights.

        """
        assert len(weights) == self.n
        weights = np.asarray(weights)
        return np.random.choice(self.n, p=weights / weights.sum())

    def __hash__(self):
        """
        Hash the Discrete Space.

        Returns:
            int: A hash of the value n.

        """
        return hash(self.n)

    @requires_tf
    def to_tf_placeholder(self, name, batch_dims):
        """Create a tensor placeholder from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            tf.Tensor: Tensor object with the same properties as
                the Discrete obj where the shape is modified by batch_dims.

        """
        return tf.placeholder(
            dtype=self.dtype,
            shape=[None] * batch_dims + [self.flat_dim],
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
                same properties as the Discrete obj where the shape is
                modified by batch_dims..

        """
        return theano.tensor.TensorType(self.dtype,
                                        (False, ) * (batch_dims + 1))(name)
