"""The abstract base class for all Space types."""

import abc

import gym.spaces


class Space(abc.ABC, gym.spaces.Space):
    """Provides a classification state spaces and action spaces.

    Allows you to write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    @abc.abstractmethod
    def flatten(self, x):
        """Return a flattened observation x.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            np.ndarray: An array of x collapsed into one dimension.

        """

    @abc.abstractmethod
    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            np.ndarray: An array of x in the shape of self.shape.

        """

    @abc.abstractmethod
    def flatten_n(self, xs):
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """

    @abc.abstractmethod
    def unflatten_n(self, xs):
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element and self.shape.

        """

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""

    @abc.abstractmethod
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

    @abc.abstractmethod
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
