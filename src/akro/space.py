"""The abstract base class for all Space types."""


class Space:
    """Provides a classification state spaces and action spaces.

    Allows you to write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self, seed=0):
        """Uniformly randomly sample a random element of this space."""
        raise NotImplementedError

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def flatten(self, x):
        """
        Return a flattened observation x.

        Returns:
            x (flattened)

        """
        raise NotImplementedError

    def unflatten(self, x):
        """
        Return an unflattened observation x.

        Returns:
            x (unflattened)

        """
        raise NotImplementedError

    def flatten_n(self, xs):
        """
        Return flattened observations xs.

        Returns:
            xs (flattened)

        """
        raise NotImplementedError

    def unflatten_n(self, xs):
        """
        Return unflattened observations xs.

        Returns:
            xs (unflattened)

        """
        raise NotImplementedError

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        raise NotImplementedError

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable given the name and extra dimensions.

        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
