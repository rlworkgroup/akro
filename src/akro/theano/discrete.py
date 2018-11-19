"""Spaces.Discrete for Theano."""
from akro.discrete import Discrete as AkroDiscrete
from akro.theano import _util


class Discrete(AkroDiscrete):
    """Theano extension of akro.Discrete."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        if self.n <= 2**8:
            return _util.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint8')
        elif self.n <= 2**16:
            return _util.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint16')
        else:
            return _util.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint32')
