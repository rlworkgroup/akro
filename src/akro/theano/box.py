"""Spaces.Box for Theano."""
from akro.box import Box as AkroBox
from akro.theano import _util


class Box(AkroBox):
    """Theano extension of akro.Box."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return _util.new_tensor(
            name=name, ndim=extra_dims + 1, dtype=self.dtype)
