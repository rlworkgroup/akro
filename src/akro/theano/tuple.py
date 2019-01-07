"""Spaces.Tuple for Theano."""

from akro.theano import _util
from akro.tuple import Tuple as AkroTuple


class Tuple(AkroTuple):
    """Theano extension of akro.Tuple."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return _util.new_tensor(
            name=name,
            ndim=extra_dims + 1,
            dtype=self._common_dtype,
        )
