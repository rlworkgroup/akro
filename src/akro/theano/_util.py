"""Private module containing utilities for working with Theno."""
import theano.tensor as TT


def new_tensor(name, ndim, dtype):
    """Return a new tensor based on the data type and name provided."""
    return TT.TensorType(dtype, (False, ) * ndim)(name)
