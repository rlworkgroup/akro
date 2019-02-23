"""Private module containing utilities for working with Theno."""
import theano.tensor


def new_tensor(name, ndim, dtype):
    """Return a new tensor based on the data type and name provided."""
    return theano.tensor.TensorType(dtype, (False, ) * ndim)(name)
