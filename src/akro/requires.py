"""Decorators used for calling tensorflow and theano functions safely."""

import functools

from akro import tf, theano


def requires_tf(func):
    """Check tf is installed before calling a function."""
    @functools.wraps(func)  # yapf: disable
    def check(*args, **kwargs):
        """Execute the function if the module is loaded.

        Args:
            args (array): positional args passed to the function.
            kwargs (dictionary): keyword args passed to the function.

        Returns:
            func: Result of the provided function.

        """
        if not tf:  # pragma: no cover
            raise RuntimeError(
                'Module "tensorflow" is required to use the function {}'.
                format(func))
        return func(*args, **kwargs)

    return check


def requires_theano(func):
    """Check theano is installed before calling a function."""
    @functools.wraps(func)  # yapf: disable
    def check(*args, **kwargs):
        """Execute the function if the module is loaded.

        Args:
            args (array): positional args passed to the function.
            kwargs (dictionary): keyword args passed to the function.

        Returns:
            func: Result of the provided function.

        """
        if not theano:  # pragma: no cover
            raise RuntimeError(
                'Module "theano" is required to use the function {}'.format(
                    func))
        return func(*args, **kwargs)

    return check
