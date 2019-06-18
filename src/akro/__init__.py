"""A library containing types of Spaces."""
import gym.spaces

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = False

try:
    import theano
except ImportError:  # pragma: no cover
    theano = False

from akro.box import Box
from akro.dict import Dict
from akro.discrete import Discrete
from akro.space import Space
from akro.tuple import Tuple


def from_gym(space):
    """
    Convert a gym.space to an akro.space.

    Args:
        space(:obj:`gym.Space`): The Space object to convert.

    Returns:
        akro.Space: The gym.Space object converted to an
            akro.Space object.

    """
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Dict):
        return Dict(space.spaces)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(list(map(from_gym, space.spaces)))
    else:  # pragma: no cover
        raise TypeError


__all__ = [
    'Space', 'Box', 'Dict', 'Discrete', 'Tuple', 'from_gym', 'tf', 'theano'
]
