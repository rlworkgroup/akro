"""A Space representing an RGB Image."""
import numpy as np

from akro.box import Box


class Image(Box):
    """An Image, represented by a Box of at most three dimensions.

    This class allows us to type check the observation input
    and decide whether to normalize. Each dimension must have
    pixel values between [0, 255].

    Args:
        shape(tuple): Shape of the observation. The shape cannot
            have more than 3 dimensions.
    """

    def __init__(self, shape):
        assert len(shape) <= 3, 'Images must have at most three dimensions'
        super(Box, self).__init__(low=0, high=255, shape=shape, dtype=np.uint8)
