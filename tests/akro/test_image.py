import pickle
import unittest

import numpy as np

from akro import Image


class TestImage(unittest.TestCase):

    def test_pickleable(self):
        obj = Image((3, 3, 3))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.shape == obj.shape
        assert np.array_equal(round_trip.bounds[0], obj.bounds[0])
        assert np.array_equal(round_trip.bounds[1], obj.bounds[1])

    def test_invalid_shape(self):
        with self.assertRaises(AssertionError):
            Image((1, 1, 1, 1))
        with self.assertRaises(TypeError):
            Image()

    def test_dtype(self):
        img = Image((1, 1, 1))
        assert img.dtype == np.uint8
