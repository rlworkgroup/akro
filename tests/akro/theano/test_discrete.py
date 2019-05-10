import pickle
import unittest

from akro import Discrete


class TestDiscrete(unittest.TestCase):
    def test_pickleable(self):
        obj = Discrete(10)
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.n == 10
