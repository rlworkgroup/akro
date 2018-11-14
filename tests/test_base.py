import pickle
import unittest

from akro import Space


class TestSpace(unittest.TestCase):
    def test_pickleable(self):
        obj = Space()
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
