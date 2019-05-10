import pickle
import unittest

from akro import Dict
from akro import Discrete


class TestDict(unittest.TestCase):
    def test_pickleable(self):
        motion_dict = {'position': Discrete(2), 'velocity': Discrete(3)}
        sample = {
            'position': 1,
            'velocity': 2,
        }
        d = Dict(motion_dict)
        round_trip = pickle.loads(pickle.dumps(d))

        assert d.contains(sample)
        assert round_trip
        assert round_trip.contains(sample)
