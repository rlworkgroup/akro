import unittest

import gym.spaces

import akro


class TestSpace(unittest.TestCase):
    def test_convert_box(self):
        obj = gym.spaces.Box(0.0, 1.0, (3, 4))
        box = akro.from_gym(obj)
        assert isinstance(box, akro.Box)
        assert (box.low == obj.low).all()
        assert (box.high == obj.high).all()
        assert box.shape == obj.shape

    def test_convert_dict(self):
        obj = gym.spaces.Dict({'foo': gym.spaces.Discrete(3)})
        dict = akro.from_gym(obj)
        assert isinstance(dict, akro.Dict)

    def test_convert_discrete(self):
        obj = gym.spaces.Discrete(3)
        disc = akro.from_gym(obj)
        assert isinstance(disc, akro.Discrete)

    def test_convert_tuple(self):
        obj = gym.spaces.Tuple((gym.spaces.Discrete(2),
                                gym.spaces.Discrete(3)))
        tup = akro.from_gym(obj)
        assert isinstance(tup, akro.Tuple)
