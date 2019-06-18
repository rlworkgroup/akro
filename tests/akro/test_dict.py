import pickle
import unittest

from akro import Dict
from akro import Discrete
from akro import tf
from akro import theano
from akro.requires import requires_tf, requires_theano


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

    def test_flat_dim(self):
        pass

    def test_flatten(self):
        pass

    def test_unflatten(self):
        pass

    def test_flatten_n(self):
        pass

    def test_unflatten_n(self):
        pass

    @requires_tf
    def test_convert_tf(self):
        d = Dict({'position': Discrete(2), 'velocity': Discrete(3)})
        tensor_dict = d.to_tf_placeholder('test', 1)
        assert isinstance(tensor_dict, Dict)
        assert all(
            [isinstance(c, tf.Tensor) for c in tensor_dict.spaces.values()])
        assert all([v.dtype == tf.int64 for v in tensor_dict.spaces.values()])

    @requires_theano
    def test_convert_theano(self):
        d = Dict({'position': Discrete(2), 'velocity': Discrete(3)})
        tensor_dict = d.to_theano_tensor('test', 1)
        assert isinstance(tensor_dict, Dict)
        assert all([
            isinstance(c, theano.tensor.TensorVariable)
            for c in tensor_dict.spaces.values()
        ])
        assert all(
            [space.dtype == 'int64' for space in tensor_dict.spaces.values()])
