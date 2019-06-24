import pickle
import unittest

import numpy as np

from akro import Box
from akro import tf
from akro import theano
from akro.requires import requires_tf, requires_theano


class TestBox(unittest.TestCase):
    def test_pickleable(self):
        obj = Box(-1.0, 1.0, (3, 4))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.shape == obj.shape
        assert np.array_equal(round_trip.bounds[0], obj.bounds[0])
        assert np.array_equal(round_trip.bounds[1], obj.bounds[1])

    def test_same_dtype(self):
        type1 = np.float32
        box = Box(0, 255, (3, 4), type1)
        assert box.dtype == type1

        type2 = np.uint8
        box = Box(0, 255, (3, 4), type2)
        assert box.dtype == type2

    def test_invalid_env(self):
        with self.assertRaises(AttributeError):
            Box(0.0, 1.0)

        with self.assertRaises(AssertionError):
            Box(np.array([-1.0, -2.0]), np.array([1.0, 2.0]), (2, 2))

    def test_default_float32_env(self):
        box = Box(0.0, 1.0, (3, 4))
        assert box.dtype == np.float32

        box = Box(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
        assert box.dtype == np.float32

    def test_flat_dim(self):
        box = Box(0.0, 1.0, (3, 4))
        assert box.flat_dim == 12

    def test_bounds(self):
        box = Box(0.0, 1.0, (3, 4))
        low, high = box.bounds
        assert low.shape == (3, 4)
        assert high.shape == (3, 4)
        assert low.dtype == np.float32
        assert high.dtype == np.float32

    def test_flatten(self):
        box = Box(0.0, 1.0, (3, 4))
        arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        flat_arr = box.flatten(arr)
        assert flat_arr.shape == (12, )
        assert flat_arr.dtype == np.int64

    def test_unflatten(self):
        box = Box(0.0, 1.0, (3, 4))
        arr = box.unflatten([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
        assert arr.shape == (3, 4)
        assert arr.dtype == np.int64

    def test_flatten_n(self):
        box = Box(0.0, 1.0, (3, 4))
        obs = np.asarray((([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                          ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                          ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9])))
        arr = box.flatten_n(obs)
        assert obs.shape == (3, 4, 3)
        assert arr.shape == (3, 12)

    def test_unflatten_n(self):
        box = Box(0.0, 1.0, (3, 4))
        obs = np.asarray((([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                          ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                          ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9])))
        arr = box.unflatten_n(obs)
        assert obs.shape == (3, 4, 3)
        assert arr.shape == (3, 3, 4)

    def test_flatten_n_tuple(self):
        box = Box(0.0, 1.0, (3, 4))
        obs = ((([1, 2, 3], [3, 4, 5], [5, 6, 7],
                 [7, 8, 9]), ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9])))
        arr = box.flatten_n(obs)
        assert arr.shape == (3, 12)

    def test_unflatten_n_tuple(self):
        box = Box(0.0, 1.0, (3, 4))
        obs = ((([1, 2, 3], [3, 4, 5], [5, 6, 7],
                 [7, 8, 9]), ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]),
                ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9])))
        arr = box.unflatten_n(obs)
        assert arr.shape == (3, 3, 4)

    def test_hash(self):
        box = Box(0.0, 1.0, (3, 4))
        assert box.__hash__() == 1213972508617964782

    @requires_tf
    def test_convert_tf(self):
        box = Box(0.0, 1.0, (3, 4))
        tensor = box.to_tf_placeholder('test', 1)
        assert isinstance(tensor, tf.Tensor)
        assert tensor.dtype == tf.float32
        assert tensor.get_shape().as_list() == [None, 3, 4]

    @requires_theano
    def test_convert_theano(self):
        box = Box(0.0, 1.0, (3, 4))
        tensor = box.to_theano_tensor('test', 1)
        assert isinstance(tensor, theano.tensor.TensorVariable)
        assert box.dtype == np.float32
        assert tensor.dtype == 'float32'
