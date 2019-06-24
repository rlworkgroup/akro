import pickle
import unittest

import numpy as np
import tensorflow as tf

from akro import Discrete
from akro import theano
from akro.requires import requires_tf, requires_theano


class TestDiscrete(unittest.TestCase):
    def test_pickleable(self):
        obj = Discrete(10)
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.n == 10

    def test_flat_dim(self):
        disc = Discrete(10)
        assert disc.flat_dim == 10

    def test_flatten(self):
        disc = Discrete(10)
        x = [3, 5, 7]
        arr = disc.flatten(x)
        assert all(arr[x] == 1)

    def test_unflatten(self):
        disc = Discrete(10)
        x = [3, 5, 7]
        arr = disc.flatten(x)
        assert disc.unflatten(arr) == 3

    def test_flatten_n(self):
        disc = Discrete(3)
        obs = np.asarray([0, 1, 2])
        arr = disc.flatten_n(obs)
        base = np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        assert np.array_equal(arr, base)

    def test_unflatten_n(self):
        disc = Discrete(3)
        obs = np.asarray([0, 1, 2])
        flat_arr = disc.flatten_n(obs)
        base = np.asarray([0, 1, 2])
        unflat_arr = disc.unflatten_n(flat_arr)
        assert np.array_equal(unflat_arr, base)

    def test_weighted_sample(self):
        disc = Discrete(4)
        weights = [0.1, 0.2, 0.3, 0.4]
        res = disc.weighted_sample(weights)
        assert res >= 0 and res < disc.n

    def test_weighted_sample_unnormalized(self):
        disc = Discrete(4)
        weights = np.array([1., 2., 3., 5.])
        res = disc.weighted_sample(weights)
        assert res >= 0 and res < disc.n

    def test_hash(self):
        disc = Discrete(10)
        assert disc.__hash__() == 10

    @requires_tf
    def test_convert_tf(self):
        disc = Discrete(10)
        tensor = disc.to_tf_placeholder('test', 1)
        assert isinstance(tensor, tf.Tensor)
        assert disc.dtype == np.int64
        assert tensor.dtype == tf.int64
        assert tensor.get_shape().as_list() == [None, 10]

    @requires_theano
    def test_convert_theano(self):
        disc = Discrete(10)
        tensor = disc.to_theano_tensor('test', 1)
        assert isinstance(tensor, theano.tensor.TensorVariable)
        assert disc.dtype == np.int64
        assert tensor.dtype == 'int64'
