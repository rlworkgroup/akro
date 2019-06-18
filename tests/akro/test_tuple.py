import pickle
import unittest

import numpy as np
import tensorflow as tf

from akro import Box
from akro import Discrete
from akro import theano
from akro import Tuple
from akro.requires import requires_tf, requires_theano


class TestTuple(unittest.TestCase):
    def test_pickleable(self):
        tup = Tuple((Discrete(3), Discrete(2)))
        round_trip = pickle.loads(pickle.dumps(tup))
        assert round_trip
        assert round_trip.spaces == tup.spaces

    def test_flat_dim(self):
        tup = Tuple((Discrete(3), Discrete(2)))
        assert tup.flat_dim == 5

    def test_flatten(self):
        tup = Tuple((Discrete(3), Discrete(2)))
        x = [2, 0]
        arr = tup.flatten(x)
        assert arr[2] == arr[3] == 1

    def test_unflatten(self):
        tup = Tuple((Discrete(3), Discrete(2)))
        x = [2, 0]
        arr = tup.flatten(x)
        assert tup.unflatten(arr) == (2, 0)

    def test_flatten_n(self):
        disc = Discrete(3)
        tup = Tuple((Discrete(2), disc))
        obs = disc.flatten_n(np.asarray([0, 1, 0, 1, 2]))
        unflat_ret = tup.unflatten_n(obs)
        flat_ret = tup.flatten_n(unflat_ret)
        base = np.asarray([[1., 0., 1., 0., 0.]])
        assert np.array_equal(flat_ret, base)

    def test_unflatten_n(self):
        disc = Discrete(3)
        tup = Tuple((Discrete(2), disc))
        obs = disc.flatten_n(np.asarray([0, 1, 0, 1, 2]))
        ret = tup.unflatten_n(obs)
        assert ret == [(0, 0)]

    def test_hash(self):
        tup = Tuple((Discrete(3), Discrete(2)))
        assert tup.__hash__() == 3713083796995235906

    @requires_tf
    def test_convert_tf(self):
        tup = Tuple((Box(0.0, 1.0, (3, 4)), Discrete(2)))
        tensor_tup = tup.to_tf_placeholder('test', 1)
        assert isinstance(tensor_tup, tuple)
        assert all([isinstance(c, tf.Tensor) for c in tensor_tup])
        assert [c.dtype for c in tensor_tup] == [tf.float32, tf.int64]
        assert [c.get_shape().as_list() for c in tensor_tup] == [[None, 3, 4],
                                                                 [None, 2]]

    @requires_theano
    def test_convert_theano(self):
        tup = Tuple((Box(0.0, 1.0, (3, 4)), Discrete(2)))
        tensor_tup = tup.to_theano_tensor('test', 1)
        assert isinstance(tensor_tup, tuple)
        assert all(
            [isinstance(c, theano.tensor.TensorVariable) for c in tensor_tup])
        assert [c.dtype for c in tensor_tup] == ['float32', 'int64']
