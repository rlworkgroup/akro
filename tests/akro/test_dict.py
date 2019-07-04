import collections
import pickle
import unittest

import numpy as np

from akro import Box
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
        d = Dict(
            collections.OrderedDict(
                position=Box(0, 10, (2, )), velocity=Box(0, 10, (3, ))))
        assert d.flat_dim == 5

    def test_flat_dim_with_keys(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        assert d.flat_dim_with_keys(['position']) == 2

    def test_flatten(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([1., 2., 3., 4., 5.])
        # Keys are intentionally in the "wrong" order.
        s = collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))])
        assert (d.flatten(s) == f).all()

    def test_unflatten(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([1., 2., 3., 4., 5.])
        # Keys are intentionally in the "wrong" order.
        s = collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))])
        assert all((s[k] == v).all() for k, v in d.unflatten(f).items())

    def test_flatten_n(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.]])
        # Keys are intentionally in the "wrong" order.
        s = [
            collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))]),
            collections.OrderedDict([('velocity', np.array([8., 9., 0.])),
                                     ('position', np.array([6., 7.]))])
        ]
        assert (d.flatten_n(s) == f).all()

    def test_unflatten_n(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.]])
        # Keys are intentionally in the "wrong" order.
        s = [
            collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))]),
            collections.OrderedDict([('velocity', np.array([8., 9., 0.])),
                                     ('position', np.array([6., 7.]))])
        ]
        for i, fi in enumerate(d.unflatten_n(f)):
            assert all((s[i][k] == v).all() for k, v in fi.items())

    def test_flatten_with_keys(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([3., 4., 5.])
        f_full = np.array([1., 2., 3., 4., 5.])
        # Keys are intentionally in the "wrong" order.
        s = collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))])
        assert (d.flatten_with_keys(s, ['velocity']) == f).all()
        assert (d.flatten_with_keys(s,
                                    ['velocity', 'position']) == f_full).all()

    def test_unflatten_with_keys(self):
        d = Dict(
            collections.OrderedDict([('position', Box(0, 10, (2, ))),
                                     ('velocity', Box(0, 10, (3, )))]))
        f = np.array([3., 4., 5.])
        f_full = np.array([1., 2., 3., 4., 5.])
        # Keys are intentionally in the "wrong" order.
        s = collections.OrderedDict([('velocity', np.array([3., 4., 5.])),
                                     ('position', np.array([1., 2.]))])
        assert all((s[k] == v).all()
                   for k, v in d.unflatten_with_keys(f, ['velocity']).items())
        assert all((s[k] == v).all() for k, v in d.unflatten_with_keys(
            f_full, ['velocity', 'position']).items())

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
