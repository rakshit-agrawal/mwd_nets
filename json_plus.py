#!/usr/bin/env python

import argparse
import base64
import datetime
import importlib
import inspect
import json
import numbers
import numpy
import traceback
import unittest

import torch
from torch.nn.parameter import Parameter
from torch_bounded_parameters import BoundedParameter
from torch.nn import Module

fallback = {}
remapper = {}

class Storage(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Serializable(object):

    # def __hash__(self):
    #     return hash(self.__dict__.__hash__)
    #
    # def __eq__(self, other):
    #     return self.__dict__ == other.__dict__

    def to_json(self, pack_ndarray=True, tolerant=True):
        return Serializable.dumps(self, pack_ndarray=pack_ndarray, tolerant=tolerant)

    @staticmethod
    def dumps(obj, pack_ndarray=True, tolerant=True):
        def custom(o):
            if isinstance(o, Serializable):
                module = o.__class__.__module__
                d = {'meta_class': '%s.%s' % (module,
                                              o.__class__.__name__)}
                fulld = {k: v for k, v in o.__dict__.items()}
                if hasattr(o, '_parameters'):
                    fulld.update(o._parameters)
                if hasattr(o, '_modules'):
                    fulld.update(o._modules)
                # print "fulld:", fulld.keys()
                d.update(item for item in fulld.items() if not item[0].startswith('_'))
                # print "serialized:", d.keys(), "orig", o._parameters.keys()
                return d
            elif isinstance(o, argparse.Namespace):
                d = {'meta_class': 'Storage',
                     'd': o.__dict__}
                return d
            elif isinstance(o, datetime.datetime):
                d = {'meta_class': 'datetime.datetime',
                     'date': o.isoformat()}
                return d
            elif isinstance(o, set):
                d = {'meta_class': 'set',
                     'set': list(o)}
                return d
            # elif isinstance(o, file):
            #     return '<file %r>' % o.name

            elif pack_ndarray and isinstance(o, numpy.ndarray):
                # This catches both numpy arrays, and CamArray.
                d = {'meta_class': 'numpy.ndarray',
                     'dtype': str(o.dtype),
                     'shape': o.shape,
                     'data': base64.b64encode(o.tobytes()).decode()}
                return d

            # We try to preserve numpy numbers.
            elif type(o).__module__ == numpy.__name__ and isinstance(o, numbers.Real):
                d = {'meta_class': 'numpy.number',
                     'dtype': str(o.dtype),
                     'data': base64.b64encode(o.tobytes()).decode()
                     }
                return d

            # Parameters.
            elif isinstance(o, BoundedParameter):
                d = {
                    'meta_class': 'BoundedParameter',
                    'data': o.cpu().data.numpy(),
                    'requires_grad': o.requires_grad,
                    'lower_bound': o.lower_bound,
                    'upper_bound': o.upper_bound,
                }
                return d

            elif isinstance(o, Parameter):
                if hasattr(o, 'lower_bound'):
                    d = {
                        'meta_class': 'BoundedParameter',
                        'data': o.cpu().data.numpy(),
                        'requires_grad': o.requires_grad,
                        'lower_bound': o.lower_bound,
                        'upper_bound': o.upper_bound,
                    }
                else:
                    d = {
                        'meta_class': 'Parameter',
                        'data': o.cpu().data.numpy(),
                        'requires_grad': o.requires_grad,
                    }
                return d

            # Normal Python types are unchanged
            # elif isinstance(o, (int, long, str, unicode, float, bool, list, tuple, dict)):
            elif isinstance(o, (int, str, float, bool, list, tuple, dict)):
                return o

            # These two defaults are catch-all
            elif isinstance(o, numbers.Integral):
                return int(o)
            elif isinstance(o, numbers.Real):
                return float(o)

            elif tolerant:
                return None
            else:
                raise ValueError("Cannot encode in json object %r" % o)
        return json.dumps(obj, default=custom, indent=2)

    @staticmethod
    def from_json(s, objectify=True, device='cpu'):
        def hook(o):
            meta_module, meta_class = None, o.get('meta_class')
            if meta_class in ('Datetime', 'datetime.datetime'):
                # 'Datetime' included for backward compatibility
                try:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S.%f')
                except Exception as e:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S')
                return tmp
            elif meta_class == 'set':
                # Set.
                return set(o['set'])
            # Numpy arrays.
            elif meta_class == 'numpy.ndarray':
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                shape = o['shape']
                v = numpy.frombuffer(data, dtype=dtype)
                v = v.reshape(shape)
                return v

            # Numpy numbers.
            elif meta_class == 'numpy.number':
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                v = numpy.frombuffer(data, dtype=dtype)[0]
                return v

            # Parameters
            elif meta_class == 'Parameter':
                p = Parameter(torch.tensor(o['data']), requires_grad=o['requires_grad'])
                return p.to(device)
            elif meta_class == 'BoundedParameter':
                p = BoundedParameter(torch.tensor(o['data']),
                                     requires_grad=o['requires_grad'],
                                     lower_bound=o['lower_bound'],
                                     upper_bound=o['upper_bound'])
                return p.to(device)

            elif meta_class == 'Storage':
                p = Storage(o['d'])
                return p

            elif meta_class and '.' in meta_class:
                # correct for classes that have migrated from one module to another
                meta_class = remapper.get(meta_class, meta_class)
                # separate the module name from the actual class name
                meta_module, meta_class = meta_class.rsplit('.',1)

            if meta_class is not None:
                del o['meta_class']
                # this option is for backward compatibility in case a module is not specified
                if meta_class in fallback:
                    meta_module = fallback.get(meta_class)

                if meta_module is not None and objectify:
                    try:
                        module = importlib.import_module(meta_module)
                        cls = getattr(module, meta_class)
                        # Figures out parameters for intializer.
                        try:
                            args = inspect.getargspec(cls.__init__)[0]
                            args = [x for x in args if x in o.keys()]
                            dd = [o[x] for x in args]
                            obj = cls(*dd)
                        except:
                            print("Incomplete rebuild:", traceback.format_exc())
                            obj = cls()
                        obj.__dict__.update(o)
                        # Restores modules.
                        if isinstance(obj, Module):
                            for k, v in obj._modules.items():
                                setattr(obj, k, v)
                        o = obj
                    except Exception as e:
                        # We need to allow the case where the class is now obsolete.
                        print(traceback.format_exc())
                        print("Could not restore: %r %r", (meta_module, meta_class))
                        o = None
            elif type(o).__name__ == 'dict':
                o = Storage(o)
            return o

        return json.loads(s, object_hook=hook)

    @staticmethod
    def loads(s):
        return Serializable.from_json(s)

class MyClass(Serializable):
    def __init__(self, a):
        self.a = a
    def __repr__(self):
        return "I am:" + repr(self.a)

class MyOtherClass(Serializable):
    def __init__(self, a, b, c=1, d=3):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    def __repr__(self):
        return "I am:" + repr(self.a) + ":" + repr(self.b) + ":" + repr(self.c) + ":" + repr(self.d)

class TestSerializable(unittest.TestCase):

    def test_simple(self):
        a = Serializable()
        a.x = 1
        a.y = 'test'
        a.z = 3.14
        b = Serializable.from_json(a.to_json())
        print(a)
        print(b)
        self.assertEqual(a, b)

    def test_class(self):
        x = MyClass(3)
        s = x.to_json()
        print(s)
        y = Serializable.from_json(s)
        print(y)
        self.assertEqual(x, y)

    def test_other_class(self):
        x = MyOtherClass(3, 4, d=6)
        s = x.to_json()
        print(s)
        y = Serializable.from_json(s)
        print(y)
        self.assertEqual(x, y)

    def test_datetime(self):
        a = Serializable()
        a.x = datetime.datetime(2015,1,3)
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_recursive(self):
        a = Serializable()
        a.x = Serializable()
        a.x.y = 'test'
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_numpy(self):
        a = Serializable()
        a.x = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - b.x)), 0)

    def test_numpy_twice(self):
        a = Serializable()
        a.x = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - b.x)), 0)
        c = Serializable.from_json(b.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - c.x)), 0)

    def test_numpy_direct(self):
        a = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        s = Serializable.dumps(a, pack_ndarray=True)
        c = Serializable.from_json(s)
        self.assertEqual(numpy.sum(numpy.abs(a - c)), 0)

    def test_float(self):
        x = numpy.float16(3.5)
        y = Serializable.from_json(Serializable.dumps(x))
        self.assertAlmostEqual(y, x, 2)

    def test_numpy_uint32(self):
        x = numpy.uint32(55)
        s = Serializable.dumps(x)
        y = Serializable.from_json(s)
        self.assertEqual(x, y)
        self.assertEqual(str(x.dtype), 'uint32')
        self.assertEqual(str(y.dtype), 'uint32')

    def test_numpy_float128(self):
        x = numpy.float128(55.3)
        s = Serializable.dumps(x)
        y = Serializable.from_json(s)
        self.assertAlmostEqual(x, y, 5)
        self.assertEqual(str(x.dtype), 'float128')
        self.assertEqual(str(y.dtype), 'float128')

    def test_set(self):
        s = set(['a', 'b', 'c'])
        x = Serializable.dumps(s)
        print("Set representation:", x)
        t = Serializable.loads(x)
        self.assertEqual(s, t)

if __name__ == '__main__':
    unittest.main()
