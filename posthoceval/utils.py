from collections.abc import Sized
from collections.abc import Iterable
from typing import Sequence  # collections.abc Sequence no subscript until 3.9
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Any

from itertools import repeat
from collections import defaultdict

import numpy as np

_RANK_AS_ARRAY_TYPE = defaultdict(lambda: 'tensor',
                                  {0: 'scalar', 1: 'vector', 2: 'matrix'})


def as_sized(x: Iterable) -> Union[Tuple, Sized]:
    if not isinstance(x, Sized):
        x = tuple(x)
    return x


def is_shape_equal(shape_a: Sequence[Optional[int]],
                   shape_b: Sequence[Optional[int]]):
    """Shape comparison allowing for `None`s for unknown/don't-care dims"""
    shape_a = as_sized(shape_a)
    shape_b = as_sized(shape_b)
    assert_same_size(len(shape_a), len(shape_b), units='dimensions')
    for da, db in zip(shape_a, shape_b):
        if not (da is None or db is None or da == db):
            return False
    return True


def assert_same_size(expected: int,
                     received: int,
                     units: str = 'values',
                     ret=None) -> Optional[Any]:
    if expected != received:
        raise ValueError('Expected %d %s but received %d instead.' %
                         (expected, units, received))
    return ret


def assert_rank(a: np.ndarray,
                rank: int,
                name: Optional[str] = None) -> np.ndarray:
    actual_rank = np.ndim(a)
    if actual_rank != rank:
        exp_type = _RANK_AS_ARRAY_TYPE[rank]
        act_type = _RANK_AS_ARRAY_TYPE[actual_rank]
        raise ValueError('Expected %s of rank %d but received %s of rank %d '
                         'instead%s.' % (exp_type, rank, act_type, actual_rank,
                                         ' for ' + name if name else ''))
    return a


def assert_shape(a: np.ndarray,
                 shape: Sequence[Optional[int]],
                 name: Optional[str] = None) -> np.ndarray:
    actual_shape = np.shape(a)
    if not is_shape_equal(actual_shape, shape):
        exp_type = _RANK_AS_ARRAY_TYPE[np.ndim(a)]
        act_type = _RANK_AS_ARRAY_TYPE[len(shape)]
        raise ValueError('Expected %s of shape %s but received %s of shape %s '
                         'instead%s.' % (exp_type, shape, act_type,
                                         actual_shape,
                                         ' for ' + name if name else ''))
    return a


def assert_same_shape(a: np.ndarray,
                      b: np.ndarray,
                      name_a: Optional[str] = 'a',
                      name_b: Optional[str] = 'b') -> Tuple[Optional[int]]:
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    if not is_shape_equal(a_shape, b_shape):
        a_type = _RANK_AS_ARRAY_TYPE[np.ndim(a)]
        b_type = _RANK_AS_ARRAY_TYPE[np.ndim(b)]
        raise ValueError(
            'Mismatched shapes: %s %s has shape %s but %s %s has shape %s.' %
            (a_type, name_a, a_shape, b_type, name_b, b_shape))
    return a_shape


def assert_same_rank(a: np.ndarray,
                     b: np.ndarray,
                     name_a: Optional[str] = 'a',
                     name_b: Optional[str] = 'b') -> int:
    a_rank = np.ndim(a)
    b_rank = np.ndim(b)
    if a_rank != b_rank:
        a_type = _RANK_AS_ARRAY_TYPE[a_rank]
        b_type = _RANK_AS_ARRAY_TYPE[b_rank]
        raise ValueError(
            'Mismatched ranks: %s %s has rank %s but %s %s has rank %s.' %
            (a_type, name_a, a_rank, b_type, name_b, b_rank))
    return a_rank


def as_iterator_of_size(values, size, units='values'):
    if isinstance(values, Iterable):
        values = as_sized(values)
        return iter(
            assert_same_size(size, len(values), units, ret=values))
    else:
        return repeat(values, size)
