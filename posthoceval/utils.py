from collections.abc import Sized
from collections.abc import Iterable
from typing import Sequence  # collections.abc Sequence no subscript until 3.9
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Any
from typing import Dict

from contextlib import contextmanager

import math

from itertools import repeat
from itertools import product
from collections import defaultdict

import joblib

import numpy as np

if hasattr(math, 'prod'):  # available in 3.8+
    prod = math.prod
else:  # functionally equivalent w/o positional argument checking
    """
    >>> %timeit reduce(mul, values)
    180 µs ± 2.15 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    
    >>> %timeit math.prod(values)
    133 µs ± 1.57 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    
    >>> math.prod(values) == reduce(mul, values)
    True
    """
    import operator
    from functools import reduce


    def prod(iterable, start=1):
        return reduce(operator.mul, iterable, start)

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


def assert_same_size(expected: Union[int, Sized],
                     received: Union[int, Sized],
                     units: str = 'values',
                     ret=None) -> Optional[Any]:
    if isinstance(expected, Sized) and not isinstance(expected, np.ndarray):
        expected = len(expected)
    if isinstance(received, Sized) and not isinstance(received, np.ndarray):
        received = len(received)
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


def dict_product(d: Dict[Any, Iterable]) -> Dict:
    keys = d.keys()
    iterables = d.values()
    for comb in product(*iterables):
        yield dict(zip(keys, comb))


def at_high_precision(func, *args, **kwargs):
    all_dtypes = set()

    args_cast = []
    kwargs_cast = {}

    for arg in args:
        if is_float(arg):
            all_dtypes.add(arg.dtype if hasattr(arg, 'dtype') else float)
            arg = np.float128(arg)
        args_cast.append(arg)

    for k, v in kwargs.items():
        if is_float(v):
            all_dtypes.add(v.dtype if hasattr(v, 'dtype') else float)
            v = np.float128(v)
        kwargs_cast[k] = v

    rets = func(*args_cast, **kwargs_cast)
    is_tuple = type(rets, tuple)
    if not is_tuple:
        rets = (rets,)

    rets_cast = []
    for ret in rets:
        if is_float(ret) and len(all_dtypes) == 1:
            # naively cast back
            ret_cast = next(iter(all_dtypes))(ret)

            if not (np.isinf(ret_cast).any() or np.isnan(ret_cast).any()):
                ret = ret_cast

        rets_cast.append(ret)

    return tuple(rets_cast) if is_tuple else rets_cast[0]


def is_int(v):
    return np.issubdtype(type(v), np.integer)


def is_float(v):
    return np.issubdtype(type(v), np.floating)


@contextmanager
def tqdm_parallel(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given
    as argument. Kudos to https://stackoverflow.com/a/58936697/6557588

    Usage:
    >>> with tqdm_parallel(tqdm(desc='My calculation', total=10)) as pbar:
    >>>     Parallel(n_jobs=-1)(delayed(sqrt)(i**2) for i in range(10))
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def safe_parse_tuple(string_, verbose=False):
    """naively parse string, only returns strings

    why did i bother with this
    """
    return tuple(_safe_parse_tuple_generator(string_, verbose=verbose))


def _safe_parse_tuple_generator(string_, verbose=False):
    # updated dynamically
    element = ''
    left = None
    tuple_open = tuple_closed = token_ready = False
    # updated consistently
    prev_token = None
    # static
    whitespace = ' \n\r\t\f\v'

    for token in string_:
        if verbose:
            print(f'token \'{token}\' [left={left}|tuple_open={tuple_open}|'
                  f'tuple_closed={tuple_closed}|token_ready={token_ready}|'
                  f'prev_token=\'{prev_token}\'|element=\'{element}\'')
        if left is not None:
            if prev_token != '\\' and token == left:
                yield element
                left, element = None, ''
            else:
                element += token
        elif token in whitespace:
            if element:
                yield element
                element, token_ready = '', False
            continue  # do not record whitespaces
        elif not tuple_open:
            assert token == '(', 'leading char must be "(" in tuple'
            tuple_open = token_ready = True
        elif tuple_closed:
            raise AssertionError('no non-whitespace allowed after closing ")"')
        elif token == ')':
            if element:
                yield element
                element = ''
            tuple_closed, token_ready = True, False
        elif token == ',':
            assert prev_token != ',', 'cannot have back to back commas'
            assert prev_token != '(', 'cannot have comma after opening "("'
            if element:
                yield element
                element = ''
            token_ready = True
        elif token_ready:
            if token in '\'\"' and prev_token != '\\':
                left = token
            else:
                element += token
        else:
            raise AssertionError(f'unexpected token \'{token}\'')
        prev_token = token
    assert tuple_closed, (f'failed to close an element with a {left}'
                          if left else 'missing ")" to close tuple')
