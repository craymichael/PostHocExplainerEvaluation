import warnings

from functools import lru_cache
from functools import wraps

import numpy as np
import sympy as sym

# from joblib import Parallel
# from joblib import delayed
from pathos.multiprocessing import ProcessingPool
from pathos.multiprocessing import cpu_count


def _maybe_cast_args_func(func, dtype, backend):
    """floating point numpy dtypes only"""

    @wraps(func)
    def wrapped(*args, **kwargs):
        maybe_cast_args = []
        for arr in args:
            if (hasattr(arr, 'dtype') and
                    np.issubdtype(arr.dtype, np.floating) and
                    arr.dtype != dtype):
                warnings.warn('Received a floating point value with dtype {} '
                              'and will be cast to {} for compatibility with '
                              'the {} backend.'.format(arr.dtype, dtype,
                                                       backend))
                arr = arr.astype(dtype)
            maybe_cast_args.append(arr)
        return func(*maybe_cast_args, **kwargs)

    return wrapped


def _parallelize_matrix_func(func):
    """Most backends should do this already, but some are stupid."""

    @wraps(func)
    def wrapped(*columns):
        # TODO: note llvm can't be pickled due to ctype pointers
        return np.asarray(func(*row) for row in zip(columns))
        # return np.asarray(ProcessingPool(nodes=cpu_count()).map(
        #     func, zip(columns)
        # ))
        # return np.asarray(Parallel(n_jobs=-1)(
        #     delayed(func)(*row) for row in zip(columns)))

    return wrapped


@lru_cache()
def theano_func(expr, symbols):
    """Theano-compiled function, 2D array as input (1D columns) and float32
    dtype. Good for GPU acceleration."""
    from sympy.printing.theanocode import theano_function

    # on_unused_input='ignore': don't freak out with dummy variables
    func = theano_function(symbols,
                           [expr.expand()],
                           dims={xi: 1 for xi in symbols},
                           dtypes={xi: 'float32' for xi in symbols},
                           on_unused_input='ignore')
    # args need to be float32s if floating
    return _maybe_cast_args_func(func, np.float32, 'theano')


@lru_cache()
def llvm_func(expr, symbols):
    """JIT"""
    from sympy.printing.llvmjitcode import llvm_callable

    expr = expr.expand()
    # llvm cannot handle things such as PI
    for num_sym in expr.atoms(sym.core.numbers.NumberSymbol):
        expr = expr.subs({num_sym: num_sym.evalf()})

    func = llvm_callable(symbols, expr.expand())
    # llvm only works for scalar inputs so far
    func = _parallelize_matrix_func(func)
    # will auto-cast to double so maybe warn user
    return _maybe_cast_args_func(func, np.float64, 'llvm')


@lru_cache()
def numpy_func(expr, symbols):
    return sym.lambdify(symbols, expr.expand(),
                        modules=['scipy', 'numpy'])


@lru_cache()
def numexpr_func(expr, symbols):
    # TODO: numexpr cannot handle Min/Max expressions
    #  detect using `expr.atoms(sym.Max, sym.Min)` and use another backend..
    #  see https://github.com/pydata/numexpr/issues/86
    #  furthermore you can replace with this glorious fix thanks to this man
    #  https://stackoverflow.com/a/60725243/6557588
    return sym.lambdify(symbols, expr.expand(), modules='numexpr')


@lru_cache()
def tensorflow_func(expr, symbols):
    expr = expr.expand()
    int_atoms = expr.atoms(sym.Integer)
    # To avoid TF typing problems, it's best to cast integers to floats
    expr = expr.subs({int_a: float(int_a) for int_a in int_atoms})
    tf_func = sym.lambdify(symbols, expr, modules='tensorflow')

    @wraps(tf_func)
    def wrapped(*args, **kwargs):
        return tf_func(*args, **kwargs).numpy()

    return wrapped


@lru_cache()
def ufuncify_numpy_func(expr, symbols):
    from sympy.utilities.autowrap import ufuncify

    return ufuncify(symbols, expr.expand(), backend='numpy')


_CYTHON_CLI_MATH_MACROS = [
    '-DM_E=2.718281828459045235360287471352662498',
    '-DM_LOG2E=1.442695040888963407359924681001892137',
    '-DM_LN2=0.693147180559945309417232121458176568',
    '-DM_LN10=2.302585092994045684017991454684364208',
    '-DM_PI=3.141592653589793238462643383279502884',
    '-DM_PI_2=1.570796326794896619231321691639751442',
    '-DM_PI_4=0.785398163397448309615660845819875721',
    '-DM_1_PI=0.318309886183790671537767526745028724',
    '-DM_2_PI=0.636619772367581343075535053490057448',
    '-DM_2_SQRTPI=1.128379167095512573896158903121545172',
    '-DM_SQRT2=1.414213562373095048801688724209698079',
    '-DM_SQRT1_2=0.707106781186547524400844362104849039',
]


@lru_cache()
def cython_func(expr, symbols):
    from sympy.utilities.autowrap import ufuncify

    # math.h header can be missing in ufunc, these are the macros SymPy uses
    # so pass as compilation args. f64 values taken from math.h for those
    # listed in this file (at least as of commit 702bcea):
    # https://github.com/sympy/sympy/blob/master/sympy/printing/c.py
    func = ufuncify(symbols, expr.expand(), backend='cython',
                    extra_compile_args=_CYTHON_CLI_MATH_MACROS)  # noqa
    # cython ufuncify expects double_t arrays and hates floats
    return _maybe_cast_args_func(func, np.float64, 'ufuncify cython')


@lru_cache()
def f2py_func(expr, symbols):
    from sympy.utilities.autowrap import ufuncify

    return ufuncify(symbols, expr.expand(), backend='f2py')


def symbolic_evaluate_func(expr, symbols, x=None, backend=None):
    if backend is None:
        backend = 'f2py'  # requires numpy
        if x is not None and len(x) > 1_100:  # empirical benchmark
            try:
                import numexpr
                backend = 'numexpr'
            except ImportError:
                pass
    if backend == 'numpy':
        eval_func = numpy_func
    elif backend == 'theano':
        eval_func = theano_func
    elif backend == 'tensorflow':
        eval_func = tensorflow_func
    elif backend == 'numexpr':
        eval_func = numexpr_func
    elif backend == 'f2py':
        eval_func = f2py_func
    elif backend == 'cython':
        eval_func = cython_func
    elif backend == 'ufuncify_numpy':
        eval_func = ufuncify_numpy_func
    elif backend == 'llvm':
        eval_func = llvm_func
    else:
        raise ValueError(backend)
    return eval_func(expr, symbols)
