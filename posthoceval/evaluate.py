import warnings

from functools import lru_cache
from functools import wraps

import numpy as np
import sympy as sp


def _maybe_cast_args_func(func, dtype, backend):
    """floating point numpy dtypes only"""

    @wraps(func)
    def wrapper(*args, **kwargs):
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

    return wrapper


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


def _parallelize_matrix_func(func):
    """Most backends should do this already, but some are stupid."""

    @wraps(func)
    def wrapper(*columns):
        # TODO: note llvm can't be pickled due to ctype pointers
        # TODO: llvm backend keeps segmentation faulting, ugh
        return np.fromiter((func(*row) for row in zip(*columns)),
                           dtype=np.float32)
        # return np.asarray(ProcessingPool(nodes=cpu_count()).map(
        #     func, zip(columns)
        # ))
        # return np.asarray(Parallel(n_jobs=-1)(
        #     delayed(func)(*row) for row in zip(columns)))

    return wrapper


@lru_cache()
def llvm_func(expr, symbols):
    """JIT"""
    from sympy.printing.llvmjitcode import llvm_callable

    expr = expr.expand()
    # llvm cannot handle things such as PI
    for num_sym in expr.atoms(sp.core.numbers.NumberSymbol):
        expr = expr.subs({num_sym: num_sym.evalf()})

    func = llvm_callable(symbols, expr.expand())
    # llvm only works for scalar inputs so far
    func = _parallelize_matrix_func(func)
    # def func_wrapper(*args):
    #     func = llvm_callable(symbols, expr.expand())
    #     return func(*args)
    #
    # def parallelize(*columns):
    #     return np.asarray(Parallel(n_jobs=-1)(
    #         delayed(func_wrapper)(*row) for row in zip(*columns)))

    # will auto-cast to double so maybe warn user
    return _maybe_cast_args_func(func, np.float64, 'llvm')


@lru_cache()
def numpy_func(expr, symbols):
    return sp.lambdify(symbols, expr.expand(),
                       modules=['numpy', 'scipy'])


@lru_cache()
def numexpr_func(expr, symbols):
    # TODO: numexpr cannot handle Min/Max expressions
    #  detect using `expr.atoms(sym.Max, sym.Min)` and use another backend..
    #  see https://github.com/pydata/numexpr/issues/86
    #  furthermore you can replace with this glorious fix thanks to this man
    #  https://stackoverflow.com/a/60725243/6557588
    return sp.lambdify(symbols, expr.expand(), modules='numexpr')


@lru_cache()
def tensorflow_func(expr, symbols, symbolic=False):
    expr = expr.expand()
    int_atoms = expr.atoms(sp.Integer)
    # To avoid TF typing problems, it's best to cast integers to floats
    # while-try-except loop necessary per issue:
    #  https://github.com/sympy/sympy/issues/21373
    n_retries = 0
    while n_retries < 50:
        try:
            expr = expr.subs({int_a: float(int_a) for int_a in int_atoms})
        except sp.PolynomialError:
            n_retries += 1
        else:
            break
    else:
        raise RuntimeError('Too many `PolynomialError`s while trying to cast '
                           f'integer atoms for tensorflow_func! Expr:\n{expr}')
    tf_func = sp.lambdify(symbols, expr, modules='tensorflow')

    if symbolic:
        return tf_func

    @wraps(tf_func)
    def wrapper(*args, **kwargs):
        ret = tf_func(*args, **kwargs)
        if hasattr(ret, 'numpy'):  # naive but faster tf.EagerTensor to ndarray
            ret = ret.numpy()
        return ret

    return wrapper


@lru_cache()
def ufuncify_numpy_func(expr, symbols):
    from sympy.utilities.autowrap import ufuncify
    if len(symbols) >= 31:
        # ufuncify (f2py) can only handle 32 total inputs and
        # outputs
        warnings.warn('ufuncify+numpy is gonna crash (can only support 32 '
                      'total inputs and outputs)')

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


# known unsupported functions on most backends
UNSUPPORTED_FUNC_REPLACEMENTS = {
    # cot
    sp.cot: lambda x: 1 / sp.tan(x),
    sp.coth: lambda x: (sp.cosh(x)) / (sp.sinh(x)),
    sp.acot: lambda x: sp.Piecewise(
        ((sp.pi / 2) - sp.atan(x), x >= 0),
        (-sp.atan(x) - (sp.pi / 2), True)
    ),
    sp.acoth: lambda x: (sp.log((x + 1) / (x - 1))) / 2,
    # csc
    sp.csc: lambda x: 1 / sp.sin(x),
    sp.csch: lambda x: 1 / (sp.sinh(x)),
    sp.acsc: lambda x: sp.asin(1 / x),
    sp.acsch: lambda x: sp.log((1 / x) + sp.sqrt((1 / (x ** 2)) + 1)),
    # sec
    sp.sec: lambda x: 1 / sp.cos(x),
    sp.sech: lambda x: 1 / (sp.cosh(x)),
    sp.asec: lambda x: sp.acos(1 / x),
    sp.asech: lambda x: sp.log((1 + sp.sqrt(1 - (x ** 2))) / x),
    # sinc
    sp.sinc: lambda x: sp.Piecewise(
        ((sp.sin(x)) / x, x != 0),
        (1, True)
    ),
}


def replace_unsupported_functions(expr):
    for k, v in UNSUPPORTED_FUNC_REPLACEMENTS.items():
        expr = expr.replace(k, v)
    return expr


def symbolic_evaluate_func(expr, symbols, x=None, backend=None, **kwargs):
    if backend is None:
        # if len(symbols) >= 32:
        if len(symbols) >= np.MAXDIMS:
            # See https://github.com/numpy/numpy/issues/4398
            # NPY_MAXARGS is set to 32, functions can not be lambdified using
            # numexpr
            backend = 'numpy'
        else:
            try:
                import numexpr
                backend = 'numexpr'
            except ImportError:
                backend = 'f2py'
        # TODO: set default backend smarter...
        # backend = 'f2py'  # requires numpy
        # if x is not None and len(x) > 1_100:  # empirical benchmark
        #     try:
        #         import numexpr
        #         backend = 'numexpr'
        #     except ImportError:
        #         pass
    if backend == 'numpy':  # or scipy...
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
        warnings.warn('Yeah, llvm is probably still broken. Don\'t be '
                      'surprised if you get a segfault...')
        eval_func = llvm_func
    else:
        raise ValueError(backend)

    # substitute known unsupported functions on most backends
    expr = replace_unsupported_functions(expr)

    return eval_func(expr, tuple(symbols), **kwargs)
