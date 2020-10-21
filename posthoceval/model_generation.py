import string
import random
from functools import partial

from collections.abc import Iterable
from collections.abc import Callable
from typing import Sequence
from typing import Optional
from typing import Union

from itertools import chain

import logging

from collections import namedtuple

import sympy as sym

from joblib import cpu_count
from joblib import Parallel
from joblib import delayed

from .utils import as_sized
from .utils import assert_same_size
from .utils import as_iterator_of_size

logger = logging.getLogger(__name__)

InteractionOp = namedtuple(
    'Interaction',
    ('name', 'n_args', 'func')
)
# For determination of continuity of functions:
#  sympy.calculus.util.continuous_domain(...)
INTERACTION_OPS = (
    InteractionOp(
        name='cos',
        n_args=1,
        func=sym.cos
    ),
    InteractionOp(
        name='cosh',
        n_args=1,
        func=sym.cosh
    ),
    InteractionOp(
        name='sin',
        n_args=1,
        func=sym.sin
    ),
    InteractionOp(
        name='sinh',
        n_args=1,
        func=sym.sinh
    ),
    InteractionOp(
        name='tan',
        n_args=1,
        func=sym.tan
    ),
    InteractionOp(
        name='tanh',
        n_args=1,
        func=sym.tanh
    ),
    InteractionOp(
        name='Abs',
        n_args=1,
        func=sym.Abs
    ),
    InteractionOp(
        name='Mul',
        n_args=2,
        func=sym.Mul
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
    InteractionOp(
        name='',
        n_args=2,
        func=sym
    ),
)


# TODO:
#  - random interaction triangular matrices, one per interaction
#  - matrix of ints, sum of elements equal to number of interaction terms
#  - int is order of interaction
#  - M_ij (i=j) --> interacts with self, e.g. x**2 (don't allow division)
#  - generate models intelligently - don't just alternate which variables
#    interact e.g. if input distributions are the same
#  - ensure interactions unique, no duplicate terms
#  - allow for integer interactions (scaling, exponent, etc.)
#  - do by classes of interaction, START WITH POLYNOMIALS


def symbol_names(n_features):
    """Generate Excel-like names for symbols"""
    assert n_features >= 1, 'Invalid number of features < 1: %d' % n_features
    alphabet = string.ascii_uppercase
    ret = []
    for d in range(1, n_features + 1):
        ret_i = ''
        while d > 0:
            d, m = divmod(d - 1, 26)
            ret_i = alphabet[m] + ret_i
        ret.append(ret_i)
    return ret


class AdditiveModel(object):
    def __init__(self,
                 n_features: int,
                 coefficients=partial(random.uniform, -1, +1),
                 interactions=None,
                 domain='real'):
        """

        :param n_features:
        :param coefficients: first coef is the bias term
        :param domain:
        """
        self.n_features = n_features

        kwargs = dict()
        if domain == 'real':
            kwargs['real'] = True
        elif domain == 'integer':
            kwargs['integer'] = True
        else:
            raise ValueError('Unknown variable domain: %s' % domain)

        # Generate n_features symbols with unique names from domain
        self.symbol_names = symbol_names(self.n_features)
        self.symbols = sym.symbols(self.symbol_names, **kwargs)
        self.expr = self._generate_model(coefficients)
        self._symbol_map = None

        self._check()

    def _check(self):
        if not isinstance(self.expr, sym.Add):
            logger.warning('Expression is not additive! Output is dependent on '
                           'all input variables: '
                           'optype {}'.format(type(self.expr)))

    @classmethod
    def from_expr(cls,
                  expr: sym.Expr,
                  symbols: Union[Sequence[sym.Symbol], sym.Symbol]):
        """Symbols needs to be ordered properly"""
        # Ensure expr symbols are a subset of symbols
        symbols = (symbols,) if isinstance(symbols, sym.Symbol) else symbols
        missing_symbols = set(expr.free_symbols) - set(symbols)
        if missing_symbols:
            raise ValueError('expr contains symbols not specified in symbols: '
                             '{}'.format(missing_symbols))

        model = cls.__new__(cls)
        model.expr = expr
        model.symbols = symbols
        model.symbol_names = tuple(s.name for s in symbols)
        model.n_features = len(symbols)
        model._symbol_map = None

        model._check()

    def get_symbol(self, symbol_name):
        if self._symbol_map is None:
            self._symbol_map = dict(zip(self.symbol_names, self.symbols))
        return self._symbol_map[symbol_name]

    @property
    def independent_terms(self):
        if isinstance(self.expr, sym.Add):
            return self.expr.args
        return self.expr,  # ',' for tuple return

    def _generate_model(self, coefficients) -> sym.Expr:
        n_coefs_expect = self.n_features + 1
        coefs = as_iterator_of_size(
            coefficients, n_coefs_expect, 'coefficients')
        bias = next(coefs)
        total = bias() if isinstance(bias, Callable) else bias
        for xi, ci in zip(self.symbols, coefs):
            total += xi * (ci() if isinstance(ci, Callable) else ci)
        return total

    def __call__(self, *x, n_jobs=cpu_count()):
        if len(x) == 1:
            x = x[0]
        if isinstance(x, Iterable):
            x = as_sized(x)
            if isinstance(x[0], Iterable):
                assert_func = partial(assert_same_size,
                                      expected=self.n_features,
                                      units='features')
                return Parallel(n_jobs=n_jobs)(
                    delayed(self.expr.subs)(
                        zip(self.symbols,
                            assert_func(received=len(xi), ret=xi))
                    ) for xi in x
                )
            else:
                assert_same_size(self.n_features, len(x), 'features')
                return self.expr.subs(zip(self.symbols, x))
        else:
            assert_same_size(self.n_features, 1, 'features')
            return self.expr.subs((self.symbols[0], x))

    def feature_attribution(self):
        # for arg in sym.preorder_traversal(self.expr):
        #     pass
        self.expr.as_independent(self.symbols, as_Add=True)

    def __repr__(self):
        return str(self.expr)


class LinearModel(AdditiveModel):
    def __init__(self, *args, **kwargs):
        super(LinearModel, self).__init__(*args, interactions=None, **kwargs)


def tsang_iclr18_models(name=None):
    all_symbols = sym.symbols('x1:10')
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = all_symbols
    synthetic_functions = dict(
        f1=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f2=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * abs(x3))
        - sym.asin(x4 / 2)
        + sym.log(abs(x3 + x5) + 1)
        + x9 / (1 + abs(x10)) * sym.sqrt(x7 / (1 + abs(x8)))
        - x2 * x7
        ,
        f3=
        sym.exp(abs(x1 - x2))
        + abs(x2 * x3)
        - x3 ** (2 * abs(x4))
        + sym.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
        + x9
        + 1 / (1 + x10 ** 2)
        ,
        f4=
        sym.exp(abs(x1 - x2))
        + abs(x2 * x3)
        - x3 ** (2 * abs(x4))
        + (x1 * x4) ** 2
        + sym.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
        + x9
        + 1 / (1 + x10 ** 2)
        ,
        f5=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f6=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f7=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f8=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f9=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
        f10=
        sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
        - sym.asin(x4)
        + sym.log(x3 + x5)
        - x9 / x10 * sym.sqrt(x7 / x8)
        - x2 * x7
        ,
    )

    if name is None:
        return synthetic_functions
    elif isinstance(name, Iterable):
        return tuple(synthetic_functions[n] for n in name)
    else:
        return synthetic_functions[name]
