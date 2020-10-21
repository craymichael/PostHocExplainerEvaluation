import string
import random
from functools import partial

from collections.abc import Iterable
from collections.abc import Callable
from typing import Sequence
from typing import Tuple
from typing import Union

from functools import cache

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


@cache
def split_effects(expr: sym.Expr,
                  symbols: Sequence[sym.Symbol]) -> (sym.Expr, Tuple[sym.Expr]):
    expr_expanded = expr.expand(add=True)
    all_symbol_set = set(symbols)
    main_effects = sym.Integer(0)
    for xi in symbols:
        all_minus_xi = all_symbol_set - {xi}
        main, _ = expr_expanded.as_independent(*all_minus_xi, as_Add=True)
        main_effects += main
    interaction_effects = set(expr_expanded.args) - set(main_effects.args)
    return main_effects, tuple(interaction_effects)


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

    def get_symbol(self, symbol_name: str) -> sym.Symbol:
        if self._symbol_map is None:
            self._symbol_map = dict(zip(self.symbol_names, self.symbols))
        return self._symbol_map[symbol_name]

    @property
    def main_effects(self):
        main_effects, _ = split_effects(self.expr, self.symbols)
        return main_effects

    @property
    def interaction_effects(self):
        _, interaction_effects = split_effects(self.expr, self.symbols)
        return interaction_effects

    @property
    def independent_terms(self) -> Tuple[sym.Expr]:
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
    all_symbols = sym.symbols('x1:11')  # 10 variables
    # TODO: figure out how many samples the authors used....otherwise sample
    #  within some expected error?
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = all_symbols
    synthetic_functions = dict(
        # TODO: figure out input data ranges from the paper for this....
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
        1 / (1 + x1 ** 2 + x2 ** 2 + x3 ** 2)
        + sym.sqrt(sym.exp(x4 + x5))
        + abs(x6 + x7)
        + x8 * x9 * x10
        ,
        f6=
        sym.exp(abs(x1 * x2) + 1)
        - sym.exp(abs(x3 + x4) + 1)
        + sym.cos(x5 + x6 - x8)
        + sym.sqrt(x8 ** 2 + x9 ** 2 + x10 ** 2)
        ,
        f7=
        (sym.atan(x1) + sym.atan(x2)) ** 2
        + sym.Max(x3 * x4 + x6, 0)
        - 1 / (1 + (x4 * x5 * x6 * x7 * x8) ** 2)
        + (abs(x7) / (1 + abs(x9))) ** 5
        + sum(all_symbols)
        ,  # sum(all_symbols) = \sum_{i=1}^{10} x_i
        f8=
        x1 * x2
        + 2 ** (x3 + x5 + x6)
        + 2 ** (x3 + x4 + x5 + x7)
        + sym.sin(x7 * sym.sin(x8 + x9))
        + sym.acos(sym.Integer(9) / sym.Integer(10) * x10)
        ,
        f9=
        sym.tanh(x1 * x2 + x3 * x4) * sym.sqrt(abs(x5))
        + sym.exp(x5 + x6)
        + sym.log((x6 * x7 * x8) ** 2 + 1)
        + x9 * x10
        + 1 / (1 + abs(x10))
        ,
        f10=
        sym.sinh(x1 + x2)
        + sym.acos(sym.tanh(x3 + x5 + x7))
        + sym.cos(x4 + x5)
        + sym.sec(x7 * x9)
        ,
    )

    if name is None:
        return synthetic_functions
    elif isinstance(name, Iterable):
        return tuple(synthetic_functions[n] for n in name)
    else:
        return synthetic_functions[name]
