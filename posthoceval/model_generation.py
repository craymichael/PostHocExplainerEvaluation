import string
import random
import logging
import warnings
from collections.abc import Callable
from typing import Sequence
from typing import Dict
from typing import Tuple
from typing import Union
from functools import partial
from functools import lru_cache
from itertools import repeat
from itertools import combinations
from itertools import product
from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict

import sympy as sym
import numpy as np

from .utils import assert_shape
from .utils import as_iterator_of_size
from .evaluate import symbolic_evaluate_func

# logger = logging.getLogger(__name__)

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
# Custom typing
Symbol1orMore = Union[sym.Symbol, Sequence[sym.Symbol]]
ContribMapping = Dict[Symbol1orMore, np.ndarray]
ExprMapping = Dict[Symbol1orMore, sym.Expr]

# Single argument ops
OPS_SINGLE_ARG = [
    sym.cos,
    sym.cosh,
    sym.sin,
    sym.sinh,
    sym.tan,
    sym.tanh,
    sym.Abs,
]
# Multiple argument ops (non-additive)
OPS_MULTI_ARG = [
    sym.Mul,
    sym.Pow,
]
# TODO: exponent of integer, mul of integer


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

def independent_terms(expr) -> Tuple[sym.Expr]:
    if isinstance(expr, sym.Add):
        return expr.args
    return expr,  # ',' for tuple return


@lru_cache()
def split_effects(
        expr: sym.Expr,
        symbols: Sequence[sym.Symbol],
) -> Tuple[Tuple[sym.Expr], Tuple[sym.Expr]]:
    """Additive effects"""
    expr_expanded = expr.expand(add=True)
    all_symbol_set = set(symbols)
    main_effects = []
    for xi in symbols:
        all_minus_xi = all_symbol_set - {xi}

        main, _ = expr_expanded.as_independent(*all_minus_xi, as_Add=True)
        main: sym.Expr  # type hint

        # Single main effect per symbol
        main_effects.append(main)

    interaction_effects = (set(independent_terms(expr_expanded)) -
                           set(main_effects))
    return tuple(main_effects), tuple(interaction_effects)


def symbol_names(n_features, excel_like=False):
    """Generate Excel-like names for symbols"""
    assert n_features >= 1, 'Invalid number of features < 1: %d' % n_features
    if excel_like:
        alphabet = string.ascii_uppercase
        ret = []
        for d in range(1, n_features + 1):
            ret_i = ''
            while d > 0:
                d, m = divmod(d - 1, 26)
                ret_i = alphabet[m] + ret_i
            ret.append(ret_i)
        return ret
    else:
        return [f'x{i}' for i in range(1, n_features + 1)]


class AdditiveModel(object):
    def __init__(self,
                 n_features: int,
                 coefficients=partial(random.uniform, -1, +1),
                 interactions=None,
                 domain: Union[str, Sequence[str]] = 'real',
                 backend=None):
        """

        :param n_features:
        :param coefficients: first coef is the bias term
        :param domain:
            negative nonnegative commutative imaginary nonzero real finite
            extended_real nonpositive extended_negative extended_nonzero
            hermitian positive extended_nonnegative zero prime infinite
            extended_nonpositive extended_positive complex composite
            See [sympy assumptions](https://docs.sympy.org/latest/modules/core.html#module-sympy.core.assumptions)  # noqa
        """
        self.n_features = n_features

        kwargs = dict()
        if isinstance(domain, str):
            kwargs[domain] = True
        else:  # assume iterable of str
            for d in domain:
                kwargs[d] = True

        # Generate n_features symbols with unique names from domain
        self.symbol_names = symbol_names(self.n_features)
        self.symbols = sym.symbols(self.symbol_names, **kwargs)
        self.expr = self._generate_model(coefficients)  # TODO interactions
        self._symbol_map = None

        self.backend = backend

        self._check()

    def _check(self):
        if not isinstance(self.expr, sym.Add):
            warnings.warn('Expression is not additive! Output is dependent on '
                          'all input variables: '
                          'optype {}'.format(type(self.expr)))

    @classmethod
    def from_expr(
            cls,
            expr: sym.Expr,
            symbols: Symbol1orMore,
            backend=None,
    ):
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
        model.backend = backend

        model._check()
        return model

    def get_symbol(self, symbol_name: str) -> sym.Symbol:
        if self._symbol_map is None:
            self._symbol_map = dict(zip(self.symbol_names, self.symbols))
        return self._symbol_map[symbol_name]

    @property
    def main_effects(self) -> Tuple[sym.Expr]:
        main_effects, _ = split_effects(self.expr, self.symbols)
        return main_effects

    @property
    def interaction_effects(self) -> Tuple[sym.Expr]:
        _, interaction_effects = split_effects(self.expr, self.symbols)
        return interaction_effects

    @property
    def independent_terms(self) -> Tuple[sym.Expr]:
        return independent_terms(self.expr)

    def _generate_model(self, coefficients) -> sym.Expr:
        n_coefs_expect = self.n_features + 1
        coefs = as_iterator_of_size(
            coefficients, n_coefs_expect, 'coefficients')
        bias = next(coefs)
        total = bias() if isinstance(bias, Callable) else bias
        for xi, ci in zip(self.symbols, coefs):
            total += xi * (ci() if isinstance(ci, Callable) else ci)
        return total

    @property
    def valid_variable_domains(self):
        """Real domains only! TODO: allow other domains?"""
        # TODO: cache this function like the other one
        domains = {}
        terms = self.independent_terms
        for term in terms:
            errored_symbols = []
            for xi in term.free_symbols:
                try:
                    domain = sym.calculus.util.continuous_domain(term, xi,
                                                                 sym.Reals)
                    domains[xi] = domains.get(xi, sym.Reals).intersect(domain)
                except NotImplementedError:
                    errored_symbols.append(xi)
            assumptions = (None, 'positive', 'negative')
            great_success = False
            # TODO: move this shit out into another function
            # Time to figure out if assumptions help automatically figure out
            # valid continuous ranges...
            for i in range(1, len(errored_symbols) + 1):
                for sym_subset in combinations(errored_symbols, i):
                    # print('sym_subset', sym_subset)
                    for sym_comb in product(
                            *(tuple(zip(repeat(xi_ss), assumptions)) for xi_ss
                              in sym_subset)):
                        try:
                            # print('sym_comb', sym_comb)
                            replacements = OrderedDict(
                                (xi_ss, (xi_ss if ass is None else sym.Symbol(
                                    xi_ss.name, **{ass: True})))
                                for xi_ss, ass in sym_comb
                            )
                            # TODO: consider existing domain of symbols...
                            intervals = (
                                sym.Interval(0, sym.oo)
                                if ass == 'positive' else
                                sym.Interval(-sym.oo, 0)
                                if ass == 'negative' else sym.Reals
                                for _, ass in sym_comb)
                            term_subs = term.subs(replacements)
                            maybe_domains = {}
                            for xi_ss, interval in zip(replacements.values(),
                                                       intervals):
                                domain = sym.calculus.util.continuous_domain(
                                    term_subs, xi_ss, interval)
                                maybe_domains[xi_ss] = domain
                            # print('maybe_domains!!!!', maybe_domains)
                            great_success = True
                            for xi_ss in errored_symbols:
                                xi_ss_new = replacements.get(xi_ss, xi_ss)
                                # TODO: replace xi_ss in expression and model!!!!! This only replaces domain........
                                # TODO: or just give the right domain for the original xi...
                                domains[xi_ss] = domains.get(xi_ss,
                                                             sym.Reals).intersect(
                                    maybe_domains.get(xi_ss_new, sym.Reals))
                            break
                        except NotImplementedError as e:
                            pass  # shit
                    if great_success:
                        break
                if great_success:
                    break
            else:
                if errored_symbols:
                    # TODO raise this shit
                    print('Failure!!!', great_success, term, errored_symbols)
        return domains

    def __call__(
            self,
            x: np.ndarray,
            backend=None,
    ):
        assert_shape(x, (None, self.n_features))
        if backend is None:
            backend = self.backend
        eval_func = symbolic_evaluate_func(self.expr, self.symbols,
                                           x=x, backend=backend)
        return eval_func(*(x[:, i] for i in range(self.n_features)))

    def feature_contributions(
            self,
            x: np.ndarray,
            main_effects=True,
            interaction_effects=True,
            return_effects=False,
            backend=None,
    ) -> Union[ContribMapping, Tuple[ContribMapping, ExprMapping]]:
        """"""
        if backend is None:
            backend = self.backend

        if not (main_effects or interaction_effects):
            raise ValueError('Must specify either main_effects or '
                             'interaction_effects')
        effects = []
        if main_effects:
            effects.extend(list(self.main_effects))
        if interaction_effects:
            effects.extend(list(self.interaction_effects))

        contributions = defaultdict(lambda: np.zeros(len(x)))
        all_effects = defaultdict(lambda: sym.Number(0))
        for effect in effects:
            effect_symbols = sorted(effect.free_symbols, key=lambda s: s.name)
            effect_symbols = tuple(effect_symbols)
            # Index x matrix (order of features)
            related_features = [x[:, self.symbols.index(s)]
                                for s in effect_symbols]
            if effect == 0:
                continue  # skip zero-effects
            eval_func = symbolic_evaluate_func(effect,
                                               effect_symbols,
                                               x=x,
                                               backend=backend)
            contribution = eval_func(*related_features)
            if len(effect_symbols) == 1:
                effect_symbols = effect_symbols[0]
            contributions[effect_symbols] = contribution
            if return_effects:
                all_effects[effect_symbols] = effect

        if return_effects:
            return contributions, all_effects
        return contributions

    def pprint(self):
        sym.pprint(self.expr)

    def __repr__(self):
        return str(self.expr)


class LinearModel(AdditiveModel):
    def __init__(self, *args, **kwargs):
        super(LinearModel, self).__init__(*args, interactions=None, **kwargs)


def tsang_iclr18_models(
        name=None
) -> Union[Dict[str, AdditiveModel], Tuple[AdditiveModel], AdditiveModel]:
    """"""
    # TODO: https://github.com/sympy/sympy/issues/11027
    #  Min/Max functions don't vectorize as expected. Here is a "beautiful" fix
    #  https://stackoverflow.com/a/60725243/6557588
    all_symbols = sym.symbols('x1:11')  # 10 variables
    # TODO:
    #  - 30k data points, 1/3 each train/valid/test
    #  - tan et al. use 50k points and a modified equation
    #  - uniformly distributed between [-1,+1] for f2-f10
    #  - Lou et al. f1: x4, x5, x8, x10 are uniformly distributed in [0.6, 1]
    #    and the other variables are uniformly distributed in [0, 1].
    #    +2 corr. datasets: ρ(x1, x6) = 0.5 and ρ(x1, x6) = 0.95
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = all_symbols

    # f7 equation intermediate component
    f7_int = x3 * x4 + x6

    synthetic_functions = dict(
        f1=AdditiveModel.from_expr(
            sym.pi ** (x1 * x2) * sym.sqrt(2 * x3)
            - sym.asin(x4)
            + sym.log(x3 + x5)
            - x9 / x10 * sym.sqrt(x7 / x8)
            - x2 * x7,
            all_symbols
        ),
        f2=AdditiveModel.from_expr(
            sym.pi ** (x1 * x2) * sym.sqrt(2 * abs(x3))
            - sym.asin(x4 / 2)
            + sym.log(abs(x3 + x5) + 1)
            + x9 / (1 + abs(x10)) * sym.sqrt(x7 / (1 + abs(x8)))
            - x2 * x7,
            all_symbols
        ),
        f3=AdditiveModel.from_expr(
            sym.exp(abs(x1 - x2))
            + abs(x2 * x3)
            - x3 ** (2 * abs(x4))
            + sym.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
            + x9
            + 1 / (1 + x10 ** 2),
            all_symbols
        ),
        f4=AdditiveModel.from_expr(
            sym.exp(abs(x1 - x2))
            + abs(x2 * x3)
            - x3 ** (2 * abs(x4))
            + (x1 * x4) ** 2
            + sym.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
            + x9
            + 1 / (1 + x10 ** 2),
            all_symbols
        ),
        f5=AdditiveModel.from_expr(
            1 / (1 + x1 ** 2 + x2 ** 2 + x3 ** 2)
            + sym.sqrt(sym.exp(x4 + x5))
            + abs(x6 + x7)
            + x8 * x9 * x10,
            all_symbols
        ),
        f6=AdditiveModel.from_expr(
            sym.exp(abs(x1 * x2) + 1)
            - sym.exp(abs(x3 + x4) + 1)
            + sym.cos(x5 + x6 - x8)
            + sym.sqrt(x8 ** 2 + x9 ** 2 + x10 ** 2),
            all_symbols
        ),
        f7=AdditiveModel.from_expr(
            (sym.atan(x1) + sym.atan(x2)) ** 2
            # Ha! You think you could do this but nooooo
            # + sym.Max(x3 * x4 + x6, 0)
            # Have to do this $#!% instead to get it vectorized. Also note that
            # the 0. has to be a float for certain backends to not complain.
            + sym.Piecewise((f7_int, f7_int > 0), (0., True))
            - 1 / (1 + (x4 * x5 * x6 * x7 * x8) ** 2)
            + (abs(x7) / (1 + abs(x9))) ** 5
            + sum(all_symbols),
            all_symbols
        ),  # sum(all_symbols) = \sum_{i=1}^{10} x_i
        f8=AdditiveModel.from_expr(
            x1 * x2
            + 2 ** (x3 + x5 + x6)
            + 2 ** (x3 + x4 + x5 + x7)
            + sym.sin(x7 * sym.sin(x8 + x9))
            + sym.acos(sym.Integer(9) / sym.Integer(10) * x10),
            all_symbols
        ),
        f9=AdditiveModel.from_expr(
            sym.tanh(x1 * x2 + x3 * x4) * sym.sqrt(abs(x5))
            + sym.exp(x5 + x6)
            + sym.log((x6 * x7 * x8) ** 2 + 1)
            + x9 * x10
            + 1 / (1 + abs(x10)),
            all_symbols
        ),
        f10=AdditiveModel.from_expr(
            sym.sinh(x1 + x2)
            + sym.acos(sym.tanh(x3 + x5 + x7))
            + sym.cos(x4 + x5)
            + 1 / sym.cos(x7 * x9),  # sec=1/cos (some backends don't have sec)
            all_symbols
        ),
    )

    if name is None:
        return synthetic_functions
    elif not isinstance(name, str):  # assume iterable
        return tuple(synthetic_functions[n] for n in name)
    else:
        return synthetic_functions[name]

# TODO: others:
#   F1(x) = 3 * x1 + x2 ** 3 - sym.pi ** x3 + sym.exp(-2 * x4 ** 2)
#    + 1 / (2 + abs(x5)) + x6 * sym.log(abs(x6)) + sym.sqrt(2 * abs(x7))
#    + sym.Max(0, x7) + x8 ** 4 + 2 * sym.cos(sym.pi * x8)
#   F2(x) = F1(x) + x1 * x2 + abs(x3) ** (2 * abs(x4)) + sym.sec(x3 * x5 * x6)
