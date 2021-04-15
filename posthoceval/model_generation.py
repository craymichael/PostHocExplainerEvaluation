import string
import random
# import logging
import warnings
from collections.abc import Callable
from typing import Sequence
from typing import Optional
from typing import Dict
from typing import Tuple
from typing import Union
from functools import partial
from functools import lru_cache
from itertools import repeat
from itertools import combinations
from itertools import product
from itertools import cycle
from collections import OrderedDict
from functools import wraps

import multiprocessing as mp
from multiprocessing import TimeoutError

import sympy as sp
from sympy.calculus.util import continuous_domain

import numpy as np
from scipy.special import comb

from posthoceval.rand import select_n_combinations
from posthoceval.rand import as_random_state
from posthoceval.rand import choice_objects
from posthoceval.utils import assert_shape
from posthoceval.utils import as_iterator_of_size
from posthoceval.utils import assert_same_size
from posthoceval.utils import is_int
from posthoceval.utils import is_float
from posthoceval.evaluate import symbolic_evaluate_func
from posthoceval.expression_tree import RandExprTree

from posthoceval.profile import profile
from posthoceval.profile import mem_profile

# logger = logging.getLogger(__name__)

# Custom typing
Symbol1orMore = Union[sp.Symbol, Sequence[sp.Symbol]]
ContribMapping = Dict[Symbol1orMore, np.ndarray]
ExprMapping = Dict[Symbol1orMore, sp.Expr]

# Single argument ops
OPS_TRIG = [
    # cos
    sp.cos,
    sp.cosh,
    # sp.acos,  # nan outside [-1, +1], remove for now...
    # sp.acosh,  # remove for now...
    # sin
    sp.sin,
    sp.sinh,
    # sp.asin,  # nan outside [-1, +1], remove for now...
    sp.asinh,
    # tan
    sp.tan,
    sp.tanh,
    sp.atan,
    # sp.atanh,  # infinite at +-1, remove for now...
    # cot
    sp.cot,
    # sp.coth,  # zoo at 0
    sp.acot,
    # sp.acoth,  # imaginary in [-1, +1], remove for now...
    # csc
    sp.csc,
    # sp.csch,  # zoo at 0
    # sp.acsc,  # imaginary in [-1, +1], remove for now...
    # sp.acsch,  # zoo at 0
    # sec
    # sp.sec,  # zoo at 0
    sp.sech,
    # sp.asec,  # imaginary in [-1, +1], remove for now...
    # sp.asech,  # imaginary in [-1, +1], remove for now...
    # special
    sp.sinc,
]
OPS_SINGLE_ARG = OPS_TRIG + [
    sp.Abs,
    sp.sqrt,
    # numpy does not have good support for powers of negatives...
    # sp.cbrt,  # disable for now, continuous_domain gives Reals
    lambda a: sp.Pow(a, 2),
    lambda a: sp.Pow(a, 3),
    sp.exp,
    sp.log,
]
# Weights in model generation
# `_trig_pct` trig
_trig_pct = .2
_len_non_trig = len(OPS_SINGLE_ARG) - len(OPS_TRIG)
OPS_SINGLE_ARG_WEIGHTS = (
        [_trig_pct / len(OPS_TRIG)] * len(OPS_TRIG) +
        [(1. - _trig_pct) / _len_non_trig] * _len_non_trig
)
# Multiple argument ops (non-additive)
OPS_MULTI_ARG_LINEAR = [
    sp.Mul,
    lambda a, b: a / b,
]
OPS_MULTI_ARG_LINEAR_WEIGHTS = [
    0.8,
    0.2,
]
OPS_MULTI_ARG_NONLINEAR = [
    # sp.Pow,  # neg. to power of val<1 --> complex (disable for now)
    # Max (which can be vectorized)
    lambda a, b: sp.Piecewise((a, a > b), (b, True)),
    # Min (which can be vectorized)
    lambda a, b: sp.Piecewise((a, a < b), (b, True)),
]
OPS_MULTI_ARG_NONLINEAR_WEIGHTS = [
    # old:
    # 0.5,
    # 0.25,
    # 0.25,
    # new:
    0.5,
    0.5,
]

# Ordered from least to most constraining on domain
ASSUMPTION_DOMAIN = OrderedDict((
    ('nonzero', sp.Union(sp.Interval.open(-sp.oo, 0),
                         sp.Interval.open(0, +sp.oo))),
    ('nonpositive', sp.Interval(-sp.oo, 0)),
    ('nonnegative', sp.Interval(0, +sp.oo)),
    ('positive', sp.Interval.open(0, +sp.oo)),
    ('negative', sp.Interval.open(-sp.oo, 0)),
))


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

def place_into_bins(n_bins, n_items, shift=0, skew=0):
    """"""
    assert -1. <= shift <= 1.

    sigmoid_offset = 0.5 * (1 + shift)
    item_proportion = 1. / (1. + np.exp(
        -skew * (np.linspace(0, 1, n_bins) - sigmoid_offset)))
    item_proportion /= item_proportion.sum()  # normalize to probabilities
    residual = 0
    n_items_per_bin = []
    items_remain = n_items
    for proportion in item_proportion:
        # carry over residual items to next bin
        n_items_bin_f = n_items * proportion + residual
        n_items_bin = max(int(round(n_items_bin_f)), min(1, items_remain))
        items_remain -= n_items_bin
        n_items_per_bin.append(n_items_bin)
        residual = n_items_bin_f - n_items_bin
    return n_items_per_bin


@profile
@mem_profile
def generate_additive_expression(
        symbols,
        n_main=None,
        n_uniq_main=None,
        n_interaction=0,
        n_uniq_interaction=None,
        interaction_ord=None,  # TODO: better default? (see code)
        n_dummy=0,
        pct_nonlinear=None,
        nonlinear_multiplier=None,
        nonlinear_shift=0,
        nonlinear_skew=0,
        nonlinear_interaction_additivity=.5,
        nonlinear_single_multi_ratio: Union[str, float] = 'balanced',
        nonlinear_single_arg_ops=None,
        nonlinear_single_arg_ops_weights=None,
        nonlinear_multi_arg_ops=None,
        nonlinear_multi_arg_ops_weights=None,
        linear_multi_arg_ops=None,
        linear_multi_arg_ops_weights=None,
        validate=False,
        validate_kwargs=None,
        validate_tries=50,
        seed=None,
) -> sp.Expr:
    """

    :param symbols:
    :param n_main: if None use a heuristic. Main effect terms
    :param n_uniq_main: number of unique main effects
    :param n_interaction: Interaction effect terms
    :param n_uniq_interaction: number of unique interactions
    :param n_dummy:
    :param pct_nonlinear: Note that this includes both linear terms as well as
        interaction terms without nonlinearities. For example, 
        $x_1 \\times x_2$ would be considered to be linear with respect to a 
        single variable (holding the other constant), whereas $sin(x_1 + x_2)$ 
        would be nonlinear.
    :param seed: For reproducibility
    :return:
    """
    validate_kwargs = validate_kwargs or {}
    assert validate_tries >= 1

    n_features = len(symbols)

    if pct_nonlinear is None:
        # default pct_nonlinear is based on nonlinear_multiplier. also the
        # logic for when multiplier < 1
        if nonlinear_multiplier is None:
            nonlinear_multiplier = 0.5  # default, case 1

        if nonlinear_multiplier >= 1:
            pct_nonlinear = 1
        else:  # nonlinear_multiplier < 1
            assert nonlinear_multiplier >= 0

            pct_nonlinear = nonlinear_multiplier
            nonlinear_multiplier = 1
    else:
        assert 0 <= pct_nonlinear <= 1
        if nonlinear_multiplier is None:  # default, case 2
            nonlinear_multiplier = 1
        else:
            assert nonlinear_multiplier >= 1, (
                'cannot have nonlinear_multiplier < 1 and pct_nonlinear '
                'specified')

    assert 0 <= nonlinear_interaction_additivity <= 1

    if isinstance(nonlinear_single_multi_ratio, str):
        nonlinear_single_multi_ratio = nonlinear_single_multi_ratio.lower()
        assert nonlinear_single_multi_ratio == 'balanced'
    else:
        assert 0 <= nonlinear_single_multi_ratio <= 1

    if is_float(n_dummy):
        assert 0. <= n_dummy < 1.
        n_dummy = min(int(round(n_dummy * n_features)), n_features - 1)
    assert n_dummy < n_features, 'Must satisfy n_dummy < n_features'

    # main effects
    max_possible_main_uniq = n_features - n_dummy
    if n_main is None:
        n_main = max_possible_main_uniq
    elif is_float(n_main):
        n_main = int(round(n_main * n_features))

    # TODO(doc) reason for both n_uniq_main and n_dummy is some can be linear
    #  and some can just be nonlinear...things like that...
    if n_uniq_main is None:
        n_uniq_main = max_possible_main_uniq
    else:
        if is_float(n_uniq_main):
            assert 0 <= n_uniq_main <= 1
            n_uniq_main = int(round(n_uniq_main * n_features))
        assert n_uniq_main <= max_possible_main_uniq, (
            'Must satisfy n_uniq_main <= n_features - n_dummy')

    if interaction_ord is None:
        interaction_ord = (2,)
    elif is_int(interaction_ord):
        interaction_ord = (interaction_ord,)
    # Filter out
    io_orig = interaction_ord
    interaction_ord = tuple(io for io in interaction_ord if io <= n_uniq_main)
    if io_orig != interaction_ord:
        print(f'Warning: provided interaction_ord {io_orig} contains '
              f'interaction orders too large for n_uniq_main ({n_uniq_main}). '
              f'Using these orders instead: {interaction_ord}')

    # Compute the possible number of unique interactions that are possible
    possible_int_ords = {
        order: comb(n_uniq_main, order, exact=True)
        for order in interaction_ord
    }
    max_possible_int_uniq = sum(possible_int_ords.values())

    if is_float(n_interaction):
        n_interaction = int(round(n_interaction * n_features))

    if n_uniq_interaction is None:  # heuristic default
        n_uniq_interaction = min(n_interaction, n_uniq_main,
                                 max_possible_int_uniq)
    else:  # validate
        if is_float(n_uniq_interaction):
            n_uniq_interaction = int(round(
                n_uniq_interaction * max_possible_int_uniq))
        assert (min(1, n_interaction) <= n_uniq_interaction <=
                max_possible_int_uniq)

    if n_uniq_interaction == 0:
        print(f'Warning: n_interaction={n_interaction} but zero interactions '
              f'are possible. Not including interactions for this expression.')
        n_interaction = 0

    # Seed uses beyond this point
    rs = as_random_state(seed)

    # Select the features
    features = choice_objects(symbols, n_uniq_main, replace=False, seed=rs)

    # Select the interactions (balanced between different orders)
    # TODO: this but also with skew+offset args?
    uniq_interactions = []
    for i, order in enumerate(interaction_ord):
        n_int_ord = min(
            possible_int_ords[order],
            int(round((n_uniq_interaction - len(uniq_interactions)) /
                      (len(interaction_ord) - i)))
        )
        # unique interactions between `order` features
        uniq_interactions.extend(select_n_combinations(
            features, k=order, n=n_int_ord, seed=rs))

    # ops
    if nonlinear_single_arg_ops is None:
        nonlinear_single_arg_ops = OPS_SINGLE_ARG
        if nonlinear_single_arg_ops_weights is None:
            nonlinear_single_arg_ops_weights = OPS_SINGLE_ARG_WEIGHTS
    if nonlinear_single_arg_ops_weights is not None:
        assert_same_size(nonlinear_single_arg_ops,
                         nonlinear_single_arg_ops_weights, 'weights')
        assert np.isclose(sum(nonlinear_single_arg_ops_weights), 1.)

    if nonlinear_multi_arg_ops is None:
        nonlinear_multi_arg_ops = OPS_MULTI_ARG_NONLINEAR
        if nonlinear_multi_arg_ops_weights is None:
            nonlinear_multi_arg_ops_weights = OPS_MULTI_ARG_NONLINEAR_WEIGHTS
    if nonlinear_multi_arg_ops_weights is not None:
        assert_same_size(nonlinear_multi_arg_ops,
                         nonlinear_multi_arg_ops_weights, 'weights')
        assert np.isclose(sum(nonlinear_multi_arg_ops_weights), 1.)

    if linear_multi_arg_ops is None:
        linear_multi_arg_ops = OPS_MULTI_ARG_LINEAR
        if linear_multi_arg_ops_weights is None:
            linear_multi_arg_ops_weights = OPS_MULTI_ARG_LINEAR_WEIGHTS
    if linear_multi_arg_ops_weights is not None:
        assert_same_size(linear_multi_arg_ops,
                         linear_multi_arg_ops_weights, 'weights')
        assert np.isclose(sum(linear_multi_arg_ops_weights), 1.)

    # Build the expression
    expr = sp.Integer(0)

    # TODO: coefficients
    # TODO: scalar additions as arguments to functions

    # Main effects
    # TODO: it is possible with large enough `n_main` some terms could repeat
    #  and simplify into a single term with a coefficient. By pigeonhole this
    #  will happen due to finite number of ops
    n_main_nonlinear = int(round(pct_nonlinear * n_main))
    # Nonlinear main effects
    # TODO: magnitude of main_ops...applying multiple to each term...
    n_main_nonlinear_ops = int(round(nonlinear_multiplier * n_main_nonlinear))
    main_nonlinear_op_counts = place_into_bins(
        n_main_nonlinear, n_main_nonlinear_ops,
        shift=nonlinear_shift, skew=nonlinear_skew
    )

    main_nonlinear_ops = choice_objects(
        nonlinear_single_arg_ops, n_main_nonlinear_ops, replace=True,
        p=nonlinear_single_arg_ops_weights, seed=rs
    )
    main_nonlinear_ops_iter = iter(main_nonlinear_ops)
    # TODO cycle --> choice all features mod len(features) times
    main_features = cycle(features)

    domains = {}  # used in validation

    for i in range(n_main_nonlinear):
        feature = next(main_features)

        validate_try = 0
        alternative_ops = None
        while validate_try < validate_tries:
            term = feature

            if alternative_ops is None:
                for _ in range(main_nonlinear_op_counts[i]):
                    op = next(main_nonlinear_ops_iter)
                    term = op(term)
            else:
                for op in alternative_ops:
                    term = op(term)

            # TODO: doc the crap out of this "elegance"
            if not validate:
                break
            try:
                domains = valid_variable_domains(term, fail_action='error',
                                                 init_domains=domains,
                                                 **validate_kwargs)
                break  # we good
            except (RuntimeError, RecursionError, TimeoutError):
                validate_try += 1

                # try again with new selection of ops...
                alternative_ops = choice_objects(
                    nonlinear_single_arg_ops, main_nonlinear_op_counts[i],
                    replace=True, p=nonlinear_single_arg_ops_weights, seed=rs
                )

            # Then we continue...
        else:
            raise RuntimeError(f'Could not validate a term of the expression '
                               f'in {validate_tries} tries...')

        expr += term

    # Linear main effects
    for _ in range(n_main - n_main_nonlinear):
        # no validation needed here...
        expr += next(main_features)

    # Nonlinear interaction effects
    n_interaction_nonlinear = int(round(pct_nonlinear * n_interaction))
    n_interaction_nonlinear_ops = int(round(
        nonlinear_multiplier * n_interaction_nonlinear))

    if nonlinear_single_multi_ratio == 'balanced':
        nonlinear_single_multi_ratio = (
                len(nonlinear_single_arg_ops) / (len(nonlinear_single_arg_ops) +
                                                 len(nonlinear_multi_arg_ops)))
    n_interaction_nonlinear_ops_single = int(round(
        nonlinear_single_multi_ratio * n_interaction_nonlinear_ops))
    n_interaction_nonlinear_ops_multi = (
            n_interaction_nonlinear_ops - n_interaction_nonlinear_ops_single)

    # TODO place_into_bins+choice_objects --> single wrapper function?
    interaction_nonlinear_op_counts_single = place_into_bins(
        n_interaction_nonlinear, n_interaction_nonlinear_ops_single,
        shift=nonlinear_shift, skew=nonlinear_skew
    )
    interaction_nonlinear_ops_single = choice_objects(
        nonlinear_single_arg_ops, n_interaction_nonlinear_ops_single,
        replace=True, p=nonlinear_single_arg_ops_weights, seed=rs
    )

    interaction_nonlinear_op_counts_multi = place_into_bins(
        n_interaction_nonlinear, n_interaction_nonlinear_ops_multi,
        shift=nonlinear_shift, skew=nonlinear_skew
    )
    interaction_nonlinear_ops_multi = choice_objects(
        nonlinear_multi_arg_ops, n_interaction_nonlinear_ops_multi,
        replace=True, p=nonlinear_multi_arg_ops_weights, seed=rs
    )
    interaction_nonlinear_ops = []
    idx_single = idx_multi = 0
    # both op count lists are length of num. nonlinear interactions
    for (count_single,
         count_multi) in zip(interaction_nonlinear_op_counts_single,
                             interaction_nonlinear_op_counts_multi):
        # tuple of (single ops, multi ops)
        ops_i = (
            interaction_nonlinear_ops_single[idx_single:
                                             idx_single + count_single],
            interaction_nonlinear_ops_multi[idx_multi:
                                            idx_multi + count_multi],
        )
        # postpone shuffling
        interaction_nonlinear_ops.append(ops_i)

        idx_single += count_single
        idx_multi += count_multi

    # TODO cycle --> choice all uniq_interactions mod len(uniq_interactions)
    #  times
    interaction_features = cycle(uniq_interactions)

    for i, (term_ops_single,
            term_ops_multi) in enumerate(interaction_nonlinear_ops):
        term_features = next(interaction_features)
        n_term_features = len(term_features)

        n_multi_ops_term = interaction_nonlinear_op_counts_multi[i]

        # there are this many linear ops needed to have all terms interact.
        # if n_interact_bridges is < 0 then term_features will by cycled
        #  through when constructing the tree as n 2-arg ops require n - 1 >
        #  n_term_features leaf nodes
        n_interact_bridges = n_term_features - 1 - n_multi_ops_term

        term_features_leaf = [*term_features]
        if n_interact_bridges < 0:
            d, r = divmod(n_term_features, abs(n_interact_bridges))
            term_features_leaf += (
                    term_features_leaf * d + term_features_leaf[:r])
            n_interact_bridges = 0

        n_additions = int(round(
            nonlinear_interaction_additivity * n_interact_bridges))

        validate_try = 0
        while validate_try < validate_tries:
            linear_bridge_ops_multi = choice_objects(
                linear_multi_arg_ops, n_interact_bridges - n_additions,
                replace=True, p=linear_multi_arg_ops_weights, seed=rs
            )
            linear_bridge_ops_multi += [sp.Add] * n_additions
            term_ops_multi += linear_bridge_ops_multi
            # shuffle 'em
            rs.shuffle(term_ops_multi)  # actually is fast on objects

            # now we have...
            #  - features in term_features_leaf
            #  - multi ops in term_ops_multi
            #  - single ops in term_ops_single
            # TODO: Abs() (and potentially others) here can be simplified out
            #  due to symbols restricted to positive domain
            term = RandExprTree(
                leaves=term_features_leaf,
                parents_with_children=term_ops_multi,
                parents_with_child=term_ops_single,
                root_blacklist=(sp.Add,),
                seed=seed
            ).to_expression()

            # TODO: doc the crap out of this "elegance"
            if not validate:
                break
            try:
                domains = valid_variable_domains(term, fail_action='error',
                                                 init_domains=domains,
                                                 **validate_kwargs)
                break  # we good
            except (RuntimeError, RecursionError, TimeoutError):
                validate_try += 1

                # try again with new selection of ops...
                term_ops_single = choice_objects(
                    nonlinear_single_arg_ops,
                    interaction_nonlinear_op_counts_single[i],
                    replace=True, p=nonlinear_single_arg_ops_weights, seed=rs
                )
                term_ops_multi = choice_objects(
                    nonlinear_multi_arg_ops,
                    interaction_nonlinear_op_counts_multi[i],
                    replace=True, p=nonlinear_multi_arg_ops_weights, seed=rs
                )

            # Then we continue...
        else:
            raise RuntimeError(f'Could not validate a term of the expression '
                               f'in {validate_tries} tries...')

        expr += term

    # Linear interaction effects
    n_interaction_linear = n_interaction - n_interaction_nonlinear
    for _ in range(n_interaction_linear):
        # no validation needed here either...
        term_features = next(interaction_features)
        linear_interaction_ops = choice_objects(
            linear_multi_arg_ops, len(term_features) - 1,
            replace=True, p=linear_multi_arg_ops_weights, seed=rs
        )
        term = term_features[0]
        for feature, op in zip(term_features[1:], linear_interaction_ops):
            term = op(feature, term)

        expr += term

    return expr


def independent_terms(expr) -> Tuple[sp.Expr]:
    if isinstance(expr, sp.Add):
        return expr.args
    return expr,  # ',' for tuple return


@lru_cache()
def split_effects(
        expr: sp.Expr,
        symbols: Sequence[sp.Symbol],
) -> Tuple[Tuple[sp.Expr], Tuple[sp.Expr]]:
    """Additive effects"""
    expr_expanded = expr.expand(add=True)
    all_symbol_set = set(symbols)
    main_effects = []
    for xi in symbols:
        all_minus_xi = all_symbol_set - {xi}

        main, _ = expr_expanded.as_independent(*all_minus_xi, as_Add=True)
        main: sp.Expr  # type hint

        # Single main effect per symbol
        main_effects.append(main)

    interaction_effects = (set(independent_terms(expr_expanded)) -
                           set(main_effects))
    return tuple(main_effects), tuple(interaction_effects)


def _bad_domain(domain, no_empty_set, simplified, no_finite_set):
    """True if bad, False if good"""
    return ((no_empty_set and domain is sp.EmptySet) or
            (simplified and (domain.free_symbols or domain.atoms(sp.Dummy))) or
            # type much faster than isinstance
            (no_finite_set and (type(domain) is sp.FiniteSet)))


def _brute_force_errored_domain(term, undesirables, errored_symbols,
                                assumptions, no_empty_set, simplified,
                                no_finite_set, fail_action, interval,
                                true_brute_force=False, verbose=False):
    """Used in the case that domain-finding for a particular sympy op is not
    implemented

    See `ASSUMPTION_TO_DOMAIN` for supported assumptions
    """
    assert fail_action in {'error', 'warn'}

    domains = {}
    # Time to figure out if assumptions help automatically figure out
    #  valid continuous ranges...
    if true_brute_force:
        # Go with fewest to most variables to be least constraining
        combination_sizes = range(1, len(errored_symbols) + 1)
    else:
        combination_sizes = range(len(errored_symbols), 0, -1)
    for i in combination_sizes:
        # Try all combinations of assumptions on each errored symbol
        #  combination (find a valid domain for each and do so with minimal
        #  number of assumptions needed)
        for symbol_subset in combinations(errored_symbols, i):
            if true_brute_force:
                # The product of each symbol and assumption with every other
                #  symbol and assumption
                symbol_combinations = product(
                    *(tuple(zip(repeat(symbol), assumptions))
                      for symbol in symbol_subset)
                )
            else:
                # Simple, naive assumption across all variables (can speed this
                #  up with a smart initial guess - side-effect may be more
                #  constricting than necessary...maybe)
                symbol_combinations = (
                    tuple((symbol, assumption) for symbol in symbol_subset)
                    for assumption in assumptions
                )
            for symbol_comb in symbol_combinations:
                if verbose:
                    print(f'start brute force of {symbol_comb} for {term}')
                try:
                    # TODO: better-consider existing assumptions from symbols...
                    replacements = OrderedDict(
                        (symbol,
                         (sp.Symbol(symbol.name, **{assumption: True,
                                                    **symbol.assumptions0})
                          ))  # Mapping entry: symbol --> symbol w/ assumption
                        for symbol, assumption in symbol_comb
                    )
                    intervals = (
                        ASSUMPTION_DOMAIN.get(assumption,
                                              sp.Reals).intersect(interval)
                        for _, assumption in symbol_comb
                    )
                    # Insert symbols with assumptions into term
                    term_subs = term.subs(replacements)
                    undesired_domain = False
                    for symbol, s_interval in zip(replacements.keys(),
                                                  intervals):
                        replacement = replacements[symbol]
                        domain = continuous_domain(term_subs, replacement,
                                                   s_interval)

                        if _bad_domain(domain, no_empty_set, simplified,
                                       no_finite_set):
                            undesired_domain = True
                            if symbol not in undesirables:
                                undesirables[symbol] = domain
                            break

                        domains[symbol] = domain

                    if undesired_domain:
                        continue

                    return domains
                except (ValueError, TypeError, NotImplementedError, KeyError,
                        AssertionError) as e:
                    # TODO(doc exceptions)
                    if verbose:
                        print('exception[brute]', symbol_comb, term, e)

    if errored_symbols:
        if not true_brute_force:
            # Return function with a true brute force run...
            return _brute_force_errored_domain(
                term, undesirables, errored_symbols, assumptions, no_empty_set,
                simplified, no_finite_set, fail_action, interval=interval,
                true_brute_force=True
            )

        failed_symbols = set(errored_symbols) - set(domains.keys())
        for symbol in failed_symbols:
            if symbol not in undesirables:
                if verbose:
                    print('undesirables', undesirables)
                raise RuntimeError(
                    f'Failed to discover a valid domain for {symbol} of term '
                    f'{term}! This means that the expression contains ops '
                    f'that are not implemented in sympy (or are but are '
                    f'broken) and naive assumptions could not coerce out an '
                    f'interval of legal values.'
                )
            domain = undesirables[symbol]
            fail_msg = (f'desirable domain (simplified={simplified}, '
                        f'no_empty_set={no_empty_set}) for symbol {symbol} of '
                        f'term {term}: {domain}')
            if fail_action == 'warn':
                warnings.warn(f'Falling back on un' + fail_msg)
            elif fail_action == 'error':
                raise RuntimeError('Failed to find ' + fail_msg)
            domains[symbol] = domain

    return domains  # empty dict if made here


# kudos https://github.com/joblib/joblib/pull/366#issuecomment-267603530
def can_timeout(decorated):
    """May raise `TimeoutError`"""

    @wraps(decorated)
    def inner(*args, timeout=None, **kwargs):
        if timeout is None:  # normal functionality
            return decorated(*args, **kwargs)
        # timeout raised if applicable
        pool = mp.pool.ThreadPool(1)
        try:
            async_result = pool.apply_async(decorated, args, kwargs)
            ret = async_result.get(timeout)
        finally:
            pool.close()
        return ret

    return inner


@lru_cache(maxsize=int(2 ** 15))
@can_timeout
def _valid_variable_domains_term(term, assumptions, no_empty_set, simplified,
                                 no_finite_set, fail_action, interval,
                                 verbose=False):
    """Real domains only! Note: @can_timeout --> timeout kwarg"""
    domains = {}
    undesirables = {}
    errored_symbols = []
    for symbol in term.free_symbols:
        if verbose:
            print(f'start term {term} symbol {symbol}')
        try:
            domain = continuous_domain(term, symbol, interval)
            if _bad_domain(domain, no_empty_set, simplified, no_finite_set):
                if verbose:
                    print(f'undesirable domain for {symbol}: {domain}')
                errored_symbols.append(symbol)
                undesirables[symbol] = domain
                continue
            domains[symbol] = domain
        except (ValueError, TypeError, NotImplementedError, KeyError,
                AssertionError) as e:
            # TODO(doc exceptions)
            # ex1:
            # ValueError:
            # sech(sqrt(x7)) > 0 contains imaginary parts which cannot be made
            # 0 for any value of x4 satisfying the inequality, leading to
            # relations like I < 0.
            # ex2:
            # File ".../sympy/solvers/diophantine/diophantine.py", line 491,
            # in <listcomp>
            #     return {tuple([t[dict_sym_index[i]] for i in var])
            # KeyError: x5
            # ex3:
            #   File ".../sympy/solvers/solveset.py", line 628, in _solve_trig2
            #     assert len(numerators) == 1
            # AssertionError
            errored_symbols.append(symbol)
            if verbose:
                print('exception[initial]', symbol, term, e)
    # Get domains for errored out symbols (not implemented) and add to domains
    if verbose:
        print(term, 'domains before brute force:\n', domains)
        print('errored_symbols before brute force:\n', errored_symbols)
    domains.update(
        _brute_force_errored_domain(term, undesirables, errored_symbols,
                                    assumptions, no_empty_set, simplified,
                                    no_finite_set, fail_action,
                                    interval=interval, verbose=verbose)
    )
    return domains


def valid_variable_domains(terms, assumptions=None, no_empty_set=True,
                           simplified=True, no_finite_set=True,
                           init_domains=None, fail_action='warn',
                           verbose=False, interval=sp.Reals, timeout=None):
    """Find the valid continuous domains of the free variables of a symbolic
    expression. Expects additive terms to be provided, but will split up a
    sympy expression too.

    fail_action error: RuntimeError

    Real domains only! TODO: allow other domains?
    """
    # TODO!!! sympy treats reals as complex - any valid domains *may be in the
    #  complex domain* - this is crap, and may lead to unexpected errors down
    #  the road. Is there a way to mitigate this??

    if isinstance(terms, sp.Expr):
        # More efficient to look at each term of expression in case of
        #  NotImplementedError in valid domain finding
        terms = independent_terms(terms)

    if assumptions is None:
        assumptions = tuple(ASSUMPTION_DOMAIN.keys())

    if init_domains is None:
        domains = {}
    else:
        domains = init_domains.copy()

    for term in terms:
        # Get valid domains
        domains_term = _valid_variable_domains_term(
            term, assumptions, no_empty_set, simplified, no_finite_set,
            fail_action, interval=interval, verbose=verbose, timeout=timeout)
        # Update valid intervals of each variable
        for symbol, domain in domains_term.items():
            domains[symbol] = domains.get(symbol, interval).intersect(domain)

    # bad domains can arise (namely empty set) from intersections of intervals
    for symbol, domain in domains.items():
        if _bad_domain(domain, no_empty_set, simplified, no_finite_set):
            fail_msg = (f'desirable domain (simplified={simplified}, '
                        f'no_empty_set={no_empty_set}) for symbol {symbol}: '
                        f'{domain}')
            if fail_action == 'warn':
                warnings.warn(f'Falling back on un' + fail_msg)
            elif fail_action == 'error':
                raise RuntimeError('Failed to find ' + fail_msg)

    return domains


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
    # TODO: AdditiveModel -> SymbolicAdditiveModel

    def __init__(self,
                 n_features: int,
                 # TODO: reproducibility...use rs instead...or just remove
                 #  this crap
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
        self.symbols = sp.symbols(self.symbol_names, **kwargs)
        self.expr = self._generate_model(coefficients)  # TODO interactions
        self._symbol_map = None

        self.backend = backend

        self._check()

    def _check(self):
        # TODO: these messages are absolutely annoying...maybe check other
        #  things?
        # if not isinstance(self.expr, sp.Add):
        #     warnings.warn('Expression is not additive! Output is dependent
        #                   'on all input variables: '
        #                   'optype {}'.format(type(self.expr)))
        pass

    @classmethod
    def from_expr(
            cls,
            expr: Union[sp.Expr, str],
            symbols: Optional[Symbol1orMore] = None,
            backend=None,
    ):
        """Symbols needs to be ordered properly"""
        # Ensure expr symbols are a subset of symbols
        if isinstance(expr, str):
            expr = sp.parse_expr(expr)

        if symbols is None:
            symbols = tuple(sorted(expr.free_symbols, key=lambda x: x.name))

        symbols = (symbols,) if isinstance(symbols, sp.Symbol) else symbols
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

    def get_symbol(self, symbol_name: str) -> sp.Symbol:
        if self._symbol_map is None:
            self._symbol_map = dict(zip(self.symbol_names, self.symbols))
        return self._symbol_map[symbol_name]

    @property
    def main_effects(self) -> Tuple[sp.Expr]:
        main_effects, _ = split_effects(self.expr, self.symbols)
        return main_effects

    @property
    def interaction_effects(self) -> Tuple[sp.Expr]:
        _, interaction_effects = split_effects(self.expr, self.symbols)
        return interaction_effects

    @property
    def independent_terms(self) -> Tuple[sp.Expr]:
        return independent_terms(self.expr)

    def valid_variable_domains(self, **kwargs):
        """See documentation of `valid_variable_domains` function"""
        return valid_variable_domains(self.independent_terms, **kwargs)

    def __call__(
            self,
            x: np.ndarray,
            backend=None,
    ):
        x = np.asarray(x)
        assert_shape(x, (None, self.n_features))
        if backend is None:
            backend = self.backend
        eval_func = symbolic_evaluate_func(self.expr, self.symbols,
                                           x=x, backend=backend)
        try:
            return eval_func(*(x[:, i] for i in range(self.n_features)))
        except FloatingPointError:

            def safe_eval_func():
                for x_i in x:
                    x_i = x_i[None, ...]
                    try:
                        yield eval_func(*(x_i[:, i]
                                          for i in range(self.n_features)))
                    except FloatingPointError:
                        # TODO: consider a multi-output AdditiveModel
                        yield np.nan

            return np.fromiter(safe_eval_func(), dtype=float)

    def predict(self, x):  # sklearn compat
        # TODO: classification vs. regression...
        return self(x)

    def predict_proba(self, x):  # sklearn compat
        # TODO: classification vs. regression...
        return self(x)

    def make_effects_dict(self,
                          main_effects=True,
                          interaction_effects=True):
        if not (main_effects or interaction_effects):
            raise ValueError('Must specify either main_effects or '
                             'interaction_effects')
        effects = []
        if main_effects:
            effects.extend(self.main_effects)
        if interaction_effects:
            effects.extend(self.interaction_effects)

        effects_dict = {}

        for effect in effects:
            # standardize (TODO use metrics func?)
            effect_symbols = sorted(effect.free_symbols, key=lambda s: s.name)
            effect_symbols = tuple(effect_symbols)
            if effect == 0:
                continue  # skip zero-effects
            effects_dict[effect_symbols] = effect

        return effects_dict

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
            effects.extend(self.main_effects)
        if interaction_effects:
            effects.extend(self.interaction_effects)

        # contributions = defaultdict(lambda: np.zeros(len(x)))
        # all_effects = defaultdict(lambda: sp.Number(0))
        contributions = {}
        all_effects = {}

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
                                               x=x,  # TODO(x)
                                               backend=backend)
            contribution = eval_func(*related_features)
            contributions[effect_symbols] = contribution
            if return_effects:
                all_effects[effect_symbols] = effect

        if return_effects:
            return contributions, all_effects
        return contributions

    def pprint(self):
        sp.pprint(self.expr)

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
    # 10 variables
    all_symbols = sp.symbols('x1:11', real=True)
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
            sp.pi ** (x1 * x2) * sp.sqrt(2 * x3)
            - sp.asin(x4)
            + sp.log(x3 + x5)
            - x9 / x10 * sp.sqrt(x7 / x8)
            - x2 * x7,
            all_symbols
        ),
        f2=AdditiveModel.from_expr(
            sp.pi ** (x1 * x2) * sp.sqrt(2 * abs(x3))
            - sp.asin(x4 / 2)
            + sp.log(abs(x3 + x5) + 1)
            + x9 / (1 + abs(x10)) * sp.sqrt(x7 / (1 + abs(x8)))
            - x2 * x7,
            all_symbols
        ),
        f3=AdditiveModel.from_expr(
            sp.exp(abs(x1 - x2))
            + abs(x2 * x3)
            - x3 ** (2 * abs(x4))
            + sp.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
            + x9
            + 1 / (1 + x10 ** 2),
            all_symbols
        ),
        f4=AdditiveModel.from_expr(
            sp.exp(abs(x1 - x2))
            + abs(x2 * x3)
            - x3 ** (2 * abs(x4))
            + (x1 * x4) ** 2
            + sp.log(x4 ** 2 + x5 ** 2 + x7 ** 2 + x8 ** 2)
            + x9
            + 1 / (1 + x10 ** 2),
            all_symbols
        ),
        f5=AdditiveModel.from_expr(
            1 / (1 + x1 ** 2 + x2 ** 2 + x3 ** 2)
            + sp.sqrt(sp.exp(x4 + x5))
            + abs(x6 + x7)
            + x8 * x9 * x10,
            all_symbols
        ),
        f6=AdditiveModel.from_expr(
            sp.exp(abs(x1 * x2) + 1)
            - sp.exp(abs(x3 + x4) + 1)
            + sp.cos(x5 + x6 - x8)
            + sp.sqrt(x8 ** 2 + x9 ** 2 + x10 ** 2),
            all_symbols
        ),
        f7=AdditiveModel.from_expr(
            (sp.atan(x1) + sp.atan(x2)) ** 2
            # Ha! You think you could do this but nooooo
            # + sym.Max(x3 * x4 + x6, 0)
            # Have to do this $#!% instead to get it vectorized. Also note that
            # the 0. has to be a float for certain backends to not complain.
            + sp.Piecewise((f7_int, f7_int > 0), (0., True))
            - 1 / (1 + (x4 * x5 * x6 * x7 * x8) ** 2)
            + (abs(x7) / (1 + abs(x9))) ** 5
            + sum(all_symbols),
            all_symbols
        ),  # sum(all_symbols) = \sum_{i=1}^{10} x_i
        f8=AdditiveModel.from_expr(
            x1 * x2
            + 2 ** (x3 + x5 + x6)
            + 2 ** (x3 + x4 + x5 + x7)
            + sp.sin(x7 * sp.sin(x8 + x9))
            + sp.acos(sp.Integer(9) / sp.Integer(10) * x10),
            all_symbols
        ),
        f9=AdditiveModel.from_expr(
            sp.tanh(x1 * x2 + x3 * x4) * sp.sqrt(abs(x5))
            + sp.exp(x5 + x6)
            + sp.log((x6 * x7 * x8) ** 2 + 1)
            + x9 * x10
            + 1 / (1 + abs(x10)),
            all_symbols
        ),
        f10=AdditiveModel.from_expr(
            sp.sinh(x1 + x2)
            + sp.acos(sp.tanh(x3 + x5 + x7))
            + sp.cos(x4 + x5)
            + 1 / sp.cos(x7 * x9),  # sec=1/cos (some backends don't have sec)
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
