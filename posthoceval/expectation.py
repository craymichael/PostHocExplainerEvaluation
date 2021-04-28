"""
expectation.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael

Unused code (atm) for expectation computation...sympy has many issues with this
so there is some naive assumptions made here to efficiently get answers and no
errors...still a WIP
"""


def naive_reduce_domain(domain, parent_is_complement=False):
    """Useful for producing Intervals used in stats computations"""
    if isinstance(domain, sp.Union):
        domain_orig = domain

        left = None
        right = None
        # this might be continuous with holes (we don't want gaps!)
        # ordering of sympy unions guarantees end points will increase
        # monotonically (if there are only holes)
        continuous_with_holes = True

        for arg in domain.args:
            if continuous_with_holes and isinstance(arg, sp.Interval):
                a, b, _, _ = arg.args
                if left is None:
                    left = a
                    right = b
                elif a == right:  # hole but not a gap
                    right = b
                else:
                    continuous_with_holes = False
            else:
                continuous_with_holes = False

                domain = domain.subs({
                    arg: naive_reduce_domain(arg, parent_is_complement)
                })
                if domain_orig != domain:
                    domain = naive_reduce_domain(domain, parent_is_complement)
                    break
        if continuous_with_holes:
            domain = sp.Interval(left, right)
    elif isinstance(domain, sp.Complement):
        to_replace = domain.args[1]
        # invert current complement
        domain = domain.subs({
            to_replace: naive_reduce_domain(to_replace,
                                            not parent_is_complement)
        })
    elif isinstance(domain, sp.ConditionSet):
        if parent_is_complement:
            # naively assume no bad conditions (empty set due to complement)
            domain = sp.EmptySet
        else:
            # naively assume no bad conditions (set given to conditionset)
            domain = domain.args[2]
    return domain


# reduced_domains = [*map(naive_reduce_domain, uniq_domains)]


########################################################

import sympy as sp
from sympy import stats
from posthoceval.evaluate import symbolic_evaluate_func
from posthoceval.evaluate import replace_unsupported_functions
from posthoceval.models.synthetic import independent_terms
from posthoceval.utils import prod
from tqdm.auto import tqdm
import numpy as np


def log_filter(*args):
    return len(args) == 1 and isinstance(args[0], sp.log) and isinstance(
        args[0].args[0], sp.Mul)


def log_replace(*args):
    assert len(args) == 1, 'this was unexpected'

    return sum(sp.log(mul_arg) for mul_arg in args[0].args[0].args)


# see https://github.com/sympy/sympy/issues/20755
wild = sp.Wild('a')
broken_exprs = {
    # sp.Abs(sp.tan(wild)): sp.Abs(1 / sp.cot(wild)),
    # sp.Abs(sp.csc(wild)): sp.Abs(1 / sp.sin(wild)),
    sp.tan(wild): (1 / sp.cot(wild)),
    sp.csc(wild): (1 / sp.sin(wild)),
    sp.cos(wild): sp.sin(np.pi / 2 - wild),
    sp.acot(wild): sp.Piecewise(
        ((sp.pi / 2) - sp.atan(wild), wild >= 0),
        (-sp.atan(wild) - (sp.pi / 2), True)
    ),
    # not broken but better this way given how i'm doing things
    log_filter: log_replace,
}

# single-arg trig functions only
all_trig = {
    # cos
    sp.cos,
    sp.cosh,
    sp.acos,
    sp.acosh,
    # sin
    sp.sin,
    sp.sinh,
    sp.asin,
    sp.asinh,
    # tan
    sp.tan,
    sp.tanh,
    sp.atan,
    sp.atanh,
    # cot
    sp.cot,
    sp.coth,
    sp.acot,
    sp.acoth,
    # csc
    sp.csc,
    sp.csch,
    sp.acsc,
    sp.acsch,
    # sec
    sp.sec,
    sp.sech,
    sp.asec,
    sp.asech,
    # special
    sp.sinc,
}


def handle_excessive_periodicity(expr, ready2kill=False):
    # things like atan(sinc(U)) have too much periodicity for sympy to handle,
    # just kill self now if the trig is too much
    visited = set()
    for sub_expr in sp.preorder_traversal(expr):
        if sub_expr in visited:
            continue
        visited.add(sub_expr)
        if type(sub_expr) in all_trig:
            if ready2kill:
                raise TypeError('SOHCAHTOA sadness')
            else:
                visited.update(
                    handle_excessive_periodicity(sub_expr.args[0],
                                                 ready2kill=True))
    return visited


def handle_expected_val(expr):
    """always raises TypeError if something bad...because"""
    if isinstance(expr, sp.Add):
        return sum(map(handle_expected_val, expr.args))
    elif isinstance(expr, sp.Mul):
        return prod(map(handle_expected_val, expr.args))
    elif len(expr.free_symbols) <= 1:
        # first check for too many periodicity units
        handle_excessive_periodicity(expr)

        # see https://github.com/sympy/sympy/issues/20756
        # also more than 1 var can take way too long...
        # yeah so many of the multi-var results are BS
        return stats.E(expr).evalf()
    elif isinstance(expr, sp.Abs):
        maybe_nan = handle_expected_val(expr.args[0])
        if maybe_nan is sp.nan:
            return maybe_nan
        else:
            raise TypeError('sadness')
    else:
        raise TypeError('sadness')


def validate(data):
    bad_expr = []
    questionable_expr = []
    for res in tqdm(data):
        sampled_e = {}
        sampled_var = {}
        # samples = np.random.uniform(-1, +1, size=(10_000, len(res.symbols)))
        to_sub = {}
        samples = {}

        for s, d in res.domains.items():
            d = naive_reduce_domain(d)
            assert isinstance(d, sp.Interval)
            a, b, _, _ = d.args
            U = stats.Uniform('U_' + s.name, a, b)
            to_sub[s] = U

            samples[s] = np.random.uniform(a, b, size=100_000)

        additive_terms = independent_terms(res.expr)

        for sub_expr in additive_terms:
            try:
                syms = [*sub_expr.free_symbols]
                # func = lambdify(syms, sub_expr, modules='numpy')
                func = symbolic_evaluate_func(sub_expr, syms, backend='numpy')
                vals = func(*(samples[sym] for sym in syms))
                sample_mean = np.mean(vals)
                sample_var = np.var(vals)
                # tqdm.write(sp.pretty(sub_expr) + ' sample mean: '
                #           f'{sample_mean}')
                sampled_e[sub_expr] = sample_mean
                sampled_var[sub_expr] = sample_var
            except Exception as e:  # TODO: too broad
                tqdm.write(f'!!!whoops: {e}')

        # expr = res.expr.subs(to_sub)
        for sub_expr in additive_terms:
            sample_mean = sampled_e.get(sub_expr)
            sample_var = None
            sub_expr = sub_expr.subs(to_sub)
            for k, v in broken_exprs.items():
                sub_expr = sub_expr.replace(k, v)
            domain_info = '|'.join(
                map(str, (u.args[1].args[1] for u in sub_expr.free_symbols)))
            exp_val = '?'

            tqdm.write('DEBUG: ' + sp.pretty(sub_expr) + ' ' + domain_info)
            try:
                exp_val = handle_expected_val(sub_expr)
                exp_val = float(exp_val)
            except TypeError:
                tqdm.write(f'estimating {exp_val} with 100k samples...')
                # bad support for some functions (original list due to
                # backends, but sympy expected value also has some trouble with
                # these...)
                sub_expr = replace_unsupported_functions(sub_expr)
                sample_var = sampled_var.get(sub_expr)
                # TODO: don't sample this until here in non-testing version
                exp_val = sample_mean
            tqdm.write(sp.pretty(
                sub_expr) + f' expected value: {exp_val} ({type(exp_val)})')

            if np.isnan(exp_val) or np.isinf(exp_val) or (
                    sample_var is not None and sample_var >= .05):
                tqdm.write(
                    f'E[{sub_expr}] is meh, moving on without deciding one '
                    f'(sample_var={sample_var})')
                exp_val = None
                bad_expr.append(sub_expr)
            elif sample_mean is None:
                tqdm.write('WARNING: ABOVE NEEDS MANUAL CHECKING!!!')
                bad_expr.append(sub_expr)
            elif abs(exp_val - sample_mean) > 1e-1:
                tqdm.write(
                    f'{exp_val} != {sample_mean} (exp_val != sample_mean)')
                exp_val = None
                bad_expr.append(sub_expr)
            elif sample_var is not None:
                questionable_expr.append(sub_expr)

            tqdm.write('')

        # TODO: cache all expected values in the metrics folder
        # TODO: parallelizing this function could be a good idea too...
        # TODO: if this function returns None for the expected value, this is
        #  an indicator to callers that you should use the sample mean for
        #  **your data** (this is most fair in comparison to something, as the
        #  sample mean for vary wildy for several less stable functions (e.g.
        #  1/x**3, x~U(0,1) ))
    return bad_expr, questionable_expr


###############################################################################

def discover_bad_piecewise(data):
    bad = []
    for i, result in enumerate(data):
        expr = replace_unsupported_functions(result.expr)
        if len(expr.atoms(sp.Piecewise)) != 0:
            bad.append(i)
    return bad


###############################################################################

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    xmax = 1000 * (0 - .63) ** 2
    x = np.linspace(0, xmax, 200)
    xp_ = np.sqrt(x / 1000) + 0.63
    xn_ = -np.sqrt(x / 1000) + 0.63
    xn = np.select([xn_ >= 0, True], [xn_, xp_])
    xp = np.select([xp_ <= 1, True], [xp_, xn])
    plt.scatter(x, xn, c='b')
    plt.scatter(x, xp, c='b')
    plt.xlabel('B')
    plt.ylabel('Bk')
    plt.show()
