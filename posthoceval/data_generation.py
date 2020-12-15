from typing import Callable

import numpy as np

import sympy as S
from sympy import stats


def sample(variables, distributions, cov=None, seed):
    # TODO: setting seed may not be possible...until 1.7.1
    #  https://github.com/sympy/sympy/pull/20528/
    U = stats.Uniform('U', a, b)
    a = np.fromiter(
        S.stats.sample_iter(U, ii.contains(U), numsamples=10000), dtype=np.float32
    )

    # TODO: note that sympy==1.6 is necessary, there is a non-public regression for
    #  some expressions in 1.7 https://github.com/sympy/sympy/issues/20563
    try:
        S.stats.sample_iter(...)
    except NameError:
        from sympy.stats.rv import sample_iter_subs

        sample_iter_subs(...)
