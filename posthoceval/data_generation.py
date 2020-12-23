from typing import Sequence
from itertools import repeat

import numpy as np

import sympy as sp
from sympy import stats
from sympy.stats.rv import sample_iter_subs


def sample(variables, distribution, n_samples, constraints=None, cov=None):
    # TODO satisfy desired covariance matrix...
    assert cov is None, 'not supported yet...'

    # TODO: setting seed may not be possible...until 1.7.1
    #  https://github.com/sympy/sympy/pull/20528/
    if isinstance(distribution, Sequence):
        assert len(distribution) == len(variables)
    else:
        distribution = repeat(distribution, len(variables))

    if constraints is None:
        constraints = {}
    elif isinstance(constraints, dict):
        assert not (set(constraints.keys()) - set(variables))
    elif isinstance(constraints, Sequence):
        assert len(constraints) == len(variables)
        constraints = dict(map(variables, constraints))
    else:
        # each variable gets same constraint
        constraints = dict(map(variables, repeat(constraints)))

    columns = []
    for v, d in zip(variables, distribution):
        constraint = constraints.get(v)

        args = ()
        if constraint:
            args = (constraint,)

        # TODO: note that sympy==1.6 is necessary, there is a non-public
        #  regression for some expressions in 1.7
        #  https://github.com/sympy/sympy/issues/20563
        try:
            samples = sp.stats.sample_iter(d, *args)
        except NameError:
            samples = sample_iter_subs(d, *args)

        samples_v = np.fromiter(
            (next(samples) for _ in range(n_samples)),
            dtype=np.float32,
        )
        columns.append(samples_v)

    return np.stack(columns, axis=1)
