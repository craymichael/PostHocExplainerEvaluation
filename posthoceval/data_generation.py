from typing import Sequence
from itertools import repeat

import random

import numpy as np

import sympy as sp
from sympy import stats
from sympy.stats.rv import sample_iter_subs


def _get_uniform_args(distribution):
    # so easy to get distribution arguments...ex
    #  U.pspace.distribution.left
    if hasattr(distribution, 'pspace'):
        pspace = distribution.pspace
        if hasattr(pspace, 'distribution'):
            actual_distribution_obj = pspace.distribution
            actual_distribution_cls = getattr(actual_distribution_obj,
                                              '__class__')
            if actual_distribution_cls is not None:
                if actual_distribution_cls.__name__ == 'UniformDistribution':
                    return (actual_distribution_obj.left,
                            actual_distribution_obj.right)
    return None


def sample(variables, distribution, n_samples, constraints=None, cov=None,
           seed=None):
    if seed is not None:
        # TODO: setting seed may not be possible...until 1.7.1
        #  https://github.com/sympy/sympy/pull/20528/
        # This may be sufficient for now...
        random.seed(seed)
        np.random.seed(seed)

    # TODO satisfy desired covariance matrix...
    assert cov is None, 'not supported yet...'

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

        args = () if constraint is None else (constraint,)

        # if uniform...
        uniform_args = _get_uniform_args(d)

        if not args and uniform_args is not None:
            # get instance of UniformDistribution
            low, high = uniform_args
            # Use numpy which is insanely faster right now
            samples_v = np.random.uniform(low, high, size=n_samples).astype(
                np.float32)
        else:
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
