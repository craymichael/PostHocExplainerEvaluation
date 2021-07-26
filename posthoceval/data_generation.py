import warnings
from typing import Sequence
from itertools import repeat

import random

import numpy as np

import sympy as sp
from sympy import stats
from sympy.stats.rv import sample_iter_subs

from posthoceval.evaluate import symbolic_evaluate_func


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
                    try:
                        return (float(actual_distribution_obj.left),
                                float(actual_distribution_obj.right))
                    except (ValueError, TypeError, NotImplementedError):
                        pass
    return None


def sample(variables, distribution, n_samples, constraints=None, cov=None,
           seed=None):
    """
    Sample variables from the given distribution and constraints

    :param variables: the random variables to sample
    :param distribution: the distribution to sample from
    :param n_samples: the number of samples to draw
    :param constraints: the constraints to impose on drawn samples
    :param cov: covariance (not implemented yet)
    :param seed: random seed
    :return: drawn samples
    """
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
        no_constraint = constraint is None

        # if uniform...
        uniform_args = _get_uniform_args(d)

        if ((no_constraint or len(constraint.free_symbols) == 1)
                and uniform_args is not None):
            # get instance of UniformDistribution
            low, high = uniform_args

            # Use numpy which is insanely faster right now (than sympy
            # sampling)
            def sample_func(n_samples_):
                return np.random.uniform(low, high, size=n_samples_).astype(
                    np.float32)

            samples_v = sample_func(n_samples)
            if not no_constraint:
                # meet constraints
                try:
                    # entirely possible that this will break in sympy
                    constraint_func = symbolic_evaluate_func(
                        constraint, [*constraint.free_symbols],
                        backend='numpy'
                    )
                    # test to make sure things work ok
                    constraint_func(samples_v[:1])
                except (NameError, ValueError, TypeError):
                    warnings.warn(f'Could not lambdify {constraint}...using '
                                  f'sympy validation instead...')
                    # next symbol (only one symbol per above check)
                    c_symbol = [*constraint.free_symbols][0]

                    def constraint_func(values):
                        return np.fromiter(
                            (constraint.subs({c_symbol: val})
                             for val in values), dtype=bool
                        )

                while True:
                    invalid_sample_idxs = np.where(
                        ~constraint_func(samples_v))[0]
                    if invalid_sample_idxs.size == 0:
                        break  # constraints met
                    samples_v[invalid_sample_idxs] = sample_func(
                        len(invalid_sample_idxs))
        else:
            args = () if no_constraint else (constraint,)

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
