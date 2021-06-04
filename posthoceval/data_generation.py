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
    if seed is not None:
        
        
        
        random.seed(seed)
        np.random.seed(seed)

    
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
        
        constraints = dict(map(variables, repeat(constraints)))

    columns = []
    for v, d in zip(variables, distribution):
        constraint = constraints.get(v)
        no_constraint = constraint is None

        
        uniform_args = _get_uniform_args(d)

        if ((no_constraint or len(constraint.free_symbols) == 1)
                and uniform_args is not None):
            
            low, high = uniform_args

            
            
            def sample_func(n_samples_):
                return np.random.uniform(low, high, size=n_samples_).astype(
                    np.float32)

            samples_v = sample_func(n_samples)
            if not no_constraint:
                
                try:
                    
                    constraint_func = symbolic_evaluate_func(
                        constraint, [*constraint.free_symbols],
                        backend='numpy'
                    )
                    
                    constraint_func(samples_v[:1])
                except (NameError, ValueError, TypeError):
                    warnings.warn(f'Could not lambdify {constraint}...using '
                                  f'sympy validation instead...')
                    
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
                        break  
                    samples_v[invalid_sample_idxs] = sample_func(
                        len(invalid_sample_idxs))
        else:
            args = () if no_constraint else (constraint,)

            
            
            
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
