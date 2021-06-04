import warnings
from itertools import combinations
from math import sqrt

import numpy as np
from scipy.special import comb


def as_random_state(seed):
    bad_rng = np.random.RandomState
    if isinstance(seed, bad_rng):
        warnings.warn(
            f'Seed is an instance of {bad_rng}: this will be slower than the '
            f'optimized RNGs available in NumPy. Unless reproducing results '
            f'that specifically use {bad_rng}, do not use this RNG. See '
            f'https://numpy.org/devdocs/reference/random/performance.html'
        )
        return seed
    return np.random.default_rng(seed)


def select_n_combinations(values, k, n, seed=None):
    n_combs = comb(len(values), k, exact=True)
    assert n <= n_combs

    rs = as_random_state(seed)

    if sqrt(n_combs) >= n:
        choice_idxs = set()
        idxs = np.arange(len(values))
        while len(choice_idxs) < n:
            choice_idxs.add(tuple(
                rs.choice(idxs, k, replace=False).tolist()
            ))
        choices = tuple(tuple(values[i] for i in idx)
                        for idx in choice_idxs)
    else:
        all_choices = tuple(combinations(values, k))
        choice_idxs = rs.choice(np.arange(len(all_choices)), n, replace=False)
        choices = tuple(all_choices[i] for i in choice_idxs)

    return choices


def choice_objects(objects, size=None, replace=True, p=None, seed=None):
    rs = as_random_state(seed)

    idxs = np.arange(len(objects))
    selected = rs.choice(idxs, size=size, replace=replace, p=p)

    return [objects[i] for i in selected]


def randint(low, high=None, size=None, dtype=int, endpoint=False, seed=None):
    rs = as_random_state(seed)
    if isinstance(rs, np.random.RandomState):
        assert not endpoint, 'RandomState endpoint must be kept False'
        return rs.randint(low, high=high, size=size, dtype=dtype)
    else:
        return rs.integers(low, high=high, size=size, dtype=dtype,
                           endpoint=endpoint)
