import warnings
from itertools import combinations
from math import sqrt

import numpy as np
from scipy.special import comb


def as_random_state(seed):
    """Turns int, Generator, etc. into Generator. Warns about RandomStates

    See docs for `numpy.random.default_rng`
    https://numpy.org/doc/stable/reference/random/generator.html

    `numpy.random.RandomState` is legacy - slow, do not use...
    https://numpy.org/devdocs/reference/random/performance.html
    """
    bad_rng = np.random.RandomState
    if isinstance(seed, bad_rng):
        warnings.warn(
            f'Seed is an instance of {bad_rng}: this will be slower than the '
            f'optimized RNGs available in NumPy. Unless reproducing results, '
            f'do not use this RNG. See '
            f'https://numpy.org/devdocs/reference/random/performance.html'
        )
        return seed
    return np.random.default_rng(seed)


def select_n_combinations(values, k, n, seed=None):
    """Randomly selects `n` combinations from `values` of size `k`. If few
    collisions are possible (see the birthday paradox and hash collision
    probabilities for background), values are randomly sampled until uniqueness
    is satisfied. Otherwise all combinations are created (memory-expensive) and
    then sampled. All combinations drawn without repetition.
    """
    n_combs = comb(len(values), k, exact=True)
    assert n <= n_combs

    rs = as_random_state(seed)

    # check if many collisions may occur
    if sqrt(n_combs) >= n:
        # not many collisions expected
        choice_idxs = set()
        idxs = np.arange(len(values))
        while len(choice_idxs) < n:
            choice_idxs.add(tuple(
                # tuple(a.tolist()) faster than tuple(a)
                rs.choice(idxs, k, replace=False).tolist()
            ))
        choices = tuple(tuple(values[i] for i in idx)
                        for idx in choice_idxs)
    else:
        # many collisions possible - get all combinations then place_into_bins n
        # randomly
        all_choices = tuple(combinations(values, k))
        choice_idxs = rs.choice(np.arange(len(all_choices)), n, replace=False)
        choices = tuple(all_choices[i] for i in choice_idxs)

    return choices


def choice_objects(objects, size=None, replace=True, p=None, seed=None):
    """
    `np.random.choice` for sets of objects. performs choice on an array of ints
    to speed things up
    """
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
