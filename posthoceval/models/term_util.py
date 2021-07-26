"""
term_util.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from scipy.special import comb

from posthoceval.rand import select_n_combinations
from posthoceval.rand import as_random_state


def generate_terms(
        n_features,
        n_main,
        n_interact=None,
        desired_interactions=None,
        min_order=None,
        max_order=None,
        seed=None,
):
    """
    Utility to generate terms given various parameters

    :param n_features: number of features
    :param n_main: number of main effects
    :param n_interact: number of interaction effects
    :param desired_interactions: the specific interaction terms desired
    :param min_order: the minimum order of interactions
    :param max_order: the maximum order of interactions
    :param seed: the random seed for reproducibility
    :return: list of terms that can be used to generate an additive model
    """
    seed = as_random_state(seed)
    if n_interact is not None and desired_interactions is not None:
        raise ValueError('Cannot specify both n_interact and '
                         'desired_interactions')
    no_interactions = n_interact is None and desired_interactions is None
    if max_order is None:
        max_order = 1 if no_interactions is None else 2
    elif no_interactions:
        max_order = 1
    if min_order is None:
        min_order = 2
    terms = []
    all_features = [*range(n_features)]
    for order in range(1, max_order + 1):
        if order == 1:
            terms.extend(select_n_combinations(
                all_features, k=1, n=n_main, seed=seed))
        elif order >= min_order or desired_interactions is not None:
            if desired_interactions is None:
                n_interact = min(n_interact - len(terms) + n_main,
                                 comb(n_features, order))
                selected_interact = select_n_combinations(
                    all_features, k=order, n=n_interact, seed=seed)
            else:
                selected_interact = desired_interactions
            terms.extend(tuple(feats) for feats in selected_interact)
            if desired_interactions is not None:
                break  # all user-specified interactions added
    return terms
