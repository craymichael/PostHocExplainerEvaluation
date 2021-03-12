"""
metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import warnings

import numpy as np
from scipy.special import comb
from posthoceval.rand import select_n_combinations
from posthoceval.rand import as_random_state


def sensitivity_n(model, explain_func, X, n_subsets=100, max_feats=0.8,
                  n_samples=None, n_points=None, seed=None):
    """
    Pros:

    Cons:
    - only works on feature attributions, so no interactions or other types of
      explanations
    -

    Aggregated version of sensitivity-n
    Max score of 1, minimum score of -1 (validate this min...)

    "Towards better understanding of gradient-based attribution methods for
     Deep Neural Networks"

    :param model:
    :param X:
    :param n_subsets:
    :return:
    """
    assert X.ndim == 2, 'i lazy gimme 2 dims for X'

    n_feats = X.shape[1]
    feats = np.arange(n_feats)

    if n_points is None:
        n_points = min(32, n_feats)
    if n_samples is None:
        n_samples = min(1000, len(X))

    rs = as_random_state(seed)

    X_eval = rs.choice(X, n_samples, replace=False)

    # TODO: do model compute out here

    # all_n = np.unique(np.round(
    #     np.geomspace(1, n_feats * max_feats, n_points)).astype(int))
    all_n = np.unique(np.round(
        np.linspace(1, n_feats * max_feats, n_points)).astype(int))

    # pearson corr coefs
    all_pccs = []

    for n in all_n:
        pccs = []
        for x in X_eval:
            max_combs = comb(n_feats, n, exact=True)
            n_subsets_n = min(max_combs, n_subsets)
            combs = select_n_combinations(feats, n, n_subsets_n, seed=rs)

            # model output for sample
            y = model(x[None, ...])

            # Create array of all perturbations of x
            all_x_s0s = np.repeat(x[None, :], len(combs), axis=0)
            # for i, feat_subset in enumerate(combs):
            #     all_x_s0s[i, [*feat_subset]] = 0
            all_x_s0s[np.arange(len(combs))[:, None],
                      [*map(list, combs)]] = 0

            # explain samples and compute attribution sum
            attribs = explain_func(all_x_s0s)
            attrib_sums = np.sum(attribs, axis=1)

            # compute model output for perturbed samples
            all_y_s0s = model(all_x_s0s)
            all_y_diffs = y - all_y_s0s

            # compute PCC
            pccs.append(
                np.corrcoef(all_y_diffs, attrib_sums)
            )
        # append average over all PCCs for this value of n
        mean_pcc = np.mean(pccs)
        if mean_pcc < 0:
            # probably not a problem just a shortcoming of method but let's
            # make user aware...
            warnings.warn(f'Negative PCC in sensitivity_n: {mean_pcc}')
        all_pccs.append(mean_pcc)

    # TODO: compute AUC of all_pccs

    return all_n, all_pccs
