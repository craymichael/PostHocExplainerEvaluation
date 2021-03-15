"""
metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import warnings

import numpy as np
from scipy.special import comb

from tqdm.auto import tqdm

from posthoceval.rand import select_n_combinations
from posthoceval.rand import as_random_state


def sensitivity_n(model, explain_func, X, n_subsets=100, max_feats=0.8,
                  n_samples=None, n_points=None, seed=None, verbose=True):
    """
    Pros:
    - can quantify the number of features attribution methods are capable of
      identifying "correctly"
    - objectively gives measure of goodness in terms of target variation, it
      at least weeds out completely off attributions
    Cons:
    - only works on feature attributions, so no interactions or other types of
      explanations
    - assumes a removed value is zero, not truly removed, 0 may not be best
      choice
    - assumes variation is linear with attribution, but attributions assume
      local linearity, locality can be breached by setting features to 0

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

    # gather all explanations
    attribs = explain_func(X_eval)
    y = model(X_eval)

    pbar_n = tqdm(all_n, desc='N', disable=not verbose, position=0)
    for n in pbar_n:
        pbar_n.set_description(f'N={n}')

        pccs = []

        pbar_x = tqdm(zip(X_eval, y, attribs),
                      desc='X', disable=not verbose, position=1)
        for x_i, y_i, attrib_i in pbar_x:
            # TODO: descriptions only here for debug - update less....
            pbar_x.set_description('Select combinations')

            max_combs = comb(n_feats, n, exact=True)
            n_subsets_n = min(max_combs, n_subsets)
            combs = select_n_combinations(feats, n, n_subsets_n, seed=rs)

            # model output for sample
            pbar_x.set_description('Call model')

            # Create array of all perturbations of x
            pbar_x.set_description('Permute x')

            all_x_s0s = np.repeat(x_i[None, :], len(combs), axis=0)
            idx_rows = np.arange(len(combs))[:, None]
            idx_feats = np.asarray(combs)
            all_x_s0s[idx_rows, idx_feats] = 0

            # explain samples and compute attribution sum
            pbar_x.set_description('Explain')

            attrib_sum_subset = attrib_i[idx_feats.ravel()]
            attrib_sum_subset = attrib_sum_subset.reshape(-1, n)
            attrib_sum_subset = attrib_sum_subset.sum(axis=1)

            # compute model output for perturbed samples
            pbar_x.set_description('Call model (permuted)')

            all_y_s0s = model(all_x_s0s)
            all_y_diffs = y_i - all_y_s0s

            # compute PCC
            pccs.append(
                np.corrcoef(all_y_diffs, attrib_sum_subset)
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
