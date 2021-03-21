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


def sensitivity_n(model, attribs, X, n_subsets=100, max_feats=0.8,
                  n_samples=None, n_points=None, aggregate=True,
                  seed=None, verbose=False):
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
    assert len(X) == len(attribs)

    n_feats = X.shape[1]
    feats = np.arange(n_feats)

    if n_points is None:
        n_points = min(16, n_feats)
    if n_samples is None:
        n_samples = min(1000, len(X))

    rs = as_random_state(seed)

    sample_idxs = rs.choice(np.arange(len(X)), n_samples, replace=False)
    X_eval = X[sample_idxs]
    attribs_eval = attribs[sample_idxs]

    # try to include at least two values of n
    max_n = max(n_feats * max_feats, min(2, n_feats))
    all_n = np.unique(np.round(
        np.linspace(1, max_n, n_points)).astype(int))

    # pearson corr coefs
    all_pccs = []

    # gather all explanations
    if verbose:
        tqdm.write('Calling model with evaluation data')
    y_eval = model(X_eval)

    pbar_n = tqdm(enumerate(all_n), desc='N', disable=not verbose, position=0)
    pbar_x = tqdm(total=len(X_eval), desc='X', disable=not verbose, position=1)

    bad_n = []

    for n_idx, n in pbar_n:
        pbar_n.set_description(f'N={n}')

        pccs = []

        pbar_x.reset()
        for x_i, y_i, attrib_i in zip(X_eval, y_eval, attribs_eval):
            pbar_x.update()

            # TODO: descriptions only here for debug - update less....
            pbar_x.set_description('Select combinations')

            max_combs = comb(n_feats, n, exact=True)
            n_subsets_n = min(max_combs, n_subsets)
            # corr not defined for <2 points...
            if n_subsets_n < 2:
                continue
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

            attrib_sum_subset = attrib_i[idx_feats.ravel()].reshape(-1, n)
            attrib_sum_subset = attrib_sum_subset.sum(axis=1)

            # compute model output for perturbed samples
            pbar_x.set_description('Call model (permuted)')

            all_y_s0s = model(all_x_s0s)
            invalid_mask = np.isnan(all_y_s0s) | np.isinf(all_y_s0s)

            if len(invalid_mask):
                if len(invalid_mask) >= (len(all_y_s0s) - 1):
                    continue

                all_y_s0s = all_y_s0s[~invalid_mask]
                attrib_sum_subset = attrib_sum_subset[~invalid_mask]

            all_y_diffs = y_i - all_y_s0s

            # compute PCC
            pccs.append(
                np.corrcoef(all_y_diffs, attrib_sum_subset)[0, 1]
            )
        pbar_x.refresh()

        if not len(pccs):
            bad_n.append(n_idx)

        # append average over all PCCs for this value of n
        mean_pcc = np.mean(pccs)
        all_pccs.append(mean_pcc)
    pbar_x.close()

    if len(bad_n):
        all_n = np.delete(all_n, bad_n)

    if aggregate:  # AUC
        if not len(all_pccs):
            return np.nan
        if len(all_pccs) == 1:
            return all_pccs[0]
        # all_n[-1] is max, 1 is min
        return np.trapz(x=all_n, y=all_pccs) / (all_n[-1] - 1)
    else:
        return all_n, all_pccs
