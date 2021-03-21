"""
expl_utils.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import os
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Union

import numpy as np
import sympy as sp
from tqdm.auto import tqdm

from posthoceval.metrics import standardize_contributions
from posthoceval.model_generation import AdditiveModel
from posthoceval.utils import safe_parse_tuple

Explanation = Dict[Tuple[sp.Symbol], np.ndarray]

# reserved name for true contributions
TRUE_CONTRIBS_NAME = '__true__'
# TODO: best way to do this?
# known explainers that give mean-centered contributions in explanation
KNOWN_MEAN_CENTERED = [
    'SHAP',
]


def is_mean_centered(explainer):
    return any(explainer.startswith(e) for e in KNOWN_MEAN_CENTERED)


def clean_explanations(
        pred_expl: Explanation,
        true_expl: Optional[Explanation] = None,
) -> Union[Tuple[Explanation, Explanation, int],
           Tuple[Explanation, int]]:
    """"""
    tqdm.write('Start cleaning explanations.')

    pred_expl = pred_expl.copy()
    pred_lens = {*map(len, pred_expl.values())}
    assert len(pred_lens) == 1, (
        'pred_expl has effect-wise explanations of non-uniform length')
    n_pred = pred_lens.pop()

    with_true = true_expl is not None
    if with_true:
        true_expl = true_expl.copy()
        true_lens = {*map(len, true_expl.values())}
        assert len(true_lens) == 1, (
            'true_expl has effect-wise explanations of non-uniform length')
        n_true = true_lens.pop()

        assert n_pred <= n_true, f'n_pred ({n_pred}) > n_true ({n_true})'

        if n_pred < n_true:
            tqdm.write(f'Truncating true_expl from {n_true} to {n_pred}')
            # truncate latter explanations to save length
            for k, v in true_expl.items():
                true_expl[k] = v[:n_pred]

    tqdm.write('Discovering pred_expl invalids')
    nan_idxs_pred = np.zeros(n_pred, dtype=np.bool)
    for v in pred_expl.values():
        nan_idxs_pred |= np.isnan(v) | np.isinf(v)

    if with_true:
        tqdm.write('Discovering true_expl invalids')
        nan_idxs_true = np.zeros(n_pred, dtype=np.bool)
        for v in true_expl.values():
            # yep, guess what - this can also happen...
            nan_idxs_true |= np.isnan(v) | np.isinf(v)

        # isnan or isinf in pred_expl but not true_expl is likely artifact of
        #  bad perturbation
        nan_idxs_pred_only = nan_idxs_pred & (~nan_idxs_true)
        if nan_idxs_pred_only.any():
            tqdm.write(f'Pred explanations has {nan_idxs_pred_only.sum()} '
                       f'nans and/or infs that true explanations do not.')
        nan_idxs = nan_idxs_pred | nan_idxs_true
    else:
        nan_idxs = nan_idxs_pred

    tqdm.write('Start removing invalids.')
    if nan_idxs.any():
        not_nan = ~nan_idxs

        for k, v in pred_expl.items():
            pred_expl[k] = v[not_nan]

        if with_true:
            for k, v in true_expl.items():
                true_expl[k] = v[not_nan]

        total_nan = nan_idxs.sum()
        tqdm.write(f'Removed {total_nan} rows from explanations '
                   f'({100 * total_nan / n_pred:.2f}%)')
        n_pred -= total_nan

    tqdm.write('Done cleaning.')

    if with_true:
        return pred_expl, true_expl, n_pred
    return pred_expl, n_pred


def load_explanation(expl_file: str, true_model: AdditiveModel):
    # TODO: intercept loading...
    if not os.path.exists(expl_file):
        raise FileNotFoundError(f'{expl_file} does not exist!')

    expl_dict = np.load(expl_file)

    if len(expl_dict) == 1 and 'data' in expl_dict:
        expl = expl_dict['data']

        assert expl.shape[1] == len(true_model.symbols), (
            f'Non-keyword explanation received with '
            f'{expl.shape[1]} features but model has '
            f'{len(true_model.symbols)} features.'
        )
        # map to model symbols and standardize
        expl = dict(zip(true_model.symbols, expl.T))
    else:
        expl = {}
        for symbols_str, expl_data in expl_dict.items():
            try:
                symbol_strs = safe_parse_tuple(symbols_str)
            except AssertionError:  # not a tuple...tisk tisk
                symbol_strs = (symbols_str,)
            # convert strings to symbols in model
            symbols = tuple(map(true_model.get_symbol, symbol_strs))

            expl[symbols] = expl_data

    expl = standardize_contributions(expl)

    return expl