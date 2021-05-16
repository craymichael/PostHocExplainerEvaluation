"""
expl_utils.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Union
from typing import Any

import os

import pickle

import numpy as np
import sympy as sp
from tqdm.auto import tqdm

from posthoceval.models.model import AdditiveModel
from posthoceval.utils import safe_parse_tuple
from posthoceval.utils import assert_same_size

# TODO: sp.Symbol --> Any
Explanation = Dict[Tuple[sp.Symbol], np.ndarray]

# reserved name for true contributions
TRUE_CONTRIBS_NAME = '__true__'
# known explainers that give mean-centered contributions in explanation
KNOWN_MEAN_CENTERED = [
    'SHAP',
    'SHAPR',
]


def is_mean_centered(explainer):
    # TODO: startswith can be dangerous here...maybe underscore split [0] check
    return any(explainer.upper().startswith(e) for e in KNOWN_MEAN_CENTERED)


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


def load_explanation(expl_path: str, true_model: AdditiveModel):
    # TODO: intercept loading...
    if not os.path.exists(expl_path):
        raise FileNotFoundError(f'{expl_path} does not exist!')

    expl_dict = np.load(expl_path)

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


def save_explanation(expl_path: str, contribs):
    if isinstance(contribs, list):
        raise NotImplementedError('Explanations for classification tasks.')

    if isinstance(contribs, dict):
        save_kwargs = {
            str(effect_symbols): contribution
            for effect_symbols, contribution in contribs.items()
        }
        np.savez_compressed(expl_path, **save_kwargs)
    elif isinstance(contribs, np.ndarray):
        np.savez_compressed(expl_path, data=contribs)
    else:
        raise TypeError(f'invalid explanation type: {type(contribs)}')


def apply_matching(matching, true_expl, pred_expl, n_explained,
                   explainer_name, always_numeric=True):
    matches = {}
    for match_true, match_pred in matching:
        if match_true:
            contribs_true = sum(true_expl[effect] for effect in match_true)
            contribs_true_mean = np.mean(contribs_true)
        else:  # no corresponding effect(s) from truth
            contribs_true_mean = np.zeros(n_explained)
            contribs_true = contribs_true_mean if always_numeric else None
        if match_pred:
            # add the mean back for these effects (this will be the
            #  same sample mean that the explainer saw before)
            contribs_pred = sum(
                (pred_expl[effect] for effect in match_pred),
                (contribs_true_mean  # start
                 if is_mean_centered(explainer_name) else 0)
            )
        else:  # no corresponding effect(s) from pred
            contribs_pred = np.zeros(n_explained) if always_numeric else None

        match_key = (tuple(match_true), tuple(match_pred))
        matches[match_key] = (contribs_true, contribs_pred)

    return matches


def standardize_effect(e):
    """sorted by str for consistency"""
    e = tuple(sorted({*e}, key=str)) if isinstance(e, (tuple, list)) else (e,)
    assert e, 'received empty effect'
    return e


def standardize_contributions(contribs_dict, remove_zeros=True, atol=1e-5):
    """standardize each effect tuple and remove effects that are 0-effects"""
    contribs_std = {}
    n_zeros = 0
    for k, v in contribs_dict.items():
        if remove_zeros and np.allclose(v, 0., atol=atol):
            n_zeros += 1
        else:
            contribs_std[standardize_effect(k)] = v
    # Future: default behavior just add dupes up? that could be janky...
    tot_contribs = len(contribs_std) + n_zeros
    assert_same_size(contribs_dict, tot_contribs, 'contributions',
                     extra='This is because there were duplicate effects in '
                           'the input.')
    return contribs_std


class CompatUnpickler(pickle.Unpickler):

    def find_class(self, module_name: str, global_name: str) -> Any:
        if module_name == '__main__' and global_name == 'ExprResult':
            from posthoceval.results import ExprResult
            return ExprResult

        if (module_name == 'posthoceval.model_generation' and
                global_name == 'AdditiveModel'):
            from posthoceval.models.synthetic import SyntheticModel
            return SyntheticModel

        return super().find_class(module_name, global_name)
