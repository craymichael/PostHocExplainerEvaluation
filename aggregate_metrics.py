"""
aggregate_metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import os
import sys
import json
import pickle
import traceback
import warnings
from typing import Dict
from typing import Tuple
from functools import partial

from tqdm.auto import tqdm

import numpy as np
import sympy as sp

from posthoceval import metrics
from posthoceval.model_generation import AdditiveModel
from posthoceval.utils import safe_parse_tuple
from posthoceval.utils import is_int
from posthoceval.utils import is_float
# Needed for pickle loading of this result type
from posthoceval.results import ExprResult

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


def compute_true_means(true_expl):
    true_means = {}
    for k, v in true_expl.items():
        # no nans or infs
        try:
            true_means[k] = np.ma.masked_invalid(v).mean()
        except (FloatingPointError, ValueError):
            true_means[k] = np.ma.masked_invalid(
                v.astype(np.float128)).mean().astype(v.dtype)
    return true_means


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, sp.Symbol):
            return obj.name
        if is_int(obj):
            return int(obj)
        if is_float(obj):
            return float(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def compute_metrics(true_expl, pred_expl, n_explained, true_means):
    # per-effect metrics
    per_match_metrics = []

    for name, effect_wise_metric in (
            ('strict_matching', metrics.strict_eval),
            ('generous_matching', metrics.generous_eval),
            ('maybe_exact_matching',
             partial(metrics.generous_eval, maybe_exact=True)),
    ):
        matching, goodness = effect_wise_metric(true_expl, pred_expl)
        matching_results = []
        agg_results = {}

        for match_goodness, (match_true, match_pred) in zip(
                goodness, matching):
            # for each pair in the match
            # the "worse" the match, the more effects will be in match
            # list needed so sum of single effect won't reduce to scalar
            if match_true:
                contribs_true = sum(
                    [true_expl[effect] for effect in match_true])
            else:
                contribs_true = np.zeros(n_explained)
            if match_pred:
                contribs_pred = sum(
                    [pred_expl[effect] for effect in match_pred])
                if true_means is not None:
                    # add the mean back for these effects (this will be the
                    #  same sample mean that the explainer saw before)
                    contribs_pred += sum(
                        [true_means[effect] for effect in match_true])
            else:
                contribs_pred = np.zeros(n_explained)

            # now we evaluate the fidelity with various error metrics!
            err_dict = {}
            for err_name, err_metric in (
                    ('rmse', metrics.rmse),
                    ('mape', metrics.mape),
                    ('mse', metrics.mse),
            ):
                try:
                    err = err_metric(contribs_true, contribs_pred)
                except (FloatingPointError, ValueError):
                    tqdm.write('               isnan isinf')
                    tqdm.write(
                        f'contribs_true: {np.isnan(contribs_true).any()} '
                        f'{np.isinf(contribs_true).any()}')
                    tqdm.write(
                        f'contribs_pred: {np.isnan(contribs_pred).any()} '
                        f'{np.isinf(contribs_pred).any()}')
                    # overflow, probably
                    dtype_orig = contribs_true.dtype
                    err = err_metric(contribs_true.astype(np.float128),
                                     contribs_pred.astype(np.float128))
                    err_low_prec = err.astype(dtype_orig)
                    if np.isinf(err_low_prec):
                        warnings.warn(f'Necessary cast from {dtype_orig} to '
                                      f'{err.dtype} to avoid '
                                      f'FloatingPointError in {err_name}')
                    else:
                        err = err_low_prec

                err_dict[err_name] = err

                err_name_agg = err_name + '_mean'
                agg_results[err_name_agg] = (
                        agg_results.get(err_name_agg, 0.) + err)

            matching_results.append({
                'error': err_dict,
                'true_effects': match_true,
                'pred_effects': match_pred,
                'goodness': match_goodness,
            })

            agg_results['goodness_mean'] = (
                    agg_results.get('goodness_mean', 0.) + match_goodness)

        for err_name_agg, err in agg_results.items():
            agg_results[err_name_agg] = err / len(matching)

        per_match_metrics.append({
            'matching_algorithm': name,
            'all_results': matching_results,
            'agg_results': agg_results,
        })

    # aggregate metrics
    effect_detection_metrics = {
        agg_name: agg_metric(true_expl, pred_expl)
        for agg_name, agg_metric in (
            ('effect_detection_jaccard_index',
             metrics.effect_detection_jaccard_index),
            ('effect_detection_precision', metrics.effect_detection_precision),
            ('effect_detection_recall', metrics.effect_detection_recall),
            ('effect_detection_f1', metrics.effect_detection_f1),
        )
    }

    return {
        'per_match_metrics': per_match_metrics,
        'effect_detection_metrics': effect_detection_metrics,
    }


def compute_true_contributions(expr_result, data_file, explainer_dir, expl_id):
    tqdm.write('Generating model')
    model = AdditiveModel.from_expr(
        expr=expr_result.expr,
        symbols=expr_result.symbols,
    )

    # check if contributions have been saved before
    cached_dir = os.path.join(explainer_dir, TRUE_CONTRIBS_NAME)
    os.makedirs(cached_dir, exist_ok=True)

    cached_path = os.path.join(cached_dir, str(expl_id) + '.npz')
    if os.path.exists(cached_path):
        tqdm.write('Loading pre-computed explanations from disk...')
        contribs = load_explanation(cached_path, model)
        tqdm.write('Loaded.')
    else:
        tqdm.write(f'Generating explanations for {expl_id}')
        tqdm.write(f'Loading data from {data_file}')
        data = np.load(data_file)['data']
        tqdm.write('Done loading.')

        contribs = model.feature_contributions(data)
        tqdm.write('Done explaining.')

        # cache contribs
        save_kwargs = {
            str(effect_symbols): contribution
            for effect_symbols, contribution in contribs.items()
        }
        tqdm.write('Saving to disk.')
        np.savez_compressed(cached_path, **save_kwargs)

        # might not be necessary, but `load_explanation` does this (so do it
        # for consistency)
        contribs = standardize_contributions(contribs)

    effects = model.make_effects_dict()

    return model, contribs, effects


def clean_explanations(
        true_expl: Explanation,
        pred_expl: Explanation,
) -> Tuple[Explanation, Explanation, int]:
    """"""
    tqdm.write('Start cleaning explanations.')

    true_expl = true_expl.copy()
    pred_expl = pred_expl.copy()

    true_lens = {*map(len, true_expl.values())}
    pred_lens = {*map(len, pred_expl.values())}

    assert len(pred_lens) == 1, (
        'pred_expl has effect-wise explanations of non-uniform length')
    assert len(true_lens) == 1, (
        'true_expl has effect-wise explanations of non-uniform length')

    n_pred = pred_lens.pop()
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

    tqdm.write('Discovering true_expl invalids')
    nan_idxs_true = np.zeros(n_pred, dtype=np.bool)
    for v in true_expl.values():
        # yep, guess what - this can also happen...
        nan_idxs_true |= np.isnan(v) | np.isinf(v)

    # isnan or isinf in pred_expl but not true_expl is likely artifact of bad
    # perturbation
    nan_idxs_pred_only = nan_idxs_pred & (~nan_idxs_true)
    if nan_idxs_pred_only.any():
        tqdm.write(f'Pred explanations has {nan_idxs_pred_only.sum()} nans '
                   f'and/or infs that true explanations do not.')

    tqdm.write('Start removing invalids.')
    nan_idxs = nan_idxs_pred | nan_idxs_true
    if nan_idxs.any():
        not_nan = ~nan_idxs
        for k, v in true_expl.items():
            true_expl[k] = v[not_nan]

        for k, v in pred_expl.items():
            pred_expl[k] = v[not_nan]

        total_nan = nan_idxs.sum()
        tqdm.write(f'Removed {total_nan} rows from explanations '
                   f'({100 * total_nan / n_pred:.2}%)')
        n_pred -= total_nan

    tqdm.write('Done cleaning.')

    return true_expl, pred_expl, n_pred


def standardize_contributions(contribs_dict):
    """standardize each effect tuple and remove effects that are 0-effects"""
    return {metrics.standardize_effect(k): v
            for k, v in contribs_dict.items()
            if not np.allclose(v, 0., atol=1e-5)}


def load_explanation(expl_file: str, true_model: AdditiveModel):
    if not os.path.exists(expl_file):
        raise IOError(f'{expl_file} does not exist!')

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


def run(expr_filename, explainer_dir, data_dir, out_dir, debug=False):
    """"""
    np.seterr('raise')  # never trust silent fp in metrics

    expr_basename = os.path.basename(expr_filename).rsplit('.', 1)[0]
    os.makedirs(out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = pickle.load(f)

    true_explanations = {}
    true_effects_all = {}
    true_models = {}

    all_results = []

    for explainer in os.listdir(explainer_dir):
        if explainer == TRUE_CONTRIBS_NAME:
            continue

        explainer_path = os.path.join(explainer_dir, explainer)
        if not os.path.isdir(explainer_path):
            continue

        explanations = os.listdir(explainer_path)
        explained = [*map(lambda x: int(x.rsplit('.', 1)[0]), explanations)]
        assert len(explained) == len({*explained})

        results_explainer = []

        # now compute metrics for each model
        for expl_id in tqdm(explained, desc=explainer):
            tqdm.write(f'\nBegin {expl_id}.')

            expr_result: ExprResult = expr_data[expl_id]

            true_expl = true_explanations.get(expl_id)
            true_effects = true_effects_all.get(expl_id)
            true_model = true_models.get(expl_id)

            if true_expl is None:
                # compute true contributions
                data_file = os.path.join(data_dir, f'{expl_id}.npz')

                # cache result for later use (by other explainers)
                try:
                    true_model, true_expl, true_effects = (
                        compute_true_contributions(expr_result, data_file,
                                                   explainer_dir, expl_id)
                    )
                except Exception as e:
                    e_module = str(getattr(e, '__module__', ''))
                    if e_module.split('.', 1)[0] != 'sympy':
                        # raise non-sympy exceptions...
                        raise

                    tqdm.write(f'Failed to compute feature contribs for '
                               f'{expl_id}!')

                    exc_lines = traceback.format_exception(
                        *sys.exc_info(), limit=None, chain=True)
                    for line in exc_lines:
                        tqdm.write(str(line), file=sys.stderr, end='')

                    continue

                true_explanations[expl_id] = true_expl
                true_effects_all[expl_id] = true_effects

                true_models[expl_id] = true_model

            tqdm.write('Loading predicted explanation')
            pred_expl_file = os.path.join(explainer_path, f'{expl_id}.npz')
            pred_expl = load_explanation(pred_expl_file, true_model)

            # check if explanation should be uncentered
            true_means = None
            if is_mean_centered(explainer):
                true_means = compute_true_means(true_expl)

            true_expl, pred_expl, n_explained = (
                clean_explanations(true_expl, pred_expl))

            if n_explained == 0:
                tqdm.write(f'Skipping {expl_id} as all instance explanations '
                           f'by {explainer} contain nans')
                continue

            tqdm.write('Begin computing metrics.')
            results = compute_metrics(true_expl, pred_expl, n_explained,
                                      true_means)
            results['model_kwargs'] = expr_result.kwargs
            results['effects'] = [
                {'symbols': effect_symbols,
                 'effect': sp.latex(true_effects[effect_symbols])}
                for effect_symbols in true_expl
            ]
            results['all_symbols'] = expr_result.symbols
            results['expl_id'] = expl_id

            results_explainer.append(results)
            tqdm.write('Done.')

            if debug:  # run once for one explainer
                break

        all_results.append({
            'explainer': explainer,
            'results': results_explainer,
        })

        if debug:  # run once for one explainer
            break

    # Save to out_dir
    out_filename = os.path.join(out_dir, expr_basename + '.json')
    print('Writing results to', out_filename)

    exists_count = 0
    while os.path.isfile(out_filename):
        exists_count += 1
        new_out_filename = os.path.join(
            out_dir, expr_basename + f'_{exists_count}.json')
        print(f'{out_filename} exists, will write to '
              f'{new_out_filename} instead')
        out_filename = new_out_filename

    with open(out_filename, 'w') as f:
        json.dump(all_results, f, cls=CustomJSONEncoder)


if __name__ == '__main__':
    import argparse


    def main():
        parser = argparse.ArgumentParser(  # noqa
            description='Compute metrics on previously produced explanations',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            'expr_filename', help='Filename of the expression pickle'
        )
        parser.add_argument(
            '--explainer-dir', '-E',
            help='Directory where generated explanations for expr_filename '
                 'exist'
        )
        parser.add_argument(
            '--data-dir', '-D',
            help='Data directory where generated data for expr_filename exist'
        )
        parser.add_argument(
            '--out-dir', '-O',
            help='Output directory to save metrics'
        )
        parser.add_argument(  # hidden debug argument
            '--debug', action='store_true',
            help=argparse.SUPPRESS
        )

        args = parser.parse_args()

        explainer_dir = args.explainer_dir
        data_dir = args.data_dir
        out_dir = args.out_dir

        err_msg = ('Could not infer --{arg} (guessed "{val}"). Please supply '
                   'this argument.')

        expr_basename = os.path.basename(args.expr_filename).rsplit('.', 1)[0]
        experiment_dir = os.path.dirname(args.expr_filename)
        if os.path.basename(experiment_dir) == 'expr':
            experiment_dir = os.path.dirname(experiment_dir)

        if explainer_dir is None:
            explainer_dir = os.path.join(
                experiment_dir, 'explanations', expr_basename)
            if not os.path.isdir(explainer_dir):
                sys.exit(err_msg.format(arg='explainer-dir',
                                        val=explainer_dir))
        if data_dir is None:
            data_dir = os.path.join(experiment_dir, 'data', expr_basename)
            if not os.path.isdir(data_dir):
                sys.exit(err_msg.format(arg='data-dir', val=data_dir))

        if out_dir is None:
            out_dir = os.path.join(experiment_dir, 'metrics', expr_basename)

        run(out_dir=out_dir,
            expr_filename=args.expr_filename,
            explainer_dir=explainer_dir,
            data_dir=data_dir,
            debug=args.debug)


    main()
