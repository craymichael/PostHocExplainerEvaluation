"""
aggregate_metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import os
import sys

import pickle
from functools import partial

from tqdm.auto import tqdm

import numpy as np

from posthoceval import metrics
from posthoceval.model_generation import AdditiveModel
# Needed for pickle loading of this result type
from posthoceval.results import ExprResult  # noqa


def compute_metrics(true_expl, pred_expl):
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
            contribs_true = sum(true_expl[effect] for effect in match_true)
            contribs_pred = sum(pred_expl[effect] for effect in match_pred)

            # now we evaluate the fidelity with various error metrics!
            err_dict = {}
            for err_name, err_metric in (
                    ('rmse', metrics.rmse),
                    ('mape', metrics.mape),
                    ('mse', metrics.mse),
            ):
                err = err_metric(contribs_true, contribs_pred)
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
    metric_dict = {}
    for agg_name, agg_metric in (
            ('effect_detection_jaccard_index',
             metrics.effect_detection_jaccard_index),
            ('effect_detection_precision', metrics.effect_detection_precision),
            ('effect_detection_recall', metrics.effect_detection_recall),
            ('effect_detection_f1', metrics.effect_detection_f1),
    ):
        pass


def compute_true_contributions(expr_result, data_file):
    tqdm.write('Generating model')
    model = AdditiveModel.from_expr(
        expr=expr_result.expr,
        symbols=expr_result.symbols,
    )

    tqdm.write(f'Loading data from {data_file}')
    data = np.load(data_file)['data']

    return model.feature_contributions(data, return_effects=True)


def run(expr_filename, explainer_dir, data_dir, out_dir):
    """"""
    basename_experiment = os.path.basename(expr_filename).rsplit('.', 1)[0]

    os.makedirs(out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = pickle.load(f)

    true_explanations = {}

    for explainer in os.listdir(explainer_dir):
        explainer_path = os.path.join(explainer_dir, explainer)
        if not os.path.isdir(explainer_path):
            continue

        explanations = os.listdir(explainer_path)
        explained = [*map(lambda x: int(x.rsplit('.', 1)[0]), explanations)]
        assert len(explained) == len(*explained)

        # now compute metrics for each model
        for expl_id in tqdm(explained, desc=explainer):
            true_expl = true_explanations.get(expl_id)
            if true_expl is None:
                # compute true contributions
                expr_result = expr_data[expl_id]
                data_file = os.path.join(data_dir, f'{expl_id}.npz')

                # cache result for later use (by other explainers)
                # note that these are defaultdicts, be careful...
                true_expl, true_effects = compute_true_contributions(
                    expr_result, data_file)
                true_explanations[expl_id] = true_expl

                # TODO: save true_expl + true_effects

            pred_expl_file = os.path.join(explainer_path, f'{expl_id}.npz')
            if not os.path.exists(pred_expl_file):
                raise IOError(f'{pred_expl_file} does not exist!')

            pred_expl_dict = np.load(pred_expl_file)
            if len(pred_expl_dict) == 1 and 'data' in pred_expl_dict:
                pred_expl = pred_expl_dict['data']
                # TODO: convert to dict based on ordering in model...fukc
                # TODO: and make sure to `standardize_effect(...)`
            else:
                # TODO: esp. in the case of interactions...
                raise NotImplementedError('Multiple-data in .npz file not '
                                          'supported yet')

            compute_metrics(true_expl, pred_expl)


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

        args = parser.parse_args()

        explainer_dir = args.explainer_dir
        data_dir = args.data_dir
        out_dir = args.out_dir

        err_msg = ('Could not infer --{arg} (guessed "{val}". Please supply '
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
            data_dir=data_dir)


    main()
