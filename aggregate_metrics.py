

import os
import sys
import json
import traceback

from tqdm.auto import tqdm

import numpy as np
import sympy as sp

from joblib import Parallel
from joblib import delayed

from posthoceval.expl_utils import TRUE_CONTRIBS_NAME, standardize_contributions
from posthoceval.expl_utils import is_mean_centered
from posthoceval.expl_utils import clean_explanations
from posthoceval.expl_utils import load_explanation
from posthoceval.expl_utils import save_explanation
from posthoceval.expl_utils import CompatUnpickler
from posthoceval import metrics
from posthoceval.models.synthetic import SyntheticModel
from posthoceval.utils import tqdm_parallel, CustomJSONEncoder
from posthoceval.utils import at_high_precision
from posthoceval.utils import atomic_write_exclusive
from posthoceval.results import ExprResult


def compute_true_means(true_expl):
    true_means = {}
    for k, v in true_expl.items():
        
        true_means[k] = at_high_precision(np.mean, np.ma.masked_invalid(v))
    return true_means


def compute_metrics(true_expl, pred_expl, n_explained, true_means):
    
    per_match_metrics = []

    
    
    for name, effect_wise_metric in (
            
            ('generous_matching', metrics.generous_eval),
            
            
    ):
        matching, goodness = effect_wise_metric(true_expl, pred_expl)
        matching_results = []
        agg_results = {}

        contribs_true_all = []
        contribs_pred_all = []

        for match_goodness, (match_true, match_pred) in zip(
                goodness, matching):
            
            
            
            if match_true:
                contribs_true = sum(
                    [true_expl[effect] for effect in match_true])
            else:
                contribs_true = np.zeros(n_explained)
            contribs_true_all.append(contribs_true)

            if match_pred:
                contribs_pred = sum(
                    [pred_expl[effect] for effect in match_pred])
                if true_means is not None:
                    
                    
                    contribs_pred += sum(
                        [true_means[effect] for effect in match_true])
            else:
                contribs_pred = np.zeros(n_explained)
            contribs_pred_all.append(contribs_pred)

            
            err_dict = {}
            for err_name, err_metric in (
                    ('rmse', metrics.rmse),
                    ('mape', metrics.mape),
                    ('mse', metrics.mse),
                    ('nrmse_std', metrics.nrmse_std),
                    ('nrmse_range', metrics.nrmse_range),
                    ('nrmse_interquartile', metrics.nrmse_interquartile),
                    ('nrmse_mean', metrics.nrmse_mean),
            ):
                try:
                    err = at_high_precision(err_metric,
                                            contribs_true, contribs_pred)
                except ValueError:
                    tqdm.write('               isnan isinf')
                    tqdm.write(
                        f'contribs_true: {np.isnan(contribs_true).any()} '
                        f'{np.isinf(contribs_true).any()}')
                    tqdm.write(
                        f'contribs_pred: {np.isnan(contribs_pred).any()} '
                        f'{np.isinf(contribs_pred).any()}')
                    raise

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

        
        contribs_true_all = np.stack(contribs_true_all, axis=1)
        contribs_pred_all = np.stack(contribs_pred_all, axis=1)

        for err_name, err_metric in (
                ('cosine_distances', metrics.cosine_distances),
                ('euclidean_distances', metrics.euclidean_distances),
        ):
            distances = err_metric(contribs_true_all, contribs_pred_all)
            agg_results[err_name + '_mean'] = distances.mean()

        per_match_metrics.append({
            'matching_algorithm': name,
            'all_results': matching_results,
            'agg_results': agg_results,
        })

    
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
    model = SyntheticModel.from_expr(
        expr=expr_result.expr,
        symbols=expr_result.symbols,
    )

    
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

        
        tqdm.write('Saving to disk.')
        save_explanation(cached_path, contribs)

        
        
        contribs = standardize_contributions(contribs)

    effects = model.make_effects_dict()

    return model, contribs, effects


def run(expr_filename, explainer_dir, data_dir, out_dir, debug=False,
        n_jobs=1):
    
    np.seterr('raise')  

    expr_basename = os.path.basename(expr_filename).rsplit('.', 1)[0]
    os.makedirs(out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = CompatUnpickler(f).load()

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

        def run_one(expl_id, expr_result: ExprResult, true_expl, true_effects,
                    true_model):
            tqdm.write(f'\nBegin {expl_id}.')

            true_expl_orig = true_expl
            if true_expl is None:
                
                data_file = os.path.join(data_dir, f'{expl_id}.npz')

                
                try:
                    true_model, true_expl, true_effects = (
                        compute_true_contributions(expr_result, data_file,
                                                   explainer_dir, expl_id)
                    )
                    true_expl_orig = true_expl
                except Exception as e:
                    e_module = str(getattr(e, '__module__', ''))
                    if e_module.split('.', 1)[0] != 'sympy':
                        
                        raise

                    tqdm.write(f'Failed to compute feature contribs for '
                               f'{expl_id}!')

                    exc_lines = traceback.format_exception(
                        *sys.exc_info(), limit=None, chain=True)
                    for line in exc_lines:
                        tqdm.write(str(line), file=sys.stderr, end='')

                    return None

            tqdm.write('Loading predicted explanation')
            pred_expl_file = os.path.join(explainer_path, f'{expl_id}.npz')
            pred_expl = load_explanation(pred_expl_file, true_model)

            
            true_means = None
            if is_mean_centered(explainer):
                true_means = compute_true_means(true_expl)

            pred_expl, true_expl, n_explained = (
                clean_explanations(pred_expl, true_expl))

            if n_explained == 0:
                tqdm.write(f'Skipping {expl_id} as all instance explanations '
                           f'by {explainer} contain nans')
                return None

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

            tqdm.write('Done.')

            return results, expl_id, true_expl_orig, true_effects, true_model

        if debug:  
            explained = explained[:1]

        jobs = (
            delayed(run_one)(
                expl_id=expl_id,
                expr_result=expr_data[expl_id],
                true_expl=true_explanations.get(expl_id),
                true_effects=true_effects_all.get(expl_id),
                true_model=true_models.get(expl_id))
            for expl_id in explained
        )

        with tqdm_parallel(tqdm(desc=explainer, total=len(explained))):
            if n_jobs == 1 or debug:
                
                packed_results = [f(*a, **kw) for f, a, kw in jobs]
            else:
                packed_results = Parallel(n_jobs=n_jobs)(jobs)

        
        explainer_results = []
        for packed_result in packed_results:
            if packed_result is None:
                continue

            
            (results, expl_id, true_expl,
             true_effects, true_model) = packed_result

            
            true_explanations[expl_id] = true_expl
            true_effects_all[expl_id] = true_effects

            true_models[expl_id] = true_model

            explainer_results.append(results)

        all_results.append({
            'explainer': explainer,
            'results': explainer_results,
        })

    
    out_filename = os.path.join(out_dir, expr_basename + '.json')
    print('Writing results to', out_filename)

    out_filename_actual = atomic_write_exclusive(
        preferred_filename=out_filename,
        data=json.dumps(all_results, cls=CustomJSONEncoder),
    )
    if out_filename_actual != out_filename:
        print('Actually wrote results to', out_filename_actual)


if __name__ == '__main__':
    import argparse


    def main():
        parser = argparse.ArgumentParser(  
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
        parser.add_argument(
            '--n-jobs', '-j', default=-1, type=int,
            help='Number of jobs to use in generation'
        )
        parser.add_argument(  
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
            n_jobs=args.n_jobs,
            debug=args.debug)


    main()
