#!/usr/bin/env python
"""
aggregate_metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import os
import sys
import json

from tqdm.auto import tqdm

import numpy as np

from joblib import Parallel
from joblib import delayed

from posthoceval.expl_utils import TRUE_CONTRIBS_NAME
from posthoceval.expl_utils import clean_explanations
from posthoceval.expl_utils import load_explanation
from posthoceval.expl_utils import CompatUnpickler
from posthoceval.utils import tqdm_parallel
from posthoceval.utils import CustomJSONEncoder
from posthoceval.utils import atomic_write_exclusive
from posthoceval.expl_eval import metrics
from posthoceval.models.synthetic import SyntheticModel
from posthoceval.results import ExprResult


def compute_metrics(model, data, pred_expl, n_explained, metric_names=None):
    if metric_names is not None:
        metric_names = [mn.lower() for mn in metric_names]

    results = {}

    # ensure explanation is compatible with metric
    assert all(len(s) == 1 and s[0] in model.symbols
               for s in pred_expl.keys()), pred_expl.keys()

    # convert to ndarray
    expl_cols = []
    for s in model.symbols:
        expl_cols.append(
            pred_expl.get((s,), np.zeros(n_explained))
        )

    expl = np.stack(expl_cols, axis=1)

    for name, metric in (
            ('sensitivity-n', metrics.sensitivity_n),
            ('faithfulness_melis', metrics.faithfulness_melis),
    ):
        if metric_names and name.lower() not in metric_names:
            continue

        # try:
        ret = metric(
            model, expl, data
        )
        # except FloatingPointError as e:
        #     # this is caused by e.g. sensitivity-n with models that don't have
        #     #  zero in input domain
        #     if 'divide by zero' in e.args[0]:
        #         continue
        #     else:
        #         raise

        results[name] = ret

    return results


def run(expr_filename, explainer_dir, data_dir, out_dir,
        explainer_names=None, metric_names=None, debug=False, n_jobs=1):
    """"""
    np.seterr('raise')  # never trust silent fp in metrics

    expr_basename = os.path.basename(expr_filename).rsplit('.', 1)[0]
    os.makedirs(out_dir, exist_ok=True)

    tqdm.write(f'Loading {expr_filename}, (this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = CompatUnpickler(f).load()

    all_results = []

    if explainer_names is not None:
        explainer_names = [en.lower() for en in explainer_names]

    for explainer in os.listdir(explainer_dir):
        # skip true contributions directory
        if (explainer == TRUE_CONTRIBS_NAME or
                (explainer_names is not None and
                 explainer.lower() not in explainer_names)):
            continue

        # skip files
        explainer_path = os.path.join(explainer_dir, explainer)
        if not os.path.isdir(explainer_path):
            continue

        explanations = os.listdir(explainer_path)
        explained = [*map(lambda x: int(x.rsplit('.', 1)[0]), explanations)]
        assert len(explained) == len({*explained})

        def run_one(expl_id, expr_result: ExprResult):
            # TODO: idk if this is necessary here or in main is ok
            np.seterr('raise')  # never trust silent fp in metrics

            tqdm.write(f'\nBegin {expl_id}.')
            tqdm.write('Loading predicted explanation')
            pred_expl_file = os.path.join(explainer_path, f'{expl_id}.npz')

            model = SyntheticModel.from_expr(
                expr=expr_result.expr,
                symbols=expr_result.symbols,
            )

            data_file = os.path.join(data_dir, f'{expl_id}.npz')
            tqdm.write(f'Loading data from {data_file}')
            data = np.load(data_file)['data']
            tqdm.write('Done loading.')

            pred_expl = load_explanation(pred_expl_file, model)
            pred_expl, n_explained = clean_explanations(pred_expl)
            # truncate data if applicable
            data = data[:n_explained]

            if n_explained == 0:
                tqdm.write(f'Skipping {expl_id} as all instance explanations '
                           f'by {explainer} contain nans')
                return None

            tqdm.write('Begin computing metrics.')
            results = compute_metrics(model, data, pred_expl, n_explained,
                                      metric_names=metric_names)
            results['model_kwargs'] = expr_result.kwargs
            results['all_symbols'] = expr_result.symbols
            results['expl_id'] = expl_id

            tqdm.write('Done.')

            return results

        if debug:  # debug --> limit to processing of 1 explanation
            explained = explained[:1]

        jobs = (
            delayed(run_one)(
                expl_id=expl_id,
                expr_result=expr_data[expl_id],
            ) for expl_id in explained
        )

        with tqdm_parallel(tqdm(desc=explainer, total=len(explained))) as pbar:
            if n_jobs == 1 or debug:
                results = []
                for f, a, kw in jobs:
                    results.append(f(*a, **kw))
                    pbar.update()
            else:
                results = Parallel(n_jobs=n_jobs)(jobs)

        # now compute metrics for each model
        explainer_results = []
        for result in results:
            if result is None:
                continue

            explainer_results.append(result)

        all_results.append({
            'explainer': explainer,
            'results': explainer_results,
        })

        if debug:  # run once for one explainer
            break

    # Save to out_dir
    out_filename = os.path.join(out_dir, expr_basename + '.json')
    print('Writing results to', out_filename)

    atomic_write_exclusive(
        preferred_filename=out_filename,
        data=json.dumps(all_results, cls=CustomJSONEncoder),
    )


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
        parser.add_argument(
            '--explainers', '-X', default=None, nargs='+',
            help='The explainers to evaluate (evaluate all if not provided)'
        )
        parser.add_argument(
            '--metrics', '-M', default=None, nargs='+',
            help='The metrics to evaluate (evaluate all if not provided)'
        )
        parser.add_argument(
            '--n-jobs', '-j', default=-1, type=int,
            help='Number of jobs to use in generation'
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
            out_dir = os.path.join(experiment_dir, 'metrics_alt',
                                   expr_basename)

        run(out_dir=out_dir,
            expr_filename=args.expr_filename,
            explainer_dir=explainer_dir,
            data_dir=data_dir,
            n_jobs=args.n_jobs,
            explainer_names=args.explainers,
            metric_names=args.metrics,
            debug=args.debug)


    main()
