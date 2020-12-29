"""
evaluate_explainers.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import os

# take no risks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import pickle
from glob import glob

from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

import numpy as np

from posthoceval import metrics
from posthoceval.model_generation import AdditiveModel
from posthoceval.utils import assert_same_size
from posthoceval.explainers.local.shap import KernelSHAPExplainer
# Needed for pickle loading of this result type
from posthoceval.results import ExprResult  # noqa


def save_explanation(data, filename):
    np.savez_compressed(filename, data=data)


def run(expr_filename, out_dir, data_dir, max_explain, seed):
    basename_experiment = os.path.basename(expr_filename).rsplit('.', 1)[0]
    # TODO: other explainers...

    explainer_out_dir = os.path.join(out_dir, basename_experiment, 'SHAP')
    os.makedirs(explainer_out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = pickle.load(f)

    print('Loading data')
    data_files = glob(os.path.join(data_dir, '*.npz'))
    assert_same_size(expr_data, data_files, f'data files (in {data_dir})')

    # grab each file ID as integer index
    file_ids = [*map(lambda fn: int(os.path.basename(fn).rsplit('.')[0]),
                     data_files)]
    assert len(file_ids) == len({*file_ids}), 'duplicate data file IDs!'
    assert min(file_ids) == 0, 'file ID index does not start at 0'
    n_results = len(expr_data)
    assert max(file_ids) == (n_results - 1), (
        f'file ID index does not end with {n_results - 1} (number of results)')

    # now that data looks good, just sort the file names so we can zip together
    # with the loaded expression data
    data_files = [fn for _, fn in sorted(zip(file_ids, data_files),
                                         key=lambda id_fn: id_fn[0])]

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, (data_file, expr_result) in tqdm(
                enumerate(zip(data_files, expr_data)), total=n_results):
            # type hint
            expr_result: ExprResult

            tqdm.write('Generating model')
            model = AdditiveModel.from_expr(
                expr=expr_result.expr,
                symbols=expr_result.symbols,
            )

            tqdm.write('Creating explainer')
            explainer = KernelSHAPExplainer(
                model,
                seed=seed,
            )
            tqdm.write(f'Loading data from {data_file}')
            data = np.load(data_file)['data']

            tqdm.write('Fitting explainer')
            explainer.fit(data)

            tqdm.write('Explaining')
            to_explain = data
            if max_explain is not None and max_explain < len(to_explain):
                to_explain = to_explain[:max_explain]
            explanation = explainer.feature_contributions(to_explain)

            # save things in parallel
            tqdm.write('Adding explanation to save queue')
            out_filename = os.path.join(explainer_out_dir, str(i)) + '.npz'
            executor.submit(save_explanation, explanation, out_filename)


if __name__ == '__main__':
    import argparse
    import sys


    def main():
        parser = argparse.ArgumentParser(  # noqa
            description='Generate explainer explanations of models and save '
                        'results to file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            'expr_filename', help='Filename of the expression pickle'
        )
        default_out_dir = os.path.join(
            os.path.dirname(__file__), 'experiment_data', 'explanations')
        parser.add_argument(
            '--out-dir', '-O', default=default_out_dir,
            help='Output directory to save explanations'
        )
        parser.add_argument(
            '--data-dir', '-D',
            help='Data directory where generated data for expr_filename exists'
        )
        parser.add_argument(
            '--max-explain', type=int,
            help='Maximum number of data points to explain per model'
        )
        # parser.add_argument(
        #     '--n-jobs', '-j', default=-1, type=int,
        #     help='Number of jobs to use in generation'
        # )
        parser.add_argument(
            '--seed', default=42, type=int,
            help='Seed for reproducibility'
        )

        args = parser.parse_args()

        data_dir = args.data_dir
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__), 'experiment_data', 'data',
                os.path.basename(args.expr_filename).rsplit('.', 1)[0]
            )
            if not os.path.isdir(data_dir):
                sys.exit(f'Could not infer --data-dir (guessed '
                         f'"{data_dir}". Please supply this argument.')

        run(out_dir=args.out_dir,
            expr_filename=args.expr_filename,
            data_dir=data_dir,
            max_explain=args.max_explain,
            # n_jobs=args.n_jobs,  # TODO: explainers should be parallel...
            seed=args.seed)


    main()
