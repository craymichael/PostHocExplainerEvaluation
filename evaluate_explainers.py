"""
evaluate_explainers.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import os

import pickle

from tqdm.auto import tqdm

from posthoceval import metrics
from posthoceval.model_generation import AdditiveModel
from posthoceval.explainers.local.shap import KernelSHAPExplainer


def run(expr_filename, out_dir, data_dir, seed):
    os.makedirs(out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = pickle.load(f)

    model = AdditiveModel.from_expr()
    KernelSHAPExplainer()


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
        # parser.add_argument(
        #     '--n-jobs', '-j', default=-1, type=int,
        #     help='Number of jobs to use in generation'
        # )
        parser.add_argument(
            '--seed', default=42, type=int,
            help='Seed for reproducibility. Technically the starting seed '
                 'from which each seed is derived per job'
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
            # n_jobs=args.n_jobs,  # TODO: explainers should be parallel...
            seed=args.seed)


    main()
