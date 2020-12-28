"""
run.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import os
import pickle

from math import sqrt
from collections import namedtuple

from joblib import Parallel
from joblib import delayed

from tqdm.auto import tqdm

from sympy import stats
import numpy as np

from posthoceval.data_generation import sample
from posthoceval.utils import tqdm_parallel

# Needed for pickle loading of this result type
ExprResult = namedtuple('ExprResult',
                        'symbols,expr,domains,state,kwargs')


def generate_data(out_filename, symbols, domains, n_samples, seed):
    if os.path.isfile(out_filename):
        return

    U = stats.Uniform('U', -1, +1)  # TODO: hard-coded...

    a = sample(
        variables=symbols,
        distribution=U,
        n_samples=n_samples,
        constraints={v: k.contains(U) for v, k in domains.items()},
        seed=seed,
    )
    np.savez_compressed(out_filename, data=a)


def run(out_dir, expr_filename, n_samples, scale_samples, n_jobs, seed):
    out_dir_full = os.path.join(
        out_dir, os.path.dirname(expr_filename).rsplit('.', 1)[0])
    os.makedirs(out_dir_full, exist_ok=True)

    print('Loading', expr_filename)
    with open(expr_filename, 'rb') as f:
        expr_data = pickle.load(f)

    print('Will save compressed numpy arrays to', out_dir_full)

    n_expr = len(expr_data)

    with tqdm_parallel(tqdm(desc='Data Generation', total=n_expr)) as pbar:

        def jobs():
            nonlocal seed

            for i, expr_result in enumerate(expr_data):
                symbols = expr_result.symbols
                domains = expr_result.domains

                n_samples_job = n_samples
                if scale_samples:
                    n_samples_job *= round(sqrt(len(symbols)))

                out_filename = os.path.join(out_dir_full, str(i))
                yield delayed(generate_data)(
                    out_filename, symbols, domains, n_samples_job, seed)
                # increment seed (don't have same RNG state per job)
                seed += 1

        if n_jobs == 1:
            [f(*a, **kw) for f, a, kw in jobs()]
        else:
            Parallel(n_jobs=n_jobs)(jobs())


if __name__ == '__main__':
    import argparse


    def main():
        parser = argparse.ArgumentParser(  # noqa
            description='Generate data and save to file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            'expr_filename', help='Filename of the expression pickle'
        )
        parser.add_argument(
            '--n-samples', '-n', default=500, type=int,
            help='Number of samples'
        )
        parser.add_argument(
            '--no-scale-samples', action='store_true',
            help='Do not scale number of samples by number of dimensions'
        )
        default_out_dir = os.path.join(
            os.path.dirname(__file__), 'experiment_data', 'data')
        parser.add_argument(
            '--out-dir', '-O', default=default_out_dir,
            help='Output directory to save generated data'
        )
        parser.add_argument(
            '--n-jobs', '-j', default=-1, type=int,
            help='Number of jobs to use in generation'
        )
        parser.add_argument(
            '--seed', default=42, type=int,
            help='Seed for reproducibility. Technically the starting seed '
                 'from which each seed is derived per job'
        )

        args = parser.parse_args()

        run(out_dir=args.out_dir,
            expr_filename=args.expr_filename,
            n_jobs=args.n_jobs,
            n_samples=args.n_samples,
            scale_samples=not args.no_scale_samples,
            seed=args.seed)


    main()
