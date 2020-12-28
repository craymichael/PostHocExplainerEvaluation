"""
evaluate_explainers.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
from posthoceval import metrics
from posthoceval.explainers.local.shap import KernelSHAPExplainer


def run():
    pass


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
