#!/usr/bin/env python
"""
evaluate_explainers.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import os
import sys

# ssshhhhhhhhhhhh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
# # lazy load tensorflow
# from posthoceval.lazy_loader import LazyLoader
#
# _ = LazyLoader('tensorflow')

from glob import glob

import traceback

from tqdm.auto import tqdm

from joblib import Parallel
from joblib import delayed

import numpy as np
import sympy as sp

from posthoceval.model_generation import SyntheticModel
from posthoceval.utils import assert_same_size
from posthoceval.expl_utils import save_explanation
from posthoceval.expl_utils import CompatUnpickler
from posthoceval.explainers.local.shap import KernelSHAPExplainer
from posthoceval.explainers.local.shapr import SHAPRExplainer
from posthoceval.explainers.local.lime import LIMEExplainer
from posthoceval.explainers.local.maple import MAPLEExplainer
from posthoceval.explainers.global_.pdp import PDPExplainer
from posthoceval.explainers import GradCAMExplainer
from posthoceval.explainers import VanillaGradientsExplainer
from posthoceval.explainers import GradientsXInputsExplainer
from posthoceval.explainers import IntegratedGradientsExplainer
from posthoceval.explainers import OcclusionExplainer
from posthoceval.explainers import XRAIExplainer
from posthoceval.explainers import BlurIntegratedGradientsExplainer
from posthoceval.utils import tqdm_parallel
from posthoceval.results import ExprResult

EXPLAINER_MAP = {
    'SHAP': KernelSHAPExplainer,
    # TODO: SHAPR for each of the conditioned distributions other than
    #  empirical
    'SHAPR': SHAPRExplainer,
    'LIME': LIMEExplainer,
    'MAPLE': MAPLEExplainer,
    'PDP': PDPExplainer,
    'GradCAM': GradCAMExplainer,
    'GradCAM-Smooth': GradCAMExplainer.smooth_grad,
    'VanillaGradients': VanillaGradientsExplainer,
    'VanillaGradients-Smooth': VanillaGradientsExplainer.smooth_grad,
    'GradientsXInputs': GradientsXInputsExplainer,
    'GradientsXInputs-Smooth': GradientsXInputsExplainer.smooth_grad,
    'IntegratedGradients': IntegratedGradientsExplainer,
    'IntegratedGradients-Smooth': IntegratedGradientsExplainer.smooth_grad,
    'Occlusion': OcclusionExplainer,
    'XRAI': XRAIExplainer,
    'XRAI-Smooth': XRAIExplainer.smooth_grad,
    'BlurIG': BlurIntegratedGradientsExplainer,
    'BlurIG-Smooth': BlurIntegratedGradientsExplainer.smooth_grad,
}


def explain(explainer_cls, out_filename, expr_result, data_file, max_explain,
            seed):
    # type hint
    expr_result: ExprResult

    if os.path.exists(out_filename):
        tqdm.write(f'{out_filename} exists, skipping!')
        return

    tqdm.write('Generating model')
    model = SyntheticModel.from_expr(
        expr=expr_result.expr,
        symbols=expr_result.symbols,
    )

    tqdm.write('Creating explainer')
    explainer = explainer_cls(
        model,
        seed=seed,
    )
    tqdm.write(f'Loading data from {data_file}')
    data = np.load(data_file)['data']
    to_explain = data
    if max_explain is not None and max_explain < len(to_explain):
        to_explain = to_explain[:max_explain]

    try:
        tqdm.write('Fitting explainer')
        explainer.fit(data)

        tqdm.write('Explaining')
        explanation = explainer.feature_contributions(to_explain)

    except (ValueError, TypeError):
        tqdm.write(f'Failed to explain model:')
        tqdm.write(sp.pretty(expr_result.expr))

        exc_lines = traceback.format_exception(
            *sys.exc_info(), limit=None, chain=True)
        for line in exc_lines:
            tqdm.write(str(line), file=sys.stderr, end='')

        return
    # save things in parallel
    tqdm.write('Saving explanation')
    save_explanation(out_filename, explanation)


def run(expr_filename, out_dir, data_dir, max_explain, seed, n_jobs,
        start_at=1, step_size=1, explainer='SHAP', debug=False):
    """"""
    try:
        explainer_cls = EXPLAINER_MAP[explainer]
    except KeyError:
        raise ValueError(
            f'{explainer} is not a valid explainer name') from None

    basename_experiment = os.path.basename(expr_filename).rsplit('.', 1)[0]

    explainer_out_dir = os.path.join(out_dir, basename_experiment, explainer)
    os.makedirs(explainer_out_dir, exist_ok=True)

    print('Loading', expr_filename, '(this may take a while)')
    with open(expr_filename, 'rb') as f:
        expr_data = CompatUnpickler(f).load()

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

    # same as indexing as [start_at - 1::step_size]
    indices = slice(start_at - 1, None, step_size)

    file_ids = range(n_results)[indices]
    data_files = data_files[indices]
    expr_data = expr_data[indices]

    with tqdm_parallel(tqdm(desc=f'Evaluating {explainer}', total=n_results)):

        def jobs():
            for i, data_file, expr_result in zip(file_ids,
                                                 data_files,
                                                 expr_data):
                out_filename = os.path.join(explainer_out_dir, str(i)) + '.npz'
                yield delayed(explain)(
                    explainer_cls, out_filename, expr_result, data_file,
                    max_explain, seed
                )

                if debug:  # one iteration
                    break

        if n_jobs == 1 or debug:
            # TODO: this doesn't update tqdm
            [f(*a, **kw) for f, a, kw in jobs()]
        else:
            Parallel(n_jobs=n_jobs)(jobs())


if __name__ == '__main__':
    import argparse


    def positive_int(x):
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{x} is not a valid integer.')

        if x < 1:
            raise argparse.ArgumentTypeError(f'{x} must be positive.')

        return x


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
            '--explainer', '-X',
            choices=[*EXPLAINER_MAP.keys()],
            default='SHAP', help='The explainer to evaluate'
        )
        parser.add_argument(
            '--max-explain', type=positive_int,
            help='Maximum number of data points to explain per model'
        )
        parser.add_argument(
            '--n-jobs', '-j', default=-1, type=int,
            help='Number of jobs to use in generation'
        )
        parser.add_argument(
            '--seed', default=42, type=int,
            help='Seed for reproducibility'
        )
        parser.add_argument(
            '--start-at', default=1, type=positive_int,
            help='start index (1-indexed) for .pkl file'
        )
        parser.add_argument(
            '--step-size', default=1, type=positive_int,
            help='Size of increment for evaluation'
        )
        parser.add_argument(
            '--debug', action='store_true',
            help=argparse.SUPPRESS
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
            n_jobs=args.n_jobs,
            seed=args.seed,
            start_at=args.start_at,
            explainer=args.explainer,
            step_size=args.step_size,
            debug=args.debug)


    main()
