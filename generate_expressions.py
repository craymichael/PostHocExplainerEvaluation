#!/usr/bin/env python
import os
import sys
import threading
import random
import pickle
from functools import wraps
from collections import namedtuple
from datetime import datetime

from tqdm.auto import tqdm
from joblib import Parallel
from joblib import delayed

import sympy as S
import numpy as np

from posthoceval.model_generation import generate_additive_expression
from posthoceval.model_generation import valid_variable_domains
from posthoceval.model_generation import as_random_state
from posthoceval.utils import tqdm_parallel

_RUNNING_PERIODICITY_IDS = {}
_MAX_RECURSIONS = 1_000

# https://bugs.python.org/issue25222
sys.setrecursionlimit(200)

ExprResult = namedtuple('ExprResult',
                        'symbols,expr,domains,state,kwargs')


def periodicity_wrapper(func):
    """A hacky fix for https://github.com/sympy/sympy/issues/20566"""
    ret_val = [None]
    raise_error = [False]
    other_exception = [None]

    @wraps(func)
    def wrapper(*args, _child_=None, **kwargs):
        ident = threading.get_ident()
        ident_present = ident in _RUNNING_PERIODICITY_IDS
        if _child_ is None:
            _child_ = ident_present

        if _child_ or ident_present:
            if not ident_present:
                _RUNNING_PERIODICITY_IDS[ident] = 0

            if _MAX_RECURSIONS < _RUNNING_PERIODICITY_IDS[ident]:
                raise_error[0] = True
                sys.exit()
            try:
                ret = func(*args, **kwargs)
            except Exception as e:
                raise_error[0] = True
                other_exception[0] = e
                sys.exit()

            _RUNNING_PERIODICITY_IDS[ident] += 1
            ret_val[0] = ret
            return ret
        else:
            kwargs['_child_'] = True
            # print(f'Enter new periodicity thread {ident}')
            thread = threading.Thread(target=wrapper, args=args, kwargs=kwargs)
            thread.start()
            thread_ident = thread.ident
            thread.join()
            # print(f'Exit periodicity thread {ident}')
            if thread_ident in _RUNNING_PERIODICITY_IDS:
                del _RUNNING_PERIODICITY_IDS[thread_ident]
            if raise_error[0]:
                if other_exception[0] is not None:
                    raise other_exception[0]
                raise RecursionError(
                    f'Maximum recursions ({_MAX_RECURSIONS}) in {func} '
                    f'exceeded!'
                )
            return ret_val[0]

    return wrapper


# 🔧 MONKEY PATCH 🔧
#      _...._
#    .-.     /
#   /o.o\ ):.\
#   \   / `- .`--._
#   // /            `-.
#  '...\     .         `.
#   `--''.    '          `.
#       .'   .'            `-.
#    .-'    /`-.._            \
#  .'    _.'      :      .-'"'/
# | _,--`       .'     .'    /
# \ \          /     .'     /
#  \///        |    ' |    /
#              \   (  `.   ``-.
#               \   \   `._    \
#             _.-`   )    .'    )
#             `.__.-'  .-' _-.-'
#                      `.__,'
# ascii credit: https://www.asciiart.eu/animals/monkeys
S.periodicity = S.calculus.util.periodicity = S.calculus.periodicity = \
    periodicity_wrapper(S.periodicity)


def generate_expression(symbols, seed, verbose=0, printer=None, **kwargs):
    """kwargs: see `generate_additive_expression`"""
    # sympy uses python random module in spots, set seed for reproducibility
    random.seed(seed, version=2)
    # I can't prove it but I think sympy also uses default numpy generator in
    # spots...
    np.random.seed(seed)
    # reproducibility, reseeded per job
    rs = as_random_state(seed)

    tries = 0
    while True:
        tries += 1
        print('Generating expression...')
        expr = generate_additive_expression(symbols, seed=rs, **kwargs)
        try:
            print('Attempting to find valid domains...')
            domains = valid_variable_domains(expr, fail_action='error',
                                             verbose=verbose)
        except (RuntimeError, RecursionError) as e:
            # import traceback
            print('Failed to find domains for:')
            print(S.pretty(expr))
            print('Yet another exception...', e, file=sys.stderr)
            # traceback.print_exc()
        else:
            break
    print(f'Generated valid expression in {tries} tries.')
    print(S.pretty(expr))

    return ExprResult(
        symbols=symbols,
        expr=expr,
        domains=domains,
        state=rs.__getstate__(),
        kwargs=kwargs
    )


def run(kwarg, kwarg_range, n_feats_range, n_runs, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)

    # default kwargs
    default_kwargs = dict(
        n_main=None,
        n_uniq_main=None,
        n_interaction=0,
        n_uniq_interaction=None,
        interaction_ord=None,
        n_dummy=0,
        pct_nonlinear=None,  # default is 0.5
        nonlinear_multiplier=None,  # default depends on pct_nonlinear
        nonlinear_shift=0,
        nonlinear_skew=0,
        nonlinear_interaction_additivity=.5,
        nonlinear_single_multi_ratio='balanced',
        nonlinear_single_arg_ops=None,
        nonlinear_single_arg_ops_weights=None,
        nonlinear_multi_arg_ops=None,
        nonlinear_multi_arg_ops_weights=None,
        linear_multi_arg_ops=None,
        linear_multi_arg_ops_weights=None,
    )

    total_expressions = len(n_feats_range) * len(kwarg_range) * n_runs

    with tqdm_parallel(tqdm(desc='Expression Generation',
                            total=total_expressions)) as pbar:
        jobs = []
        for n_feat in n_feats_range:
            symbols = S.symbols(f'x1:{n_feat + 1}', real=True)

            for kw_val in kwarg_range:
                kwargs = default_kwargs.copy()
                kwargs[kwarg] = kw_val

                for _ in range(n_runs):
                    jobs.append(
                        delayed(generate_expression)(symbols, seed, **kwargs)
                    )
                    # increment seed (don't have same RNG state per job)
                    seed += 1

        results = Parallel(n_jobs=-1)(jobs)

    now_str = datetime.now().isoformat(timespec='seconds').replace(':', '_')
    out_file = os.path.join(out_dir, f'generated_expressions_{now_str}.pkl')

    print('Saving results to', out_file)
    with open(out_file, 'wb') as f:
        pickle.dump(
            results, f,
            protocol=4  # 4 is compatible with python 3.3+, 5 with 3.8+
        )


if __name__ == '__main__':
    import argparse
    import re
    from math import ceil
    from math import sqrt

    rx_range_t = re.compile(
        r'^\s*'
        r'([-+]?\d+\.?(?:\d+)?|(?:\d+)?\.?\d+)'
        r'\s*,\s*'
        r'([-+]?\d+\.?(?:\d+)?|(?:\d+)?\.?\d+)'
        r'\s*'
        r'(?:,\s*(log|linear))?'
        r'\s*$'
    )
    range_pattern = 'a,b[,log|linear], e.g., "1,10" or "0,.9,log"'


    def range_type(value):
        m = rx_range_t.match(value)
        if m is None:
            raise argparse.ArgumentTypeError(
                f'{value} does not match pattern "{range_pattern}"'
            )
        a, b, scale = m.groups()
        if '.' in a or '.' in b:
            dtype = float
        else:
            dtype = int
        if scale is None:
            scale = 'linear'

        a, b = float(a), float(b)
        if a >= b:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid range ({a} is not less than {b})'
            )
        return a, b, scale, dtype


    def arg_val_to_range(n, a, b, scale, inferred_dtype, dtype):
        is_int = (dtype == 'int')

        if dtype == 'infer' and inferred_dtype is int:
            is_int = ((n is None) or ((b - a) >= (n / 2)))
            print(f'Inferred range [{a},{b}] as '
                  f'a{"n int" if is_int else " float"} interval.')

        if scale == 'linear':
            if n is None:
                if not is_int:
                    sys.exit(f'Cannot infer the number of samples from a '
                             f'space with float dtype for range [{a},{b}]')
                # otherwise
                n = int(b - a + 1)
            space = np.linspace(a, b, n)
        elif scale == 'log':
            if n is None:
                sys.exit(f'Cannot use a log space without a defined number of '
                         f'samples for range [{a},{b}]')
            space = np.geomspace(a, b, n)
        else:
            raise ValueError(scale)

        if is_int:
            space = np.around(space).astype(int)

        return space


    def main():
        parser = argparse.ArgumentParser(  # noqa
            description='Generate expressions and save to file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument(
            '--n-runs', type=int, required=True,
            help='Number of runs (expressions generated) per data point'
        )
        parser.add_argument(
            '--n-feats-range', type=range_type, required=True,
            help=f'Range of number of features in expressions. '
                 f'Expected format: {range_pattern}'
        )
        parser.add_argument(
            '--n-feats-range-size', type=int,
            help=f'number of features sampled in the range'
        )
        # TODO: interaction_ord, all ops args
        parser.add_argument(
            '--kwarg', required=True,
            choices=['n_main', 'n_uniq_main', 'n_interaction',
                     'n_uniq_interaction', 'interaction_ord', 'n_dummy',
                     'pct_nonlinear', 'nonlinear_multiplier', 'nonlinear_shift',
                     'nonlinear_skew', 'nonlinear_interaction_additivity',
                     'nonlinear_single_multi_ratio'],
            help='Name of the expression generation parameter that value range '
                 'refers to'
        )
        parser.add_argument(
            '--kwarg-range', required=True, type=range_type,
            help=f'Range for kwarg. Expected format: {range_pattern}'
        )
        parser.add_argument(
            '--kwarg-range-size', required=True, type=int,
            help=f'number of values taken from the kwarg range'
        )
        parser.add_argument(
            '--kwarg-dtype', default='infer',
            choices=('infer', 'int', 'float'),
            help=f'dtype for kwarg'
        )
        default_out_dir = os.path.join(
            os.path.dirname(__file__), 'experiment_data', 'expr')
        parser.add_argument(
            '--out-dir', '-O', default=default_out_dir,
            help='Output directory to save generated expressions'
        )
        parser.add_argument(
            '--seed', default=42, type=int,
            help='Seed for reproducibility. Technically the starting seed '
                 'from which each seed is derived per job'
        )

        args = parser.parse_args()

        kwarg_range = arg_val_to_range(args.kwarg_range_size, *args.kwarg_range,
                                       dtype=args.kwarg_dtype)

        n_feats_range = arg_val_to_range(args.n_feats_range_size,
                                         *args.n_feats_range, dtype='int')

        run(
            kwarg=args.kwarg,
            kwarg_range=kwarg_range,
            n_feats_range=n_feats_range,
            n_runs=args.n_runs,
            out_dir=args.out_dir,
            seed=args.seed,
        )


    main()
