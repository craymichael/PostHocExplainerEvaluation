#!/usr/bin/env python
import os
import sys
import threading
import random
import pickle
import math
import traceback
from functools import wraps
from collections import namedtuple
from datetime import datetime

from multiprocessing import TimeoutError
from multiprocessing import Lock

from tqdm.auto import tqdm
from joblib import Parallel
from joblib import delayed

import sympy as sp
import numpy as np

from posthoceval.model_generation import generate_additive_expression
from posthoceval.model_generation import valid_variable_domains
from posthoceval.model_generation import as_random_state
from posthoceval.utils import tqdm_parallel
from posthoceval.utils import dict_product

_RUNNING_PERIODICITY_IDS = {}
_MAX_RECURSIONS = 1_000

# # https://bugs.python.org/issue25222
# sys.setrecursionlimit(200)

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
sp.periodicity = sp.calculus.util.periodicity = sp.calculus.periodicity = \
    periodicity_wrapper(sp.periodicity)


def tqdm_write(*args, sep=' ', **kwargs):
    tqdm.write(sep.join(map(str, args)), **kwargs)


def generate_expression(symbols, seed, verbose=0, timeout=None, **kwargs):
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
        expr = None
        tqdm_write('Generating expression...')
        try:
            expr = generate_additive_expression(symbols, seed=rs, **kwargs)
            tqdm_write('Attempting to find valid domains...')
            domains = valid_variable_domains(expr, fail_action='error',
                                             verbose=verbose, timeout=timeout)
        except (RuntimeError, RecursionError, TimeoutError) as e:
            if expr is None:
                tqdm_write('Failed to find domains for:')
                tqdm_write(sp.pretty(expr))
            else:
                tqdm_write('Wow...failed to generate expression...')
            # tqdm_write('Yet another exception...', e, file=sys.stderr)
            exc_lines = traceback.format_exception(
                *sys.exc_info(), limit=None, chain=True)
            for line in exc_lines:
                tqdm_write(line, file=sys.stderr, end='')
        else:
            break
    tqdm_write(f'Generated valid expression in {tries} tries.')
    tqdm_write(sp.pretty(expr))

    return ExprResult(
        symbols=symbols,
        expr=expr,
        domains=domains,
        state=rs.__getstate__(),
        kwargs=kwargs
    )


def run(n_feats_range, n_runs, out_dir, seed, kwargs, timeout=3):
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

    total_expressions = math.prod((
        n_runs, len(n_feats_range), *(len(v) for v in kwargs.values())
    ))

    with tqdm_parallel(tqdm(desc='Expression Generation',
                            total=total_expressions)) as pbar:
        import inspect

        tqdm.set_lock(Lock())
        inspect.builtins.print = tqdm_write

        def jobs():
            nonlocal seed

            for n_feat in n_feats_range:
                for kw_val in dict_product(kwargs):
                    job_kwargs = default_kwargs.copy()
                    job_kwargs.update(kw_val)

                    symbols = sp.symbols(f'x1:{n_feat + 1}', real=True)

                    print(f'{n_feat} features, generate expr with:')
                    print(job_kwargs)
                    for _ in range(n_runs):
                        yield delayed(generate_expression)(
                            symbols, seed, timeout=timeout, **job_kwargs)
                        # increment seed (don't have same RNG state per job)
                        seed += 1

        results = Parallel(n_jobs=-1)(jobs())

    param_str = '-vs-'.join(k for k in kwargs.keys())

    now_str = datetime.now().isoformat(timespec='seconds').replace(':', '_')
    out_file = os.path.join(out_dir,
                            f'generated_expressions_{param_str}_{now_str}.pkl')

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
        # a: int/float
        r'^\s*'
        r'([-+]?\d+\.?(?:\d+)?|(?:\d+)?\.?\d+)'
        r'\s*'
        # b: int/float (optional, default: None, range comprises `a` only)
        r'(?:,\s*'
        r'([-+]?\d+\.?(?:\d+)?|(?:\d+)?\.?\d+)'
        r'\s*'
        # n: int (optional, default: infer)
        r'(?:,\s*'
        r'(\d+)'
        r'\s*)?'
        # scale: str (optional, default: linear)
        r'(?:,\s*(log|linear))?'
        r'\s*)?'
        r'$'
    )
    range_pattern = ('"a[,b[,n][,log|linear]]" e.g. "1,10" or "-.5,.9,10,log" '
                     'or "0.5" or "1,10,5"')


    def range_type(value):
        m = rx_range_t.match(value)
        if m is None:
            raise argparse.ArgumentTypeError(
                f'{value} does not match pattern {range_pattern}'
            )
        a, b, n, scale = m.groups()
        if '.' in a or (b and '.' in b):
            dtype = float
        else:
            dtype = int

        if n is not None:
            n = int(n)
            if n < 1:
                raise argparse.ArgumentTypeError(
                    f'{value} contains an invalid range size ({n} cannot be '
                    f'less than 1)'
                )

        if scale is None:
            scale = 'linear'

        a = float(a)
        if b:
            b = float(b)
            if a >= b:
                raise argparse.ArgumentTypeError(
                    f'{value} is an invalid range ({a} is not less than {b})'
                )
        return a, b, n, scale, dtype


    def arg_val_to_range(a, b, n, scale, inferred_dtype, dtype):
        is_int = (dtype == 'int')
        inferred_int = (dtype == 'infer' and inferred_dtype is int)

        range_msg = f'[{a},{b}]'

        if is_int:
            int_msg = ('Range was explicitly specified as integer, however, '
                       'the {} value {} is not an integer in range ' +
                       range_msg)
            if not a.is_integer():
                sys.exit(int_msg.format('a', a))
            if not (b is None or b.is_integer()):
                sys.exit(int_msg.format('b', b))

        if b is None:
            if is_int or inferred_int:
                a = int(a)
            if n is not None and n != 1:
                sys.exit(f'Specified single value as the range ({a}) but the '
                         f'size is not 1 ({n}).')
            return [a]

        if inferred_int:
            is_int = ((n is None) or ((b - a) >= (n / 2)))
            print(f'Inferred range {range_msg} as '
                  f'a{"n int" if is_int else " float"} interval.')

        if scale == 'linear':
            if n is None:
                if not is_int:
                    sys.exit(f'Cannot infer the number of samples from a '
                             f'space with float dtype for range {range_msg}')
                # otherwise
                n = int(b - a + 1)
            space = np.linspace(a, b, n)
        elif scale == 'log':
            if n is None:
                sys.exit(f'Cannot use a log space without a defined number of '
                         f'samples for range {range_msg}')
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
        # TODO: interaction_ord, all ops args
        parser.add_argument(
            '--kwarg', required=True, action='append',
            choices=['n_main', 'n_uniq_main', 'n_interaction',
                     'n_uniq_interaction', 'interaction_ord', 'n_dummy',
                     'pct_nonlinear', 'nonlinear_multiplier', 'nonlinear_shift',
                     'nonlinear_skew', 'nonlinear_interaction_additivity',
                     'nonlinear_single_multi_ratio'],
            help='Name of the expression generation parameter that value range '
                 'refers to'
        )
        parser.add_argument(
            '--kwarg-range', required=True, type=range_type, action='append',
            help=f'Range for kwarg. Expected format: {range_pattern}'
        )
        parser.add_argument(
            '--kwarg-dtype', default='infer', action='append',
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

        if len(args.kwarg_range) != len(args.kwarg):
            sys.exit('The arguments --kwarg and --kwarg-range '
                     'must all have the same number of arguments. Received: '
                     f'{len(args.kwarg)}, {len(args.kwarg_range)}')

        if isinstance(args.kwarg_dtype, str):
            kwarg_dtype = [args.kwarg_dtype] * len(args.kwarg)
        elif len(args.kwarg_dtype) == len(args.kwarg):
            kwarg_dtype = args.kwarg_dtype
        elif len(args.kwarg_dtype) == 1:
            print('Using provided --kwarg-dtype for all kwarg arguments')
            kwarg_dtype = args.kwarg_dtype * len(args.kwarg)
        else:
            sys.exit('Provided --kwarg-dtype must be the same size as --kwarg '
                     'arguments, or a single value to be applied to all '
                     '--kwarg arguments.')

        n_feats_range = arg_val_to_range(*args.n_feats_range, dtype='int')

        kwargs = {}
        for kwarg, range_t, dtype in zip(
                args.kwarg, args.kwarg_range, kwarg_dtype):
            kwargs[kwarg] = arg_val_to_range(*range_t, dtype=dtype)

        run(
            n_feats_range=n_feats_range,
            n_runs=args.n_runs,
            out_dir=args.out_dir,
            seed=args.seed,
            kwargs=kwargs
        )


    main()
