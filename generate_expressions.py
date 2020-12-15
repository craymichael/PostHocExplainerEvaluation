import sys
from functools import wraps
import threading
import random

import sympy as S

from posthoceval.model_generation import generate_additive_expression
from posthoceval.model_generation import valid_variable_domains
from posthoceval.model_generation import as_random_state

_RUNNING_PERIODICITY_IDS = {}
_MAX_RECURSIONS = 1_000


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
            thread = threading.Thread(target=wrapper, args=args, kwargs=kwargs)
            thread.start()
            thread_ident = thread.ident
            thread.join()
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


# MONKEY PATCH
S.periodicity = S.calculus.util.periodicity = S.calculus.periodicity = \
    periodicity_wrapper(S.periodicity)

symbols = S.symbols('x1:11', real=True)
tries = 0

# reproducibility
seed = 42
rs = as_random_state(seed)
# sympy uses python random module in spots, set seed for reproducibility
random.seed(seed, version=2)

while True:
    tries += 1
    try:
        expr = generate_additive_expression(symbols, pct_nonlinear=.25,
                                            n_interaction=4,
                                            interaction_ord=(2, 3),
                                            nonlinear_multiplier=4,
                                            nonlinear_skew=0,
                                            seed=rs)
        print('Attempting to find valid domains for:')
        S.pprint(expr)
        domains = valid_variable_domains(expr, fail_action='error',
                                         verbose=True)
    except (RuntimeError, RecursionError) as e:
        import traceback
        print('Yet another exception...', e, file=sys.stderr)
        # traceback.print_exc()
    else:
        break
print(f'Generated valid expression in {tries} tries.')
S.pprint(expr)
print(domains)
