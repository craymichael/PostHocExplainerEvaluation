import string
import random
from functools import partial
from itertools import repeat
from collections.abc import Iterable
from collections.abc import Callable

import sympy as sym

from joblib import cpu_count
from joblib import Parallel
from joblib import delayed

from .utils import as_sized
from .utils import assert_same_size


def symbol_names(n_features):
    """Generate Excel-like names for symbols"""
    assert n_features >= 1, 'Invalid number of features < 1: %d' % n_features
    alphabet = string.ascii_uppercase
    ret = []
    for d in range(1, n_features + 1):
        ret_i = ''
        while d > 0:
            d, m = divmod(d - 1, 26)
            ret_i = alphabet[m] + ret_i
        ret.append(ret_i)
    return ret


class LinearModel(object):
    def __init__(self,
                 n_features: int,
                 coefficients=partial(random.uniform, -1, +1),
                 domain='real'):
        """

        :param n_features:
        :param coefficients: first coef is the bias term
        :param domain:
        """
        self.n_features = n_features
        self.domain = domain
        self.coefficients = coefficients

        kwargs = dict()
        if self.domain == 'real':
            kwargs['real'] = True
        elif self.domain == 'integer':
            kwargs['integer'] = True
        else:
            raise ValueError('Unknown variable domain: %s' % self.domain)

        # Generate n_features symbols with unique names from domain
        self.symbols = sym.symbols(symbol_names(self.n_features), **kwargs)
        self.expr = self._generate_model()

    def _generate_model(self) -> sym.Expr:
        n_coefs_expect = self.n_features + 1
        if isinstance(self.coefficients, Iterable):
            self.coefficients = as_sized(self.coefficients)
            assert_same_size(n_coefs_expect, len(self.coefficients),
                             'coefficients')
            coefs = iter(self.coefficients)
        else:
            coefs = repeat(self.coefficients, n_coefs_expect)

        bias = next(coefs)
        total = bias() if isinstance(bias, Callable) else bias
        for xi, ci in zip(self.symbols, coefs):
            total += xi * (ci() if isinstance(ci, Callable) else ci)
        return total

    def __call__(self, *x, n_jobs=cpu_count()):
        if len(x) == 1:
            x = x[0]
        if isinstance(x, Iterable):
            x = as_sized(x)
            if isinstance(x[0], Iterable):
                assert_func = partial(assert_same_size,
                                      expected=self.n_features,
                                      units='features')
                return Parallel(n_jobs=n_jobs)(
                    delayed(self.expr.subs)(
                        zip(self.symbols,
                            assert_func(received=len(xi), ret=xi))
                    ) for xi in x
                )
            else:
                assert_same_size(self.n_features, len(x), 'features')
                return self.expr.subs(zip(self.symbols, x))
        else:
            assert_same_size(self.n_features, 1, 'features')
            return self.expr.subs((self.symbols[0], x))

    def feature_attribution(self):
        # for arg in sym.preorder_traversal(self.expr):
        #     pass
        self.expr.as_independent(self.symbols, as_Add=True)

    def __repr__(self):
        return str(self.expr)
