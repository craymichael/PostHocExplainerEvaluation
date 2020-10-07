import string
import random
from functools import partial
from itertools import repeat
from collections.abc import Iterable
from collections.abc import Sized
from collections.abc import Callable

import sympy as sym


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
                 n_features,
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
        self.expr = self.generate_model()

    def generate_model(self):
        n_coefs_expect = self.n_features + 1
        if isinstance(self.coefficients, Iterable):
            if not isinstance(self.coefficients, Sized):
                self.coefficients = tuple(self.coefficients)
            assert len(self.coefficients) == n_coefs_expect, (
                    'len(coefficients) != (n_features + 1) (%d != %d)' %
                    (len(self.coefficients), n_coefs_expect))
            coefs = iter(self.coefficients)
        else:
            coefs = repeat(self.coefficients, n_coefs_expect)

        bias = next(coefs)
        total = bias() if isinstance(bias, Callable) else bias
        for xi, ci in zip(self.symbols, coefs):
            total += xi * (ci() if isinstance(ci, Callable) else ci)
        return total

    def feature_attribution(self):
        for arg in sym.preorder_traversal(self.expr):
            pass
