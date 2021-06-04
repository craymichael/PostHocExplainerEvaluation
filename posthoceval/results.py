
from collections import namedtuple

import sympy as sp

ExprResult = namedtuple('ExprResult',
                        'symbols,expr,domains,state,kwargs')


class MetricResult(object):
    __slots__ = 'effect_symbols', 'effect_latex'

    def __init__(self,
                 symbols,
                 effect):
        if not isinstance(symbols, (list, tuple)):
            symbols = (symbols,)

        symbols = [s.name if hasattr(s, 'name') else s
                   for s in symbols]
        self.effect_symbols = symbols
        self.effect_latex = (effect if isinstance(effect, str) else
                             sp.latex(effect))

    def as_dict(self):
        return {
            'effect_symbols': self.effect_symbols,
            'effect_latex': self.effect_latex,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        return obj
