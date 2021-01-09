"""
gam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from pygam import LinearGAM
from pygam import LogisticGAM as _LogisticGAM
from pygam import GammaGAM
from pygam import ExpectileGAM
from pygam import InvGaussGAM
from pygam import PoissonGAM
from pygam import GAM

__all__ = ['GAM', 'InvGaussGAM', 'PoissonGAM', 'ExpectileGAM', 'GammaGAM',
           'LogisticGAM', 'LinearGAM']


class LogisticGAM(_LogisticGAM):
    __slots__ = 'gridsearch',

    def __init__(self, *args, **kwargs):
        self._fit_with_gridsearch = kwargs.pop('gridsearch', True)
        super().__init__(*args, **kwargs)

    def fit(self, X, y, weights=None):
        if self._fit_with_gridsearch:
            return self.gridsearch(X, y, weights=weights, progress=False)
        else:
            return super().fit(X, y, weights=weights)

    def predict_proba(self, *args, **kwargs):
        probas = super().predict_proba(*args, **kwargs)
        return np.stack([1 - probas, probas], axis=1)
