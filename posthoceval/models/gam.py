"""
gam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import List
from typing import Optional

import numpy as np

from sklearn.preprocessing import OneHotEncoder

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
    # __slots__ = '_fit_with_gridsearch',

    # def __init__(self, *args, **kwargs):
    #     self._fit_with_gridsearch = kwargs.pop('fit_with_gridsearch', True)
    #     super().__init__(*args, **kwargs)

    # def fit(self, X, y, weights=None):
    #     if self._fit_with_gridsearch:
    #         # TODO: ugly
    #         self._fit_with_gridsearch = False
    #         ret = self.gridsearch(X, y, weights=weights, progress=False)
    #         self._fit_with_gridsearch = True
    #         return ret
    #     else:
    #         return super().fit(X, y, weights=weights)

    def predict_proba(self, *args, **kwargs):
        probas = super().predict_proba(*args, **kwargs)
        return np.stack([1 - probas, probas], axis=1)


class MultiClassLogisticGAM:
    def __init__(self, *args, **kwargs):
        self.fit_with_gridsearch = kwargs.pop('fit_with_gridsearch', False)

        self._estimator_args = args
        self._estimator_kwargs = kwargs
        self._estimators: Optional[List[LogisticGAM]] = None

        self._y_encoder = None
        self._n_classes = None

    @property
    def _is_fitted(self):
        return (self._estimators is not None and
                all(estimator._is_fitted  # noqa
                    for estimator in self._estimators))

    def _standardize_y(self, y):
        y = np.asarray(y)
        if y.ndim == 1:
            self._y_encoder = OneHotEncoder(sparse=False)
            y = self._y_encoder.fit_transform(y.reshape(-1, 1))
        else:
            assert y.ndim == 2

        self._n_classes = y.shape[1]

        if self._n_classes == 2:
            # single class in this case
            y = y[:, 1]
            self._n_classes = 1

        return y

    def fit(self, X, y, weights=None):
        y = self._standardize_y(y)
        self._estimators = [LogisticGAM(*self._estimator_args,
                                        **self._estimator_kwargs)
                            for _ in range(self._n_classes)]
        for i, estimator in enumerate(self._estimators):
            yi = y if self._n_classes == 1 else y[:, i]
            if self.fit_with_gridsearch:
                estimator.gridsearch(X, yi, weights=weights)
            else:
                estimator.fit(X, yi, weights=weights)

        return self

    def predict(self, X):
        assert self._is_fitted

        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        # y : np.array of shape (n_samples,)
        assert self._is_fitted

        for i, estimator in enumerate(self._estimators):
            pass
