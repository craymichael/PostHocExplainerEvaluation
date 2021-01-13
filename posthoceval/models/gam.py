"""
gam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import warnings

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
from pygam import terms

from posthoceval.metrics import standardize_effect
from posthoceval.utils import at_high_precision
from posthoceval.model_generation import AdditiveModel

__all__ = ['GAM', 'InvGaussGAM', 'PoissonGAM', 'ExpectileGAM', 'GammaGAM',
           'LogisticGAM', 'LinearGAM', 'MultiClassLogisticGAM', 'Terms', 'T']


class Terms:
    __slots__ = ()

    te = terms.te
    s = terms.s
    f = terms.f


T = Terms


class LogisticGAM(_LogisticGAM):
    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return np.stack([1 - probas, probas], axis=1)


class MultiClassLogisticGAM(AdditiveModel):
    def __init__(self, symbols, symbol_names=None, **kwargs):
        self.fit_with_gridsearch = kwargs.pop('fit_with_gridsearch', False)

        self._estimator_kwargs = kwargs
        self._estimators: Optional[List[LogisticGAM]] = None

        self._y_encoder = None
        self._n_classes = None

        # TODO: re-abstract AdditiveModel...this is quite sloppy
        # for compatibility
        self.symbols = tuple(symbols)
        if symbol_names is None:
            symbol_names = tuple(s.name if hasattr(s, 'name') else str(s)
                                 for s in symbols)
        else:
            assert len(symbol_names) == len(symbols)

        self.symbol_names = symbol_names
        self.n_features = len(symbols)
        self._symbol_map = None
        self.expr = self.backend = None

    def __call__(self, X, backend=None):
        if backend is not None:
            warnings.warn(f'{self.__class__} ignores kwarg "backend" '
                          f'({backend}) - this is N/A here')
        return self.predict_proba(X)

    # TODO this and a bunch of others will have issues per sympy-related
    #  AdditiveModel methods...
    def pprint(self):
        raise NotImplementedError

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
        self._estimators = [LogisticGAM(**self._estimator_kwargs)
                            for _ in range(self._n_classes)]
        for i, estimator in enumerate(self._estimators):
            yi = y if self._n_classes == 1 else y[:, i]
            if self.fit_with_gridsearch:
                estimator.gridsearch(X, yi, weights=weights)
            else:
                estimator.fit(X, yi, weights=weights)

        return self

    def feature_contributions(self, X, **kwargs):
        if kwargs:
            warnings.warn(f'Ignoring all kwargs {kwargs} - these are N/A '
                          f'here.')
        contribs = []
        intercepts = []
        for e_i in self._estimators:
            contribs_i = {}
            for i, term in enumerate(e_i.terms):
                if term.isintercept:
                    assert (i + 1) == len(e_i.terms)
                    assert term.n_coefs == 1

                    intercepts.append(e_i.coef_[-1])
                else:
                    # term.feature TODO
                    effect = standardize_effect(term.feature)
                    contribs_i[effect] = e_i.partial_dependence(i, X)

            contribs.append(contribs_i)

        if len(contribs) == 1:
            # invert log odds
            contribs_1 = contribs[0]
            contribs_0 = {
                effect: at_high_precision(opposite_log_odd, pd)
                for effect, pd in contribs_1.items()
            }
            contribs = [contribs_0, contribs_1]

        return contribs

    def predict(self, X):
        assert self._is_fitted

        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        # y : np.array of shape (n_samples,)
        assert self._is_fitted

        if self._n_classes == 1:
            return self._estimators[0].predict_proba(X)

        probas = []
        for i, estimator in enumerate(self._estimators):
            # get probability at 1 (not 0)
            p_i = estimator.predict_proba(X)[:, 1]
            probas.append(p_i)

        return np.stack(probas, axis=1)


def opposite_log_odd(values):
    exp_vals = np.exp(values)
    # logistic
    log_odds = 1 - (exp_vals / (1 + exp_vals))  # val --> p
    # logit
    prob_vals = np.log(log_odds / (1 - log_odds))  # p --> val
    return prob_vals
