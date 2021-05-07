"""
parzen.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael

Adapted from:
https://github.com/marcotcr/lime-experiments/blob/master/parzen_windows.py
at 1d7c19f
"""
from typing import Optional

import numpy as np
import scipy as sp
import scipy.sparse

from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel


class ParzenWindowExplainer(BaseExplainer):

    def __init__(self,
                 model: AdditiveModel,
                 seed=None,
                 task: str = 'classification',
                 verbose=True):
        import warnings
        warnings.warn(
            f'This class ({self.__class__.__name__}) was never finished...'
            f'use at your own risk'
        )
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self._tabular = True

        if self.task != 'classification':
            raise ValueError(f'The {self.__class__.__name__} only supports '
                             f'classification tasks, not {self.task}.')

        # self.kernel = lambda x, sigma: np.exp(
        #     -.5 * x.dot(x.T)[0, 0] / sigma ** 2) / (
        #                                    np.sqrt(2 * np.pi * sigma ** 2))
        self.kernel = lambda x, sigma: np.asarray(
            np.exp(-.5 * x.power(2).sum(axis=1) / sigma ** 2) /
            np.sqrt(2 * np.pi * sigma ** 2)).flatten()

        self.verbose = verbose

        self.X = self.y = self.ones = self.zeros = self.sigma = None

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ):
        self.X = X.toarray() if sp.sparse.issparse(X) else X
        if y is None:
            y = self.model(self.X)
        # https://github.com/marcotcr/lime-experiments/blob/master/parzen_windows.py
        self.find_sigma(
            [0.1, .25, .5, .75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            self.X, y
        )
        self.y = y
        self.ones = y == 1
        self.zeros = y == 0

        n_valid = self.ones.sum() + self.zeros.sum()
        if n_valid < len(y):
            raise ValueError(f'{len(y) - n_valid} of {len(y)} labels produced '
                             f'by the model were not 0/1. These should be '
                             f'predictions for two-class classification only.')

    def predict(self, x):  # TODO: this is for one instance only
        probs = self.predict_proba(x)
        return int(probs[1] > .5)

    def predict_proba(self, x):  # TODO: this is for one instance only
        b = sp.sparse.csr_matrix(x - self.X)
        # pr = np.array([self.kernel(z, self.sigma) for z in b])
        pr = self.kernel(b, self.sigma)
        # prob = sum(pr[self.ones]) / sum(pr)
        # TODO: validate this
        prob = np.sum(pr[self.ones]) / np.sum(pr)
        return np.array([1 - prob, prob])

    def find_sigma(self, sigmas_to_try, cv_X, cv_y):
        self.sigma = sigmas_to_try[0]
        best_mistakes = np.iinfo(int).max
        best_sigma = self.sigma
        for sigma in sorted(sigmas_to_try):
            self.sigma = sigma
            preds = []
            for i in range(cv_X.shape[0]):
                preds.append(self.predict(cv_X[i]))
            mistakes = sum(cv_y != np.array(preds))
            # print(sigma, mistakes)
            if mistakes < best_mistakes:
                best_mistakes = mistakes
                best_sigma = sigma
        if self.verbose:
            print(f'Best sigma ({best_sigma}) achieves {best_mistakes} '
                  f'mistakes. Disagreement={best_mistakes / cv_X.shape[0]}')
        self.sigma = best_sigma

    def explain_instance(self, x):
        minus = self.X - x
        b = sp.sparse.csr_matrix(minus)
        ker = self.kernel(b, self.sigma)
        # ker = np.array([self.kernel(z, self.sigma) for z in b])
        times = np.multiply(minus, ker[:, np.newaxis])
        sumk_0 = sum(ker[self.zeros])
        sumk_1 = sum(ker[self.ones])
        sumt_0 = sum(times[self.zeros])
        sumt_1 = sum(times[self.ones])
        sumk_total = sumk_0 + sumk_1
        exp = ((sumk_0 * sumt_1 - sumk_1 * sumt_0) /
               (self.sigma ** 2 * sumk_total ** 2))
        features = x.nonzero()[1]
        # values = np.array(exp[0, features])[0]
        values = np.array(exp[features])
        return values

    def _call_explainer(self, X):
        contribs = []
        for xi in X:
            contribs.append(
                self.explain_instance(xi[None, :])
            )

        contribs = np.asarray(contribs)
        # TODO: this ([] * 2) is kinda stupid but here for compat...
        return {
            'contribs': [contribs] * 2
        }
