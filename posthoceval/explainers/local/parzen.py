"""
parzen.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael

Adapted from https://github.com/marcotcr/lime-experiments/blob/master/parzen_windows.py
at 1d7c19f
"""
import numpy as np
import scipy as sp
import scipy.sparse

from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel


class ParzenWindowExplainer(BaseExplainer):

    def __init__(self,
                 model: AdditiveModel,
                 seed=None,
                 task: str = 'regression',
                 verbose=True):
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        # self.kernel = lambda x, sigma: np.exp(
        #     -.5 * x.dot(x.T)[0, 0] / sigma ** 2) / (
        #                                    np.sqrt(2 * np.pi * sigma ** 2))
        self.kernel = lambda x, sigma: (
                np.exp(-.5 * x.power(2).sum(axis=1) / sigma ** 2) /
                np.sqrt(2 * np.pi * sigma ** 2)).flatten()

        self.verbose = verbose

        self.X = self.y = self.ones = self.zeros = self.sigma = None

    def fit(self, X, y=None):
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

    def predict(self, x):
        probs = self.predict_proba(x)
        print(probs.shape)
        return int(probs[1] > .5)

    def predict_proba(self, x):
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
            print(sigma, mistakes)
            if mistakes < best_mistakes:
                best_mistakes = mistakes
                best_sigma = sigma
        if self.verbose:
            print(f'Best sigma achieves {best_mistakes} mistakes. '
                  f'Disagreement={best_mistakes / cv_X.shape[0]}')
        self.sigma = best_sigma

    def explain_instance(self, x, num_features):
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
        values = np.array(exp[0, x.nonzero()[1]])[0]
        return sorted(
            zip(features, values), key=lambda xx: np.abs(xx[1]), reverse=True
        )[:num_features]

    def feature_contributions(self, X, return_y=False):
        contribs = []
        for xi in X:
            contribs.append(
                self.explain_instance(xi, self.model.n_features)
            )
        return contribs
