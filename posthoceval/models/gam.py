"""
gam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Union
from typing import List

import warnings

from abc import abstractmethod
from abc import ABCMeta

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from pygam import LinearGAM as _LinearGAM
from pygam import LogisticGAM as _LogisticGAM
# from pygam import GammaGAM
# from pygam import ExpectileGAM
# from pygam import InvGaussGAM
# from pygam import PoissonGAM
# from pygam import GAM
from pygam import terms as pygam_terms

from posthoceval.expl_utils import standardize_effect
from posthoceval.utils import at_high_precision
from posthoceval.models.model import AdditiveModel

# __all__ = ['GAM', 'InvGaussGAM', 'PoissonGAM', 'ExpectileGAM', 'GammaGAM',
#            'LogisticGAM', 'LinearGAM', 'MultiClassLogisticGAM', 'Terms', 'T']
__all__ = ['LinearGAM', 'MultiClassLogisticGAM', 'Terms', 'T']


class Terms:
    """Wrapper around PyGAM terms"""
    __slots__ = ()

    te = pygam_terms.te
    s = pygam_terms.s
    f = pygam_terms.f


T = Terms


class LogisticGAM(_LogisticGAM):
    """Wrapper around the PyGAM Logistic GAM class that is compatible with
    sklearn"""

    def predict_proba(self, X):
        """
        Binary logistic regression predicted probabilities

        :param X: the 2D input data (N x F)
        :return: predicted probabilities
        """
        probas = super().predict_proba(X)
        return np.stack([1 - probas, probas], axis=1)


class BaseGAM(AdditiveModel, metaclass=ABCMeta):
    def __init__(
            self,
            n_features=None,
            symbols=None,
            symbol_names=None,
            terms: Union[str, List] = 'auto',
            **kwargs,
    ):
        """
        The base GAM class compatible with PyGAM models

        :param n_features: the number of features
        :param symbols: sequence of symbols, one for each feature
        :param symbol_names: the name of each feature/symbol
        :param terms: each term is a sequence of indices of features (symbols)
        :param kwargs: passed to underlying PyGAM model
        """
        super().__init__(
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )

        if not isinstance(terms, (str, pygam_terms.TermList)):
            converted_terms = []
            for term in terms:
                if isinstance(term, pygam_terms.SplineTerm):
                    continue
                if len(term) == 1:
                    gam_term = T.s(term[0], n_splines=25)
                else:
                    gam_term = T.te(*term, n_splines=10)
                converted_terms.append(gam_term)
            terms = sum(converted_terms[1:], converted_terms[0])

        self.fit_with_gridsearch = kwargs.pop('fit_with_gridsearch', False)
        kwargs['terms'] = terms
        kwargs.setdefault('max_iter', 100)
        kwargs.setdefault('verbose', True)
        self._estimator_kwargs = kwargs
        self._estimator_ = None

    @property
    @abstractmethod
    def is_classifier(self):
        """Set by children - whether model is a classifier"""
        raise NotImplementedError

    @property
    def _estimator(self):
        return (self._estimator_
                if isinstance(self._estimator_, (list, tuple)) else
                (self._estimator_,))

    @abstractmethod
    def __call__(self, X):
        """Call the PyGAM model"""
        raise NotImplementedError

    @property
    def _is_fitted(self):
        return (self._estimator is not None and
                all(estimator._is_fitted  # noqa
                    for estimator in self._estimator))

    @abstractmethod
    def fit(self, X, y, weights=None):
        """
        Fit the model to the training data.

        :param X: 2D input data (N x F)
        :param y: labels
        :param weights: sample weights
        :return: self
        """
        for i, estimator in enumerate(self._estimator):
            self._do_fit(estimator, i, X, y, weights=weights)

        return self

    def _do_fit(self, estimator, i, X, y, weights=None):
        if self.fit_with_gridsearch:
            estimator.gridsearch(X, y, weights=weights)
        else:
            estimator.fit(X, y, weights=weights)

    def feature_contributions(self, X, return_intercepts=False, **kwargs):
        """
        Return the feature contributions of each term of the GAM.

        :param X: 2d data (N x F)
        :param return_intercepts: also return the GAM intercepts
        :param kwargs: ignored
        :return:
        """
        if kwargs:
            warnings.warn(f'Ignoring all kwargs {kwargs} - these are N/A '
                          f'here.')
        contribs = []
        intercepts = []
        for e_i in self._estimator:
            contribs_i = {}
            for i, term in enumerate(e_i.terms):
                if term.isintercept:
                    assert (i + 1) == len(e_i.terms)
                    assert term.n_coefs == 1

                    intercepts.append(e_i.coef_[-1])
                else:
                    if isinstance(term.feature, list):
                        effect_symbols = [self.symbols[i]
                                          for i in term.feature]
                    else:
                        effect_symbols = [self.symbols[term.feature]]

                    effect = standardize_effect(effect_symbols)
                    contribs_i[effect] = e_i.partial_dependence(i, X)

            contribs.append(contribs_i)

        if len(contribs) == 1:
            if self.is_classifier:
                # TODO: this should work for more than two classes...
                # invert log odds
                contribs_1 = contribs[0]
                contribs_0 = {
                    effect: at_high_precision(inverse_log_odd, pd)
                    for effect, pd in contribs_1.items()
                }
                contribs = [contribs_0, contribs_1]
                intercepts = intercepts * 2
            else:
                contribs = contribs[0]
                intercepts = intercepts[0]

        if return_intercepts:
            return contribs, intercepts
        return contribs

    def predict(self, X):
        """
        Make predictions on the data

        :param X: 2d input data (N x F)
        :return: predictions
        """
        assert self._is_fitted

        y_pred = self(X)
        if self.is_classifier:
            y_pred = y_pred.argmax(axis=1)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for classifiers, raises a TypeError if a regressor

        :param X: 2d input data (N x F)
        :return: predicted probabilities
        """
        if self.is_classifier:
            return self(X)
        else:
            raise TypeError('Regression models do not have probabilities')


class MultiClassLogisticGAM(BaseGAM):
    """GAM for multi-class logistic classification"""

    def __init__(
            self,
            n_features=None,
            symbols=None,
            symbol_names=None,
            **kwargs,
    ):
        """
        GAM for multi-class logistic classification

        :param n_features: the number of features
        :param symbols: sequence of symbols, one for each feature
        :param symbol_names: the name of each feature/symbol
        :param terms: each term is a sequence of indices of features (symbols)
        :param kwargs: passed to BaseGAM
        """
        super().__init__(n_features=n_features, symbols=symbols,
                         symbol_names=symbol_names, **kwargs)

        self._y_encoder = None
        self._n_classes = None

    @property
    def is_classifier(self):
        return True

    def __call__(self, X):
        return self.predict_proba(X)

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
        """
        Fit the model to the training data.

        :param X: 2D input data (N x F)
        :param y: labels
        :param weights: sample weights
        :return: self
        """
        y = self._standardize_y(y)
        self._estimator_ = [LogisticGAM(**self._estimator_kwargs)
                            for _ in range(self._n_classes)]

        return super().fit(X, y, weights=weights)

    def _do_fit(self, estimator, i, X, y, weights=None):
        yi = y if self._n_classes == 1 else y[:, i]

        super()._do_fit(estimator, i, X, yi, weights=weights)

    def predict_proba(self, X):
        """
        Make predictions for classifiers

        :param X: 2d input data (N x F)
        :return: predicted probabilities
        """
        # y : np.array of shape (n_samples,)
        assert self._is_fitted

        if self._n_classes == 1:
            return self._estimator_[0].predict_proba(X)

        probas = []
        for i, estimator in enumerate(self._estimator_):
            # get probability at 1 (not 0)
            p_i = estimator.predict_proba(X)[:, 1]
            probas.append(p_i)

        return np.stack(probas, axis=1)


def inverse_log_odd(values):
    exp_vals = np.exp(values)
    # logistic
    log_odds = 1 - (exp_vals / (1 + exp_vals))  # val --> p
    # logit
    prob_vals = np.log(log_odds / (1 - log_odds))  # p --> val
    return prob_vals


class LinearGAM(BaseGAM):
    """Linear GAM for regression"""

    @property
    def is_classifier(self):
        return False

    def __call__(self, X):
        return self._estimator_.predict(X)

    def fit(self, X, y, weights=None):
        """
        Fit the model to the training data.

        :param X: 2D input data (N x F)
        :param y: labels
        :param weights: sample weights
        :return: self
        """
        self._estimator_ = _LinearGAM(**self._estimator_kwargs)

        return super().fit(X, y, weights=weights)
