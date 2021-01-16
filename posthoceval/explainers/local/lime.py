"""
lime.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import logging

from typing import Optional
from typing import Union

import numpy as np

from interpret.blackbox import LimeTabular

from posthoceval.profile import profile
from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """"""

    def __init__(self,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 verbose: Union[int, bool] = 1):
        """"""
        self.model = model
        self.verbose = verbose
        self.seed = seed
        if task != 'regression':
            # you'd think that this wrapper would make things easier...
            raise NotImplementedError('Backend `interpret` does not have '
                                      'proper classification support....')
        self.task = task

        # will be initialized in fit
        self._explainer = None

        self._mean = None
        self._scale = None

    def fit(self, X, y=None):
        if self.verbose > 0:
            logger.info('Fitting LIME')

        self._explainer = LimeTabular(
            predict_fn=self.model,
            data=X,
            feature_names=range(self.model.n_features),
            mode=self.task,
            random_state=self.seed,
            discretize_continuous=False,
            explain_kwargs={'num_features': X.shape[1],
                            'num_samples': 5000}
        )

        # used in un-normalizing contributions
        self._mean = self._explainer.lime.scaler.mean_
        self._scale = self._explainer.lime.scaler.scale_

    def predict(self, X):
        pass  # TODO: n/a

    @profile
    def feature_contributions(self, X: np.ndarray, as_dict=False,
                              return_intercepts=False):
        # Note that LIME must have sample_around_instance=False otherwise
        #  the inverse normalization is invalid

        if self._explainer is None:
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        if self.verbose > 0:
            logger.info('Fetching LIME explanations')

        # passes explain_kwargs from LIME init to LIME explain_instance
        explanation = self._explainer.explain_local(X)

        contribs_lime = []
        intercepts = []

        for i, xi in enumerate(X):
            expl_i = explanation.data(i)

            # sort explanation values...
            expl_names = expl_i['names']  # feature indices
            expl_scores = expl_i['scores']  # unsorted coefficients
            # get all items and sort them to match order of features
            coefs_i, _ = zip(*sorted(zip(expl_scores, range(len(expl_names))),
                                     key=lambda x: expl_names[x[1]]))
            # LIME scales only (StandardScale with `with_mean=False`)
            coefs_i = np.asarray(coefs_i) / self._scale

            # if desired, intercept can be fetched and adjusted via:
            if return_intercepts:
                intercept = (expl_i['extra']['scores'][0] -
                             np.sum(coefs_i * self._mean))
                intercepts.append(intercept)

            contribs_i = coefs_i * xi
            contribs_lime.append(contribs_i)

        contribs_lime = np.asarray(contribs_lime)
        # TODO: move this logic to super? likely redundant between explainers
        if as_dict:
            contribs_lime = dict(zip(self.model.symbols, contribs_lime.T))

        if return_intercepts:
            return contribs_lime, intercepts

        return contribs_lime
