"""
lime.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import logging

from typing import Optional
from typing import Union

from operator import itemgetter

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
                 verbose: Union[int, bool] = 1):
        """"""
        self.model = model
        self.verbose = verbose
        self.seed = seed

        # will be initialized in fit
        self._explainer = None

    def fit(self, X, y=None):
        if self.verbose > 0:
            logger.info('Fitting LIME')

        self._explainer = LimeTabular(
            predict_fn=self.model,
            data=X,
            feature_names=range(self.model.n_features),
            mode='regression',  # TODO: add classification to API
            random_state=self.seed,
            discretize_continuous=False,
            explain_kwargs={'num_features': X.shape[1],
                            'num_samples': 1000}
        )

    def predict(self, X):
        pass  # TODO: n/a

    @profile
    def feature_contributions(self, X, as_dict=False):
        if self._explainer is None:
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        if self.verbose > 0:
            logger.info('Fetching LIME explanations')

        # passes explain_kwargs from LIME init to LIME explain_instance
        explanation = self._explainer.explain_local(X)

        contribs_lime = []

        for i, xi in enumerate(X):
            expl_i = explanation.data(i)
            # sort explanation values...
            coefs_i = itemgetter(*sorted(expl_i['names']))(expl_i['scores'])
            contribs_i = [coef_ij * xij
                          for coef_ij, xij in zip(coefs_i, xi)]
            contribs_lime.append(contribs_i)

        contribs_lime = np.asarray(contribs_lime)
        # TODO: move this logic to super? likely redundant between explainers
        if as_dict:
            contribs_lime = dict(zip(self.model.symbols, contribs_lime.T))

        return contribs_lime
