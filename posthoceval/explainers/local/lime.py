"""
lime.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import logging

from typing import Optional
from typing import Union

import numpy as np

from posthoceval.profile import profile
from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel

logger = logging.getLogger(__name__)


class LIMETabularExplainer(BaseExplainer):
    """LIME explainer for tabular data"""

    def __init__(self,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 num_samples: int = 5000,
                 verbose: Union[int, bool] = 1):
        """
        LIME explainer for tabular data.

        :param model: the model to explain
        :param seed: the RNG seed for reproducibility
        :param task: the task, either "classification" or "regression"
        :param num_samples: the size of the neighborhood to learn the linear
            model
        :param verbose: print more messages if True
        """
        super().__init__(
            model=model,
            tabular=True,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self.num_samples = num_samples

        self._mean = None
        self._scale = None

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ):
        if self.verbose > 0:
            logger.info('Fitting LIME')

        # TODO: (using grouped_feature_names)
        #  categorical_features: list of indices (ints) corresponding to the
        #    categorical columns. Everything else will be considered
        #    continuous. Values in these columns MUST be integers.
        #  categorical_names: map from int to list of names, where
        #    categorical_names[x][y] represents the name of the yth value of
        #    column x.

        # lazy load
        from lime.lime_tabular import (
            LimeTabularExplainer as _LimeTabularExplainer)

        self._explainer: Optional[_LimeTabularExplainer]
        self._explainer = _LimeTabularExplainer(
            training_data=X,
            feature_names=range(self.model.n_features),
            mode=self.task,
            random_state=self.seed,
            discretize_continuous=False,
        )

        # used in un-normalizing contributions
        self._mean = self._explainer.scaler.mean_
        self._scale = self._explainer.scaler.scale_

    def predict(self, X):
        return NotImplementedError

    def _process_explanation(self, expl_vals_i, xi, intercept_i=None):
        # sort explanation values...
        _, coefs_i = zip(*sorted(expl_vals_i, key=lambda x: x[0]))
        # LIME scales with z-score (StandardScaler)
        coefs_i = np.asarray(coefs_i) / self._scale
        # feature-wise contributions to output
        contribs_i = coefs_i * xi

        # if desired, intercept can be fetched and adjusted
        if intercept_i is not None:
            intercept_i = (intercept_i -
                           np.sum(coefs_i * self._mean))
            return contribs_i, intercept_i
        return contribs_i

    @profile
    def _call_explainer(self, X: np.ndarray):
        # Note that LIME must have sample_around_instance=False otherwise
        #  the inverse normalization is invalid

        if self._explainer is None:  # TODO: to base...
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        if self.verbose > 0:
            logger.info('Fetching LIME explanations')

        contribs_lime = []
        intercepts = []

        explain_kwargs = {
            'predict_fn': self.model,
            'num_features': X.shape[-1],
            'num_samples': self.num_samples,
        }
        if self.task == 'classification':
            # get number of classes, obtain feature contributions for each
            #  class
            explain_kwargs['top_labels'] = self.model(X[:1]).shape[-1]

        for i, xi in enumerate(X):
            expl_i = self._explainer.explain_instance(xi, **explain_kwargs)
            expl_i_map = expl_i.as_map()

            if self.task == 'classification':
                contribs_i = []
                intercept_i = []
                for k in range(explain_kwargs['top_labels']):
                    # explanation coefs for class k
                    expl_ik = expl_i_map[k]

                    contribs_ik, intercept_ik = self._process_explanation(
                        expl_ik, xi, intercept_i=expl_i.intercept[k])
                    intercept_i.append(intercept_ik)
                    contribs_i.append(contribs_ik)
            else:  # regression (one output)
                expl_i1 = expl_i_map[1]

                contribs_i, intercept_i = self._process_explanation(
                    expl_i1, xi, intercept_i=expl_i.intercept[1])

            # store contributions and intercept(s)
            contribs_lime.append(contribs_i)
            intercepts.append(intercept_i)

        contribs_lime = np.asarray(contribs_lime)
        intercepts = np.asarray(intercepts)

        if self.task == 'classification':
            # samples x classes x features --> classes x samples x features
            contribs_lime = np.moveaxis(contribs_lime, 0, 1)
            # samples x classes --> classes x samples
            intercepts = intercepts.T
            # predictions
            y_expl = contribs_lime.sum(axis=2) + intercepts
        else:
            # predictions
            y_expl = contribs_lime.sum(axis=1) + intercepts

        return {'contribs': contribs_lime, 'intercepts': intercepts,
                'predictions': y_expl}


LIMEExplainer = LIMETabularExplainer
