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
                 num_samples: int = 5000,
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

        self.num_samples = num_samples

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
                            'num_samples': self.num_samples}
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
            # LIME scales with z-score (StandardScaler)
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

# aids

# TODO: Make kwargs explicit.
class LimeTabular(ExplainerMixin):

    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        feature_names=None,
        feature_types=None,
        **kwargs
    ):
        """ Initializes class.
        Args:
            predict_fn: Function of blackbox that takes input, and returns prediction.
            data: Data used to initialize LIME with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs that will be sent to lime at initialization time.
        """
        from lime.lime_tabular import LimeTabularExplainer

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = predict_fn

        self.kwargs = kwargs
        final_kwargs = {"mode": "regression"}
        if self.feature_names:
            final_kwargs["feature_names"] = self.feature_names
        final_kwargs.update(self.kwargs)

        self.lime = LimeTabularExplainer(self.data, **final_kwargs)

    def explain_local(self, X, y=None):
        """ Generates local explanations for provided instances.
        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        pred_fn = self.predict_fn

        data_dicts = []
        scores_list = []
        for i, instance in enumerate(X):
            lime_explanation = self.lime.explain_instance(
                instance, pred_fn, **self.explain_kwargs
            )

            names = []
            scores = []
            values = []
            feature_idx_imp_pairs = lime_explanation.as_map()[1]
            for feat_idx, imp in feature_idx_imp_pairs:
                names.append(self.feature_names[feat_idx])
                scores.append(imp)
                values.append(instance[feat_idx])
            intercept = lime_explanation.intercept[1]

            scores_list.append(scores)

            data_dict = {
                "names": names,
                "scores": scores,
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            data_dicts.append(data_dict)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "local_feature_importance",
                    "value": {
                        "scores": scores_list,
                        "intercept": intercept,
                    },
                }
            ],
        }

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
        )
