"""
lime.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import logging

from typing import Optional
from typing import Union

import numpy as np

from lime.lime_tabular import LimeTabularExplainer

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
        self.task = task

        self.num_samples = num_samples

        # will be initialized in fit
        self._explainer: Optional[LimeTabularExplainer] = None

        self._mean = None
        self._scale = None

    def fit(self, X, y=None):
        if self.verbose > 0:
            logger.info('Fitting LIME')

        # TODO:
        #  categorical_features: list of indices (ints) corresponding to the
        #    categorical columns. Everything else will be considered
        #    continuous. Values in these columns MUST be integers.
        #  categorical_names: map from int to list of names, where
        #    categorical_names[x][y] represents the name of the yth value of
        #    column x.

        self._explainer = LimeTabularExplainer(
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
        pass  # TODO: n/a

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
    def feature_contributions(self, X: np.ndarray, as_dict=False,
                              return_intercepts=False):
        # Note that LIME must have sample_around_instance=False otherwise
        #  the inverse normalization is invalid

        if self._explainer is None:
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

                    if return_intercepts:
                        contribs_ik, intercept_ik = self._process_explanation(
                            expl_ik, xi, intercept_i=expl_i.intercept[k])
                        intercept_i.append(intercept_ik)
                    else:
                        contribs_ik = self._process_explanation(expl_ik, xi)

                    contribs_i.append(contribs_ik)
            else:
                expl_i = expl_i_map[1]

                if return_intercepts:
                    contribs_i, intercept_i = self._process_explanation(
                        expl_i, xi, intercept_i=expl_i.intercept[1])
                else:
                    contribs_i = self._process_explanation(expl_i, xi)

            # store contributions and maybe intercept(s)
            contribs_lime.append(contribs_i)
            if return_intercepts:
                # noinspection PyUnboundLocalVariable
                intercepts.append(intercept_i)

        contribs_lime = np.asarray(contribs_lime)

        if self.task == 'classification':
            # samples x classes x features --> classes x samples x features
            contribs_lime = np.moveaxis(contribs_lime, 0, 1)

        # TODO: move this logic to super? likely redundant between explainers
        if as_dict:
            contribs_lime = dict(zip(self.model.symbols, contribs_lime.T))

        if return_intercepts:
            intercepts = np.asarray(intercepts)
            if self.task == 'classification':
                # samples x classes --> classes x samples
                intercepts = intercepts.T
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
                "extra": {"names": ["Intercept"], "scores": [intercept],
                          "values": [1]},
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
