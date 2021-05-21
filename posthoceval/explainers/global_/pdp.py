"""
pdp.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Union
from typing import List
from typing import Optional
from typing import Dict
from typing import Any

from itertools import chain

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from pdpbox.pdp import pdp_isolate
# from pdpbox.pdp import pdp_interact  # TODO- future work...

from posthoceval.rand import as_random_state
from posthoceval.rand import randint
from posthoceval.explainers._base import BaseExplainer
from posthoceval.explainers.global_.global_util import (
    MultivariateInterpolation)
from posthoceval.models.model import AdditiveModel


class _PDPBoxModelCompatRegression(object):
    """
    From pdpbox/utils.py:

    def _check_model(model):
    '''Check model input, return class information and predict function'''
    try:
        n_classes = len(model.classes_)
        predict = model.predict_proba
    except:
        n_classes = 0
        predict = model.predict

    return n_classes, predict
    """

    # noinspection PyUnusedLocal
    def __init__(self, model: AdditiveModel, X=None):
        self.model = model  # wrapped model

    # noinspection PyPep8Naming
    @staticmethod
    def _handle_X(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X

    def predict(self, X):
        return self.model.predict(self._handle_X(X))


class _PDPBoxModelCompatClassification(_PDPBoxModelCompatRegression):
    def __init__(self, model: AdditiveModel, X: np.ndarray):
        super().__init__(model=model)
        # sniff the number of classes
        y_subset = self.predict_proba(X[:1])
        if y_subset.ndim != 2:
            raise ValueError(
                f'Classifier models with output ndim != 2 are not supported. '
                f'Wrapped model {model.__class__.__name__} has ndim == '
                f'{y_subset.ndim}'
            )
        self.classes_ = [*range(y_subset.shape[1])]

    def predict_proba(self, X):
        return self.model.predict_proba(self._handle_X(X))


class PDPExplainer(BaseExplainer):
    """https://github.com/SauceCat/PDPbox/blob/master/pdpbox/pdp.py"""
    # TODO: interactions using pdp_interact???? how do

    _explainer: Union[MultivariateInterpolation,
                      List[MultivariateInterpolation]]

    def __init__(self,
                 model: AdditiveModel,
                 seed=None,
                 task: str = 'regression',
                 n_grid_points: int = 100,
                 interpolation='linear',
                 n_jobs: int = 1,
                 max_samples: int = 10000,
                 verbose=True):
        super().__init__(
            tabular=True,
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self.n_grid_points = n_grid_points
        self.interpolation = interpolation
        self.max_samples = max_samples
        self.n_jobs = n_jobs

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ) -> None:
        # needs to be list, not tuple
        feature_names = [*self.model.symbol_names]
        if grouped_feature_names is None:
            grouped_feature_names = feature_names

        # TODO: rename max_samples........
        n_samples = round(self.max_samples * 25 / X.shape[1])
        if len(X) > n_samples:
            rng = as_random_state(self.seed)
            X = X[randint(0, len(X), size=n_samples, seed=rng)]

        dataset = pd.DataFrame(
            columns=feature_names,
            data=X,
        )

        wrapper_cls = (_PDPBoxModelCompatRegression
                       if self.task == 'regression' else
                       _PDPBoxModelCompatClassification)
        wrapped_model = wrapper_cls(model=self.model, X=X)

        all_x = []
        all_y = []
        for feature in tqdm(grouped_feature_names, unit='feature',
                            disable=not self.verbose):
            is_grouped = not isinstance(feature, str)
            if is_grouped:
                # one-hot encoded feature
                # TODO: this is very bad-looking (" = ")
                feature = [f'{feature[0]} = {val}' for val in feature[1]]
            # classification vs. regression automatically handled by pdpbox
            pdp_feat = pdp_isolate(
                model=wrapped_model,
                dataset=dataset,
                model_features=feature_names,
                feature=feature,
                num_grid_points=self.n_grid_points,
                grid_type='percentile',
                n_jobs=self.n_jobs,
            )
            if self.task == 'regression':
                if is_grouped:
                    all_x.extend(pdp_feat.feature_grids[i:i + 1]
                                 for i in range(len(feature)))
                    all_y.extend(pdp_feat.pdp[i:i + 1]
                                 for i in range(len(feature)))
                else:
                    all_x.append(pdp_feat.feature_grids)
                    all_y.append(pdp_feat.pdp)
            else:
                if len(wrapped_model.classes_) == 2:
                    # TODO: this is not correct....
                    if is_grouped:
                        for i in range(len(feature)):
                            all_x.append(
                                [pdp_feat.feature_grids[i:i + 1]] * 2)
                            all_y.append(
                                [pdp_feat.pdp[i:i + 1]] * 2)
                    else:
                        all_x.append([pdp_feat.feature_grids] * 2)
                        all_y.append([pdp_feat.pdp] * 2)
                else:
                    if is_grouped:
                        for i in range(len(feature)):
                            all_x.append([p.feature_grids[i:i + 1]
                                          for p in pdp_feat])
                            all_y.append([p.pdp[i:i + 1]
                                          for p in pdp_feat])
                    else:
                        all_x.append([p.feature_grids for p in pdp_feat])
                        all_y.append([p.pdp for p in pdp_feat])

        if self.task == 'regression':
            self._explainer = MultivariateInterpolation(
                x=np.asarray(all_x).T,
                y=np.asarray(all_y).T,
                interpolation=self.interpolation,
            )
        else:
            n_classes = len(all_x[0])
            assert all(len(fx) == n_classes
                       for fx in chain(all_x[1:], all_y))

            self._explainer = [MultivariateInterpolation(
                x=np.asarray([all_xf[k] for all_xf in all_x]).T,
                y=np.asarray([all_yf[k] for all_yf in all_y]).T,
                interpolation=self.interpolation,
            ) for k in range(n_classes)]

    def predict(self, X):
        raise NotImplementedError

    def _call_explainer(
            self,
            X: np.ndarray,
    ) -> Dict[str, Any]:
        if self.task == 'regression':
            contribs = self._explainer.interpolate(X)
            predictions = np.sum(contribs, axis=1)
        else:
            contribs = [expl.interpolate(X)
                        for expl in self._explainer]
            predictions = np.sum(contribs, axis=2)

        return {'contribs': contribs,
                'predictions': predictions}
