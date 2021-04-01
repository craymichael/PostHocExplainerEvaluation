"""
pdp.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from itertools import chain
from typing import Union
from typing import List

import pandas as pd

from pdpbox.pdp import pdp_isolate
from pdpbox.pdp import pdp_interact

from posthoceval.explainers._base import BaseExplainer
from posthoceval.explainers.global_.global_util import MultivariateInterpolation
from posthoceval.model_generation import AdditiveModel


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
                 verbose=True):
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self.n_grid_points = n_grid_points
        self.interpolation = interpolation
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        feature_names = self.model.symbol_names

        # TODO: consider categorical stuff here - PDP sets up groups by having
        #  categorical feature name "foo" suffixed by "_value" - e.g., feature
        #  "color" with values "red" and "blue" would result in the columns
        #  "color_red" and "color_blue" and be handled accordingly/
        #  automatically within PDPBox lib

        dataset = pd.DataFrame(
            columns=feature_names,
            data=X,
        )

        all_x = []
        all_y = []
        for feature in feature_names:
            # classification vs. regression automatically handled by pdpbox
            pdp_feat = pdp_isolate(
                model=self.model,
                dataset=dataset,
                model_features=feature_names,
                feature=feature,
                num_grid_points=self.n_grid_points,
                grid_type='percentile',
                n_jobs=self.n_jobs,
            )
            if self.task == 'regression':
                all_x.append(pdp_feat.feature_grids)
                all_y.append(pdp_feat.pdp)
            else:
                all_x.extend(p.feature_grids for p in pdp_feat)
                all_y.extend(p.pdp for p in pdp_feat)

        if self.task == 'regression':
            self._explainer = MultivariateInterpolation(
                x=all_x, y=all_y, interpolation=self.interpolation,
            )
        else:
            n_classes = len(all_x[0])
            assert all(len(fx) == n_classes
                       for fx in chain(all_x[1:], all_y))

            self._explainer = [MultivariateInterpolation(
                x=[all_xf[k] for all_xf in all_x],
                y=[all_yf[k] for all_yf in all_y],
                interpolation=self.interpolation,
            ) for k in range(n_classes)]

    def predict(self, X):
        pass  # TODO

    def feature_contributions(self, X, return_y=False, as_dict=False):
        if self.task == 'regression':
            contribs = self._explainer.interpolate(X)
        else:
            contribs = [expl.interpolate(X)
                        for expl in self._explainer]

        if as_dict:
            contribs = self._contribs_as_dict(contribs)

        if return_y:
            return contribs, self.model(X)
        return contribs
