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

from pdpbox.pdp import pdp_isolate
# from pdpbox.pdp import pdp_interact  # TODO- future work...

from posthoceval.explainers._base import BaseExplainer
from posthoceval.explainers.global_.global_util import (
    MultivariateInterpolation)
from posthoceval.models.model import AdditiveModel


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

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ) -> None:
        # needs to be list, not tuple
        feature_names = [*self.model.symbol_names]

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
        pass  # TODO

    def _call_explainer(
            self,
            X: np.ndarray,
    ) -> Dict[str, Any]:
        if self.task == 'regression':
            contribs = self._explainer.interpolate(X)
        else:
            contribs = [expl.interpolate(X)
                        for expl in self._explainer]

        return {'contribs': contribs}
