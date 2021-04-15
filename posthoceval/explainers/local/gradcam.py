import logging

from typing import Union
from typing import Optional

from posthoceval.model_generation import AdditiveModel
from posthoceval.explainers._base import BaseExplainer

logger = logging.getLogger(__name__)


class GradCAMExplainer(BaseExplainer):

    def __init__(self,
                 model: AdditiveModel,
                 task: str = 'regression',
                 seed: Optional[int] = None,
                 verbose: Union[int, bool] = 1,
                 **explainer_kwargs):
        """"""
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self._explainer = None

    def fit(self, X, y=None):
        if self.verbose > 0:
            logger.info('Fitting GradCAM')
        self._explainer.fit(X)

    def predict(self, X):
        pass  # TODO: n/a atm

    def feature_contributions(self, X, return_y=False, as_dict=False):
        if self.verbose > 0:
            logger.info('Fetching GradCAM explanations')

        # TODO so much to do
        explanation = self._explainer.explain(X)

        if self.task == 'regression':
            contribs = explanation[0]
        else:
            contribs = explanation

        if as_dict:
            contribs = self._contribs_as_dict(contribs)

        if return_y:
            return contribs, self.model(X)
        return contribs
