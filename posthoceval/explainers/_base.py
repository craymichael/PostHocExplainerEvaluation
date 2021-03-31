from abc import ABC
from abc import abstractmethod

from typing import Union
from typing import Optional

from posthoceval.metrics import standardize_effect
from posthoceval.model_generation import AdditiveModel


class BaseExplainer(ABC):

    def __init__(self,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 verbose: Union[int, bool] = 1):
        self.model = model

        task = task.lower()
        if task not in ['regression', 'classification']:
            raise ValueError(f'Invalid task name: {task}')
        self.task = task

        self.verbose = verbose
        self.seed = seed

        # initialized in fit
        self._explainer = None

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def feature_contributions(self, X, return_y=False):
        raise NotImplementedError

    def _contribs_as_dict(self, contribs):
        # TODO: this assumes per-feature attribution
        symbols = map(standardize_effect, self.model.symbols)
        if self.task == 'regression':
            contribs = dict(zip(symbols, contribs.T))
        else:
            contribs = [dict(zip(symbols, contribs_k.T))
                        for contribs_k in contribs]
        return contribs
