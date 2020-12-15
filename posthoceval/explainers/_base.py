from abc import ABC
from abc import abstractmethod


class BaseExplainer(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def feature_contributions(self, X, return_y=False):
        raise NotImplementedError
