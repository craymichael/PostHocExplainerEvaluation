"""
model.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from abc import ABC
from abc import abstractmethod

import numpy as np


class AdditiveModel(ABC):

    def __init__(
            self,
            n_features,
            symbol_names,
            symbols,
    ):
        # TODO: validation...maybe
        self.n_features = n_features
        self.symbol_names = symbol_names
        self.symbols = symbols

    @abstractmethod
    def __call__(
            self,
            x: np.ndarray,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):  # sklearn compat
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x):  # sklearn compat
        raise NotImplementedError

    @abstractmethod
    def feature_contributions(
            self,
            X: np.ndarray,
    ):
        raise NotImplementedError
