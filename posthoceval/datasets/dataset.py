"""
dataset.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import List

import logging

import numpy as np
import pandas as pd

from posthoceval.utils import is_float
from posthoceval.utils import is_int
from posthoceval.utils import prod
from posthoceval.utils import is_pandas
from posthoceval.utils import is_df

logger = logging.getLogger(__name__)


def _needs_load(f):
    called_name = '__loaded_called__'

    def inner(self, *args, **kwargs):
        # Call _load() if not called before
        if not getattr(self, called_name, False):
            self._load()
            setattr(self, called_name, True)
        # Call wrapped function
        return f(self, *args, **kwargs)

    return inner


class Dataset(ABC):

    @abstractmethod
    def __init__(self, task):
        # TODO: standardize tasks between data and models
        task = task.lower()
        if task not in ['classification', 'regression']:
            raise ValueError(f'Unknown task: {task}')

        self.task: str = task
        # These will be set lazily
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._data: Optional[pd.DataFrame] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_types: Optional[List[str]] = None
        self._label_col: Optional[str] = None

    @property
    def is_classification(self):
        return self.task == 'classification'

    @property
    def is_regression(self):
        return self.task == 'regression'

    @property
    def n_features(self):
        return prod(self.input_shape)

    @property
    def input_shape(self):
        return self.X.shape[1:]

    @property
    @_needs_load
    def X(self):  # noqa
        if self._X is None:  # infer
            self.X = self.data[self.feature_names].values

        return self._X

    @X.setter
    def X(self, val):  # noqa
        if is_pandas(val):
            val = val.values
        else:
            val = np.asarray(val)
        assert val.ndim > 1, 'X needs to have ndim > 1'
        val = val.astype(np.float32)
        self._X = val
        self._validate_sizes()

    @property
    @_needs_load
    def y(self):
        if self._y is None:  # infer
            self.y = self.data[self.label_col].values
        return self._y

    @y.setter
    def y(self, val):
        if is_pandas(val):
            val = val.values
        else:
            val = np.asarray(val)

        if self.is_regression and not is_float(self._y):
            val = val.astype(np.float32)
        else:
            assert is_int(val), 'for classification y must be integers'

        # ensure it is a vector
        val = val.squeeze(axis=tuple(range(1, val.ndim)))
        assert val.ndim == 1

        self._y = val
        self._validate_sizes()

    @property
    @_needs_load
    def data(self):
        if self._data is None:  # infer
            data = pd.DataFrame(
                columns=self.feature_names,
                data=self.X,
            )
            data[self.label_col] = self.y
            self.data = data
        return self._data

    @data.setter
    def data(self, val):
        assert is_df(val)

        self._data = val
        assert all(feat_name in self._data for feat_name in self.feature_names)
        assert self.label_col in self._data

        self._validate_sizes()

    @property
    @_needs_load
    def feature_names(self):
        if self._feature_names is None:  # infer
            if self._data is None:
                self.feature_names = [*map(str, range(self.n_features))]
            else:
                self.feature_names = [name for name in self.data
                                      if name != self.label_col]
        return self._feature_names

    @feature_names.setter
    def feature_names(self, val):
        val = [*val]
        assert all(isinstance(v, str) for v in val)
        assert len(val) == len(set(val))
        self._feature_names = val
        self._validate_sizes()

    @property
    @_needs_load
    def feature_types(self):
        if self._feature_types is None:  # infer
            # TODO: dtypes do
            self.feature_types = ['float'] * self.n_features
        return self._feature_types

    @feature_types.setter
    def feature_types(self, val):
        val = [*val]
        # TODO: validate dtypes
        # assert all(isinstance(v, str) for v in val)
        self._feature_types = val
        self._validate_sizes()

    @property
    @_needs_load
    def label_col(self):
        if self._label_col is None:  # infer
            assert self._data is None
            self.label_col = 'target'
        return self._label_col

    @label_col.setter
    def label_col(self, val):
        assert isinstance(val, str)
        self._label_col = val

    def _validate_sizes(self):
        if self._X is not None:
            if self._y is not None:
                assert len(self._X) == len(self._y)
            if self._data is not None:
                assert len(self._X) == len(self._data)
                assert self.n_features >= (len(self._data.columns) - 1)
            if self._feature_names is not None:
                assert self.n_features == len(self._feature_names)
            if self._feature_types is not None:
                assert self.n_features == len(self._feature_types)
        elif self._y is not None:
            if self._data is not None:
                assert len(self._y) == len(self._data)

    @abstractmethod
    def _load(self):
        raise NotImplementedError
