"""
dataset.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Iterable

import logging

import numpy as np
import pandas as pd

from posthoceval.utils import prod
from posthoceval.utils import is_float
from posthoceval.utils import is_int
from posthoceval.utils import is_pandas
from posthoceval.utils import is_df

__all__ = ['Dataset', 'CustomDataset']

logger = logging.getLogger(__name__)


def _needs_load(f):
    called_name = '__loaded_called__'

    def inner(self, *args, **kwargs):
        # Call _load() if not called before
        if not getattr(self, called_name, False):
            setattr(self, called_name, True)
            self._load()
        # Call wrapped function
        return f(self, *args, **kwargs)

    return inner


class Dataset(ABC):

    def __init__(self, task: str):
        # TODO: standardize tasks between data and models
        task = task.lower()
        if task not in ['classification', 'regression']:
            raise ValueError(f'Unknown task: {task}')

        self.task: str = task
        # These will be set lazily
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._data: Optional[pd.DataFrame] = None
        self._grouped_feature_names: Optional[
            List[Union[str, Dict[str, List[str]]]]] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_types: Optional[List[str]] = None
        self._label_col: Optional[str] = None
        # These will be set lazily and a function of those provided in _load
        self._X_df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        # return length without triggering lazy-loading, if possible
        if self._y is not None:
            return len(self.y)
        elif self._X is not None:
            return len(self.X)
        else:
            return len(self.data)

    @classmethod
    def from_data(cls, task: str, *args, **kwargs) -> 'Dataset':
        dataset = cls(task=task)
        # >:)
        Dataset._load(dataset, *args, **kwargs)
        return dataset

    @property
    def is_classification(self) -> bool:
        return self.task == 'classification'

    @property
    def is_regression(self) -> bool:
        return self.task == 'regression'

    @property
    def n_features(self) -> int:
        return prod(self.input_shape)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.X.shape[1:]

    @property
    def categorical_features(self) -> List[str]:
        return [
            feat_name
            for feat_name, feat_type in zip(
                self.feature_names, self.feature_types)
            if feat_type == 'categorical'
        ]

    @property
    def numerical_features(self) -> List[str]:
        return [
            feat_name
            for feat_name, feat_type in zip(
                self.feature_names, self.feature_types)
            if feat_type == 'numerical'
        ]

    @property
    def X_df(self) -> pd.DataFrame:  # noqa
        if self._X_df is None:
            self._X_df = self.data[self.feature_names]
        return self._X_df

    @property
    @_needs_load
    def X(self) -> np.ndarray:  # noqa
        if self._X is None:  # infer
            self.X = self.data[self.feature_names].values

        return self._X

    @X.setter
    def X(self, val: Union[pd.DataFrame, np.ndarray]):  # noqa
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
    def y(self) -> np.ndarray:
        if self._y is None:  # infer
            self.y = self.data[self.label_col].values
        return self._y

    @y.setter
    def y(self, val: Union[pd.Series, pd.DataFrame, np.ndarray]):
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
        assert val.ndim == 1, 'y ndim is not 1'

        self._y = val
        self._validate_sizes()

    @property
    @_needs_load
    def data(self) -> pd.DataFrame:
        if self._data is None:  # infer
            data = pd.DataFrame(
                columns=self.feature_names,
                data=self.X,
            )
            data[self.label_col] = self.y
            self.data = data
        return self._data

    @data.setter
    def data(self, val: pd.DataFrame):
        assert is_df(val), 'val is not df'

        self._data = val
        assert all(feat_name in self._data
                   for feat_name in self.feature_names), (
            'feat names not in data')
        assert self.label_col in self._data, 'label_col not in data'

        self._validate_sizes()

    @property
    @_needs_load
    def feature_names(self) -> List[str]:
        if self._feature_names is None:  # infer
            if self._data is None:
                self.feature_names = [*map(str, range(self.n_features))]
            else:
                self.feature_names = self.data.columns.drop(
                    self.label_col).to_list()
        return self._feature_names

    @feature_names.setter
    def feature_names(self, val: Iterable[str]):
        val = [*val]
        assert all(isinstance(v, str) for v in val), 'feat names not all str'
        assert len(val) == len(set(val)), 'non-unique feat names'
        self._feature_names = val
        self._validate_sizes()

    @property
    @_needs_load
    def grouped_feature_names(self) -> List[Union[str, Dict[str, List[str]]]]:
        if self._grouped_feature_names is None:  # infer
            self.grouped_feature_names = self.feature_names.copy()
        return self._grouped_feature_names

    @grouped_feature_names.setter
    def grouped_feature_names(
            self,
            val: Iterable[Union[str, Dict[str, List[str]]]]
    ):
        val = [*val]
        self._grouped_feature_names = val
        self._validate_sizes()

    @property
    @_needs_load
    def feature_types(self) -> List[str]:
        if self._feature_types is None:  # infer
            # TODO: dtypes do
            self.feature_types = ['numerical'] * self.n_features
        return self._feature_types

    @feature_types.setter
    def feature_types(self, val: Iterable[str]):
        valid_types = ['numerical', 'categorical']
        val = [*val]
        assert all(v in valid_types for v in val)
        self._feature_types = val
        self._validate_sizes()

    @property
    @_needs_load
    def label_col(self) -> str:
        if self._label_col is None:  # infer
            assert self._data is None, 'data is not None'
            self.label_col = 'target'
        return self._label_col

    @label_col.setter
    def label_col(self, val: str):
        assert isinstance(val, str), 'val is not str'
        self._label_col = val

    def _validate_sizes(self):
        if self._X is not None:
            if self._y is not None:
                assert len(self._X) == len(self._y), 'X and y lengths differ'
            if self._data is not None:
                assert len(self._X) == len(self._data), (
                    'X and data lengths differ')
                assert self.n_features >= (len(self._data.columns) - 1), (
                    'n_features differs from # data columns')
            if self._feature_names is not None:
                assert self.n_features == len(self._feature_names), (
                    'n_features differs from # feat names')
            if self._feature_types is not None:
                assert self.n_features == len(self._feature_types), (
                    'n_features differs from # feat types')
            if self._grouped_feature_names is not None:
                tot_grouped_names = sum(
                    1 if isinstance(name, str) else len(name.keys())
                    for name in self._grouped_feature_names
                )
                assert self.n_features == tot_grouped_names, (
                    'n_features differs from total # of grouped_feature_names')
        elif self._y is not None:
            if self._data is not None:
                assert len(self._y) == len(self._data), (
                    'y and data lengths differ')

    @abstractmethod
    def _load(self, *args, **kwargs) -> None:
        if not args:
            assert kwargs, 'need kwargs if no load_dict'
            maybe_load_dict = kwargs.get('load_dict')
            if len(kwargs) == 1 and maybe_load_dict is not None:
                load_dict = maybe_load_dict
            else:
                load_dict = kwargs
        else:
            assert len(args) == 1, 'only 1 value for load_dict can be provided'
            assert not kwargs, 'cannot have both load_dict and kwargs'
            load_dict = args[0]

        loaded_keys = set(load_dict.keys())
        expect_keys = ['label_col', 'grouped_feature_names', 'feature_names',
                       'feature_types', 'X', 'y', 'data']
        assert not (loaded_keys - set(expect_keys)), (
            f'unknown keys in load_dict: {loaded_keys}')

        # set in the correct order
        for key in expect_keys:
            loaded_val = load_dict.get(key)
            if loaded_val is None:
                continue
            setattr(self, key, loaded_val)


class CustomDataset(Dataset):
    """
    Usage:
    >>> ds1 = CustomDataset(task='classification', X=X, y=y)
    >>> ds2 = CustomDataset(task='regression', data=df, label_col='target')
    """

    def __init__(self, task, *args, **kwargs):
        super().__init__(task)
        self._load(*args, **kwargs)

    def _load(self, *args, **kwargs):
        super()._load(*args, **kwargs)
