"""
dataset.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Optional
from typing import List
from typing import Any
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Sequence
from typing import NoReturn

from abc import ABC
from abc import abstractmethod

from copy import deepcopy

import logging

import numpy as np
import pandas as pd

from posthoceval.utils import prod
from posthoceval.utils import as_int
from posthoceval.utils import is_float
from posthoceval.utils import is_pandas
from posthoceval.utils import is_df

__all__ = ['Dataset', 'CustomDataset']

logger = logging.getLogger(__name__)


def _needs_load(f):
    """
    Wrapper for properties that require than `_load()` be called exactly once
    before a property is accessed. Specific for Dataset objects

    :param f: the property method
    :return: wrapper that calls `_load()`
    """

    def inner(self: 'Dataset', *args, **kwargs):
        # Call _load() if not called before
        if not self.__load_called__:
            logger.info(f'Loading bas begun for {self.__class__.__name__}')
            self.__load_called__ = True
            self._load()
            logger.info(f'Loading has finished for {self.__class__.__name__}')
        # Call wrapped function
        return f(self, *args, **kwargs)

    return inner


class Dataset(ABC):
    """Base Dataset object"""

    __expect_keys = ['label_col', 'grouped_feature_names', 'feature_names',
                     'feature_types', 'X', 'y', 'data']

    def __init__(self, task: str):
        """
        Base Dataset object

        :param task: either "classification" or "regression"
        """
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
            List[Union[str, Tuple[str, Sequence[Any]]]]] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_types: Optional[List[str]] = None
        self._label_col: Optional[str] = None
        # These will be set lazily and a function of those provided in _load
        self._X_df: Optional[pd.DataFrame] = None

        self.__load_called__ = False

    def _raise_bad_task(self) -> NoReturn:
        """Raise exception stating specified task is unsupported"""
        raise ValueError(f'{self.task} is not supported for '
                         f'{self.__class__.__name__}')

    def __len__(self) -> int:
        """
        :return: number of samples in the dataset
        """
        # return length without triggering lazy-loading, if possible
        if self._y is not None:
            return len(self.y)
        elif self._X is not None:
            return len(self.X)
        else:
            return len(self.data)

    @classmethod
    def from_data(cls, task: str, *args, **kwargs) -> 'Dataset':
        """
        Creates a new dataset from the provided data

        :param task: either "classification" or "regression"
        :param args: data, see _load documentation
        :param kwargs: data, see _load documentation
        :return: a new Dataset instance
        """
        dataset = cls(task=task)
        # >:)
        dataset.__load_called__ = True
        Dataset._load(dataset, *args, **kwargs)
        return dataset

    def copy(self) -> 'Dataset':
        """
        :return: deep copy of the dataset
        """
        return deepcopy(self)

    @property
    def is_classification(self) -> bool:
        """
        :return: whether the task is classification
        """
        return self.task == 'classification'

    @property
    def is_regression(self) -> bool:
        """
        :return: whether the task is regression
        """
        return self.task == 'regression'

    @property
    def n_features(self) -> int:
        """
        :return: the number of features in the dataset
        """
        return prod(self.input_shape)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        :return: the shape of a data sample
        """
        return self.X.shape[1:]

    @property
    def categorical_features(self) -> List[str]:
        """

        :return: list of categorical feature names
        """
        return [
            feat_name
            for feat_name, feat_type in zip(
                self.feature_names, self.feature_types)
            if feat_type == 'categorical'
        ]

    @property
    def numerical_features(self) -> List[str]:
        """

        :return: list of numerical feature names
        """
        return [
            feat_name
            for feat_name, feat_type in zip(
                self.feature_names, self.feature_types)
            if feat_type == 'numerical'
        ]

    @property
    def X_df(self) -> pd.DataFrame:  # noqa
        """
        Returns the data without labels as a pandas DataFrame. If the data
        is not tabular, then the data is flattened first

        :return: the dataset as a tabular dataframe
        """
        if self._X_df is None:
            self._X_df = self.data[self.feature_names]
        return self._X_df

    @property
    @_needs_load
    def X(self) -> np.ndarray:  # noqa
        """

        :return: the dataset without labels
        """
        if self._X is None:  # infer
            self.X = self.data[self.feature_names].values

        return self._X

    @X.setter
    def X(self, val: Union[pd.DataFrame, np.ndarray]):  # noqa
        """
        Update and validate X

        :param val: value which will update X
        """
        if is_pandas(val):
            val = val.values
        else:
            val = np.asarray(val)
        assert val.ndim > 1, 'X needs to have ndim > 1'
        try:
            val = val.astype(np.float32)
        except ValueError:
            logger.debug(f'{self.__class__.__name__} X contains non-float '
                         f'column(s)')
        self._X = val
        self._validate_sizes()

    @property
    @_needs_load
    def y(self) -> np.ndarray:
        """

        :return: the labels of the dataset
        """
        if self._y is None:  # infer
            self.y = self.data[self.label_col].values
        return self._y

    @y.setter
    def y(self, val: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Update and validate y

        :param val: the values which will update the labels
        """
        if is_pandas(val):
            val = val.values
        else:
            val = np.asarray(val)

        if self.is_regression and not is_float(self._y):
            val = val.astype(np.float32)
        else:
            try:
                val = as_int(val)
            except (ValueError, TypeError):
                logger.debug(f'{self.__class__.__name__} y is of type '
                             f'{val.dtype} and not integer for {self.task}')

        # ensure it is a vector
        val = val.squeeze(axis=tuple(range(1, val.ndim)))
        assert val.ndim == 1, 'y ndim is not 1'

        self._y = val
        self._validate_sizes()

    @property
    @_needs_load
    def data(self) -> pd.DataFrame:
        """
        Returns the dataset as a pandas DataFrame. This includes X values, y
        values (labels), and potentially other columns depending on the dataset

        :return: the dataset as a pandas DataFrame
        """
        if self._data is None:  # infer
            data = pd.DataFrame(
                columns=self.feature_names,
                data=self.X.reshape(-1, self.n_features),
            )
            data[self.label_col] = self.y
            self.data = data
        return self._data

    @data.setter
    def data(self, val: pd.DataFrame):
        """
        Update and validate data

        :param val: the DataFrame to update data with
        """
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
        """

        :return: the names of all features
        """
        if self._feature_names is None:  # infer
            if self._data is None:
                self.feature_names = [*map(str, range(self.n_features))]
            else:
                self.feature_names = self.data.columns.drop(
                    self.label_col).to_list()
        return self._feature_names

    @feature_names.setter
    def feature_names(self, val: Iterable[str]):
        """
        Update and validate feature names

        :param val: values to update the feature names
        """
        val = [*val]
        assert all(isinstance(v, str) for v in val), 'feat names not all str'
        assert len(val) == len(set(val)), 'non-unique feat names'
        self._feature_names = val
        self._validate_sizes()

    @property
    @_needs_load
    def grouped_feature_names(
            self,
    ) -> List[Union[str, Tuple[str, List[Any]]]]:
        """
        Grouped feature names. These are the feature names with hierarchical
        structure in the case it is present in the dataset. The most common
        use case in this library is for grouped feature names is one-hot
        encoded categorical features where each element is either a string
        (feature name) or a size-2 tuple (feature name, list of feature names).
        The order of these features corresponds to the order of the features in
        the data.

        :return: grouped feature names
        """
        if self._grouped_feature_names is None:  # infer
            self.grouped_feature_names = self.feature_names.copy()
        return self._grouped_feature_names

    @grouped_feature_names.setter
    def grouped_feature_names(
            self,
            val: Iterable[Union[str, Tuple[str, Sequence[Any]]]]
    ):
        """
        Update and validate feature names

        :param val: value with which to update grouped_feature_names
        """
        val = [v if isinstance(v, str) else (v[0], [*v[1]])
               for v in val]
        self._grouped_feature_names = val
        self._validate_sizes()

    @property
    @_needs_load
    def feature_types(self) -> List[str]:
        """

        :return: the type of each feature (either "numerical" or "categorical")
        """
        if self._feature_types is None:  # infer
            # TODO: dtypes do
            self.feature_types = ['numerical'] * self.n_features
        return self._feature_types

    @feature_types.setter
    def feature_types(self, val: Iterable[str]):
        """
        Update and validate feature types

        :param val: values with which to update the feature types
        """
        valid_types = ['numerical', 'categorical']
        val = [*val]
        assert all(v in valid_types for v in val)
        self._feature_types = val
        self._validate_sizes()

    @property
    @_needs_load
    def label_col(self) -> str:
        """
        :return: The label column name
        """
        if self._label_col is None:  # infer
            assert self._data is None, 'data is not None'
            self.label_col = 'target'
        return self._label_col

    @label_col.setter
    def label_col(self, val: str):
        """
        Update and validate label_col

        :param val: value with which to update label_col
        """
        assert isinstance(val, str), 'val is not str'
        self._label_col = val

    def _validate_sizes(self):
        """Validates the sizes of all dynamically set/loaded dataset values"""
        if self._X is not None:
            if self._y is not None:
                assert len(self._X) == len(self._y), 'X and y lengths differ'
            if self._data is not None:
                assert len(self._X) == len(self._data), (
                    'X and data lengths differ')
                assert self.n_features <= (len(self._data.columns) - 1), (
                    'n_features differs from # data columns')
            if self._feature_names is not None:
                assert self.n_features == len(self._feature_names), (
                    'n_features differs from # feat names')
            if self._feature_types is not None:
                assert self.n_features == len(self._feature_types), (
                    'n_features differs from # feat types')
            if self._grouped_feature_names is not None:
                tot_grouped_names = sum(
                    1 if isinstance(name, str) else len(name[1])
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
        """
        Loads the dataset according to provided arguments. Loading will lazily
        load attributes that are not provided, but ensure that enough data has
        been provided to infer unprovided attributes.

        Subclasses should override this method and can call super() in any of
        the following ways:

        1.
        >>> super()._load(X=X, y=y, ...)
        2.
        >>> load_dict = dict(X=X, y=y, ...)
        >>> super()._load(load_dict=load_dict)
        3.
        >>> load_dict = dict(X=X, y=y, ...)
        >>> super()._load(load_dict)

        Valid kwargs that can be provided:
            X, y, X_df, label_col, feature_names, feature_types,
            grouped_feature_names, data

        Enough kwargs need to be provided to infer what is needed in the task.
        For example, the following are minimally sufficient for supervised
        learning:
            - data, feature_names, label_col
            - X, y
            - X_df, y
            - data, X, label_col

        :param args: load_dict or nothing
        :param kwargs: just load_dict or any of the above specified attributes
        """
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
        assert not (loaded_keys - set(self.__expect_keys)), (
            f'unknown keys in load_dict: {loaded_keys}')

        # set in the correct order
        for key in self.__expect_keys:
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
        self.__load_called__ = True
        self._load(*args, **kwargs)

    def _load(self, *args, **kwargs):
        super()._load(*args, **kwargs)
