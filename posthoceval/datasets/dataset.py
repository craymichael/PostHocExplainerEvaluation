"""
dataset.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import List
from typing import Tuple
from typing import Union
from typing import Iterable

import logging

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from posthoceval.utils import UNPROVIDED
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
        self._feature_names: Optional[List[str]] = None
        self._feature_types: Optional[List[str]] = None
        self._label_col: Optional[str] = None
        # These will be set lazily and a function of those provided in _load
        self._X_df: Optional[pd.DataFrame] = None

        # Transformers
        self.data_transformer: Optional[TransformerMixin] = None
        self.label_encoder: Optional[TransformerMixin] = None
        # TODO: do not transform the entire dataset....only training....
        #  this maybe shouldn't be embedded here....or support get_item and len
        #  for train test split then transform + transform support DataSet
        #  objects
        # TODO: nah we are just going to make a transformer class that works
        #  with Dataset objects
        self._untransformed = {}

    def __len__(self) -> int:
        # return length without triggering lazy-loading, if possible
        if self._y is not None:
            return len(self.y)
        elif self._X is not None:
            return len(self.X)
        else:
            return len(self.data)

    @classmethod
    def from_data(cls, task, *args, **kwargs):
        dataset = cls(task=task)
        dataset._load(*args, **kwargs)
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
        elif self._y is not None:
            if self._data is not None:
                assert len(self._y) == len(self._data), (
                    'y and data lengths differ')

    @abstractmethod
    def _load(self, *args, **kwargs):
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
        expect_keys = ['label_col', 'feature_names', 'feature_types', 'X', 'y',
                       'data']
        assert not (loaded_keys - set(expect_keys)), (
            f'unknown keys in load_dict {loaded_keys}')

        # set in the correct order
        for key in expect_keys:
            loaded_val = load_dict.get(key)
            if loaded_val is None:
                continue
            setattr(self, key, loaded_val)

    def transform(
            self,
            numerical_transformer: Optional[TransformerMixin] = UNPROVIDED,
            categorical_transformer: Optional[TransformerMixin] = UNPROVIDED,
            label_encoder: Optional[TransformerMixin] = UNPROVIDED,
    ):
        numerical_cols = self.numerical_features
        categorical_cols = self.categorical_features

        column_transformers = []

        if numerical_transformer is not None:
            if numerical_transformer is UNPROVIDED:
                numerical_transformer = StandardScaler()
            column_transformers.append(
                ('numerical', numerical_transformer, numerical_cols)
            )

        if categorical_transformer is not None:
            if categorical_transformer is UNPROVIDED:
                categorical_transformer = OneHotEncoder(sparse=False)
            column_transformers.append(
                ('categorical', categorical_transformer, categorical_cols)
            )

        if column_transformers:
            data_transformer = ColumnTransformer(column_transformers)

            X = data_transformer.fit_transform(self.X_df)
        else:
            data_transformer = None
            X = self.X

        if label_encoder is None:
            if label_encoder is UNPROVIDED:
                label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(self.y)
        else:
            y = self.y

        transformed_feature_names = numerical_cols.copy()
        if 'categorical' in data_transformer.named_transformers_:
            categories = \
                data_transformer.named_transformers_['categorical'].categories_

            for cat, names in zip(categorical_cols, categories):
                transformed_feature_names.extend(
                    cat + f' = {name}'
                    for name in names
                )
        else:
            categories = None
            transformed_feature_names += categorical_cols

        if categories is not None:
            groups = [[i] for i in range(len(numerical_cols))]

            start = len(groups)

            # column index -> categories
            category_map = dict(zip(
                range(start, start + len(categories)), categories
            ))

            for cat in categories:
                end = start + len(cat)
                groups.append([*range(start, end)])
                start = end

            group_names = numerical_cols + categorical_cols

            # TODO: these are both SHAP-only
            expl_init_kwargs = dict(categorical_names=category_map)
            expl_fit_kwargs = dict(group_names=group_names, groups=groups)
        else:
            expl_init_kwargs = {}
            expl_fit_kwargs = {}

        # store transformers
        self.data_transformer = data_transformer
        self.label_encoder = label_encoder

        # store untransformed data
        self.X
        self.y
        self.X_df
        self.data
        self.feature_names
        self.feature_types

    def inverse_transform(self):
        pass
