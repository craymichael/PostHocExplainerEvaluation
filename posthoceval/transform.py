"""
transform.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Optional
from typing import Callable
from typing import Protocol
from typing import List
from typing import Tuple
from typing import Sequence
from typing import Any
from typing import Union

import warnings

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from posthoceval.datasets.dataset import Dataset
from posthoceval.utils import UNPROVIDED

__all__ = ['Transformer']


class SklearnBaseTransformer(Protocol):
    def fit(self, X: np.ndarray) -> 'SklearnBaseTransformer': pass

    def transform(self, X: np.ndarray) -> np.ndarray: pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray: pass


class SklearnTransformer(SklearnBaseTransformer, TransformerMixin): pass


class Transformer(TransformerMixin):
    """"""

    _numerical_transformer: Optional[SklearnTransformer]
    _categorical_transformer: Optional[SklearnTransformer]
    _data_transformer: Optional[ColumnTransformer]
    _target_transformer: Optional[SklearnTransformer]
    _numerical_features: Optional[List[str]]
    _categorical_features: Optional[List[str]]

    def __init__(
            self,
            numerical_transformer: Optional[SklearnTransformer] = UNPROVIDED,
            categorical_transformer: Optional[SklearnTransformer] = UNPROVIDED,
            target_transformer: Optional[SklearnTransformer] = UNPROVIDED,
    ):
        """"""
        if numerical_transformer is UNPROVIDED:
            numerical_transformer = StandardScaler()
        self._numerical_transformer = numerical_transformer

        if categorical_transformer is UNPROVIDED:
            categorical_transformer = OneHotEncoder(sparse=False)
        self._categorical_transformer = categorical_transformer

        self._data_transformer = None
        self._target_transformer = target_transformer

        self._numerical_features = None
        self._categorical_features = None

    def _build_data_transformer(
            self,
            dataset: Dataset,
    ) -> None:
        """"""
        self._numerical_features = dataset.numerical_features
        self._categorical_features = dataset.categorical_features

        column_transformers = []
        if self._numerical_transformer is not None:
            column_transformers.append(
                ('numerical',
                 self._numerical_transformer,
                 self._numerical_features)
            )
        if self._categorical_transformer is not None:
            column_transformers.append(
                ('categorical',
                 self._categorical_transformer,
                 self._categorical_features)
            )
        if column_transformers:
            self._data_transformer = ColumnTransformer(column_transformers)
        else:
            self._data_transformer = None

    def _infer_default_target_transformer(
            self,
            dataset: Dataset,
    ):
        # Note: calling this twice changes this value, so if for some silly
        #  reason you call fit with two Datasets with different tasks then you
        #  may have the wrong transformer if target_transformer is UNPROVIDED
        if self._target_transformer is UNPROVIDED:
            if dataset.is_classification:
                self._target_transformer = LabelEncoder()
            else:
                self._target_transformer = StandardScaler()

    def _handle_y_transformer(
            self,
            transformer_func: Callable,
            y: np.ndarray,
    ) -> np.ndarray:
        shape_orig = y.shape
        if y.ndim == 1:  # vector
            y = y[:, np.newaxis]
        elif y.ndim == 0:  # scalar
            y = y[np.newaxis, np.newaxis]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='A column-vector y was passed when a 1d '
                                  'array was expected')
            y = transformer_func(y)
        if isinstance(y, TransformerMixin):
            return y
        y = y.reshape(shape_orig)
        return y

    @property
    def transforms_numerical(self) -> bool:
        return self._numerical_transformer is not None

    @property
    def transforms_categorical(self) -> bool:
        return self._categorical_transformer is not None

    @property
    def transforms_data(self) -> bool:
        return self.transforms_numerical or self.transforms_categorical

    @property
    def transforms_target(self) -> bool:
        return self._target_transformer is not None

    def fit(
            self,
            dataset: Dataset,
    ) -> 'Transformer':
        """"""
        self._build_data_transformer(dataset)
        self._infer_default_target_transformer(dataset)

        if self.transforms_data:
            self._data_transformer.fit(dataset.X_df)

        if self.transforms_target:
            self._handle_y_transformer(
                self._target_transformer.fit, dataset.y
            )

        return self

    def transform(
            self,
            dataset: Dataset,
    ) -> Dataset:
        """"""
        if self.transforms_data:
            X = self._data_transformer.transform(dataset.X_df)
        else:
            X = dataset.X

        if self.transforms_target:
            y = self._handle_y_transformer(
                self._target_transformer.transform, dataset.y
            )
        else:
            y = dataset.y

        transformed_feature_names = self._numerical_features.copy()
        if self.transforms_categorical:
            # safely get the categorical transformer (possibly could differ
            #  from self._categorical_transformer)
            categorical_transformer_ = (
                self._data_transformer.named_transformers_['categorical'])
            categories = categorical_transformer_.categories_
            assert len(self._categorical_features) == len(categories)

            # record grouped feature names, retaining flat features with
            #  mappings to unique categorical values (new columns in
            #  transformed data)
            grouped_feature_names: List[Union[str, Tuple[str, Sequence[Any]]]]
            grouped_feature_names = transformed_feature_names.copy()

            for cat, names in zip(self._categorical_features, categories):
                # cat:   categorical feature name
                # names: unique values for that categorical feature
                transformed_feature_names.extend(
                    cat + f' = {name}'
                    for name in names
                )
                grouped_feature_names.append((cat, names))
        else:
            transformed_feature_names += self._categorical_features
            grouped_feature_names = transformed_feature_names.copy()

        dataset_transformed = dataset.from_data(
            task=dataset.task,
            X=X,
            y=y,
            label_col=dataset.label_col,
            feature_names=transformed_feature_names,
            grouped_feature_names=grouped_feature_names,
        )
        return dataset_transformed

    def inverse_transform(self):
        raise NotImplementedError