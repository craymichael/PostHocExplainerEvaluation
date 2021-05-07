"""
transform.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple
from typing import Sequence
from typing import Any
from typing import Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from posthoceval.datasets.dataset import Dataset
from posthoceval.utils import UNPROVIDED

__all__ = ['Transformer']


class SklearnTransformer(Protocol):
    def fit(self, X: np.ndarray) -> 'SklearnTransformer': pass

    def transform(self, X: np.ndarray) -> np.ndarray: pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray: pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray: pass


class IdentityTransformer(BaseEstimator, TransformerMixin):
    __slots__ = ()

    # noinspection PyUnusedLocal
    def fit(self, X: np.ndarray) -> 'IdentityTransformer': return self

    @staticmethod
    def transform(X: np.ndarray) -> np.ndarray: return X

    @staticmethod
    def inverse_transform(X: np.ndarray) -> np.ndarray: return X


class Transformer(TransformerMixin):
    """"""

    _numerical_transformer: Optional[SklearnTransformer]
    _categorical_transformer: Optional[SklearnTransformer]
    _data_transformer: Optional[ColumnTransformer]
    _target_transformer: Optional[SklearnTransformer]
    numerical_features_: Optional[List[str]]
    categorical_features_: Optional[List[str]]

    def __init__(
            self,
            numerical_transformer: Optional[SklearnTransformer] = UNPROVIDED,
            categorical_transformer: Optional[SklearnTransformer] = UNPROVIDED,
            target_transformer: Optional[SklearnTransformer] = UNPROVIDED,
            impute_numerical: Union[str, bool, TransformerMixin] = UNPROVIDED,
            impute_categorical: Union[str, bool, TransformerMixin] = UNPROVIDED,
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

        self._impute_numerical = impute_numerical
        self._impute_categorical = impute_categorical

        # Set during fit(...)
        self.numerical_features_: List[str]
        self.categorical_features_: List[str]
        self.transformed_feature_names_: List[str]
        self.grouped_feature_names_: List[Union[str,
                                                Tuple[str, Sequence[Any]]]]

    def _build_data_transformer(
            self,
            dataset: Dataset,
    ) -> None:
        """"""
        self.numerical_features_ = dataset.numerical_features
        self.categorical_features_ = dataset.categorical_features

        # numerical and categorical transformers
        numerical_transformer = (
            self._numerical_transformer if self.transforms_numerical else
            IdentityTransformer()
        )
        categorical_transformer = (
            self._categorical_transformer if self.transforms_categorical else
            IdentityTransformer()
        )

        # Handle numerical imputer
        if self._impute_numerical is UNPROVIDED:
            if dataset.X_df[self.numerical_features_].isna().any(axis=None):
                self._impute_numerical = 'median'

        numerical_imputer = None
        if isinstance(self._impute_numerical, str):
            numerical_imputer = SimpleImputer(strategy=self._impute_numerical)
        elif self._impute_numerical is True:
            numerical_imputer = SimpleImputer()
        elif isinstance(self._impute_numerical, TransformerMixin):
            numerical_imputer = self._impute_numerical

        # Handle categorical imputer
        if (self._impute_categorical is UNPROVIDED or
                self._impute_categorical is True):
            if dataset.X_df[self.categorical_features_].isna().any(axis=None):
                self._impute_categorical = 'most_frequent'

        categorical_imputer = None
        if isinstance(self._impute_categorical, str):
            categorical_imputer = SimpleImputer(
                strategy=self._impute_categorical)
        elif isinstance(self._impute_categorical, TransformerMixin):
            categorical_imputer = self._impute_categorical

        # Add imputers if specified
        if numerical_imputer is not None:
            numerical_transformer = Pipeline(steps=[
                ('numerical_imputer', numerical_imputer),
                ('numerical_transformer', numerical_transformer),
            ])
        if categorical_imputer is not None:
            categorical_transformer = Pipeline(steps=[
                ('categorical_imputer', categorical_imputer),
                ('categorical_transformer', categorical_transformer),
            ])

        self._data_transformer = ColumnTransformer([
            ('numerical',
             numerical_transformer,
             self.numerical_features_),
            ('categorical',
             categorical_transformer,
             self.categorical_features_)
        ])

    def _infer_default_target_transformer(
            self,
            dataset: Dataset,
    ) -> None:
        # Note: calling this twice changes this value, so if for some silly
        #  reason you call fit with two Datasets with different tasks then you
        #  may have the wrong transformer if target_transformer is UNPROVIDED
        if self._target_transformer is UNPROVIDED:
            if dataset.is_classification:
                self._target_transformer = LabelEncoder()
            else:
                self._target_transformer = StandardScaler()

    @staticmethod
    def _handle_y_transformer(
            transformer_func: Callable,
            y: np.ndarray,
    ) -> Union[np.ndarray, TransformerMixin]:
        """"""
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
        if not isinstance(y, np.ndarray):
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

    def class_name(self, idx: int) -> Any:
        if not hasattr(self._target_transformer, 'classes_'):
            raise ValueError('Target transformer does not have the classes_ '
                             'attribute!')
        # noinspection PyUnresolvedReferences
        return self._target_transformer.classes_[idx]

    def fit(
            self,
            dataset: Dataset,
    ) -> 'Transformer':
        """"""
        self._build_data_transformer(dataset)
        self._infer_default_target_transformer(dataset)

        self._data_transformer.fit(dataset.X_df)
        self._set_transformed_feature_names()

        if self.transforms_target:
            self._handle_y_transformer(
                self._target_transformer.fit, dataset.y
            )
        return self

    def _set_transformed_feature_names(self):
        transformed_feature_names = self.numerical_features_.copy()
        if self.transforms_categorical:
            # NOTE: case of drop_idx_ is not handled here...

            # safely get the categorical transformer (possibly could differ
            #  from self._categorical_transformer)
            categorical_transformer = (
                self._data_transformer.named_transformers_['categorical'])
            if isinstance(categorical_transformer, Pipeline):
                categorical_transformer = (
                    categorical_transformer.named_steps[
                        'categorical_transformer'])
            if hasattr(categorical_transformer, 'categories_'):
                categories = categorical_transformer.categories_
            else:
                categories = []
            assert len(self.categorical_features_) == len(categories)

            # record grouped feature names, retaining flat features with
            #  mappings to unique categorical values (new columns in
            #  transformed data)
            grouped_feature_names: List[Union[str, Tuple[str, Sequence[Any]]]]
            grouped_feature_names = transformed_feature_names.copy()

            for cat, names in zip(self.categorical_features_, categories):
                # cat:   categorical feature name
                # names: unique values for that categorical feature
                transformed_feature_names.extend(
                    cat + f' = {name}'
                    for name in names
                )
                grouped_feature_names.append((cat, names))
        else:
            transformed_feature_names += self.categorical_features_
            grouped_feature_names = transformed_feature_names.copy()

        self.transformed_feature_names_ = transformed_feature_names
        self.grouped_feature_names_ = grouped_feature_names

    @staticmethod
    def _validate_transform_inputs(
            dataset: Optional[Dataset],
            X_df: Optional[pd.DataFrame],
            y: Optional[np.ndarray],
    ):
        # input validation
        ds_missing = dataset is None
        X_missing = (ds_missing and X_df is None)
        y_missing = (ds_missing and y is None)
        non_ds_missing = (X_df is None and y is None)
        if ds_missing:
            assert not non_ds_missing, (
                'X_df or y should be provided if dataset is not')
        else:
            assert non_ds_missing, (
                'X_df and y should not be provided if dataset is')
        return ds_missing, X_missing, y_missing

    def transform(
            self,
            dataset: Optional[Dataset] = None,
            X_df: Optional[pd.DataFrame] = None,
            y: Optional[np.ndarray] = None,
    ) -> Union[Dataset,
               pd.DataFrame,
               np.ndarray,
               Tuple[pd.DataFrame, np.ndarray]]:
        """Must provide either 1) dataset alone or 2) X_df and/or y.

        :param dataset:
        :param X_df:
        :param y:
        :return: The transformed input(s):
            dataset -> Dataset
            X_df    -> pd.DataFrame
            y       -> np.ndarray
            X_df, y -> Tuple[pd.DataFrame, np.ndarray]
        """
        ds_missing, X_missing, y_missing = self._validate_transform_inputs(
            dataset, X_df, y)

        if not X_missing:
            if self.transforms_data:
                X = self._data_transformer.transform(
                    X_df if ds_missing else dataset.X_df)
            else:
                X = X_df.values if ds_missing else dataset.X

        if not y_missing:
            if self.transforms_target:
                y = self._handle_y_transformer(
                    self._target_transformer.transform,
                    y if ds_missing else dataset.y,
                )
            elif not ds_missing:
                y = dataset.y

            if X_missing:  # nothing else to do here
                return y

        if ds_missing:
            # X guaranteed at this point, y not
            # noinspection PyUnboundLocalVariable
            X_df_transformed = pd.DataFrame(
                data=X,  # not unbound
                columns=self.transformed_feature_names_,
            )
            if y_missing:
                return X_df_transformed
            else:
                return X_df_transformed, y
        else:
            if not ds_missing:
                # noinspection PyUnboundLocalVariable
                X = X.reshape(-1, *dataset.input_shape)
            # noinspection PyUnboundLocalVariable
            dataset_transformed = dataset.from_data(
                task=dataset.task,
                X=X,  # not unbound
                y=y,
                label_col=dataset.label_col,
                feature_names=self.transformed_feature_names_,
                grouped_feature_names=self.grouped_feature_names_,
            )
            return dataset_transformed

    def inverse_transform(
            self,
            dataset: Optional[Dataset] = None,
            X_df: Optional[pd.DataFrame] = None,
            y: Optional[np.ndarray] = None,
            transform_numerical: bool = True,
            transform_categorical: bool = False,
            transform_target: bool = False,
    ):
        ds_missing, X_missing, y_missing = self._validate_transform_inputs(
            dataset, X_df, y)

        if transform_categorical:
            raise NotImplementedError('transform_categorical')

        if not y_missing:
            y = y if ds_missing else dataset.y
            if transform_target and self.transforms_target:
                y = self._handle_y_transformer(
                    self._target_transformer.inverse_transform, y)
            if X_missing:
                return y

        if not X_missing:
            if transform_numerical and self.transforms_numerical:
                # safely get the categorical transformer (possibly could differ
                #  from self._categorical_transformer)
                numerical_transformer = (
                    self._data_transformer.named_transformers_['numerical'])
                if isinstance(numerical_transformer, Pipeline):
                    numerical_transformer = (
                        numerical_transformer.named_steps[
                            'numerical_transformer'])
                X_df = X_df if ds_missing else dataset.X_df
                X = numerical_transformer.inverse_transform(
                    X_df[self.numerical_features_].values)
                if X.shape[1] != len(self.numerical_features_):
                    raise NotImplementedError(
                        'inverse_transform for numerical transformers that '
                        'change the number of columns.'
                    )
                X_cat = X_df.drop(columns=self.numerical_features_).values
                X = np.concatenate([X, X_cat], axis=1)
            else:
                X = X_df.values if ds_missing else dataset.X

        if ds_missing:
            if y_missing:
                # noinspection PyUnboundLocalVariable
                return X
            else:
                # noinspection PyUnboundLocalVariable
                return X, y
        else:
            # NOTE: this assumes numerical transformer does not alter the
            #  number of columns
            if not ds_missing:
                # noinspection PyUnboundLocalVariable
                X = X.reshape(-1, *dataset.input_shape)
            # noinspection PyUnboundLocalVariable
            return dataset.from_data(
                task=dataset.task,
                X=X,  # not unbound
                y=y,
                label_col=dataset.label_col,
                feature_names=dataset.feature_names,
                grouped_feature_names=dataset.grouped_feature_names,
            )
