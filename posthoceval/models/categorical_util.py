"""
categorical_util.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from posthoceval.datasets.dataset import Dataset
from posthoceval.utils import UNPROVIDED


def deal_with_categorical_shit(
        dataset: Dataset,
        numerical_transformer: Optional[TransformerMixin] = UNPROVIDED,
        categorical_transformer: Optional[TransformerMixin] = UNPROVIDED,
        label_encoder: Optional[TransformerMixin] = UNPROVIDED,
):
    # TODO everything
    numerical_cols = dataset.numerical_features
    categorical_cols = dataset.categorical_features

    column_transformers = []

    if numerical_transformer is not None:
        if numerical_transformer is UNPROVIDED:
            numerical_transformer = StandardScaler()
        column_transformers.append(
            ('num', numerical_transformer, numerical_cols)
        )

    if categorical_transformer is not None:
        if categorical_transformer is UNPROVIDED:
            categorical_transformer = OneHotEncoder(sparse=False)
        column_transformers.append(
            ('cat', categorical_transformer, categorical_cols)
        )

    if column_transformers:
        data_transformer = ColumnTransformer(column_transformers)

        X = data_transformer.fit_transform(dataset.X_df)
    else:
        data_transformer = None
        X = dataset.X

    if label_encoder is None:
        if label_encoder is UNPROVIDED:
            label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(dataset.y)
    else:
        y = dataset.y

    transformed_feature_names = numerical_cols.copy()
    if 'cat' in data_transformer.named_transformers_:
        categories = data_transformer.named_transformers_['cat'].categories_

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

    return expl_init_kwargs, expl_fit_kwargs
