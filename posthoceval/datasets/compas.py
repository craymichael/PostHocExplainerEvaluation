"""
compas.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import json

import pandas as pd

from posthoceval.datasets.dataset import Dataset
from posthoceval.datasets.dataset_utils import LOCAL_DATA_PATH


class COMPASDataset(Dataset):
    def __init__(self, task='regression'):
        super().__init__(task=task)

    def _load(self):
        data_path = LOCAL_DATA_PATH / 'compas_two_year_filtered.csv'
        data_df = pd.read_csv(data_path)

        metadata_path = LOCAL_DATA_PATH / 'compas_metadata.json'
        with metadata_path.open('r') as f:
            compas_meta = json.load(f)

        def pretty_name(ugly_name):
            if isinstance(ugly_name, str):
                return compas_meta[ugly_name]['name']
            return [compas_meta[un]['name'] for un in ugly_name]

        # rename and get feature columns
        if self.is_regression:
            label_col = 'decile_score'
            alt_label = 'score_text'
        elif self.is_classification:
            label_col = 'score_text'
            alt_label = 'decile_score'
        else:
            self._raise_bad_task()

        # TODO: https://youtrack.jetbrains.com/issue/PY-24273
        # noinspection PyUnboundLocalVariable
        feature_names = data_df.columns.drop(
            [label_col, 'age_cat', alt_label])
        feature_types = [
            'categorical' if compas_meta[name]['categorical'] else 'numerical'
            for name in feature_names
        ]
        # rename, make all pretty
        data_df.rename(columns=pretty_name, inplace=True)
        feature_names = pretty_name(feature_names)
        label_col = pretty_name(label_col)

        super()._load(
            data=data_df,
            feature_names=feature_names,
            feature_types=feature_types,
            label_col=label_col,
        )
