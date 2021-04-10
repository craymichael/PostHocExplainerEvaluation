"""
boston.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import pandas as pd

from posthoceval.datasets.dataset import Dataset
from posthoceval.datasets.dataset_utils import LOCAL_DATA_DIR


class BostonDataset(Dataset):
    def __init__(self):
        # TODO: also support classification (thresholded MEDV)
        super().__init__(
            task='regression',
        )

    def _load(self):
        data_path = LOCAL_DATA_DIR / 'boston'
        data_df = pd.read_csv('data/boston', delimiter=' ')
        label_col = 'MEDV'

        X_df = data_df.drop(columns=label_col)

        self.data = data_df
        self.X = X_df.values
        self.y = data_df[label_col].values
        self.feature_names = [*X_df.keys()]

    def as_X_y(self):
        return
