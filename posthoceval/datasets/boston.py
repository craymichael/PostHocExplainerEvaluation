
import pandas as pd

from posthoceval.datasets.dataset import Dataset
from posthoceval.datasets.dataset_utils import LOCAL_DATA_PATH


class BostonDataset(Dataset):
    def __init__(self, task='regression', medv_threshold=None):
        super().__init__(
            task=task,
        )
        if self.is_classification:
            if medv_threshold is None:
                medv_threshold = 22.5
        elif self.is_regression:
            assert medv_threshold is None
        else:
            self._raise_bad_task()

        self.medv_threshold = medv_threshold

    def _load(self):
        data_path = LOCAL_DATA_PATH / 'boston'
        data_df = pd.read_csv(data_path, delimiter=' ')
        if self.is_classification:
            thresh = self.medv_threshold
            label_col = f'MEDV>={thresh}'
            data_df[label_col] = (data_df.loc[:, 'MEDV'] >= thresh).astype(int)
            to_drop = ['MEDV', label_col]
        elif self.is_regression:
            to_drop = label_col = 'MEDV'
        else:
            self._raise_bad_task()

        
        
        feature_names = data_df.columns.drop(to_drop).to_list()

        
        
        super()._load(
            data=data_df,
            feature_names=feature_names,
            label_col=label_col,
        )
