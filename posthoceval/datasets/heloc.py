
import pandas as pd

from posthoceval.datasets.dataset import Dataset
from posthoceval.datasets.dataset_utils import LOCAL_DATA_PATH


class HELOCDataset(Dataset):
    def __init__(self, task='classification'):
        super().__init__(task=task)
        if not self.is_classification:
            self._raise_bad_task()

    def _load(self):
        data_path = LOCAL_DATA_PATH / 'HELOC' / 'heloc_dataset_v1.csv'
        
        
        
        
        data_df = pd.read_csv(
            data_path,
            na_values=['-9', '-8', '-7']
        )

        label_col = 'RiskPerformance'
        feature_names = data_df.columns.drop([label_col])

        categorical_names = ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
        feature_types = [
            'categorical' if name in categorical_names else 'numerical'
            for name in feature_names
        ]

        super()._load(
            data=data_df,
            feature_names=feature_names,
            feature_types=feature_types,
            label_col=label_col,
        )
