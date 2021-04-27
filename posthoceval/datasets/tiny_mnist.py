"""
tiny_mnist.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import os

from joblib import Memory

import numpy as np

from skimage.transform import resize
from sklearn.datasets import fetch_openml as uncached_fetch_openml
from sklearn.datasets import get_data_home

from posthoceval.datasets.dataset import Dataset
from posthoceval.utils import UNPROVIDED

__all__ = ['load_tiny_mnist', 'TinyMNISTDataset']

# Data not actually cached (as of sklearn 0.23.2 at least)
# https://github.com/scikit-learn/scikit-learn/issues/18783
# https://github.com/scikit-learn/scikit-learn/pull/14855
OPENML_CACHE_DIR = os.path.join(get_data_home(), 'openml-cache')
fetch_openml_memory = Memory(OPENML_CACHE_DIR)
fetch_openml = fetch_openml_memory.cache(uncached_fetch_openml)


# TODO: https://youtrack.jetbrains.com/issue/PY-24273
class TinyMNISTDataset(Dataset):

    def __init__(self, task='classification', *args, **kwargs):
        super().__init__(task='classification')
        if not self.is_classification:
            # TODO: https://youtrack.jetbrains.com/issue/PY-24273
            self._raise_bad_task()

    def _load(self, *args, **kwargs) -> None:
        pass


def load_tiny_mnist(
        class_subset=None,
        crop_top_rows=3,
        crop_bottom_rows=2,
        crop_left_cols=5,
        crop_right_cols=3,
        downscale=0.5,
):
    """"""
    # load MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                        as_frame=False)
    X = X.astype(float)
    y = y.astype(int)

    if class_subset is not None:
        mask = (y == class_subset[0])
        for k in class_subset[1:]:
            mask |= (y == k)
        X = X[mask]
        y = y[mask]

    # reshape to image dims
    X = X.reshape(-1, 28, 28)

    # crop border
    n_rows, n_cols = X.shape[1:]
    X = X[
        :,  # samples
        crop_top_rows:n_rows - crop_bottom_rows,  # rows
        crop_left_cols:n_cols - crop_right_cols,  # cols
        ]

    if downscale is not None and downscale != 1:
        # resize images (downscale)
        output_shape = (round(X.shape[1] * downscale),
                        round(X.shape[2] * downscale))
        X = np.asarray(
            [resize(xi, output_shape, anti_aliasing=True)
             for xi in X]
        )
    return X, y
