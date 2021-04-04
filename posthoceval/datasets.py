"""
datasets.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from math import floor

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from skimage.transform import resize


# TODO: add COMPAS/Boston/others here

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


def show_crop_stats(X, verbose=True):
    nonzero = X.astype(bool)
    nonzero_tot = nonzero.sum()

    max_rows = floor(X.shape[1] / 2)
    max_cols = floor(X.shape[2] / 2)

    crop_stat_data = []

    for i in range(max_rows + 1):
        for j in range(max_cols + 1):
            for k in range(max_cols + 1):
                for m in range(max_cols + 1):
                    nonzero_border_tot = (
                            nonzero[:, :i].sum() +  # top rows (inclusive)
                            # bottom rows (incl.)
                            nonzero[:, X.shape[1] - j:].sum() +
                            # left cols (excl.)
                            nonzero[:, i:X.shape[1] - j, :k].sum() +
                            # right cols (excl.)
                            nonzero[:, i:X.shape[1] - j, X.shape[2] - m:].sum()
                    )

                    n_removed = ((i + j) * X.shape[1] +
                                 (k + m) * (X.shape[2] - (i + j)))

                    pct_removed_img = n_removed / X[0].size * 100
                    pct_tot_removed_nonzero = nonzero_border_tot / X.size * 100
                    pct_of_nonzero = nonzero_border_tot / nonzero_tot * 100
                    pct_removed_nonzero = (
                            nonzero_border_tot / (n_removed * len(X)) * 100)

                    crop_stat_data.append([
                        i, j, k, m,
                        pct_removed_img,
                        pct_tot_removed_nonzero,
                        pct_of_nonzero,
                        pct_removed_nonzero,
                    ])

                    if not verbose:
                        continue

                    print(
                        f'\tRemoving {i:>2} top rows, {j:>2} bottom rows, '
                        f'{k:>2} left cols, and {m:>2} right cols '
                        f'({pct_removed_img:.2f}% of each image) '
                        f'gets rid of '
                        f'{pct_tot_removed_nonzero:.2f}% '
                        f'of total pixels that were nonzero, '
                        f'{pct_of_nonzero:.2f}% of '
                        f'total nonzero pixels, and '
                        f'{pct_removed_nonzero:.2f}'
                        f'% of removed pixels are nonzero.'
                    )

    df = pd.DataFrame(
        columns=[
            'n_top_rows', 'n_bottom_rows', 'n_left_cols', 'n_right_cols',
            'pct_removed_img',
            'pct_tot_removed_nonzero',
            'pct_of_nonzero_removed',
            'pct_removed_are_nonzero',
        ],
        data=crop_stat_data,
    )

    if not verbose:
        return df

    # do some empirical analysis
    df_anal = df.copy()
    # we care about this metric the most
    df_anal['metric'] = (df_anal['pct_removed_img'] /
                         df_anal['pct_of_nonzero_removed'])
    df_anal = df.sort_values(by='pct_of_nonzero_removed', ascending=True)
    df_anal = df_anal.loc[
        (df_anal['pct_removed_img'] >= df_anal['pct_removed_img'].cummax())
        # & (df_anal['pct_of_nonzero_removed'] <= 50)
    ]
    df_anal = df_anal.loc[
        df_anal.groupby(['pct_removed_img'])['pct_of_nonzero_removed'].idxmin()
    ]
    df_anal = df_anal.sort_values(by='metric', ascending=False)  # .iloc[:50]

    print('Here are some good border crop values:')
    print(df_anal)

    return df
