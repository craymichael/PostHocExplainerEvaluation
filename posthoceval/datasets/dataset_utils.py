"""
dataset_utils.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from math import floor
from pathlib import Path

import pandas as pd

import posthoceval

LOCAL_DATA_PATH = Path(posthoceval.__file__).parent.parent / 'data'


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
