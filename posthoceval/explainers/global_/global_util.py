"""
util.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from functools import partial
from typing import Callable
from typing import Union
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from posthoceval import utils


class MultivariateInterpolation(object):

    def __init__(self,
                 x: Union[np.ndarray, List[np.ndarray]],
                 y: Union[np.ndarray, List[np.ndarray]],
                 interpolation: Union[str, Callable] = 'linear'):
        """

        :param interpolation: 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic', 'previous', 'next', or a callable
        """
        # n data point, k features
        utils.assert_rank(y, 2, name='y')
        n, k = utils.assert_same_shape(x, y)

        interpolation = (
            interpolation if isinstance(interpolation, Callable) else
            partial(interp1d, kind=interpolation, fill_value='extrapolate',
                    assume_sorted=True)
        )
        interp_funcs = []
        for i in range(k):
            x_i, y_i = self._get_feat(x, i), self._get_feat(y, i)
            # Handle duplicate values properly. Sort both arrays by feat values
            # ascending. Then take unique feature values with the output being
            # the average value.
            idx_sort = np.argsort(x_i)
            x_i_sort = x_i[idx_sort]
            y_i_sort = y_i[idx_sort]
            x_i_uniq, idx_start = np.unique(x_i_sort, return_index=True)
            y_i_agg = np.fromiter(
                map(np.mean, np.split(y_i_sort, idx_start[1:])),
                dtype=y_i_sort.dtype
            )

            interp_funcs.append(interpolation(x_i_uniq, y_i_agg))

        self.interp_funcs = interp_funcs

    @staticmethod
    def _get_feat(mat, idx):
        if isinstance(mat, list):
            return mat[idx]
        else:
            return mat[:, idx]

    def interpolate(self, x: np.ndarray):
        utils.assert_shape(x, (None, len(self.interp_funcs)), name='x')

        y_interp = [
            interp_func(x[:, i])
            for i, interp_func in enumerate(self.interp_funcs)
        ]
        # f x n --> n x f
        return np.asarray(y_interp).T
