"""
util.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from functools import partial
from typing import Callable
from typing import Union

import numpy as np
from scipy.interpolate import interp1d

from posthoceval import utils


def _get_unary_func(scalar_value):
    def identity_like(x):
        return np.repeat(scalar_value, len(x))

    return identity_like


class MultivariateInterpolation(object):
    """Multivariate interpolation: a wrapper around interpolation and
    extrapolation for multiple features"""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 interpolation: Union[str, Callable] = 'linear'):
        """
        Multivariate interpolation: a wrapper around interpolation and
        extrapolation for multiple features

        :param x: the input of an unspecified function
        :param y: the output of an unspecified function
        :param interpolation: 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic', 'previous', 'next', or a callable
        """
        if x.dtype is np.dtype('O'):
            assert y.dtype is np.dtype('O')
            is_object = True
            # object detected, assume ragged k x variable n
            utils.assert_rank(y, 1, name='y')
            k = utils.assert_same_shape(x, y)[0]
        else:
            # n data point, k features
            is_object = False
            utils.assert_rank(y, 2, name='y')
            n, k = utils.assert_same_shape(x, y)

        interpolation = (
            interpolation if isinstance(interpolation, Callable) else
            partial(interp1d, kind=interpolation, fill_value='extrapolate',
                    assume_sorted=True)
        )
        interp_funcs = []
        for i in range(k):
            if is_object:
                x_i, y_i = x[i], y[i]
                utils.assert_rank(y_i, 1, name='y_i')
                utils.assert_same_shape(x_i, y_i)
            else:
                x_i, y_i = x[:, i], y[:, i]

            # check case that only single value provided
            if len(x_i) == 1:
                interp_funcs.append(_get_unary_func(y_i[0]))
                continue

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

    def interpolate(self, x: np.ndarray):
        """
        Interpolates y given x

        :param x: ndarray of input values of the interpolated function
        :return: interpolated values y for each feature
        """
        utils.assert_shape(x, (None, len(self.interp_funcs)), name='x')

        y_interp = [
            interp_func(x[:, i])
            for i, interp_func in enumerate(self.interp_funcs)
        ]
        # f x n --> n x f
        return np.asarray(y_interp).T
