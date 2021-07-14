"""
Acknowledgements in help coding things well:
https://stackoverflow.com/a/30003565/6557588
"""
from collections.abc import Callable
from typing import Union
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from posthoceval import utils


class GlobalKernelSHAP(object):
    """Very much not part of any notion of a public API."""

    def __init__(self,
                 data: np.ndarray,
                 shap_values: np.ndarray,
                 expected_value: Union[int, float, np.ndarray],
                 interpolation: Union[str, Callable] = 'linear'):
        """

        :param interpolation: 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic', 'previous', 'next', or a callable
        """
        # n data point, k features
        utils.assert_rank(shap_values, 2, name='shap_values')
        n, k = utils.assert_same_shape(data, shap_values)  # noqa

        # TODO: validation of expected_value, rank 0 or 1 (> is not supported
        #  now but 100% possible)
        self.expected_value = expected_value

        interpolation = (
            interpolation if isinstance(interpolation, Callable) else
            partial(interp1d, kind=interpolation, fill_value='extrapolate',
                    assume_sorted=True)
        )
        interp_funcs = []
        for i in range(k):
            feat, shap_val = data[:, i], shap_values[:, i]
            # Handle duplicate values properly. Sort both arrays by feat values
            # ascending. Then take unique feature values with the output being
            # the average value.
            idx_sort = np.argsort(feat)
            feat_sort = feat[idx_sort]
            shap_val_sort = shap_val[idx_sort]
            vals, idx_start = np.unique(feat_sort, return_index=True)
            feat_unique = feat_sort[idx_start]
            shap_val_agg = np.asarray(
                [*map(np.mean,
                      np.split(shap_val_sort, idx_start[1:]))])

            interp_funcs.append(interpolation(feat_unique, shap_val_agg))

        self.interp_funcs = interp_funcs

    def predict(self, x, return_shap_values=False):
        utils.assert_shape(x, (None, len(self.interp_funcs)), name='x')

        shap_values = []
        for i, interp_func in enumerate(self.interp_funcs):
            shap_values_feat = interp_func(x[:, i])
            shap_values.append(shap_values_feat)

        shap_values = np.asarray(shap_values).T
        predictions = np.sum(shap_values, axis=1) + self.expected_value
        if return_shap_values:
            return predictions, shap_values
        return predictions
