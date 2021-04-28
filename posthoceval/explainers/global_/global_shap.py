"""
Acknowledgements in help coding things well:
https://stackoverflow.com/a/30003565/6557588
"""
from collections.abc import Callable
from typing import Union
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from alibi.explainers import KernelShap

from posthoceval import utils


# @profile
# def gshap_explain(model, data_train, data_test, n_background_samples=100):
#     explainer = KernelShap(
#         model,
#         feature_names=model.gen_symbol_names,
#         task='regression',
#         distributed_opts={
#             # https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/
#             'n_cpus': cpu_count(),
#             # If batch_size set to `None`, an input array is split in (roughly)
#             # equal parts and distributed across the available CPUs
#             'batch_size': None,
#         }
#     )
#     fit_kwargs = {}
#     if n_background_samples < len(data_train):
#         print('Intending to summarize background data as n_samples > '
#               '{}'.format(n_background_samples))
#         fit_kwargs['summarise_background'] = True
#         fit_kwargs['n_background_samples'] = n_background_samples
#
#     print('Explainer fit')
#     explainer.fit(data_train, **fit_kwargs)
#
#     # Note: explanation.raw['importances'] has aggregated scores per output with
#     # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
#     # for the aggregated scores over all outputs
#     print('Explain')
#     explanation = explainer.explain(data_train, silent=True)
#     expected_value = explanation.expected_value.squeeze()
#     shap_values = explanation.shap_values[0]
#     outputs_train = explanation.raw['raw_prediction']
#     shap_values_g = explanation.raw['importances']['0']
#
#     print('Global SHAP')
#     gshap = GlobalKernelSHAP(data_train, shap_values, expected_value)
#
#     gshap_preds, gshap_vals = gshap.predict(data_train, return_shap_values=True)
#     print('RMSE global error train', metrics.rmse(outputs_train, gshap_preds))
#
#     gshap_preds, gshap_vals = gshap.predict(data_test, return_shap_values=True)
#     outputs_test = model(data_test)
#     print('RMSE global error test', metrics.rmse(outputs_test, gshap_preds))
#
#     contribs_gshap = dict(zip(model.symbols, gshap_vals.T))
#
#     return contribs_gshap

class GlobalKernelSHAP(object):

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
