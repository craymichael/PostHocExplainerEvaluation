"""
All functions of format:
`func(y_true, y_pred, *args, **kwargs)`
"""
import logging
from functools import partial
from typing import Iterable

from sklearn import metrics
from sklearn.metrics import pairwise
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'effect_detection_f1', 'effect_detection_jaccard_index',
    'effect_detection_precision', 'effect_detection_recall',
    'mean_squared_error', 'mse',
    'root_mean_squared_error', 'rmse',
    'mean_absolute_percentage_error', 'mape',
    'accuracy', 'balanced_accuracy',
    'cosine_distances', 'euclidean_distances',
]


def _standardize_effect(e):
    e = tuple(sorted({*e})) if isinstance(e, tuple) else (e,)
    assert e, 'received empty effect'
    return e


def strict_eval(y_true: Iterable, y_pred: Iterable):
    """TODO: also return goodness - as arg"""
    y_true = {*map(_standardize_effect, y_true)}
    y_pred = {*map(_standardize_effect, y_pred)}

    matching = [((match,), (match,)) for match in y_true & y_pred]
    matching.extend(((miss,), (None,)) for miss in y_true - y_pred)
    matching.extend(((None,), (miss,)) for miss in y_pred - y_true)

    return matching


def _effects_confusion_matrix(y_true, y_pred, effects='all'):
    """Excludes true negatives - this information is usually useless in this
    context"""
    effects = effects.lower()
    assert effects in {'main', 'interaction', 'all'}

    if effects == 'main':
        # Main effects
        y_true = {_standardize_effect(x) for x in y_true if len(x) == 1}
        y_pred = {_standardize_effect(x) for x in y_pred if len(x) == 1}
    elif effects == 'interaction':
        # Interaction effects
        y_true = {_standardize_effect(x) for x in y_true if len(x) >= 2}
        y_pred = {_standardize_effect(x) for x in y_pred if len(x) >= 2}
    else:
        y_true = {*map(_standardize_effect, y_true)}
        y_pred = {*map(_standardize_effect, y_pred)}

    tp = len(y_true & y_pred)
    fn = len(y_true - y_pred)
    fp = len(y_pred - y_true)

    return tp, fn, fp


def _handle_zero_divisor(numerator, denominator, metric, value):
    if denominator == 0:
        logger.warning(f'Divisor is 0 in metric {metric}. Defining {metric} '
                       f'as {value}')
        return value
    # Otherwise:
    return numerator / denominator


def effect_detection_f1(y_true, y_pred, effects='all'):
    """
    harmonic mean of precision and recall (aka sensitivity or true positive
    rate)
    """
    tp, fn, fp = _effects_confusion_matrix(y_true, y_pred, effects=effects)
    tp2 = 2 * tp
    return _handle_zero_divisor(tp2, tp2 + fn + fp,
                                'f1 (effect detection)', 1.)


def effect_detection_jaccard_index(y_true, y_pred, effects='all'):
    """
    Jaccard index
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    tp, fn, fp = _effects_confusion_matrix(y_true, y_pred, effects=effects)
    return _handle_zero_divisor(tp, tp + fn + fp,
                                'jaccard index (effect detection)', 1.)


def effect_detection_precision(y_true, y_pred, effects='all'):
    tp, fn, fp = _effects_confusion_matrix(y_true, y_pred, effects=effects)
    return _handle_zero_divisor(tp, tp + fp,
                                'precision (effect detection)', 1.)


def effect_detection_recall(y_true, y_pred, effects='all'):
    tp, fn, fp = _effects_confusion_matrix(y_true, y_pred, effects=effects)
    return _handle_zero_divisor(tp, tp + fn,
                                'recall (effect detection)', 1.)


cosine_distances = partial(pairwise.paired_distances, metric='cosine')
euclidean_distances = partial(pairwise.paired_distances, metric='euclidean')

accuracy = metrics.accuracy_score
balanced_accuracy = metrics.balanced_accuracy_score

mean_squared_error = metrics.mean_squared_error
mse = mean_squared_error

root_mean_squared_error = partial(metrics.mean_squared_error, squared=False)
rmse = root_mean_squared_error

if hasattr(metrics, 'mean_absolute_percentage_error'):
    mean_absolute_percentage_error = metrics.mean_absolute_percentage_error
    mape = mean_absolute_percentage_error
else:
    # copy pasta from 0.24 (unreleased at time of copy)
    from sklearn.utils.validation import check_consistent_length
    from sklearn.metrics._regression import _check_reg_targets  # noqa


    def mean_absolute_percentage_error(y_true, y_pred,
                                       sample_weight=None,
                                       multioutput='uniform_average'):
        """Mean absolute percentage error regression loss.
        Note here that we do not represent the output as a percentage in range
        [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in
        the :ref:`User Guide <mean_absolute_percentage_error>`.
        .. versionadded:: 0.24
        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        multioutput : {'raw_values', 'uniform_average'} or array-like
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            If input is list then the shape must be (n_outputs,).
            'raw_values' :
                Returns a full set of errors in case of multioutput input.
            'uniform_average' :
                Errors of all outputs are averaged with uniform weight.
        Returns
        -------
        loss : float or ndarray of floats in the range [0, 1/eps]
            If multioutput is 'raw_values', then mean absolute percentage error
            is returned for each output separately.
            If multioutput is 'uniform_average' or an ndarray of weights, then
            the weighted average of all output errors is returned.
            MAPE output is non-negative floating point. The best value is 0.0.
            But note the fact that bad predictions can lead to arbitarily large
            MAPE values, especially if some y_true values are very close to
            zero. Note that we return a large value instead of `inf` when
            y_true is zero.
        Examples
        --------
        >>> from sklearn.metrics import mean_absolute_percentage_error  # noqa
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_absolute_percentage_error(y_true, y_pred)
        0.3273...
        >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
        >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
        >>> mean_absolute_percentage_error(y_true, y_pred)
        0.5515...
        >>> mean_absolute_percentage_error(y_true, y_pred,
                                           multioutput=[0.3, 0.7])
        0.6198...
        """
        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput)
        check_consistent_length(y_true, y_pred, sample_weight)
        epsilon = np.finfo(np.float64).eps
        mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
        output_errors = np.average(mape,
                                   weights=sample_weight, axis=0)
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                return output_errors
            elif multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                multioutput = None

        return np.average(output_errors, weights=multioutput)
