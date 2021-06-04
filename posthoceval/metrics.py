
import logging
from functools import partial as partial_
from functools import update_wrapper
from typing import Iterable
from typing import List
from typing import Tuple

import sympy
from sklearn import metrics as sk_metrics
from sklearn.metrics import pairwise

import numpy as np
import scipy.stats

from posthoceval.expl_utils import standardize_effect

logger = logging.getLogger(__name__)

__all__ = [
    'strict_eval', 'generous_eval',
    'effect_detection_f1', 'effect_detection_jaccard_index',
    'effect_detection_precision', 'effect_detection_recall',
    'mean_squared_error', 'mse',
    'root_mean_squared_error', 'rmse',
    'mean_absolute_percentage_error', 'mape',
    'normalized_root_mean_squared_error', 'nrmse', 'nrmse_range', 'nrmse_std',
    'nrmse_interquartile', 'nrmse_mean',
    'accuracy', 'balanced_accuracy',
    'cosine_distances', 'euclidean_distances',
    'spearmanr', 'spearman_corr', 'spearman_rank_correlation',
    'pearson_correlation_coef', 'corr', 'corrcoef'
]


def partial(*args, **kwargs):
    
    if isinstance(args[0], str):
        name = args[0]
        func = args[1]
        args = args[2:]
    else:
        name = None
        func = args[0]
        args = args[1:]

    partial_func = partial_(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name is None:
        partial_func.__name__ += '__' + '__'.join(
            f'{k}_{v}' for k, v in kwargs.items())
    else:
        partial_func.__name__ = name
    return partial_func


def strict_eval(y_true: Iterable, y_pred: Iterable):
    
    y_true = {*map(standardize_effect, y_true)}
    y_pred = {*map(standardize_effect, y_pred)}

    intersection = y_true & y_pred
    residual_true = y_true - y_pred
    residual_pred = y_pred - y_true

    matching = [([match], [match]) for match in intersection]
    matching.extend(([miss], []) for miss in residual_true)
    matching.extend(([], [miss]) for miss in residual_pred)

    goodness = [1.] * len(intersection)
    goodness.extend([0.] * (len(residual_true) + len(residual_pred)))

    return matching, goodness


def sorted_sym(iterable, reverse=False):
    

    
    
    
    
    
    

    return sorted(iterable, key=str, reverse=reverse)


def generous_eval(
        y_true: Iterable,
        y_pred: Iterable,
        maybe_exact=False
) -> Tuple[List[Tuple[List[Tuple], List[Tuple]]], List[float]]:
    
    y_true = {*map(standardize_effect, y_true)}
    y_pred = {*map(standardize_effect, y_pred)}

    
    
    
    
    
    
    components: List[Tuple[List[Tuple], List[Tuple]]] = []
    
    goodness = []

    if maybe_exact:
        
        
        common = y_true & y_pred
        components.extend(([effect], [effect]) for effect in common)
        
        goodness.extend([1.] * len(common))
        y_true -= common
        y_pred -= common

    
    y_true = [*y_true]
    y_pred = [*y_pred]

    y_true_sets = [*map(set, y_true)]
    y_pred_sets = [*map(set, y_pred)]

    n_true = len(y_true)
    n_pred = len(y_pred)

    visited_t = set()
    visited_p = set()

    
    adj = np.fromiter((effect_t & effect_p
                       for effect_t in y_true_sets
                       for effect_p in y_pred_sets),
                      dtype=bool).reshape(n_true, n_pred)

    
    for i0 in range(len(y_true)):
        if i0 in visited_t:  
            continue
        visited_t.add(i0)  

        component_t = [y_true[i0]]
        component_p = []

        stack = [i0]
        stack_t = True

        
        goodness_edges = []

        
        while stack:
            new_stack = []
            if stack_t:  
                for i in stack:
                    y_true_i = y_true[i]
                    for j in range(n_pred):
                        
                        if not adj[i, j] or j in visited_p:
                            continue
                        visited_p.add(j)  
                        y_pred_j = y_pred[j]
                        component_p.append(y_pred_j)
                        new_stack.append(j)

                        goodness_edges.append(
                            effect_detection_jaccard_index(y_true_i, y_pred_j)
                        )
            else:  
                for j in stack:
                    y_pred_j = y_pred[j]
                    for i in range(n_true):
                        
                        if not adj[i, j] or i in visited_t:
                            continue
                        visited_t.add(i)  
                        y_true_i = y_true[i]
                        component_t.append(y_true_i)
                        new_stack.append(i)

                        goodness_edges.append(
                            effect_detection_jaccard_index(y_true_i, y_pred_j)
                        )
            stack = new_stack
            stack_t = not stack_t  

        component_t = sorted_sym(component_t)
        component_p = sorted_sym(component_p)
        if component_t == component_p:
            components.extend(
                ([ct], [cp]) for ct, cp in zip(component_t, component_p)
            )
            goodness.extend([1.] * len(component_t))
        else:
            components.append((component_t, component_p))
            goodness.append(np.mean(goodness_edges) if goodness_edges else 0.)

    
    
    for j, effect_p in enumerate(y_pred):
        if j not in visited_p:
            components.append(([], [effect_p]))
            goodness.append(0.)  

    return components, goodness


def _effects_confusion_matrix(y_true, y_pred, effects='all'):
    effects = effects.lower()
    assert effects in {'main', 'interaction', 'all'}

    y_true = map(standardize_effect, y_true)
    y_pred = map(standardize_effect, y_pred)

    if effects == 'main':
        
        y_true = {x for x in y_true if len(x) == 1}
        y_pred = {x for x in y_pred if len(x) == 1}
    elif effects == 'interaction':
        
        y_true = {x for x in y_true if len(x) >= 2}
        y_pred = {x for x in y_pred if len(x) >= 2}
    else:
        y_true = {*y_true}
        y_pred = {*y_pred}

    tp = len(y_true & y_pred)
    fn = len(y_true - y_pred)
    fp = len(y_pred - y_true)

    return tp, fn, fp


def _handle_zero_divisor(numerator, denominator, metric, value):
    if denominator == 0:
        logger.warning(f'Divisor is 0 in metric {metric}. Defining {metric} '
                       f'as {value}')
        return value
    
    return numerator / denominator


def effect_detection_f1(y_true, y_pred, effects='all'):
    tp, fn, fp = _effects_confusion_matrix(y_true, y_pred, effects=effects)
    tp2 = 2 * tp
    return _handle_zero_divisor(tp2, tp2 + fn + fp,
                                'f1 (effect detection)', 1.)


def effect_detection_jaccard_index(y_true, y_pred, effects='all'):
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


cosine_distances = partial('cosine_distances', pairwise.paired_distances,
                           metric='cosine')
euclidean_distances = partial('euclidean_distances', pairwise.paired_distances,
                              metric='euclidean')

accuracy = sk_metrics.accuracy_score
balanced_accuracy = sk_metrics.balanced_accuracy_score

mean_squared_error = sk_metrics.mean_squared_error
mse = mean_squared_error

root_mean_squared_error = partial('root_mean_squared_error',
                                  sk_metrics.mean_squared_error, squared=False)
rmse = root_mean_squared_error


def normalized_root_mean_squared_error(y_true, y_pred, sample_weight=None,
                                       multioutput='uniform_average',
                                       normalize='range'):
    rmse_ = root_mean_squared_error(y_true, y_pred,
                                    sample_weight=sample_weight,
                                    multioutput=multioutput)
    epsilon = np.finfo(np.float64).eps

    if normalize == 'mean':
        divisor = np.mean(y_true)
    elif normalize == 'range':
        divisor = np.max(y_true) - np.min(y_true)
    elif normalize == 'interquartile':
        q25, q75 = np.percentile(y_true, [25, 75])
        divisor = q75 - q25
    elif normalize == 'std':
        divisor = np.std(y_true)
    else:
        raise ValueError('`normalize` must be "mean", "range", '
                         '"interquartile", or "std", but received '
                         f'"{normalize}"')
    return rmse_ / np.maximum(divisor, epsilon)


nrmse = normalized_root_mean_squared_error
nrmse_mean = partial(normalized_root_mean_squared_error, normalize='mean')
nrmse_range = partial(normalized_root_mean_squared_error, normalize='range')
nrmse_interquartile = partial(normalized_root_mean_squared_error,
                              normalize='interquartile')
nrmse_std = partial(normalized_root_mean_squared_error, normalize='std')


def pearson_correlation_coef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


corrcoef = pearson_correlation_coef
corr = pearson_correlation_coef


def spearman_rank_correlation(y_true, y_pred, p_val=False):
    result = scipy.stats.spearmanr(y_true, y_pred)
    if p_val:
        return result
    else:
        return result[0]


spearmanr = spearman_rank_correlation
spearman_corr = spearman_rank_correlation

if hasattr(sk_metrics, 'mean_absolute_percentage_error'):
    mean_absolute_percentage_error = sk_metrics.mean_absolute_percentage_error
else:
    
    from sklearn.utils.validation import check_consistent_length
    from sklearn.metrics._regression import _check_reg_targets  


    def mean_absolute_percentage_error(y_true, y_pred,
                                       sample_weight=None,
                                       multioutput='uniform_average'):
        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput)
        check_consistent_length(y_true, y_pred, sample_weight)
        epsilon = np.finfo(np.float64).eps
        mape_ = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
        output_errors = np.average(mape_,
                                   weights=sample_weight, axis=0)
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                return output_errors
            elif multioutput == 'uniform_average':
                
                multioutput = None

        return np.average(output_errors, weights=multioutput)

mape = mean_absolute_percentage_error
