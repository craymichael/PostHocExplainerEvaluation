"""
viz.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import List
from typing import Set
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Any
from typing import Callable
from typing import Sequence

import warnings
from itertools import chain

import numpy as np
import pandas as pd

from posthoceval.expl_utils import apply_matching
from posthoceval.expl_utils import standardize_contributions
from posthoceval.models.model import AdditiveModel
from posthoceval.datasets.dataset import Dataset
from posthoceval.transform import Transformer
from posthoceval import metrics

# TODO: standardize...
# Dict[explainer_name, contribs]
RegressionContribs = Dict[Any, np.ndarray]
ClassificationContribs = List[RegressionContribs]
Contribs = Union[RegressionContribs, ClassificationContribs]


def make_tex_str(features, start_i, explained=False):
    out_strs = []
    for feats in features:
        feats_str = ','.join(
            f'x_{{{feat}}}' if isinstance(feat, int) else str(feat)
            for feat in feats
        )
        if explained:
            out_str = fr'\hat{{f}}_{{{start_i}}}({feats_str})'
        else:
            out_str = f'f_{{{start_i}}}({feats_str})'
        out_strs.append(out_str)
        start_i += 1
    return '$' + ('+'.join(out_strs) or '0') + '$'


def gather_viz_data(
        model: AdditiveModel,
        dataset: Dataset,
        transformer: Transformer,
        true_contribs: Contribs,
        pred_contribs_map: Dict[str, Contribs],
        dataset_sample_idxs: Sequence[int] = None,
        err_func: Union[List[Callable], Callable] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """pred_contribs: Dict[explainer_name, contribs]"""

    if err_func is None:
        err_func = [metrics.rmse, metrics.nrmse_interquartile,
                    metrics.nrmse_range, metrics.mape, metrics.corr,
                    metrics.spearmanr]
    elif not isinstance(err_func, list):
        err_func = [err_func]

    # dataframe rows (main effects and order-2 interaction effects)
    dfs = []
    dfs_3d = []
    ignored_true_effects = set()

    if isinstance(true_contribs, dict):
        true_contribs = [true_contribs]

    for explainer_name, pred_contribs in pred_contribs_map.items():
        if isinstance(pred_contribs, dict):
            pred_contribs = [pred_contribs]
        assert len(pred_contribs) == len(true_contribs), explainer_name

        # iterate over each class
        for class_k, (true_contribs_k, pred_contribs_k) in enumerate(
                zip(true_contribs, pred_contribs)):
            # Create target string
            target_str = f'{dataset.label_col}'
            if dataset.is_classification:
                try:
                    class_name = transformer.class_name(class_k)
                except ValueError:
                    class_name = str(class_k)
                target_str = target_str + ' = ' + class_name

            dfs_class, dfs_3d_class = _gather_viz_data_single_output(
                true_contribs_k=true_contribs_k,
                pred_contribs_k=pred_contribs_k,
                dataset=dataset,
                transformer=transformer,
                model=model,
                explainer_name=explainer_name,
                target_str=target_str,
                err_func=err_func,
                dataset_sample_idxs=dataset_sample_idxs,
                ignored_true_effects=ignored_true_effects,
            )
            dfs += dfs_class
            dfs_3d += dfs_3d_class

    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df_3d = pd.concat(dfs_3d, ignore_index=True) if dfs_3d else None

    return df, df_3d


def _gather_viz_data_single_output(
        true_contribs_k,
        pred_contribs_k,
        dataset: Dataset,
        transformer: Transformer,
        model: AdditiveModel,
        explainer_name: str,
        target_str: str,
        err_func: List[Callable],
        dataset_sample_idxs: Sequence[int],
        ignored_true_effects: Set,
):
    dfs = []
    dfs_3d = []

    # shed zero elements
    true_contribs_k = standardize_contributions(
        true_contribs_k, remove_zeros=False)
    pred_contribs_k = standardize_contributions(
        pred_contribs_k, remove_zeros=False)
    # find and apply matching
    components, goodness = metrics.generous_eval(true_contribs_k,
                                                 pred_contribs_k)
    n_explained = (len(dataset) if dataset_sample_idxs is None else
                   len(dataset_sample_idxs))
    matches = apply_matching(
        matching=components,
        true_expl=true_contribs_k,
        pred_expl=pred_contribs_k,
        n_explained=n_explained,
        explainer_name=explainer_name,
        always_numeric=False,
    )
    # Transform data to original space
    dataset_inv = transformer.inverse_transform(dataset,
                                                transform_numerical=True,
                                                transform_categorical=False,
                                                transform_target=False)
    X_inv = dataset_inv.X
    if dataset_sample_idxs is not None:
        X_inv = X_inv[dataset_sample_idxs]

    # "for the effects and corresponding contributions of each match..."
    for ((true_feats, pred_feats),
         (true_contrib_i, pred_contrib_i)) in matches.items():

        # Check if contribution is all zeros (which may be returned by
        #  apply_matching, e.g., pred has effect that true does not so
        #  true gives zeros for that effect)
        no_true_contrib = (true_contrib_i is None)
        no_pred_contrib = (pred_contrib_i is None)
        # gather all features (union of matched effects)
        all_feats = [*{*chain(chain.from_iterable(true_feats),
                              chain.from_iterable(pred_feats))}]
        f_idxs = [model.symbols.index(fi) for fi in all_feats]
        feature_str = ' & '.join(dataset.feature_names[fi] for fi in f_idxs)

        # print some metrics for the matched effect(s)
        if no_true_contrib:
            true_contrib_i = np.zeros_like(pred_contrib_i)
        if no_pred_contrib:
            pred_contrib_i = np.zeros_like(true_contrib_i)
        err_scores = {ef.__name__: ef(true_contrib_i, pred_contrib_i)
                      for ef in err_func}
        score_str = ' | '.join(f'{name}={score:.3g}'
                               for name, score in err_scores.items())
        print(f'{explainer_name} {feature_str}| {score_str}\n')

        if len(all_feats) > 2:
            warnings.warn(
                f'skipping match with {all_feats} for because it is an '
                f'interaction with order > 2.\n\ttrue_feats: {true_feats}'
                f'\n\tpred_feats: {pred_feats}'
            )
            continue

        # Do nothing if pred_contrib_is_zero and we are not recording data
        #  for true contribs
        ignore_true_effect = (true_feats in ignored_true_effects)
        if no_pred_contrib and ignore_true_effect:
            continue
        ignored_true_effects.add(true_feats)
        # inverse transform regression targets to original space (helps with
        #  interpretability)
        if dataset.is_regression and transformer.transforms_target:
            if not no_true_contrib:
                true_contrib_i = transformer.inverse_transform(
                    y=true_contrib_i)
            if not no_pred_contrib:
                pred_contrib_i = transformer.inverse_transform(
                    y=pred_contrib_i)

        # Gather relevant data
        xi = X_inv[:, f_idxs]

        # base data for visualization
        base = {
            'Class': target_str,
            'True Effect': [true_feats] * n_explained,
            'Predicted Effect': [pred_feats] * n_explained,
            'Match': feature_str,
        }
        if len(all_feats) == 1:
            # main effect otherwise interaction effect (order of 2 per earlier
            #  check)
            base['Feature Value'] = xi.squeeze(axis=1)
            store_target = dfs
        else:
            base['Feature Value x'] = xi[:, 0]
            base['Feature Value y'] = xi[:, 1]
            store_target = dfs_3d

        # predicted contributions (explainer)
        if not no_pred_contrib:
            pred_df_data = base.copy()
            pred_df_data['Explainer'] = explainer_name
            pred_df_data['Contribution'] = pred_contrib_i
            store_target.append(pd.DataFrame(pred_df_data))
        # true contributions (model)
        if not (ignore_true_effect or no_true_contrib):
            true_df_data = base.copy()
            true_df_data['Explainer'] = 'True'
            true_df_data['Contribution'] = true_contrib_i
            store_target.append(pd.DataFrame(true_df_data))

    return dfs, dfs_3d


def plot_fit():
    if task == 'regression':
        y_pred = model(X)

        # TODO: unify
        model_intercepts = 0
        if model_type == 'gam':
            contribs_full, model_intercepts = model.feature_contributions(
                X, return_intercepts=True)
        else:
            contribs_full = model.feature_contributions(X)

        print(f'GT vs. {model_type}')
        print(f' RMSE={metrics.rmse(y, y_pred)}')
        print(f'NRMSE={nrmse_func(y, y_pred)}')

        # This should be 0
        print(f'{model_type} Out vs. {model_type} Contribs')
        y_contrib_pred = np.asarray([*contribs_full.values()]).sum(axis=0)
        y_contrib_pred += model_intercepts
        print(f' RMSE={metrics.rmse(y_pred, y_contrib_pred)}')
        print(f'NRMSE={nrmse_func(y_pred, y_contrib_pred)}')

        print(f'{model_type} vs. Explainer')
        y_pred_trunc = model(X_trunc)
        print(f' RMSE={metrics.rmse(y_pred_trunc, y_expl)}')
        print(f'NRMSE={nrmse_func(y_pred_trunc, y_expl)}')

        fig, ax = plt.subplots()
        ax.scatter(sample_idxs_all,
                   y,
                   alpha=.65,
                   label='GT')
        ax.scatter(sample_idxs_all,
                   # sample_idxs,
                   y_pred,
                   # y_pred_trunc,
                   alpha=.65,
                   label=f'{model_type}')
        ax.scatter(sample_idxs,
                   y_expl,
                   alpha=.65,
                   label='Explainer')
        ax.set_xlabel('Sample idx')
        ax.set_ylabel('Predicted value')
        fig.legend()

        if plt.get_backend() == 'agg':
            fig.savefig(
                nonexistent_filename(
                    f'prediction_comparison_{model_type}_{explainer_name}.pdf'
                )
            )
        else:
            plt.show()
