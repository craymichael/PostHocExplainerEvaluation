"""
viz.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Any

import warnings
from itertools import chain
from itertools import repeat

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


def gather_viz_data(
        model: AdditiveModel,
        data: Union[Dataset, np.ndarray],
        transformer: Transformer,
        true_contribs: Contribs,
        all_pred_contribs: Dict[str, Contribs],
        feature_names=None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """"""
    # TODO: give me the untransformed data here...or use transformer here for
    #  inverse transform minus one-hot categorical data........
    #  If we do latter option then we can pass transformer to other func and
    #  use to inverse_transform y from explainers! >:(

    # pred_contribs: Dict[explainer_name, contribs]
    if feature_names is None:
        feature_names = model.symbol_names
    else:
        feature_names = [*map(str, feature_names)]

    # dataframe rows (main effects and order-2 interaction effects)
    dfs = []
    dfs_3d = []

    is_dataset = isinstance(data, Dataset)

    for explainer_name, pred_contribs in all_pred_contribs.items():
        if isinstance(pred_contribs, dict):
            pred_contribs = [pred_contribs]

        # iterate over each class
        for class_num, (e_true_i, e_pred_i) in enumerate(
                zip(true_contribs, pred_contribs)):
            # Create target string
            target_str = f'{data.label_col}' if is_dataset else None
            if len(true_contribs) == 1:
                target_str = target_str or 'Target'
            else:
                try:
                    class_name = transformer.class_name(class_num)
                except ValueError:
                    class_name = str(class_num)
                target_str = target_str or 'Class'
                target_str = target_str + ' = ' + class_name

            dfs_class, dfs_3d_class = _gather_viz_data_single_output(
                e_true_i=e_true_i,
                e_pred_i=e_pred_i,
                data=data.X if is_dataset else data,
                model=model,
                feature_names=feature_names,
                explainer_name=explainer_name,
                target_str=target_str,
            )
            dfs += dfs_class
            dfs_3d += dfs_3d_class

    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df_3d = pd.concat(dfs_3d, ignore_index=True) if dfs_3d else None

    return df, df_3d


def _gather_viz_data_single_output(
        e_true_i,
        e_pred_i,  # TODO: Dict[explainer_name, contribs]
        data: np.ndarray,
        model: AdditiveModel,
        feature_names: List,
        explainer_name: str,
        target_str: str,
):
    dfs = []
    dfs_3d = []

    # shed zero elements
    e_true_i = standardize_contributions(e_true_i)
    e_pred_i = standardize_contributions(e_pred_i)

    components, goodness = metrics.generous_eval(e_true_i, e_pred_i)
    n_explained = len(data)
    matches = apply_matching(
        matching=components,
        true_expl=e_true_i,
        pred_expl=e_pred_i,
        n_explained=n_explained,
        explainer_name=explainer_name,
    )

    # "for the effects and corresponding contributions of each match..."
    for ((true_feats, pred_feats),
         (true_contrib_i, pred_contrib_i)) in matches.items():

        # Check if contribution is all zeros (which may be returned by
        #  apply_matching, e.g., pred has effect that true does not so
        #  true gives zeros for that effect)
        true_contrib_is_zero = (true_contrib_i == 0.).all()
        pred_contrib_is_zero = (pred_contrib_i == 0.).all()
        # gather all features (union of matched effects)
        all_feats = [*{*chain(chain.from_iterable(true_feats),
                              chain.from_iterable(pred_feats))}]
        f_idxs = [model.symbols.index(fi) for fi in all_feats]
        feature_str = ' & '.join(feature_names[fi] for fi in f_idxs)

        # print some metrics for the matched effect(s)
        print(feature_str, ' RMSE',
              metrics.rmse(true_contrib_i, pred_contrib_i))
        nrmse_score = nrmse_func(true_contrib_i, pred_contrib_i)
        print(feature_str, 'NRMSE', nrmse_score)
        print()

        if len(all_feats) > 2:
            warnings.warn(
                f'skipping match with {all_feats} for because it is an '
                f'interaction with order > 2.\n\ttrue_feats: {true_feats}'
                f'\n\tpred_feats: {pred_feats}'
            )
            continue
        # main effect otherwise interaction effect (order of 2 per
        #  earlier check)
        is_main_effect = (len(all_feats) == 1)

        # Transform data to original space
        data_inverse = data
        # TODO...transformer
        if x_scaler is not None:
            if hasattr(x_scaler, 'inverse_transform'):
                data_inverse = x_scaler.inverse_transform(data_inverse)
            else:
                warnings.warn('x_scaler provided but it does not have '
                              'the inverse_transform() method: '
                              f'{x_scaler}')
        if y_scaler is not None:
            if not true_contrib_is_zero:
                true_contrib_i = scale_y(
                    y_scaler.inverse_transform, true_contrib_i)
            if not pred_contrib_is_zero:
                pred_contrib_i = scale_y(
                    y_scaler.inverse_transform, pred_contrib_i)

        # Gather relevant data
        xi = data_inverse[:, f_idxs]

        # TODO do nothing if pred_contrib_is_zero and we are not recording data
        #  for true contribs
        # if pred_contrib_is_zero and true_contrib_is_zero:
        #     continue

        # base data for visualization
        base = {
            'Class': target_str,
            'True Effect': repeat(true_feats, times=n_explained),
            'Predicted Effect': repeat(pred_feats, times=n_explained),
            'Match': feature_str,
        }
        if is_main_effect:
            base['Feature Value'] = xi
            store_target = dfs
        else:
            base['Feature Value x'] = xi[:, 0]
            base['Feature Value y'] = xi[:, 1]
            store_target = dfs_3d
        # predicted contributions (explainer)
        if not pred_contrib_is_zero:
            pred_df_data = base.copy()
            pred_df_data['Explainer'] = explainer_name
            pred_df_data['Contribution'] = pred_contrib_i
            store_target.append(pd.DataFrame(pred_df_data))
        # true contributions (model)
        if not true_contrib_is_zero:
            true_df_data = base.copy()
            true_df_data['Explainer'] = 'True'
            true_df_data['Contribution'] = true_contrib_i
            store_target.append(pd.DataFrame(true_df_data))

    return dfs, dfs_3d
