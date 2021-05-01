"""
viz.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import List

import warnings
from itertools import chain

import numpy as np
import pandas as pd

from posthoceval.expl_utils import apply_matching
from posthoceval.expl_utils import standardize_contributions
from posthoceval.models.model import AdditiveModel
from posthoceval.transform import Transformer
from posthoceval import metrics


def gather_viz_data(model: AdditiveModel,
                    data: np.ndarray,
                    transformer: Transformer,
                    true_contribs,
                    pred_contribs,  # TODO: Dict[explainer_name, contribs]
                    explainer_name,
                    feature_names=None):
    """"""
    if feature_names is None:
        feature_names = model.symbol_names
    else:
        feature_names = [*map(str, feature_names)]

    # dataframe rows (main effects and order-2 interaction effects)
    rows = []
    rows_3d = []

    # TODO: this iterates over classes - consider case of regression...
    # iterate over each class
    for class_num, (e_true_i, e_pred_i) in enumerate(
            zip(true_contribs, pred_contribs)):
        rows_class, rows_3d_class = _gather_viz_data_single_output(
            e_true_i=e_true_i,
            e_pred_i=e_pred_i,  # TODO: Dict[explainer_name, contribs]
            data=data,
            model=model,
            feature_names=feature_names,
            explainer_name=explainer_name,
        )
        rows += rows_class
        rows_3d += rows_3d_class

    df = pd.DataFrame(rows) if rows else None
    df_3d = pd.DataFrame(rows_3d) if rows_3d else None

    return df, df_3d


def _gather_viz_data_single_output(
        e_true_i,
        e_pred_i,  # TODO: Dict[explainer_name, contribs]
        data: np.ndarray,
        model: AdditiveModel,
        feature_names: List,
        explainer_name: str,
):
    rows = []
    rows_3d = []

    # shed zero elements
    e_true_i = standardize_contributions(e_true_i)
    e_pred_i = standardize_contributions(e_pred_i)

    components, goodness = metrics.generous_eval(e_true_i, e_pred_i)

    matches = apply_matching(
        matching=components,
        true_expl=e_true_i,
        pred_expl=e_pred_i,
        n_explained=len(data),
        explainer_name=explainer_name,
    )

    true_func_idx = pred_func_idx = 1
    for ((true_feats, pred_feats),
         (true_contrib_i, pred_contrib_i)) in matches.items():

        # Check if contribution is all zeros (which may be returned by
        #  apply_matching, e.g., pred has effect that true does not so
        #  true gives zeros for that effect)
        true_contrib_is_zero = (true_contrib_i == 0.).all()
        pred_contrib_is_zero = (pred_contrib_i == 0.).all()

        all_feats = [*{*chain(chain.from_iterable(true_feats),
                              chain.from_iterable(pred_feats))}]
        f_idxs = [model.symbols.index(fi) for fi in all_feats]

        feature_str = ' & '.join(feature_names[fi] for fi in f_idxs)

        match_str = (
            feature_str  # + '\n' +
            # TODO: depression
            # 'True: ' +
            # make_tex_str(true_feats, true_func_idx, False) +
            # ' | Predicted: ' +
            # ' vs. ' +
            # make_tex_str(pred_feats, pred_func_idx, True)
        )
        # TODO: these are used to attempt to have consistent features as
        #  headers but is broken at the moment...probably get rid of and
        #  just change the way multiple explainer results are combined
        true_func_idx += len(true_feats)
        pred_func_idx += len(pred_feats)

        # TODO: fix this (feature-wise error metrics)
        print(match_str, ' RMSE',
              metrics.rmse(true_contrib_i, pred_contrib_i))
        nrmse_score = nrmse_func(true_contrib_i, pred_contrib_i)
        print(match_str, 'NRMSE', nrmse_score)
        print()

        # pretty format score
        # TODO: sad times we have here
        # match_str += ('\nNRMSE = ' + (f'{nrmse_score:.3f}'
        #                               if (1e-3 < nrmse_score < 1e3) else
        #                               f'{nrmse_score:.3}'))

        if len(all_feats) > 2:
            warnings.warn(
                f'skipping match with {all_feats} for because it is an '
                f'interaction with order > 2.\n\ttrue_feats: {true_feats}'
                f'\n\tpred_feats: {pred_feats}'
            )
            continue

        data_inverse = data
        # TODO...transformer
        if x_scaler is not None:
            if hasattr(x_scaler, 'inverse_transform'):
                data_inverse = x_scaler.inverse_transform(data_inverse)
            else:
                warnings.warn('x_scaler provided but it does not have '
                              'the inverse_transform() method: '
                              f'{x_scaler}')

        xi = data_inverse[:, f_idxs]
        base = {
            'Class': class_num,  # TODO: class_num --> class name
            'True Effect': true_feats,
            'Predicted Effect': pred_feats,
            'Match': match_str,
        }
        true_row = base.copy()
        true_row['explainer'] = 'True'

        pred_row = base
        pred_row['explainer'] = explainer_name

        if y_scaler is not None:
            if not true_contrib_is_zero:
                true_contrib_i = scale_y(
                    y_scaler.inverse_transform, true_contrib_i)
            if not pred_contrib_is_zero:
                pred_contrib_i = scale_y(
                    y_scaler.inverse_transform, pred_contrib_i)

        for true_contrib_ik, pred_contrib_ik, xik in zip(
                true_contrib_i, pred_contrib_i, xi):
            # main effect otherwise interaction effect (order of 2 per
            #  earlier check)
            is_main_effect = (len(all_feats) == 1)

            # do not add zero effects
            if not pred_contrib_is_zero:
                pred_row_i = pred_row.copy()
                pred_row_i['contribution'] = pred_contrib_ik

                if is_main_effect:
                    pred_row_i['feature value'] = xik[0]
                    rows.append(pred_row_i)
                else:
                    pred_row_i['feature value x'] = xik[0]
                    pred_row_i['feature value y'] = xik[1]
                    rows_3d.append(pred_row_i)

            # do not add zero effects
            if (not true_contrib_is_zero and
                    (expl_i + 1) == len(explainer_array)):  # TODO(refactor)
                true_row_i = true_row.copy()
                true_row_i['contribution'] = true_contrib_ik

                if is_main_effect:
                    true_row_i['feature value'] = xik[0]
                    rows.append(true_row_i)
                else:
                    true_row_i['feature value x'] = xik[0]
                    true_row_i['feature value y'] = xik[1]
                    rows_3d.append(true_row_i)

    return rows, rows_3d
