"""
viz.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from itertools import chain

from posthoceval.expl_utils import apply_matching
from posthoceval.metrics import standardize_contributions
from posthoceval.metrics import generous_eval
from posthoceval.model_generation import AdditiveModel


# TODO naming of function
def gather_data(model: AdditiveModel,
                data,
                true_contribs,
                pred_contribs,
                explainer_name,
                feature_names=None):
    """"""
    if feature_names is None:
        feature_names = [*map(str, model.symbols)]

    for i, (e_true_i, e_pred_i) in enumerate(zip(true_contribs, pred_contribs)):
        # shed zero elements
        e_true_i = standardize_contributions(e_true_i)
        e_pred_i = standardize_contributions(e_pred_i)
        components, goodness = generous_eval(e_true_i, e_pred_i)
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

            # TODO: blegh
            true_contrib_is_zero = (true_contrib_i == 0.).all()
            pred_contrib_is_zero = (pred_contrib_i == 0.).all()

            all_feats = [*{*chain(chain.from_iterable(true_feats),
                                  chain.from_iterable(pred_feats))}]
            f_idxs = [model.symbols.index(fi) for fi in all_feats]

            feature_str = ' & '.join(map(str, (feature_names[fi] for fi in f_idxs)))

            match_str = (
                feature_str  # + '\n' +
                # TODO: depression
                # 'True: ' +
                # make_tex_str(true_feats, true_func_idx, False) +
                # ' | Predicted: ' +
                # ' vs. ' +
                # make_tex_str(pred_feats, pred_func_idx, True)
            )
            true_func_idx += len(true_feats)
            pred_func_idx += len(pred_feats)

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
                print(
                    f'skipping match with {all_feats} for now as is interaction '
                    f'with order > 2 true_feats {true_feats} | pred_feats '
                    f'{pred_feats}')
                continue
            data_inverse = data
            # TODO!!!inverse_transform
            if scaler is not None and hasattr(scaler, 'inverse_transform'):
                data_inverse = scaler.inverse_transform(data_inverse)
            xi = data_inverse[:, f_idxs]
            base = {
                'class': i,
                'true_effect': true_feats,
                'pred_effect': pred_feats,
                'Match': match_str,
            }
            true_row = base.copy()
            true_row['explainer'] = 'True'

            pred_row = base
            pred_row['explainer'] = explainer_name

            for true_contrib_ik, pred_contrib_ik, xik in zip(
                    true_contrib_i, pred_contrib_i, xi):
                true_row_i = true_row.copy()
                if scale_y is not None and y.ndim == 1:
                    true_contrib_ik = float(scale_y(
                        y_scaler.inverse_transform, true_contrib_ik))
                true_row_i['contribution'] = true_contrib_ik

                pred_row_i = pred_row.copy()
                if scale_y is not None and y.ndim == 1:
                    pred_contrib_ik = float(scale_y(
                        y_scaler.inverse_transform, pred_contrib_ik))
                pred_row_i['contribution'] = pred_contrib_ik

                if len(all_feats) == 1:  # main effect
                    if not pred_contrib_is_zero:
                        pred_row_i['feature value'] = xik[0]
                        rows.append(pred_row_i)

                    if (not true_contrib_is_zero and
                            (expl_i + 1) == len(explainer_array)):
                        true_row_i['feature value'] = xik[0]
                        rows.append(true_row_i)
                else:  # interaction effect (order of 2)
                    if not pred_contrib_is_zero:
                        pred_row_i['feature value x'] = xik[0]
                        pred_row_i['feature value y'] = xik[1]
                        rows_3d.append(pred_row_i)

                    if (not true_contrib_is_zero and
                            (expl_i + 1) == len(explainer_array)):
                        true_row_i['feature value x'] = xik[0]
                        true_row_i['feature value y'] = xik[1]
                        rows_3d.append(true_row_i)

    df = pd.DataFrame(rows)
