from math import sqrt
from math import ceil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from alibi.explainers import KernelShap
from alibi.explainers.shap_wrappers import KERNEL_SHAP_BACKGROUND_THRESHOLD

from posthoceval.global_shap import GlobalKernelShap
from posthoceval.model_generation import AdditiveModel
from posthoceval.model_generation import tsang_iclr18_models
from posthoceval.evaluate import symbolic_evaluate_func
from posthoceval import metrics

sns.set()


def grid_size(n):
    x = sqrt(n)
    return ceil(x), round(x)


def plot_feature_shapes():
    main_effects = model.main_effects
    plot_cols = []

    iv = 'Feature Value'
    iv_name = 'Feature Effect'
    dv = 'Output Contribution'
    typ = 'Output Type'
    plot_headers = (iv_name, iv, typ, dv)

    print()
    errs = []
    errs_centered = []
    for i in range(model.n_features):
        # TODO: I'm too stupid to figure out what do with +gshap.expected_value
        # pred_feat_contribution = gshap_vals[:, i] + gshap.expected_value
        pred_feat_contribution = gshap_vals[:, i]
        # TODO: only main effects considered atm
        main_effect_i = main_effects[i]
        data_test_i = data_test[:, i]
        eval_func = symbolic_evaluate_func(main_effect_i,
                                           (model.symbols[i],),
                                           x=data_test_i)
        real_feat_contribution = eval_func(data_test_i)

        # feat_name = [model.symbols[i].name] * len(data_test_i)
        feat_name = (['[' + model.symbols[i].name + ']: ' +
                      str(main_effect_i)] * len(data_test_i))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['True'] * len(data_test_i),
                          real_feat_contribution))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['gSHAP'] * len(data_test_i),
                          pred_feat_contribution))
        real_feat_contribution_centered = (
                real_feat_contribution - np.mean(real_feat_contribution))
        pred_feat_contribution_centered = (
                pred_feat_contribution - np.mean(pred_feat_contribution))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['True (Centered)'] * len(data_test_i),
                          real_feat_contribution_centered))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['gSHAP (Centered)'] * len(data_test_i),
                          pred_feat_contribution_centered))

        err = metrics.rmse(real_feat_contribution, pred_feat_contribution)
        err_centered = metrics.rmse(real_feat_contribution_centered,
                                    pred_feat_contribution_centered)
        errs.append(err)
        errs_centered.append(err_centered)
        print(model.symbols[i].name + ' feature shape error:', err)
        print(model.symbols[i].name + ' feature shape error centered:',
              err_centered)
        print('Abs error difference from centering:', abs(err - err_centered))
        print()
    print('Mean feature shape error:', np.mean(errs))
    print('Mean feature shape error centered:', np.mean(errs_centered))
    plot_data = np.hstack(plot_cols).T
    df = pd.DataFrame(plot_data, columns=plot_headers)
    # hstack with string values converts dtype to object, fix
    df.loc[:, iv] = df.loc[:, iv].astype(float)
    df.loc[:, dv] = df.loc[:, dv].astype(float)

    sns.relplot(data=df, x=iv, y=dv, col=iv_name, hue=typ, style=typ,
                kind='line', col_wrap=round(sqrt(model.n_features)))
    plt.show()


def evaluate_shap(debug=False):
    # TODO: gSHAP-linear, gSHAP-spline, etc.

    # Make model
    print('Model')
    # model = tsang_iclr18_models('f9')
    # TODO: this is temporary as I still haven't figured out the whole
    #  attributing interaction effects as main effects thing...
    import sympy as sym
    symbols = sym.symbols('x1:11')
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols
    expr = (sym.log(abs(x1)) * x1 - x2 + x3 ** 3 - x4 - 2 * x5 + sym.sin(x6) +
            sym.cos(x7) + 1 / (1 + x8))
    model = AdditiveModel.from_expr(expr, symbols)
    model.pprint()

    # Make data
    print('Data')
    n_samples = 500 if not debug else 20  # TODO 30_000
    # TODO: better data ranges based on continuity of function
    # TODO: use valid_variable_domains when properly+cleanly integrated
    # data = np.random.uniform(-1, +1, size=(n_samples, model.n_features))
    data = np.random.uniform(0, +1, size=(n_samples, model.n_features))

    # Split
    train_split_pct = 2 / 3
    split_idx = int(train_split_pct * len(data))
    data_train = data[:split_idx]
    data_test = data[split_idx:]

    # Explainer
    print('Explainer')
    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
    )
    fit_kwargs = {}
    if KERNEL_SHAP_BACKGROUND_THRESHOLD < n_samples:
        fit_kwargs['summarise_background'] = True

    print('Explainer fit')
    explainer.fit(data_train, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
    print('Explain')
    explanation = explainer.explain(data_train)
    expected_value = explanation.expected_value
    shap_values = explanation.shap_values[0]
    outputs_train = explanation.raw['raw_prediction']
    shap_values_g = explanation.raw['importances']['0']

    print('expected_value', expected_value)

    # print('outputs', outputs)
    # print('shap_values', shap_values)
    # print('shap_values_g', shap_values_g)

    shap_preds = np.sum(shap_values, axis=1) + expected_value
    print('sum of abs local error', (shap_preds - outputs_train).sum())

    print('Global SHAP')
    gshap = GlobalKernelShap(data_train, shap_values, expected_value)

    gshap_preds, gshap_vals = gshap.predict(data_train, return_shap_values=True)
    print('RMSE global error train', metrics.rmse(outputs_train, gshap_preds))

    gshap_preds, gshap_vals = gshap.predict(data_test, return_shap_values=True)
    outputs_test = model(data_test)
    print('RMSE global error test', metrics.rmse(outputs_test, gshap_preds))

    print('DFICK', model.feature_contributions(data_test))

    main_effects = model.main_effects
    plot_cols = []

    iv = 'Feature Value'
    iv_name = 'Feature Effect'
    dv = 'Output Contribution'
    typ = 'Output Type'
    plot_headers = (iv_name, iv, typ, dv)

    print()
    errs = []
    errs_centered = []
    for i in range(model.n_features):
        # TODO: I'm too stupid to figure out what do with +gshap.expected_value
        # pred_feat_contribution = gshap_vals[:, i] + gshap.expected_value
        pred_feat_contribution = gshap_vals[:, i]
        # TODO: only main effects considered atm
        main_effect_i = main_effects[i]
        data_test_i = data_test[:, i]
        eval_func = symbolic_evaluate_func(main_effect_i,
                                           (model.symbols[i],),
                                           x=data_test_i)
        real_feat_contribution = eval_func(data_test_i)

        # feat_name = [model.symbols[i].name] * len(data_test_i)
        feat_name = (['[' + model.symbols[i].name + ']: ' +
                      str(main_effect_i)] * len(data_test_i))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['True'] * len(data_test_i),
                          real_feat_contribution))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['gSHAP'] * len(data_test_i),
                          pred_feat_contribution))
        real_feat_contribution_centered = (
                real_feat_contribution - np.mean(real_feat_contribution))
        pred_feat_contribution_centered = (
                pred_feat_contribution - np.mean(pred_feat_contribution))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['True (Centered)'] * len(data_test_i),
                          real_feat_contribution_centered))
        plot_cols.append((feat_name,
                          data_test_i,
                          ['gSHAP (Centered)'] * len(data_test_i),
                          pred_feat_contribution_centered))

        err = metrics.rmse(real_feat_contribution, pred_feat_contribution)
        err_centered = metrics.rmse(real_feat_contribution_centered,
                                    pred_feat_contribution_centered)
        errs.append(err)
        errs_centered.append(err_centered)
        print(model.symbols[i].name + ' feature shape error:', err)
        print(model.symbols[i].name + ' feature shape error centered:',
              err_centered)
        print('Abs error difference from centering:', abs(err - err_centered))
        print()
    print('Mean feature shape error:', np.mean(errs))
    print('Mean feature shape error centered:', np.mean(errs_centered))
    plot_data = np.hstack(plot_cols).T
    df = pd.DataFrame(plot_data, columns=plot_headers)
    # hstack with string values converts dtype to object, fix
    df.loc[:, iv] = df.loc[:, iv].astype(float)
    df.loc[:, dv] = df.loc[:, dv].astype(float)

    sns.relplot(data=df, x=iv, y=dv, col=iv_name, hue=typ, style=typ,
                kind='line', col_wrap=round(sqrt(model.n_features)))
    plt.show()


if __name__ == '__main__':
    def main():
        import argparse

        parser = argparse.ArgumentParser(
            description=''
        )
        parser.add_argument('--debug', '-B', action='store_true',
                            help='Use fewer samples to make sure things work.')
        args = parser.parse_args()

        evaluate_shap(debug=args.debug)


    main()
