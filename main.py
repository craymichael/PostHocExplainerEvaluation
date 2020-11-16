from math import sqrt
from math import ceil

import cProfile
from functools import wraps
from collections import defaultdict

from itertools import chain

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from alibi.explainers import KernelShap
from alibi.explainers.shap_wrappers import KERNEL_SHAP_BACKGROUND_THRESHOLD

from interpret.glassbox import ExplainableBoostingRegressor
# from interpret.glassbox import LinearRegression  TODO
# from interpret.blackbox import PartialDependence  TODO

# TODO: commented out in source???? why
# from interpret.blackbox import PermutationImportance

from posthoceval.global_shap import GlobalKernelShap
from posthoceval.model_generation import AdditiveModel
from posthoceval.model_generation import tsang_iclr18_models
from posthoceval.evaluate import symbolic_evaluate_func
from posthoceval import metrics

sns.set()

PROFILE = False


# def grid_size(n):
#     x = sqrt(n)
#     return ceil(x), round(x)

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PROFILE:
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                ret = func(*args, **kwargs)
                profiler.disable()
                return ret
            finally:
                filename = func.__name__ + '.pstat'
                profiler.dump_stats(filename)
        else:
            return func(*args, **kwargs)

    return wrapper


def eval_and_plot(data, contribs_true, effects_true, contribs_explainers, symbols=None):
    if symbols is None:
        # By default use all provided symbols (symbols can for example be a
        # subset, like just main effects)
        symbols = contribs_true.keys()

    if not isinstance(contribs_explainers, dict):
        # Assume iterable otherwise...
        contribs_explainers = {'Explainer': contribs_explainers}

    iv = 'Feature Value'
    iv_name = 'Feature Effect'
    dv = 'Output Contribution'
    typ = 'Output Type'
    plot_headers = (iv_name, iv, typ, dv)

    errs = defaultdict(lambda: [])
    plot_cols = []
    n_samples = len(data)

    max_expl_name = max(len(explainer_name)
                        for explainer_name in contribs_explainers.keys())
    all_feat_names = chain(
        s.name for contribs_explainer in contribs_explainers.values()
        for s in contribs_explainer.keys()
    )
    max_feat_name = max(len(feature_name)
                        for feature_name in all_feat_names)
    explainer_str = '[{{:>{}}}]'.format(max_expl_name)
    feature_str = '[{{:>{}}}]'.format(max_feat_name)

    for i, symbol in enumerate(symbols):
        data_feat_i = data[:, i]

        contrib_sym_true = contribs_true[symbol]
        contrib_eff_true = effects_true[symbol]

        feat_name = (['[' + symbol.name + ']: ' +
                      str(contrib_eff_true)] * n_samples)
        plot_cols.append((feat_name,
                          data_feat_i,
                          ['True'] * n_samples,
                          contrib_sym_true))

        for explainer_name, contribs_explainer in contribs_explainers.items():
            contrib_sym_explainer = contribs_explainer[symbol]
            plot_cols.append((feat_name,
                              data_feat_i,
                              [explainer_name] * n_samples,
                              contrib_sym_explainer))

            err = metrics.rmse(contrib_sym_true, contrib_sym_explainer)
            errs[explainer_name].append(err)
            explainer_prefix = explainer_str.format(explainer_name)
            feature_prefix = feature_str.format(symbol.name)
            print(explainer_prefix + ' ' + feature_prefix +
                  f' feature shape RMSE: {err:.6f}')

    print()
    for explainer_name in contribs_explainers:
        explainer_prefix = explainer_str.format(explainer_name)
        print(explainer_prefix + f' Mean feature shape RMSE: '
                                 f'{np.mean(errs[explainer_name]):.6f}')

    plot_data = np.hstack(plot_cols).T
    df = pd.DataFrame(plot_data, columns=plot_headers)
    # hstack with string values converts dtype to object, fix
    df.loc[:, iv] = df.loc[:, iv].astype(float)
    df.loc[:, dv] = df.loc[:, dv].astype(float)

    sns.relplot(data=df, x=iv, y=dv, col=iv_name, hue=typ, style=typ,
                kind='line', col_wrap=round(sqrt(len(symbols))))
    plt.show()


@profile
def gshap_explain(model, data_train, data_test):
    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
    )
    fit_kwargs = {}
    if KERNEL_SHAP_BACKGROUND_THRESHOLD < len(data_train):
        fit_kwargs['summarise_background'] = True

    print('Explainer fit')
    explainer.fit(data_train, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
    print('Explain')
    explanation = explainer.explain(data_train)
    expected_value = explanation.expected_value.squeeze()
    shap_values = explanation.shap_values[0]
    outputs_train = explanation.raw['raw_prediction']
    shap_values_g = explanation.raw['importances']['0']

    print('Global SHAP')
    gshap = GlobalKernelShap(data_train, shap_values, expected_value)

    gshap_preds, gshap_vals = gshap.predict(data_train, return_shap_values=True)
    print('RMSE global error train', metrics.rmse(outputs_train, gshap_preds))

    gshap_preds, gshap_vals = gshap.predict(data_test, return_shap_values=True)
    outputs_test = model(data_test)
    print('RMSE global error test', metrics.rmse(outputs_test, gshap_preds))

    contribs_gshap = dict(zip(model.symbols, gshap_vals.T))

    return contribs_gshap


def ebm_explain(model, data_train, data_test):
    y_train = model(data_train)
    y_test = model(data_test)

    ebm = ExplainableBoostingRegressor(
        feature_names=model.symbol_names,
        feature_types=['continuous'] * model.n_features,  # TODO
        interactions=0,  # TODO
    )
    ebm.fit(data_train, y_train)

    ebm_preds_train = ebm.predict(data_train)
    ebm_preds_test = ebm.predict(data_test)

    # TODO: evaluate global explanations using scores (rankings of feature
    #  contributions, whether feature/interaction present, etc.) - coarser
    #  metrics
    # ebm_expl = ebm.explain_global('EBM')
    ebm_expl = ebm.explain_local(data_test)

    # Expected value
    # intercept = ebm.intercept_

    ebm_contribs = defaultdict(lambda: [])
    for i in range(len(data_test)):
        # Additive contributions for sample i
        expl_i = ebm_expl.data(i)
        # TODO make sure this looks good for interactions
        feat_names = expl_i['names']
        feat_scores = expl_i['scores']
        feat_contribs = dict(zip(feat_names, feat_scores))
        # TODO this only does main effects right now...
        for symbol in model.symbols:
            ebm_contribs[symbol].append(feat_contribs[symbol.name])

    print('RMSE global error train', metrics.rmse(y_train, ebm_preds_train))
    print('RMSE global error test', metrics.rmse(y_test, ebm_preds_test))

    return dict(ebm_contribs)  # don't return defaultdict


def evaluate_explainers(debug=False):
    # TODO: gSHAP-linear, gSHAP-spline, etc.

    # Make model
    print('Model')
    # model = tsang_iclr18_models('f9')
    # TODO: this is temporary as I still haven't figured out the whole
    #  attributing interaction effects as main effects thing...
    import sympy as sym
    from sympy import stats
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
    data_sym_stats = {
        s: stats.Uniform(s.name, 0, +1)
        for s in model.symbols
    }

    # Split
    train_split_pct = 2 / 3
    split_idx = int(train_split_pct * len(data))
    data_train = data[:split_idx]
    data_test = data[split_idx:]

    # True contributions, mean-adjusted
    contribs_true, effects_true = model.feature_contributions(
        data_test, return_effects=True)
    # Mean-adjusted output values for fair comparison
    contribs_true_adj = contribs_true.copy()
    for i, symbol in enumerate(model.symbols):
        # TODO: only main effects considered atm
        contrib_sym_true = contribs_true_adj[symbol]
        contrib_eff_true = effects_true[symbol]
        # SHAP values are relative to expected output value:
        expected_sym_val = float(stats.E(
            contrib_eff_true.subs({symbol: data_sym_stats[symbol]})
        ))
        contrib_sym_true -= expected_sym_val
        contribs_true_adj[symbol] = contrib_sym_true

    # Explainers
    # TODO: other models
    #  - logistic regression
    #  - xgboost
    #  - Neural Additive Models: Interpretable Machine Learning with Neural Nets
    #  -
    contribs_expls = {}
    for explainer_name, explainer_func in (
            ('gSHAP', gshap_explain),
            ('EBM', ebm_explain),
    ):
        print('Gathering Explanations from', explainer_name)
        contribs_expl = explainer_func(model, data_train, data_test)
        contribs_expls[explainer_name] = contribs_expl

    print()
    eval_and_plot(
        data=data_test,
        contribs_true=contribs_true_adj,
        effects_true=effects_true,
        contribs_explainers=contribs_expls,
        symbols=model.symbols,  # TODO: incorporate interaction effects
    )


if __name__ == '__main__':
    def main():
        global PROFILE

        import argparse

        parser = argparse.ArgumentParser(
            description=''
        )
        parser.add_argument('--debug', '-D', action='store_true',
                            help='Use fewer samples to make sure things work.')
        parser.add_argument('--profile', '-P', action='store_true',
                            help='Run code profiling.')
        args = parser.parse_args()

        PROFILE = args.profile

        evaluate_explainers(debug=args.debug)


    main()
