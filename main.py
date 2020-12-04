from math import sqrt
from math import ceil

import cProfile
from functools import wraps
from collections import defaultdict

from itertools import chain

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import seaborn as sns

from alibi.explainers import KernelShap
from alibi.explainers.shap_wrappers import KERNEL_SHAP_BACKGROUND_THRESHOLD

from interpret.glassbox import ExplainableBoostingRegressor
from interpret.glassbox import LinearRegression
# from interpret.blackbox import PartialDependence  TODO

# TODO: commented out in source???? why
# from interpret.blackbox import PermutationImportance

from sklearn.model_selection import KFold

from joblib import cpu_count

from posthoceval.global_shap import GlobalKernelShap
from posthoceval.model_generation import AdditiveModel
from posthoceval.model_generation import tsang_iclr18_models
from posthoceval.evaluate import symbolic_evaluate_func
from posthoceval import metrics

sns.set(
    context='talk'
)

PROFILE = False


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


def _standardize_effects(effects):
    # Squeeze effects tuples
    if isinstance(effects, dict):
        d = effects.copy()
        for e, v in d.items():
            if isinstance(e, tuple) and len(e) == 1:
                e_squeeze = e[0]
                d[e_squeeze] = d.pop(e)
        return d
    else:  # assume iterable
        return tuple(e if (not isinstance(e, tuple) or len(e) > 1) else e[0]
                     for e in effects)


def eval_and_plot(data, contribs_true, effects_true, contribs_explainers,
                  symbols=None):
    contribs_true = _standardize_effects(contribs_true)
    effects_true = _standardize_effects(effects_true)
    if symbols is None:
        # By default use all provided symbols (symbols can for example be a
        # subset, like just main effects)
        symbols = contribs_true.keys()
    symbols = _standardize_effects(symbols)

    if not isinstance(contribs_explainers, dict):
        # Assume iterable otherwise...
        contribs_explainers = {'Explainer': contribs_explainers}

    iv = 'Value'  # feature value
    iv_name = 'Feature Effect'
    dv = 'Contribution'
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

    # Compute number of interaction terms
    n_interactions = sum(isinstance(symbol, tuple) for symbol in symbols)
    cols_3d = rows_3d = 0
    idx_3d = 1
    if n_interactions:
        # 1. Matching interactions
        # 2. All terms with variables in interaction
        n_plots = n_interactions * 2
        # rows_3d = round(sqrt(n_plots))
        rows_3d = 2
        cols_3d = int(ceil(n_plots / rows_3d))

    fig_3d = plt.figure() if n_interactions > 0 else None

    for symbol in symbols:
        contrib_sym_true = contribs_true[symbol]
        contrib_eff_true = effects_true[symbol]

        if isinstance(symbol, tuple):
            # Interactions effects

            do_plot = True
            if len(symbol) != 2:
                print('Skipping interaction effect {}. Effects of only 1 or 2 '
                      'variables can be visualized.'.format(contrib_eff_true))
                do_plot = False

            if do_plot:
                sym0, sym1 = symbol
                data_feat_sym0 = data[sym0]
                data_feat_sym1 = data[sym1]

                ax = fig_3d.add_subplot(rows_3d, cols_3d, idx_3d,
                                        projection='3d')

                ax_title = f'[{sym0.name},{sym1.name}]: {contrib_eff_true}'
                ax.set_title(ax_title)
                ax.set_xlabel(sym0.name)
                ax.set_ylabel(sym1.name)
                ax.set_zlabel(dv)

                # True contribution
                ax.scatter(data_feat_sym0,
                           data_feat_sym1,
                           contrib_sym_true,
                           # c='skyblue',
                           s=10,
                           label='True')

                idx_3d += 1

            # Explainer contributions
            for (explainer_name,
                 contribs_explainer) in contribs_explainers.items():
                # True contribution or all zeros
                # TODO: show this and/or contributions by all effects including
                #  the true symbols?
                contrib_sym_explainer = contribs_explainer.get(
                    symbol, np.zeros_like(contrib_sym_true))

                err = metrics.rmse(contrib_sym_true, contrib_sym_explainer)
                errs[explainer_name].append(err)
                explainer_prefix = explainer_str.format(explainer_name)
                feature_prefix = feature_str.format(
                    ','.join(s.name for s in symbol))
                print(explainer_prefix + ' ' + feature_prefix +
                      f' feature shape RMSE: {err:.6f}')

                if do_plot:
                    ax.scatter(data_feat_sym0,
                               data_feat_sym1,
                               contrib_sym_explainer,
                               # c='skyblue',
                               s=10,
                               label=explainer_name)

            if idx_3d == 2:
                fig_3d.legend()

            symbol_set = {*symbol}

            relevant_symbols = []
            for s in symbols:
                s_set = set(s) if isinstance(s, tuple) else {s}
                if not (s_set - symbol_set):  # empty set
                    # This means all symbols are contained in `symbol`
                    relevant_symbols.append(s)
            if not relevant_symbols:
                class ZachIsBrainDeadError(Exception):
                    pass

                raise ZachIsBrainDeadError('The last brain cell has '
                                           'finally deceased')

            contrib_sym_true_all = sum(contribs_true[s]
                                       for s in relevant_symbols)
            contrib_eff_true_all = sum(effects_true[s]
                                       for s in relevant_symbols)

            if do_plot:
                # TODO: this ignores explainers that attribute relevant symbols
                #  in conjunction with irrelevant symbols with respect to the
                #  interaction effect in question
                ax = fig_3d.add_subplot(rows_3d, cols_3d, idx_3d,
                                        projection='3d')

                ax_title = f'[{sym0.name},{sym1.name}]: {contrib_eff_true_all}'
                ax.set_title(ax_title)
                ax.set_xlabel(sym0.name)
                ax.set_ylabel(sym1.name)
                ax.set_zlabel(dv)

                # True contribution
                ax.scatter(data_feat_sym0,
                           data_feat_sym1,
                           contrib_sym_true_all,
                           # c='skyblue',
                           s=10,
                           label='True')

                # Explainer contributions
                for (explainer_name,
                     contribs_explainer) in contribs_explainers.items():
                    # True contribution or all zeros
                    contrib_sym_explainer = sum(
                        np.asarray(
                            contribs_explainer.get(
                                s, np.zeros_like(contrib_sym_true))
                        ) for s in relevant_symbols
                    )
                    ax.scatter(data_feat_sym0,
                               data_feat_sym1,
                               contrib_sym_explainer,
                               # c='skyblue',
                               s=10,
                               label=explainer_name)
                idx_3d += 1

            for (explainer_name,
                 contribs_explainer) in contribs_explainers.items():
                # True contribution or all zeros
                contrib_sym_explainer = sum(
                    np.asarray(
                        contribs_explainer.get(
                            s, np.zeros_like(contrib_sym_true))
                    ) for s in relevant_symbols
                )
                # TODO: standardize...
                print(contrib_eff_true_all,
                      explainer_name,
                      metrics.rmse(contrib_sym_true_all,
                                   contrib_sym_explainer))

        else:
            # Main effects

            data_feat_sym = data[symbol]

            # ax_title = symbol.name
            ax_title = '[' + symbol.name + ']: ' + str(contrib_eff_true)

            feat_name = [ax_title] * n_samples
            plot_cols.append((feat_name,
                              data_feat_sym,
                              ['True'] * n_samples,
                              contrib_sym_true))

            for (explainer_name,
                 contribs_explainer) in contribs_explainers.items():
                contrib_sym_explainer = contribs_explainer[symbol]
                plot_cols.append((feat_name,
                                  data_feat_sym,
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

    g = sns.relplot(data=df, x=iv, y=dv, col=iv_name, hue=typ, style=typ,
                    kind='line', col_wrap=round(sqrt(len(symbols))),
                    facet_kws={'sharey': False, 'sharex': False})
    g.legend.set_title(None)
    g.fig.tight_layout()
    for ax in g.axes:  # de-verbosify subplot titles
        ax.set_title(ax.get_title().split('= ', 1)[-1])

    plt.show()


@profile
def gshap_explain(model, data_train, data_test,
                  n_background_samples=300):
    # TODO: gSHAP-linear, gSHAP-spline, etc.

    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
        distributed_opts={
            # https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/
            'n_cpus': cpu_count(),
            # If batch_size set to `None`, an input array is split in (roughly)
            # equal parts and distributed across the available CPUs
            'batch_size': None,
        }
    )
    fit_kwargs = {}
    if n_background_samples < len(data_train):
        print('Intending to summarize background data as n_samples > '
              '{}'.format(n_background_samples))
        fit_kwargs['summarise_background'] = True
        fit_kwargs['n_background_samples'] = n_background_samples

    print('Explainer fit')
    explainer.fit(data_train, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
    print('Explain')
    explanation = explainer.explain(data_train, silent=True)
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

    # TODO: mean absolute percentage error, consolidate metric eval code...
    print('RMSE global error train', metrics.rmse(y_train, ebm_preds_train))
    print('RMSE global error test', metrics.rmse(y_test, ebm_preds_test))

    return dict(ebm_contribs)  # don't return defaultdict


def linear_explain(model, data_train, data_test):  # TODO: dupe code...
    expected_value = np.mean(data_train)

    y_train = model(data_train)
    y_test = model(data_test)

    # Common linear regressor parameters
    lr_params = dict(
        max_iter=10_000,
        feature_names=model.symbol_names,
        feature_types=['continuous'] * model.n_features,  # TODO
        fit_intercept=False,  # TODO(easier to deal with...)
    )

    # Center expected values in training/prediction
    y_train -= expected_value
    y_test -= expected_value

    alphas = (1e-6, 1e-5, 1e-4, 1e-3, 1e-1, 1., 1e1)
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    cv_errs = [0.] * len(alphas)
    for cv_train_idx, cv_val_idx in kf.split(data_train):
        X_cv_train = data_train[cv_train_idx]
        y_cv_train = y_train[cv_train_idx]
        X_cv_val = data_train[cv_val_idx]
        y_cv_val = y_train[cv_val_idx]

        for i, alpha in enumerate(alphas):
            lr = LinearRegression(
                alpha=alpha,
                **lr_params,
            )
            lr.fit(X_cv_train, y_cv_train)
            cv_errs[i] += metrics.rmse(y_cv_val, lr.predict(X_cv_val))
    cv_errs = [cv_err / n_splits for cv_err in cv_errs]
    # noinspection PyTypeChecker
    alpha_best = alphas[np.argmin(cv_errs)]

    print(f'Best Alpha ({n_splits}-fold CV)', alpha_best)

    # Use best alpha according to RMSE-scored CV
    lr = LinearRegression(
        alpha=alpha_best,
        **lr_params,
    )
    lr.fit(data_train, y_train)

    lr_preds_train = lr.predict(data_train)
    lr_preds_test = lr.predict(data_test)

    # TODO: evaluate global explanations using scores (rankings of feature
    #  contributions, whether feature/interaction present, etc.) - coarser
    #  metrics
    # lr_expl = lr.explain_global('Linear')
    lr_expl = lr.explain_local(data_test)

    # TODO intercept in explanations, not class attr - this is
    #  bias term, not expected value
    # intercept = lr.intercept_

    lr_contribs = defaultdict(lambda: [])
    for i in range(len(data_test)):
        # Additive contributions for sample i
        expl_i = lr_expl.data(i)

        feat_names = expl_i['names']
        feat_scores = expl_i['scores']
        feat_contribs = dict(zip(feat_names, feat_scores))

        for symbol in model.symbols:
            lr_contribs[symbol].append(feat_contribs[symbol.name])

    print('RMSE global error train', metrics.rmse(y_train, lr_preds_train))
    print('RMSE global error test', metrics.rmse(y_test, lr_preds_test))

    return dict(lr_contribs)  # don't return defaultdict


def evaluate_explainers(debug=False):
    generate_with_correlation = False
    dummy_model_with_interactions = True

    # Make model
    print('Model')
    # model = tsang_iclr18_models('f9')
    # TODO: I still haven't figured out the whole attributing interaction
    #  effects as main effects thing...
    import sympy as sym
    from sympy import stats
    symbols = sym.symbols('x1:11')
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols

    expr = (sym.log(abs(x1)) * x1 - x2 + x3 ** 3 - x4 - 2 * x5 + sym.sin(x6) +
            sym.cos(x7) + 1 / (1 + x8))
    if dummy_model_with_interactions:
        expr += x1 * x2

    model = AdditiveModel.from_expr(expr, symbols)

    if dummy_model_with_interactions:  # TODO
        model = tsang_iclr18_models('f1')

    model.pprint()

    # Make data
    print('Data')
    # n_samples = 30_000 if not debug else 20
    n_samples = 1_000 if not debug else 20  # TODO
    # TODO: better data ranges based on continuity of function
    # TODO: use valid_variable_domains when properly+cleanly integrated
    # data = np.random.uniform(-1, +1, size=(n_samples, model.n_features))
    data = np.random.uniform(0, +1, size=(n_samples, model.n_features))
    data_sym_stats = {
        s: stats.Uniform(s.name, 0, +1)
        for s in model.symbols
    }
    if generate_with_correlation:
        data[:, 2] = data[:, 0] + 2 * data[:, 1]  # TODO...
        data[:, 4] = data[:, 2] * data[:, 3]  # TODO...
        data[:, 9] = data[:, 5] + data[:, 6]  # TODO...
        data_sym_stats[model.symbols[2]] = \
            data_sym_stats[model.symbols[0]] + \
            2 * data_sym_stats[model.symbols[1]]  # TODO...
        data_sym_stats[model.symbols[4]] = \
            data_sym_stats[model.symbols[2]] * \
            data_sym_stats[model.symbols[3]]  # TODO...
        data_sym_stats[model.symbols[9]] = \
            data_sym_stats[model.symbols[5]] + \
            data_sym_stats[model.symbols[6]]  # TODO...

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
    #  - xgboost
    #  - Neural Additive Models: Interpretable Machine Learning with Neural Nets
    #  -
    contribs_expls = {}
    for explainer_name, explainer_func in (
            ('gSHAP', gshap_explain),
            ('EBM', ebm_explain),
            ('Linear', linear_explain),
    ):
        print('Gathering Explanations from', explainer_name)
        contribs_expl = explainer_func(model, data_train, data_test)
        contribs_expls[explainer_name] = contribs_expl

    data_test_df = pd.DataFrame(
        data=data_test,
        columns=model.symbols
    )

    print()
    eval_and_plot(
        data=data_test_df,
        contribs_true=contribs_true_adj,
        effects_true=effects_true,
        contribs_explainers=contribs_expls,
        # symbols=model.symbols,
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
