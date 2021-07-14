"""
linear.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold

from interpret.glassbox import LinearRegression

import posthoceval.models.model
from posthoceval import metrics


def linear_explain(model, data_train, data_test):
    """Very much not part of any notion of a public API."""
    expected_value = np.mean(data_train)

    y_train = model(data_train)
    y_test = model(data_test)

    # Common linear regressor parameters
    lr_params = dict(
        max_iter=10_000,
        feature_names=posthoceval.models.model.gen_symbol_names,
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
