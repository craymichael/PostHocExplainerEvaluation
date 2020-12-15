"""
ebm.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
from collections import defaultdict

from interpret.glassbox import ExplainableBoostingRegressor

from posthoceval import metrics


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