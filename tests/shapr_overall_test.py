"""
shapr_overall_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import faulthandler

faulthandler.enable()

if MWE := True:

    from collections import OrderedDict

    import numpy as np
    import pandas as pd

    import rpy2.rinterface as ri
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import globalenv
    from rpy2.robjects import ListVector
    from rpy2.robjects import StrVector
    from rpy2.robjects import NULL
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    pandas2ri.activate()

    shapr_lib = importr('shapr')

    X = np.arange(3 * 100).reshape(-1, 3).astype(float)
    feature_names = [*map(str, range(X.shape[-1]))]


    def model(X):
        return X.sum(axis=1)[..., None]


    @ri.rternalize
    def get_model_specs_compat(x):  # noqa
        labels = StrVector(feature_names)
        classes = StrVector(['numeric'] * len(feature_names))
        classes.names = labels

        feature_list = ListVector({
            'labels': labels,
            'classes': classes,
            'factor_levels': ListVector(OrderedDict(zip(
                feature_names, [NULL] * len(feature_names)
            ))),
        })
        return feature_list


    globalenv['get_model_specs.mwe'] = get_model_specs_compat


    @ri.rternalize
    def predict_model_compat(x, newdata):  # noqa
        if isinstance(newdata, ri.ListSexpVector):
            newdata = np.asarray(newdata).T
        return numpy2ri.py2rpy(model(newdata))


    predict_model_func_name = f'predict_model.mwe'
    globalenv[predict_model_func_name] = predict_model_compat


    @ri.rternalize
    def inner(_):
        raise RuntimeError('This method should never have been called.')


    inner.rclass = StrVector(('mwe', 'function'))
    wrapped_model = inner

    # Start fit
    y = model(X)

    X_df = pd.DataFrame(data=X, columns=feature_names)
    _explainer = shapr_lib.shapr(X_df, wrapped_model)

    # expected value - the prediction value for unseen data, typically
    #  equal to the mean of the response.
    prediction_zero_ = np.mean(y, axis=0)
    # end fit

    # shapr wants a dataframe...
    X_df = pd.DataFrame(data=X, columns=feature_names)

    explanation = shapr_lib.explain(
        X_df,
        explainer=_explainer,
        approach='empirical',
        prediction_zero=prediction_zero_,
    )

    print('explanation')
    print(explanation)
    expl_dict = dict(explanation.items())
    print('expl_dict')
    print(expl_dict)
    print('explanation.dt')
    print(expl_dict['dt'])

else:

    import numpy as np
    import pandas as pd
    from posthoceval.explainers.local.shapr import SHAPRExplainer
    from posthoceval.model_generation import AdditiveModel

    # import gc

    # gc.disable()
    model = AdditiveModel.from_expr('x1 + x2 - 2 * x3')
    explainer = SHAPRExplainer(model=model)
    data = np.random.rand(10, 3)
    explainer.fit(data)
    explainer.feature_contributions(data, as_dict=True)
