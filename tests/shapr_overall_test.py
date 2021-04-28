"""
shapr_overall_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import faulthandler

faulthandler.enable()

# if MWE := False:
if False:

    if False:
        # ok this is issue https://github.com/rpy2/rpy2/issues/563
        print('start i kill it all')
        import sys
        import rpy2.rinterface as ri
        from rpy2.robjects import globalenv
        from rpy2 import __version__

        print(f'Python version={sys.version}')
        print(f'rpy2 version={__version__}')


        @ri.rternalize
        def test_ok():
            print('I am an okay test')
            return 0


        globalenv['test_ok'] = test_ok
        globalenv['test_ok']()


        def give_me_test():
            @ri.rternalize
            def test_bad():
                print('I am a bad test')
                return 1

            return test_bad


        globalenv['test_bad'] = give_me_test()
        globalenv['test_bad']()

        print('end i kill it all')

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


    class ShaprDummy:
        def _install_shapr_r_compat(self):
            @ri.rternalize
            def get_model_specs_compat(x):
                feature_names = [*x.do_slot('feature_names')]
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
                return x(newdata)

            predict_model_func_name = f'predict_model.mwe'
            globalenv[predict_model_func_name] = predict_model_compat


    obj = ShaprDummy()
    obj._install_shapr_r_compat()


    def model(X):
        return X.sum(axis=1)[..., None]


    @ri.rternalize
    def inner(X):
        if isinstance(X, ri.ListSexpVector):
            X = np.asarray(X).T
        return numpy2ri.py2rpy(model(X))


    inner.rclass = StrVector(('mwe', 'function'))
    inner.do_slot_assign('feature_names', StrVector(feature_names))
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
    expl_dict = dict(explanation.items())

    # print('explanation')
    # print(explanation)
    # print('expl_dict')
    # print(expl_dict)
    print('expl_dict.keys()')
    print(expl_dict.keys())
    print('explanation.dt')
    print(expl_dict['dt'])

    # https://stackoverflow.com/questions/5199334/clearing-memory-used-by-rpy2
    import gc

    gc.collect()

else:

    import numpy as np
    import pandas as pd
    from posthoceval.explainers.local.shapr import SHAPRExplainer
    from posthoceval.explainers.local.shap import KernelSHAPExplainer
    from posthoceval.models.synthetic import SyntheticModel

    # import gc

    # gc.disable()
    data = np.random.rand(20, 3)

    model = SyntheticModel.from_expr('x1 + x2 - 2 * x3')

    for explainer_cls in [KernelSHAPExplainer, SHAPRExplainer]:
        explainer = explainer_cls(model=model)
        explainer.fit(data)
        contribs = explainer.feature_contributions(data, as_dict=False)

        print(explainer_cls.__name__)
        print(contribs)
        print()

    print('True')
    print(model.feature_contributions(data))
