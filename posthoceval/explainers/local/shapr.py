"""
shapr.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel

__all__ = ['SHAPRExplainer']

logger = logging.getLogger(__name__)


class SHAPRExplainer(BaseExplainer):
    """Explain the output of machine learning models with more accurately
    estimated Shapley values"""

    __r_class_name_model__ = 'pymodel'

    def __new__(cls, *args, **kwargs):
        shapr = None
        try:
            import rpy2

            rpy2_available = True
        except ImportError:
            rpy2 = None
            rpy2_available = False
        else:
            try:
                from rpy2.robjects.packages import importr

                # R imports
                shapr = importr('shapr')
            except rpy2.robjects.packages.PackageNotInstalledError:
                logger.error(
                    'You must install the shapr R package if you want to '
                    'use the explainer: `install.packages("shapr")`'
                )

        if rpy2 is None or shapr is None:
            raise EnvironmentError(
                f'{cls.__name__} is not available because '
                f'{"R package shapr" if rpy2_available else "rpy2"} '
                f'is not installed'
            )
        obj = super().__new__(cls)
        obj.shapr_lib = shapr
        return obj

    def __init__(self,
                 model: AdditiveModel,
                 seed=None,
                 task: str = 'regression',
                 verbose=True,
                 **kwargs):
        # keep imports to inside class to avoid some multiprocessing issues
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import pandas2ri

        # auto-convert numpy arrays to R data structures
        numpy2ri.activate()
        # auto-convert pandas dataframes to R data structures
        pandas2ri.activate()

        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self.prediction_zero_ = None

        # handle kwargs
        self.approach = kwargs.pop('approach', 'empirical')

        if kwargs:
            raise ValueError(f'Unexpected keyword arguments: {kwargs}')

        # handle shapr compat things
        # TODO: install every instantiation or just after first import?
        # TODO: do not recreate model every time either...
        self._install_shapr_r_compat()

    def _install_shapr_r_compat(self):
        import rpy2.rinterface as ri
        from rpy2.robjects import globalenv
        from rpy2.robjects import ListVector
        from rpy2.robjects import StrVector
        from rpy2.robjects import NULL

        @ri.rternalize
        def get_model_specs_compat(x):  # noqa
            """
            labels: character vector with the feature names to compute Shapley
                values for
            classes: a named character vector with the labels as names and the
                class type as elements
            factor_levels: a named list with the labels as names and character
                vectors with the factor levels as elements (NULL if the feature
                is not a factor)

            feature_list <- list()
            feature_list$labels <- labels(x$Terms)
            m <- length(feature_list$labels)
            feature_list$classes <- attr(x$Terms, "dataClasses")[-1]
            feature_list$factor_levels <- setNames(vector("list", m),
                                                   feature_list$labels)
            # the model object don't contain factor levels info
            feature_list$factor_levels[feature_list$classes == "factor"] <- NA
            return(feature_list)
            """
            # TODO: categorical data...
            # classes can realistically be "numeric" or "factor" here for
            #  continuous/categorical, respectively. factor_levels will need
            #  to be set properly as well
            feature_list = ListVector({
                'labels': StrVector(self.model.symbol_names),
                'classes': ListVector(OrderedDict(zip(
                    self.model.symbol_names,
                    ['numeric'] * self.model.n_features
                ))),
                'factor_levels': ListVector(OrderedDict(zip(
                    self.model.symbol_names,
                    [NULL] * self.model.n_features
                ))),
            })
            return feature_list

        model_spec_func_name = f'get_model_specs.{self.__r_class_name_model__}'
        globalenv[model_spec_func_name] = get_model_specs_compat

        @ri.rternalize
        def predict_model_compat(x, newdata):  # noqa
            """
            TODO WIP
            """
            print(newdata)
            return self.model(newdata)

        predict_model_func_name = f'predict_model.{self.__r_class_name_model__}'
        globalenv[predict_model_func_name] = predict_model_compat

    @property
    def _wrapped_model(self):
        import rpy2.rinterface as ri
        from rpy2.robjects import StrVector

        @ri.rternalize
        def inner(x):  # TODO...
            raise RuntimeError('This method should never have been called.')
            return self.model(x)

        inner.rclass = StrVector((self.__r_class_name_model__,
                                  'function'))

        return inner

    def fit(self, X, y=None):
        if y is None:
            y = self.model(X)

        self._explainer = self.shapr_lib.shapr(
            X, self._wrapped_model
        )

        # expected value - the prediction value for unseen data, typically
        #  equal to the mean of the response.
        self.prediction_zero_ = np.mean(y, axis=0)

    def predict(self, X):
        pass

    def feature_contributions(self, X, return_y=False, as_dict=False):
        """
        Arguments:

        x: A matrix or data.frame. Contains the the features, whose
                  predictions ought to be explained (test data).

        explainer: An ‘explainer’ object to use for explaining the
                  observations. See ‘shapr’.

        approach: Character vector of length ‘1’ or ‘n_features’.
                  ‘n_features’ equals the total number of features in the
                  model. All elements should either be ‘"gaussian"’,
                  ‘"copula"’, ‘"empirical"’, or ‘"ctree"’. See details for
                  more information.

        prediction_zero: Numeric. The prediction value for unseen data,
                  typically equal to the mean of the response.

             ...: Additional arguments passed to ‘prepare_data’

            type: Character. Should be equal to either ‘"independence"’,
                  ‘"fixed_sigma"’, ‘"AICc_each_k"’ or ‘"AICc_full"’.

        fixed_sigma_vec: Numeric. Represents the kernel bandwidth. Note
                  that this argument is only applicable when ‘approach =
                  "empirical"’, and ‘type = "fixed_sigma"’

        n_samples_aicc: Positive integer. Number of samples to consider in
                  AICc optimization. Note that this argument is only
                  applicable when ‘approach = "empirical"’, and ‘type’ is
                  either equal to ‘"AICc_each_k"’ or ‘"AICc_full"’

        eval_max_aicc: Positive integer. Maximum number of iterations when
                  optimizing the AICc. Note that this argument is only
                  applicable when ‘approach = "empirical"’, and ‘type’ is
                  either equal to ‘"AICc_each_k"’ or ‘"AICc_full"’

        start_aicc: Numeric. Start value of ‘sigma’ when optimizing the
                  AICc. Note that this argument is only applicable when
                  ‘approach = "empirical"’, and ‘type’ is either equal to
                  ‘"AICc_each_k"’ or ‘"AICc_full"’

        w_threshold: Positive integer between 0 and 1.

              mu: Numeric vector. (Optional) Containing the mean of the
                  data generating distribution. If ‘NULL’ the expected
                  values are estimated from the data. Note that this is
                  only used when ‘approach = "gaussian"’.

         cov_mat: Numeric matrix. (Optional) Containing the covariance
                  matrix of the data generating distribution. ‘NULL’ means
                  it is estimated from the data if needed (in the Gaussian
                  approach).

        mincriterion: Numeric value or vector where length of vector is the
                  number of features in model. Value is equal to 1 - alpha
                  where alpha is the nominal level of the conditional
                  independence tests. If it is a vector, this indicates
                  which mincriterion to use when conditioning on various
                  numbers of features.

        minsplit: Numeric value. Equal to the value that the sum of the
                  left and right daughter nodes need to exceed.

        minbucket: Numeric value. Equal to the minimum sum of weights in a
                  terminal node.

          sample: Boolean. If TRUE, then the method always samples
                  ‘n_samples’ from the leaf (with replacement). If FALSE
                  and the number of obs in the leaf is less than
                  ‘n_samples’, the method will take all observations in the
                  leaf. If FALSE and the number of obs in the leaf is more
                  than ‘n_samples’, the method will sample ‘n_samples’
                  (with replacement). This means that there will always be
                  sampling in the leaf unless ‘sample’ = FALSE AND the
                  number of obs in the node is less than ‘n_samples’.

        Details:

             The most important thing to notice is that ‘shapr’ has
             implemented four different approaches for estimating the
             conditional distributions of the data, namely ‘"empirical"’,
             ‘"gaussian"’, ‘"copula"’ and ‘"ctree"’.

             In addition, the user also has the option of combining the
             four approaches. E.g. if you're in a situation where you have
             trained a model the consists of 10 features, and you'd like to
             use the ‘"gaussian"’ approach when you condition on a single
             feature, the ‘"empirical"’ approach if you condition on 2-5
             features, and ‘"copula"’ version if you condition on more than
             5 features this can be done by simply passing
             ‘approach = c("gaussian", rep("empirical", 4),
              rep("copula", 5))’. If ‘"approach[i]" = "gaussian"’ it means
              that you'd like to use the ‘"gaussian"’ approach when
              conditioning on ‘i’ features.
        """
        # TODO: massive WIP

        # shapr wants a dataframe...
        X_df = pd.DataFrame(data=X, columns=self.model.symbol_names)

        return self.shapr_lib.explain(
            X_df,
            explainer=self._explainer,
            approach=self.approach,
            prediction_zero=self.prediction_zero_,
        ).dt
