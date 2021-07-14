"""
shapr.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael

Referenced a lot:
https://github.com/NorskRegnesentral/shapr/blob/master/tests/testthat/manual_test_scripts/test_custom_models.R
"""
from typing import Optional

import logging
import gc

from collections import OrderedDict
from multiprocessing import Lock

import numpy as np
import pandas as pd

from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel

__all__ = ['SHAPRExplainer']

logger = logging.getLogger(__name__)

# https://github.com/rpy2/rpy2/issues/563
_ref_get_model_specs_compat = None
_ref_predict_model_compat = None
# lock
_shapr_install_lock = Lock()
# there can be multiple of these so we update globals AND this for later
#  deletion
_ref_make_wrapped_model_r_inner_dict = {}
# lock
_make_wrapped_model_r_inner_lock = Lock()


class SHAPRExplainer(BaseExplainer):
    """
    SHAPR Explainer

    Explain the output of machine learning models with more accurately
    estimated Shapley values

    shapr/R/features.R
    ```R
      if (m > 13 & is.null(n_combinations)) {
        stop(
          paste0(
            "Due to computational complexity, we recommend setting n_combinations = 10 000\n",
            "if the number of features is larger than 13. Note that you can force the use of the exact\n",
            "method (i.e. n_combinations = NULL) by setting n_combinations equal to 2^m,\n",
            "where m is the number of features."
          )
        )
      }

      # Not supported for m > 30
      if (m > 30) {
        stop("Currently we are not supporting cases where the number of features is greater than 30.")
      }
    ```
    """

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
        """
        SHAPR explainer. rpy2 used to provide interface to R implementation
        through Python. You must have R and the R package shapr installed
        in order for this class to work properly. Don't worry, it'll yell at
        you if you don't. See the README for this library for more details.


        :param model: the model to explain
        :param seed: the RNG seed for reproducibility
        :param task: the task, either "classification" or "regression"
        :param verbose: print more messages
        :param approach: the shapr approach to use (assumption of conditional
            distribution, see SHAPR docs). Default: "empirical"
        """
        # keep imports to inside class to avoid some multiprocessing issues
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import pandas2ri

        # auto-convert numpy arrays to R data structures
        numpy2ri.activate()
        # auto-convert pandas dataframes to R data structures
        pandas2ri.activate()

        super().__init__(
            model=model,
            tabular=True,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self._x_fit_size = None

        self.prediction_zero_ = None

        # handle kwargs
        self.approach = kwargs.pop('approach', 'empirical')

        if kwargs:
            raise ValueError(f'Unexpected keyword arguments: {kwargs}')

        self.__wrapped_model = {}
        # handle shapr compat things
        with _shapr_install_lock:
            self._install_shapr_r_compat()

    def __del__(self):
        with _make_wrapped_model_r_inner_lock:
            for model_key in self.__wrapped_model:
                model_str = self._wrapped_model_ref_str + f'_{model_key}'
                # decrement count by 1
                _ref_make_wrapped_model_r_inner_dict[model_str] -= 1
                # if no more references then also delete the reference to the
                #  model
                if _ref_make_wrapped_model_r_inner_dict[model_str] == 0:
                    del globals()[model_str]
            gc.collect()

    @property
    def _wrapped_model_ref_str(self):
        return (f'_ref_make_wrapped_model_r_inner_{id(self.model)}_'
                f'{self.__r_class_name_model__}')

    def _make_wrapped_model_r(self, key):
        import rpy2.rinterface as ri
        from rpy2.robjects import StrVector
        from rpy2.robjects import numpy2ri

        @ri.rternalize
        def inner(X):
            if isinstance(X, ri.ListSexpVector):
                X = np.asarray(X).T
            if key is None:
                out = self.model(X)
            else:
                # assuming key is an int
                out = self.model(X)[:, key]
            return numpy2ri.py2rpy(out)

        inner.rclass = StrVector((self.__r_class_name_model__, 'function'))
        inner.do_slot_assign('feature_names',
                             StrVector(self.model.symbol_names))

        with _make_wrapped_model_r_inner_lock:
            # add reference, increment counter for the model by 1
            model_str = self._wrapped_model_ref_str + f'_{key}'
            cur_count = _ref_make_wrapped_model_r_inner_dict.get(model_str, 0)
            _ref_make_wrapped_model_r_inner_dict[model_str] = cur_count + 1
            if cur_count == 0:
                globals()[model_str] = inner

        return inner

    def _install_shapr_r_compat(self):
        from rpy2.robjects import globalenv

        model_spec_func_name = f'get_model_specs.{self.__r_class_name_model__}'
        predict_model_func_name = f'predict_model.{self.__r_class_name_model__}'

        if (model_spec_func_name in globalenv and
                predict_model_func_name in globalenv):
            return

        import rpy2.rinterface as ri
        from rpy2.robjects import ListVector
        from rpy2.robjects import StrVector
        from rpy2.robjects import NULL

        @ri.rternalize
        def get_model_specs_compat(x):
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
            # TODO: categorical data...approach="ctree" supports this only
            # classes can realistically be "numeric" or "factor" here for
            #  continuous/categorical, respectively. factor_levels will need
            #  to be set properly as well
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

        @ri.rternalize
        def predict_model_compat(x, newdata):
            return x(newdata)

        # install
        globalenv[model_spec_func_name] = get_model_specs_compat
        globalenv[predict_model_func_name] = predict_model_compat

        # add global references
        # https://github.com/rpy2/rpy2/issues/563
        global _ref_get_model_specs_compat, _ref_predict_model_compat

        _ref_get_model_specs_compat = get_model_specs_compat
        _ref_predict_model_compat = predict_model_compat

    def _wrapped_model(self, key=None):
        __wrapped_model_from_key = self.__wrapped_model.get(key)
        if __wrapped_model_from_key is not None:
            return __wrapped_model_from_key

        __wrapped_model_from_key = self._make_wrapped_model_r(key)
        self.__wrapped_model[key] = __wrapped_model_from_key
        return __wrapped_model_from_key

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ):
        if y is None:
            y = self.model(X)

        if grouped_feature_names and any(not isinstance(feat, str)
                                         for feat in grouped_feature_names):
            # TODO: use groups...and ctree
            logger.warning('SHAPR does not current have proper support for '
                           'categorical variables!')

        X_df = pd.DataFrame(data=X, columns=self.model.symbol_names)
        self._x_fit_size = len(X_df)
        # From SHAPR:
        # Due to computational complexity, we recommend setting
        # n_combinations = 10 000 if the number of features is larger than 13.
        # Note that you can force the use of the exact method (i.e.
        # n_combinations = NULL) by setting n_combinations equal to 2^m,
        # where m is the number of features.
        kwargs = {}
        if self.model.n_features > 13:
            # TODO: parameterize in init
            kwargs['n_combinations'] = 10000

        if self.task == 'classification':
            assert len(y.shape) == 2, f'{y.shape} ndim != 2 for model output'
            n_classes = y.shape[1]

            self._explainer = [self.shapr_lib.shapr(
                X_df, self._wrapped_model(k), **kwargs
            ) for k in range(n_classes)]
        else:
            self._explainer = self.shapr_lib.shapr(
                X_df, self._wrapped_model(), **kwargs
            )

        # expected value - the prediction value for unseen data, typically
        #  equal to the mean of the response.
        self.prediction_zero_ = np.mean(y, axis=0)

    def predict(self, X):
        raise NotImplementedError

    def _call_explainer(self, X):
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
                  default: fixed_sigma

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
        # from rpy2.robjects import pandas2ri

        # shapr wants a dataframe...
        X_df = pd.DataFrame(data=X, columns=self.model.symbol_names)

        n_explain = len(X_df)
        # TODO: bad impl
        '''
        # need to deactivate pandas for this
        pandas2ri.deactivate()
        n_comb = len(self._explainer.rx2('X'))  # i.e. n_slices
        pandas2ri.activate()

        # https://github.com/RcppCore/RcppArmadillo/blob/bfce33f905f8b4b55c2184ea53a4ce8c63bc854c/inst/include/armadillo_bits/typedef_elem.hpp
        # Assume ARMA_64BIT_WORD undefined
        # #define ARMA_MAX_UWORD  0xffffffff
        ARMA_MAX_UWORD = 0xffffffff
        # Error raised if too large of a Cube in init (dist func in shapr)
        # (double(n_rows) * double(n_cols) * double(n_slices)) >
        #   double(ARMA_MAX_UWORD)
        cube_size = self._x_fit_size * n_explain * n_comb
        # 10 is a fudge factor
        # batch_size = max(
        #     int(n_explain / (10 * cube_size / ARMA_MAX_UWORD)), 1)
        '''
        batch_size = 1

        contribs = []
        for idx in range(0, n_explain, batch_size):
            X_df_batch = X_df.iloc[idx:idx + batch_size].reset_index(drop=True)

            if self.task == 'classification':
                contribs_batch = []
                # for each class's explainer
                for k, explainer in enumerate(self._explainer):
                    contribs_batch_k = self._explain_batch_one_class(
                        X_df_batch, explainer, self.prediction_zero_[k])
                    contribs_batch.append(contribs_batch_k)
            else:
                contribs_batch = self._explain_batch_one_class(
                    X_df_batch, self._explainer, self.prediction_zero_)
            contribs.append(contribs_batch)

        if self.task == 'classification':
            contribs = np.concatenate(contribs, axis=1)
            intercepts = np.expand_dims(self.prediction_zero_, 1)
            predictions = np.sum(contribs, axis=2) + intercepts
        else:
            contribs = np.concatenate(contribs, axis=0)
            predictions = np.sum(contribs, axis=1) + self.prediction_zero_

        return {'contribs': contribs,
                'intercepts': self.prediction_zero_,
                'predictions': predictions}

    def _explain_batch_one_class(self, X_df_batch, explainer, prediction_zero):
        explanation_batch = self.shapr_lib.explain(
            X_df_batch,
            explainer=explainer,
            approach=self.approach,
            prediction_zero=prediction_zero,
        )
        expl_dict_batch = dict(explanation_batch.items())

        # https://stackoverflow.com/questions/5199334/clearing-memory-used-by-rpy2
        gc.collect()

        expl_df_batch = expl_dict_batch['dt']
        # ignore prediction_zero col "none"
        # expl_df.drop(columns='none', inplace=True)
        # contribs = expl_df.values
        # either a pandas dataframe or a numpy recarray are returned, this
        #  code should handle both cases...
        contribs_batch = np.stack(
            [expl_df_batch[name] for name in self.model.symbol_names],
            axis=1
        )
        return contribs_batch
