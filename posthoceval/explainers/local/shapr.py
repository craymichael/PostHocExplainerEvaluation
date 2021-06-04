
from typing import Optional

import logging
import gc

from collections import OrderedDict
from multiprocessing import Lock

from math import ceil

import numpy as np
import pandas as pd

from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel

__all__ = ['SHAPRExplainer']

logger = logging.getLogger(__name__)

 
_ref_get_model_specs_compat = None
_ref_predict_model_compat = None
 
_shapr_install_lock = Lock()
 
 
_ref_make_wrapped_model_r_inner_dict = {}
 
_make_wrapped_model_r_inner_lock = Lock()


class SHAPRExplainer(BaseExplainer):

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
         
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import pandas2ri

         
        numpy2ri.activate()
         
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

         
        self.approach = kwargs.pop('approach', 'empirical')

        if kwargs:
            raise ValueError(f'Unexpected keyword arguments: {kwargs}')

        self.__wrapped_model = {}
         
        with _shapr_install_lock:
            self._install_shapr_r_compat()

    def __del__(self):
        with _make_wrapped_model_r_inner_lock:
            for model_key in self.__wrapped_model:
                model_str = self._wrapped_model_ref_str + f'_{model_key}'
                 
                _ref_make_wrapped_model_r_inner_dict[model_str] -= 1
                 
                 
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
                 
                out = self.model(X)[:, key]
            return numpy2ri.py2rpy(out)

        inner.rclass = StrVector((self.__r_class_name_model__, 'function'))
        inner.do_slot_assign('feature_names',
                             StrVector(self.model.symbol_names))

        with _make_wrapped_model_r_inner_lock:
             
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

         
        globalenv[model_spec_func_name] = get_model_specs_compat
        globalenv[predict_model_func_name] = predict_model_compat

         
         
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
             
            logger.warning('SHAPR does not current have proper support for '
                           'categorical variables!')

        X_df = pd.DataFrame(data=X, columns=self.model.symbol_names)
        self._x_fit_size = len(X_df)
         
         
         
         
         
         
        kwargs = {}
        if self.model.n_features > 13:
             
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

         
         
        self.prediction_zero_ = np.mean(y, axis=0)

    def predict(self, X):
        raise NotImplementedError

    def _call_explainer(self, X):
         

         
        X_df = pd.DataFrame(data=X, columns=self.model.symbol_names)

        n_explain = len(X_df)

        batch_size = 1

        contribs = []
        for idx in range(0, n_explain, batch_size):
            X_df_batch = X_df.iloc[idx:idx + batch_size].reset_index(drop=True)

            if self.task == 'classification':
                contribs_batch = []
                 
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

         
        gc.collect()

        expl_df_batch = expl_dict_batch['dt']
         
         
         
         
         
        contribs_batch = np.stack(
            [expl_df_batch[name] for name in self.model.symbol_names],
            axis=1
        )
        return contribs_batch
