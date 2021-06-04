import logging

from typing import Union
from typing import Optional
from typing import List
from typing import Tuple
from typing import Any
from typing import Dict

import numpy as np

from alibi.explainers import KernelShap
from joblib import cpu_count

from posthoceval.profile import profile
from posthoceval.models.model import AdditiveModel
from posthoceval.models.base_dnn import BaseAdditiveDNN
from posthoceval.explainers._base import BaseExplainer
from posthoceval.expl_utils import standardize_effect
from posthoceval.utils import at_high_precision

_HAS_RAY = None

logger = logging.getLogger(__name__)


def init_ray():
    global _HAS_RAY

    if _HAS_RAY is not None:
        try:
            import ray
        except ImportError:
            logger.warning('Could not import ray for multiprocessing.')
            _HAS_RAY = False
        else:
            
            ray.init(
                include_dashboard=False,
                logging_level=logging.WARNING,
            )
            _HAS_RAY = True
    return _HAS_RAY


class KernelSHAPExplainer(BaseExplainer):


    def __init__(self,
                 model: AdditiveModel,
                 n_background_samples: int = 100,
                 n_cpus: int = 1,
                 task: str = 'regression',
                 link: Optional[str] = None,
                 seed: Optional[int] = None,
                 group_categorical: bool = False,
                 verbose: Union[int, bool] = 1,
                 **explainer_kwargs):
         
        super().__init__(
            model=model,
            tabular=True,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self.n_background_samples = n_background_samples
        if link is None:
            link = ('identity'
                    if (self.task == 'regression' or
                        isinstance(model, BaseAdditiveDNN)) else
                    'logit')
            logger.info(f'Inferred link as "{link}"')
        self.link = link

        if n_cpus < 0:
            n_cpus = max(cpu_count() + n_cpus + 1, 1)

        self._n_cpus = n_cpus
        self._explainer_kwargs = explainer_kwargs
        self._group_categorical = group_categorical
         
        self.expected_value_ = None
        self._groups = None
        self._category_map = None

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names: Optional[
                List[Union[str, Tuple[str, List[Any]]]]] = None,
            **fit_kwargs,
    ):
         
        explainer_kwargs = self._explainer_kwargs.copy()
        if grouped_feature_names is not None:
            init_kwargs, extra_fit_kwargs = self._handle_categorical(
                grouped_feature_names)
            explainer_kwargs.update(init_kwargs)
            fit_kwargs.update(extra_fit_kwargs)

        use_ray = self._n_cpus > 1
        if use_ray:
            use_ray = init_ray()

         
         
        self._explainer = KernelShap(
            self.model,
            feature_names=self.model.symbol_names,
            task=self.task,
            link=self.link,
            distributed_opts={
                 
                'n_cpus': self._n_cpus,
                 
                 
                 
                'batch_size': None,
            } if use_ray else None,
            seed=self.seed,
            **explainer_kwargs,
        )

        if self.n_background_samples < len(X):
            if self.verbose > 1:
                logger.info(f'Intending to summarize background data as '
                            f'n_samples > {self.n_background_samples}')
            fit_kwargs.setdefault(
                'summarise_background', True)
            fit_kwargs.setdefault(
                'n_background_samples', self.n_background_samples)

        if self.verbose > 0:
            logger.info('Fitting KernelSHAP')
        self._explainer.fit(X, **fit_kwargs)

        self.expected_value_ = self._explainer.expected_value

     
    def _handle_categorical(
            self,
            grouped_feature_names: List[Union[str, Tuple[str, List[Any]]]],
    ) -> Tuple[Dict, Dict]:
         
        groups: List[List[int]] = []
        group_names: List[str] = []
        category_map: Dict[int, List[Any]] = {}

        cat_vars_start_idx: List[int] = []
        cat_vars_enc_dim: List[int] = []

        column_idx = 0   
        for orig_column_idx, item in enumerate(grouped_feature_names):
             
            if isinstance(item, str):
                if self._group_categorical:
                    groups.append([column_idx])
                    group_names.append(item)
                else:
                     
                    pass
                column_idx += 1
            else:
                feature_name, categories = item

                if self._group_categorical:
                     
                    column_idx_end = column_idx + len(categories)
                    groups.append([*range(column_idx, column_idx_end)])
                    group_names.append(feature_name)
                    column_idx = column_idx_end
                     
                    category_map[orig_column_idx] = categories
                else:
                    cat_vars_start_idx.append(column_idx)
                    cat_vars_enc_dim.append(len(categories))
                    column_idx += len(categories)

        if self._group_categorical:
            self._groups = groups
            self._category_map = category_map
            init_kwargs = (dict(categorical_names=category_map)
                           if category_map else {})
            fit_kwargs = (dict(group_names=group_names, groups=groups)
                          if category_map else {})
        else:
            init_kwargs = {}
            fit_kwargs = (dict(cat_vars_start_idx=cat_vars_start_idx,
                               cat_vars_enc_dim=cat_vars_enc_dim)
                          if cat_vars_start_idx else {})
        return init_kwargs, fit_kwargs

    def predict(self, X):
        raise NotImplementedError

    @profile
    def _call_explainer(
            self,
            X: np.ndarray,
    ):
        if self.verbose > 0:
            logger.info('Fetching KernelSHAP explanations')

         
         
         
        explanation = self._explainer.explain(X, silent=self._n_cpus != 1,
                                              l1_reg=False)

         
         
         
         
        if self.task == 'regression':
            assert len(explanation.shap_values) == 1
            contribs_shap = explanation.shap_values[0]
            predictions = np.sum(contribs_shap, axis=1) + self.expected_value_
        else:
            contribs_shap = explanation.shap_values
            intercepts = np.expand_dims(self.expected_value_, 1)
            predictions = np.sum(contribs_shap, axis=2) + intercepts

         
        if self._group_categorical and self._category_map:
            if self.task == 'regression':
                contribs_shap = self._handle_grouped_categorical_as_dict(
                    contribs_shap, X)
            else:
                contribs_shap = [
                    self._handle_grouped_categorical_as_dict(contribs_k, X)
                    for contribs_k in contribs_shap
                ]

        if self.link == 'logit':
             
            link_inv = self._explainer._explainer.link.finv
            predictions = at_high_precision(link_inv, predictions)

        return {'contribs': contribs_shap,
                'intercepts': self.expected_value_,
                'predictions': predictions}

    def _handle_grouped_categorical_as_dict(self, contribs, X: np.ndarray):
        contribs_dict = {}
        symbols = [*map(standardize_effect, self.model.symbols)]
        n_explained = len(X)
        for idx, group in enumerate(self._groups):
            if len(group) > 1:
                 
                contrib = contribs[:, idx]
                for orig_idx in group:
                    symbol = symbols[orig_idx]
                    contrib_cat = np.zeros(n_explained, dtype=float)
                    feat_mask = X[:, orig_idx].astype(bool)
                    assert (X[:, orig_idx] == feat_mask.astype(int)).all()
                    contrib_cat[feat_mask] = contrib[feat_mask]
                    contribs_dict[symbol] = contrib_cat
            else:
                symbol = symbols[group[0]]
                contribs_dict[symbol] = contribs[:, idx]

        return contribs_dict


 
SHAPExplainer = KernelSHAPExplainer
