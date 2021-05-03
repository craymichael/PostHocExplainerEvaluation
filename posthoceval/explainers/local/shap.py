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
from posthoceval.explainers._base import BaseExplainer

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
            # tell ray to shut up and not launch the dashboard...
            ray.init(
                include_dashboard=False,
                logging_level=logging.WARNING,
            )
            _HAS_RAY = True
    return _HAS_RAY


class KernelSHAPExplainer(BaseExplainer):
    """
    https://github.com/slundberg/shap/issues/624
    https://github.com/slundberg/shap/blob/e1d0314e5eed0825fb99d8ef01e8cab5b3d45d54/notebooks/kernel_explainer/Squashing%20Effect.ipynb
    """

    def __init__(self,
                 model: AdditiveModel,
                 n_background_samples: int = 100,
                 n_cpus: int = 1,
                 task: str = 'regression',
                 link: Optional[str] = None,
                 seed: Optional[int] = None,
                 verbose: Union[int, bool] = 1,
                 **explainer_kwargs):
        """"""
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self.n_background_samples = n_background_samples
        if link is None:
            link = 'identity' if self.task == 'regression' else 'logit'
            logger.info(f'Inferred link as "{link}"')
        self.link = link

        if n_cpus < 0:
            n_cpus = max(cpu_count() + n_cpus + 1, 1)

        self.n_cpus = n_cpus
        self._explainer_kwargs = explainer_kwargs
        # attributes set after fit
        self.expected_value_ = None

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names: Optional[
                List[Union[str, Tuple[str, List[Any]]]]] = None,
            **fit_kwargs,
    ):
        """"""
        explainer_kwargs = self._explainer_kwargs.copy()
        if grouped_feature_names is not None:
            init_kwargs, extra_fit_kwargs = self._handle_categorical(
                grouped_feature_names)
            explainer_kwargs.update(init_kwargs)
            fit_kwargs.update(extra_fit_kwargs)

        use_ray = self.n_cpus > 1
        if use_ray:
            use_ray = init_ray()

        # TODO: we have dataset objects with feature names, and models with
        #  symbol names for each feature. This gap needs to be bridged...
        self._explainer = KernelShap(
            self.model,
            feature_names=self.model.symbol_names,
            task=self.task,
            link=self.link,
            distributed_opts={
                # https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/
                'n_cpus': self.n_cpus,
                # If batch_size set to `None`, an input array is split in
                # (roughly) equal parts and distributed across the available
                # CPUs
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

    # noinspection PyMethodMayBeStatic
    def _handle_categorical(
            self,
            grouped_feature_names: List[Union[str, Tuple[str, List[Any]]]],
    ) -> Tuple[Dict, Dict]:
        """"""
        groups: List[List[int]] = []
        group_names: List[str] = []
        category_map: Dict[int, List[Any]] = {}

        column_idx = 0  # transformed column index
        for orig_column_idx, item in enumerate(grouped_feature_names):
            # orig_column_idx: original index in untransformed data
            if isinstance(item, str):
                groups.append([column_idx])
                group_names.append(item)
                column_idx += 1
            else:
                feature_name, categories = item
                # groups
                column_idx_end = column_idx + len(categories)
                groups.append([*range(column_idx, column_idx_end)])
                group_names.append(feature_name)
                column_idx = column_idx_end
                # category map
                category_map[orig_column_idx] = categories

        init_kwargs = (dict(categorical_names=category_map)
                       if category_map else {})
        fit_kwargs = (dict(group_names=group_names, groups=groups)
                      if category_map else {})
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

        # Explain with n_cpus > 1 and silent=False gives awful output
        # unfortunately (multiple processes using tqdm in parallel)
        # l1_reg=False --> explain all features, not a subset
        explanation = self._explainer.explain(X, silent=self.n_cpus != 1,
                                              l1_reg=False)

        # Note: explanation.raw['importances'] has aggregated scores per output
        # with corresponding keys, e.g., '0' & '1' for two outputs. Also has
        # 'aggregated' for the aggregated scores over all outputs. Same applies
        # to e.g. `explanation.shap_values[0]`
        if self.task == 'regression':
            assert len(explanation.shap_values) == 1
            contribs_shap = explanation.shap_values[0]
            predictions = np.sum(contribs_shap, axis=1) + self.expected_value_
        else:
            contribs_shap = explanation.shap_values
            predictions = np.sum(contribs_shap, axis=2) + self.expected_value_

        y = explanation.raw['raw_prediction']

        if self.link == 'logit':
            # noinspection PyProtectedMember
            link_inv = np.vectorize(self._explainer._explainer.link.finv)
            y = link_inv(y)
            predictions = link_inv(y)

        return {'contribs': contribs_shap, 'y': y,
                'intercepts': self.expected_value_,
                'predictions': predictions}


# Alias
SHAPExplainer = KernelSHAPExplainer
