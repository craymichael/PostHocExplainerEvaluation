import logging

from typing import Union
from typing import Optional

from alibi.explainers import KernelShap
from joblib import cpu_count

from posthoceval.profile import profile
from posthoceval.model_generation import AdditiveModel
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
            n_cpus = cpu_count() + n_cpus + 1

        self.n_cpus = n_cpus

        use_ray = n_cpus > 1

        if use_ray:
            use_ray = init_ray()

        self._explainer = KernelShap(
            self.model,
            feature_names=self.model.symbol_names,
            task=self.task,
            link=self.link,
            distributed_opts={
                # https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/
                'n_cpus': n_cpus,
                # If batch_size set to `None`, an input array is split in
                # (roughly) equal parts and distributed across the available
                # CPUs
                'batch_size': None,
            } if use_ray else None,
            seed=self.seed,
            **explainer_kwargs
        )
        # attributes set after fit
        self.expected_value_ = None

    def fit(self, X, y=None, **fit_kwargs):
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

    def predict(self, X):
        pass  # TODO: n/a atm

    @profile
    def feature_contributions(self, X, return_y=False, as_dict=False):
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
        else:
            contribs_shap = explanation.shap_values

        if as_dict:
            contribs_shap = self._contribs_as_dict(contribs_shap)

        if return_y:
            y = explanation.raw['raw_prediction']

            return contribs_shap, y
        return contribs_shap


SHAPExplainer = KernelSHAPExplainer
