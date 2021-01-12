import logging

from typing import Union
from typing import Optional

from alibi.explainers import KernelShap
from joblib import cpu_count

from posthoceval import metrics
from posthoceval.profile import profile
from posthoceval.model_generation import AdditiveModel
from posthoceval.explainers._base import BaseExplainer
from posthoceval.explainers.global_.global_shap import GlobalKernelSHAP
from posthoceval.metrics import standardize_effect

_RAY_INIT = False

logger = logging.getLogger(__name__)


def init_ray():
    global _RAY_INIT

    if _RAY_INIT:
        return
    _RAY_INIT = True
    try:
        import ray
    except ImportError:
        logger.warning('Could not import ray for multiprocessing.')
    else:
        # tell ray to shut up and not launch the dashboard...
        ray.init(
            include_dashboard=False,
            logging_level=logging.WARNING,
        )


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
                 verbose: Union[int, bool] = 1):
        """"""
        self.model = model
        self.n_background_samples = n_background_samples
        self.verbose = verbose
        self.task = task.lower()
        if link is None:
            link = 'identity' if self.task == 'regression' else 'logit'
        self.link = link

        if n_cpus < 0:
            n_cpus = cpu_count() + n_cpus + 1

        self.n_cpus = n_cpus

        use_ray = n_cpus > 1

        if use_ray:
            init_ray()

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
            seed=seed,
        )
        # attributes set after fit
        self.expected_value_ = None

    def fit(self, X, y=None):
        fit_kwargs = {}
        if self.n_background_samples < len(X):
            if self.verbose > 1:
                logger.info(f'Intending to summarize background data as '
                            f'n_samples > {self.n_background_samples}')
            fit_kwargs.update(
                summarise_background=True,
                n_background_samples=self.n_background_samples,
            )

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
        # TODO: multi-class SHAP values (non-regression)
        if self.task == 'regression':
            shap_values = explanation.shap_values[0]
        else:
            shap_values = explanation.shap_values

        # shap_values_g = explanation.raw['importances']['0']

        # TODO: move this logic to super? likely redundant between explainers
        if as_dict:
            symbols = [*map(standardize_effect, self.model.symbols)]
            if self.task == 'regression':
                contribs_shap = dict(zip(symbols, shap_values.T))
            else:
                contribs_shap = [dict(zip(symbols, vals_i.T))
                                 for vals_i in shap_values]
        else:
            contribs_shap = shap_values

        if return_y:
            y = explanation.raw['raw_prediction']

            return contribs_shap, y
        return contribs_shap


@profile
def gshap_explain(model, data_train, data_test, n_background_samples=100):
    # TODO n_background_samples=300 ??

    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
        distributed_opts={
            # https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/
            'n_cpus': cpu_count(),
            # If batch_size set to `None`, an input array is split in (roughly)
            # equal parts and distributed across the available CPUs
            'batch_size': None,
        }
    )
    fit_kwargs = {}
    if n_background_samples < len(data_train):
        print('Intending to summarize background data as n_samples > '
              '{}'.format(n_background_samples))
        fit_kwargs['summarise_background'] = True
        fit_kwargs['n_background_samples'] = n_background_samples

    print('Explainer fit')
    explainer.fit(data_train, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
    print('Explain')
    explanation = explainer.explain(data_train, silent=True)
    expected_value = explanation.expected_value.squeeze()
    shap_values = explanation.shap_values[0]
    outputs_train = explanation.raw['raw_prediction']
    shap_values_g = explanation.raw['importances']['0']

    print('Global SHAP')
    gshap = GlobalKernelSHAP(data_train, shap_values, expected_value)

    gshap_preds, gshap_vals = gshap.predict(data_train, return_shap_values=True)
    print('RMSE global error train', metrics.rmse(outputs_train, gshap_preds))

    gshap_preds, gshap_vals = gshap.predict(data_test, return_shap_values=True)
    outputs_test = model(data_test)
    print('RMSE global error test', metrics.rmse(outputs_test, gshap_preds))

    contribs_gshap = dict(zip(model.symbols, gshap_vals.T))

    return contribs_gshap
