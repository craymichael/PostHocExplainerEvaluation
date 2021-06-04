
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class IntegratedGradientsExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self._explainer = saliency.IntegratedGradients()
        self._expected_keys = saliency.IntegratedGradients.expected_keys
