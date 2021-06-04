
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class VanillaGradientsExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._explainer = saliency.GradientSaliency()
        self._expected_keys = saliency.GradientSaliency.expected_keys
