
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class BlurIntegratedGradientsExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self._explainer = saliency.BlurIG()
        self._expected_keys = saliency.BlurIG.expected_keys
        self._atleast_3d = True


BlurIGExplainer = BlurIntegratedGradientsExplainer
