
from saliency.core.occlusion import Occlusion

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class OcclusionExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        
        
        super().__init__(*args, **kwargs, size=kwargs.pop('size', 1))
        self._explainer = Occlusion()
        self._expected_keys = Occlusion.expected_keys
        self._atleast_2d = True
