"""
occlusion.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from saliency.core.occlusion import Occlusion

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class OcclusionExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self._explainer = Occlusion()
