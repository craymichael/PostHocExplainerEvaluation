"""
vanilla_gradients.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class VanillaGradientsExplainer(SalienceMapExplainer):
    """Vanilla gradients explainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._explainer = saliency.GradientSaliency()
        self._expected_keys = saliency.GradientSaliency.expected_keys
