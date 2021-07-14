"""
integrated_gradients.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class IntegratedGradientsExplainer(SalienceMapExplainer):
    """Integrated Gradients (IG) Explainer"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self._explainer = saliency.IntegratedGradients()
        self._expected_keys = saliency.IntegratedGradients.expected_keys
