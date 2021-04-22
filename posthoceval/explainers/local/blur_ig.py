"""
blur_ig.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class BlurIntegratedGradientsExplainer(SalienceMapExplainer):

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self._explainer = saliency.BlurIG()
        self._expected_keys = saliency.BlurIG.expected_keys


BlurIGExplainer = BlurIntegratedGradientsExplainer
