"""
grad_cam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import saliency.core as saliency

from posthoceval.explainers.local.saliency_base import SalienceMapExplainer


class GradCAMExplainer(SalienceMapExplainer):

    def __init__(self,
                 *args,
                 **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self._explainer = saliency.GradCam()
        self._expected_keys = saliency.GradCam.expected_keys
