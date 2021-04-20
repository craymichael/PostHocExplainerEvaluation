"""
grad_cam.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from tf_explain.core.grad_cam import GradCAM

from posthoceval.explainers.local.tf_explain_compat import TFExplainer


class GradCAMExplainer(TFExplainer):

    def __init__(self,
                 *args,
                 **kwargs):
        """"""
        super().__init__(
            *args,
            needs_layer_name=True,
            **kwargs
        )

        self._explainer = GradCAM()
