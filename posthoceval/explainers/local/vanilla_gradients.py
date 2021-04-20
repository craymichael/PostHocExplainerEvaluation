"""
vanilla_gradients.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from tf_explain.core.vanilla_gradients import VanillaGradients

from posthoceval.explainers.local.tf_explain_compat import TFExplainer


class VanillaGradientsExplainer(TFExplainer):

    def __init__(self,
                 *args,
                 **kwargs):
        """"""
        super().__init__(
            *args,
            needs_layer_name=False,
            **kwargs
        )

        self._explainer = VanillaGradients()
