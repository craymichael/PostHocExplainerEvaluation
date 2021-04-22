"""
gradients_x_inputs.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from posthoceval.explainers.local.vanilla_gradients import (
    VanillaGradientsExplainer)


class GradientsXInputsExplainer(VanillaGradientsExplainer):

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs, multiply_by_input=True)
