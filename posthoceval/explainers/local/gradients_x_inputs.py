
from posthoceval.explainers.local.vanilla_gradients import (
    VanillaGradientsExplainer)


class GradientsXInputsExplainer(VanillaGradientsExplainer):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs, multiply_by_input=True)
