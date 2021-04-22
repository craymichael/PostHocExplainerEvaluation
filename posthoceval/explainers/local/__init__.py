"""
__init__.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from posthoceval.explainers.local.blur_ig import (
    BlurIntegratedGradientsExplainer)
from posthoceval.explainers.local.blur_ig import BlurIGExplainer
from posthoceval.explainers.local.grad_cam import GradCAMExplainer
from posthoceval.explainers.local.gradients_x_inputs import (
    GradientsXInputsExplainer)
from posthoceval.explainers.local.integrated_gradients import (
    IntegratedGradientsExplainer)
from posthoceval.explainers.local.lime import LIMETabularExplainer
from posthoceval.explainers.local.lime import LIMEExplainer
from posthoceval.explainers.local.maple import MAPLEExplainer
from posthoceval.explainers.local.occlusion import OcclusionExplainer
# from posthoceval.explainers.local.parzen import ParzenWindowExplainer
from posthoceval.explainers.local.shap import KernelSHAPExplainer
from posthoceval.explainers.local.shap import SHAPExplainer
from posthoceval.explainers.local.shapr import SHAPRExplainer
from posthoceval.explainers.local.vanilla_gradients import (
    VanillaGradientsExplainer)
from posthoceval.explainers.local.xrai import XRAIExplainer


__all__ = [
    'BlurIGExplainer',
    'BlurIntegratedGradientsExplainer',
    'GradCAMExplainer',
    'GradientsXInputsExplainer',
    'IntegratedGradientsExplainer',
    'BlurIGExplainer',
    'LIMETabularExplainer',
    'LIMEExplainer',
    'MAPLEExplainer',
    'OcclusionExplainer',
    # 'ParzenWindowExplainer',
    'KernelSHAPExplainer',
    'SHAPExplainer',
    'SHAPRExplainer',
    'VanillaGradientsExplainer',
    'XRAIExplainer',
]
