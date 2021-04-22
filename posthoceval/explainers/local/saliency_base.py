"""
saliency_base.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Union
from typing import Optional

from abc import ABCMeta

import numpy as np
import sympy as sp

import saliency.core as saliency

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel
from posthoceval.evaluate import symbolic_evaluate_func


class SalienceMapExplainer(BaseExplainer, metaclass=ABCMeta):
    _explainer: saliency.base.CoreSaliency

    def __init__(self,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 verbose: Union[int, bool] = 1,
                 smooth: bool = False,
                 multiply_by_input: bool = False,
                 **explain_kwargs):
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self.smooth = smooth
        self.multiply_by_input = multiply_by_input

        self._tf_model = None
        self._target_layer = None
        self._n_classes = None
        self._explain_kwargs = explain_kwargs

    def predict(self, X):
        pass  # TODO: n/a atm

    def fit(self, X, y=None):
        if (hasattr(self.model, '_dnn') and
                isinstance(self.model._dnn, Model)):  # noqa
            # TODO: AdditiveModel -> TFAdditiveModel....
            tf_model = self.model._dnn  # noqa
            target_layer = None
            n_classes = self._infer_n_classes(tf_model.output_shape)

        elif (hasattr(self.model, 'expr') and
              isinstance(self.model.expr, sp.Expr)):
            # TODO: AdditiveModel -> SymbolicAdditiveModel
            expr = self.model.expr
            symbols = self.model.symbols

            output_shape = self.model(X[:1]).shape
            scalar_out = (len(output_shape) < 2)
            n_classes = self._infer_n_classes(output_shape)

            in_layer = Input([self.model.n_features])
            f = symbolic_evaluate_func(expr=expr,
                                       symbols=symbols,
                                       backend='tensorflow',
                                       symbolic=True)
            target_layer = 'target_layer'
            layer = Lambda(
                (lambda x:
                 f(*(x[:, i] for i in range(self.model.n_features)))[:, None])
                if scalar_out else
                (lambda x:
                 f(*(x[:, i] for i in range(self.model.n_features)))),
                name=target_layer
            )
            output = layer(in_layer)
            tf_model = Model(in_layer, output)

        else:
            # Otherwise
            raise TypeError(f'Incompatible model type {type(self.model)} for '
                            f'{self.__class__.__name__} explainer: '
                            f'{self.model}')
        if n_classes > 1 and self.task != 'classification':
            raise ValueError(f'Task is not classification (task={self.task}) '
                             f'but model has {n_classes} outputs!')
        self._tf_model = tf_model
        self._target_layer = target_layer
        self._n_classes = n_classes

    @staticmethod
    def _infer_n_classes(output_shape):
        if len(output_shape) < 2:
            n_classes = 1
        elif len(output_shape) == 2:
            n_classes = output_shape[-1]
        else:
            raise ValueError(f'Unexpected number of dimensions in output of '
                             f'model: {len(output_shape)}')
        return n_classes

    def feature_contributions(self, X, return_y=False, as_dict=False):
        explain_func = (self._explainer.GetSmoothedMask if self.smooth else
                        self._explainer.GetMask)

        explanation = [
            [
                explain_func(
                    x_i,
                    self._saliency_call_func,
                    call_model_args={'target_class_idx': k},
                    **self._explain_kwargs,
                )
                for x_i in X
            ]
            for k in range(self._n_classes)
        ]

        if self.task == 'regression':
            contribs = explanation[0]
        else:
            contribs = explanation

        contribs = np.asarray(contribs)
        if self.multiply_by_input:
            if self.task == 'regression':
                contribs *= X
            else:
                contribs *= X[None, ...]

        if as_dict:
            contribs = self._contribs_as_dict(contribs)

        if return_y:
            return contribs, self.model(X)
        return contribs

    def _saliency_call_func(self,
                            data,
                            call_model_args=None,
                            expected_keys=None):
        target_class_idx = call_model_args.get('target_class_idx')
        data = tf.convert_to_tensor(data)
        with tf.GradientTape() as tape:
            if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
                tape.watch(data)
                model_out = self._tf_model(data)
                if isinstance(model_out, tuple):
                    _, output_layer = model_out
                else:
                    output_layer = model_out
                if target_class_idx is not None:
                    output_layer = output_layer[:, target_class_idx]
                gradients = tape.gradient(output_layer, data).numpy()
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                raise NotImplementedError('Methods requiring a conv_layer...')
                # TODO: infer conv layer from model. raise TypeError for sympy
                #  models?
                # TODO: use _target_layer here?
                conv_layer, output_layer = self._tf_model(data)
                gradients = tape.gradient(output_layer, conv_layer).numpy()
                return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}
