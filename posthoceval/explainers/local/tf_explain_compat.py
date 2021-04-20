"""
tf_explain_compat.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Union
from typing import Optional

from abc import ABCMeta

import sympy as sp

from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel


class TFExplainer(BaseExplainer, metaclass=ABCMeta):

    def __init__(self,
                 needs_layer_name,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 verbose: Union[int, bool] = 1,
                 **explain_kwargs):
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self._tf_model = None
        self._target_layer = None
        self._needs_layer_name = needs_layer_name
        self._n_classes = None
        self._explain_kwargs = explain_kwargs

    def predict(self, X):
        pass  # TODO: n/a atm

    def fit(self, X, y=None):
        (self._tf_model,
         self._target_layer,
         self._n_classes) = self._convert_model_to_keras_model(X)

    def feature_contributions(self, X, return_y=False, as_dict=False):
        # TODO so much to do
        extra_kwargs = self._explain_kwargs.copy()
        if self._needs_layer_name:
            extra_kwargs['layer_name'] = self._target_layer

        explanation = [
            [
                self._explainer.explain(
                    ([x_i], None),
                    model=self._tf_model,
                    class_index=k,
                    **extra_kwargs,
                )
                for x_i in X
            ]
            for k in range(self._n_classes)
        ]

        if self.task == 'regression':
            contribs = explanation[0]
        else:
            contribs = explanation

        if as_dict:
            contribs = self._contribs_as_dict(contribs)

        if return_y:
            return contribs, self.model(X)
        return contribs

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

    def _convert_model_to_keras_model(self, data):
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(f'You must have tensorflow installed to use '
                               f'tf-explain explainers')

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import Lambda

        if hasattr(self.model, '_dnn') and isinstance(self.model._dnn, Model):
            # TODO: AdditiveModel -> TFAdditiveModel
            tf_model = self.model._dnn
            return tf_model, None, self._infer_n_classes(tf_model.output_shape)

        if (hasattr(self.model, 'expr') and
                isinstance(self.model.expr, sp.Expr)):
            # TODO: AdditiveModel -> SymbolicAdditiveModel
            expr = self.model.expr
            symbols = self.model.symbols

            output_shape = self.model(data[:1]).shape
            scalar_out = (len(output_shape) < 2)
            n_classes = self._infer_n_classes(output_shape)

            in_layer = Input([self.model.n_features])
            f = sp.lambdify(symbols, expr, 'tensorflow')
            layer = Lambda(
                (lambda x:
                 f(*(x[:, i] for i in range(self.model.n_features)))[:, None])
                if scalar_out else
                (lambda x:
                 f(*(x[:, i] for i in range(self.model.n_features)))),
                name='target_layer'
            )
            output = layer(in_layer)

            tf_model = Model(in_layer, output)

            return tf_model, 'target_layer', n_classes

        # Otherwise
        raise TypeError(f'Incompatible model type {type(self.model)} for '
                        f'tf-explain explainer: {self.model}')
