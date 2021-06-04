
from typing import Union
from typing import Optional
from typing import List

from abc import ABCMeta

import numpy as np
import sympy as sp

import saliency.core as saliency

from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel
from posthoceval.evaluate import symbolic_evaluate_func


class SalienceMapExplainer(BaseExplainer, metaclass=ABCMeta):
    _explainer: saliency.CoreSaliency

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
            tabular=False,
            seed=seed,
            task=task,
            verbose=verbose,
        )
        self.smooth = smooth
        self.multiply_by_input = multiply_by_input
        
        self._expected_keys: Optional[List[str]] = None
        self._atleast_2d = False
        self._atleast_3d = False

        self._tf_model = None
        self._target_layer = None
        self._n_classes = None
        self._explain_kwargs = explain_kwargs

    @classmethod
    def smooth_grad(cls, *args, **kwargs):
        return cls(*args, smooth=True, **kwargs)

    def predict(self, X):
        raise TypeError('Salience map explainers are not model-based: '
                        'predict() is unavailable.')

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ):
        if (saliency.CONVOLUTION_LAYER_VALUES in self._expected_keys or
                saliency.CONVOLUTION_OUTPUT_GRADIENTS in self._expected_keys):
            raise NotImplementedError(
                'Convolutional-based salience map methods')

        if (hasattr(self.model, '_dnn') and
                isinstance(self.model._dnn, Model)):  
            
            tf_model = self.model._dnn  
            target_layer = None
            n_classes = self._infer_n_classes(tf_model.output_shape)

        elif (hasattr(self.model, 'expr') and
              isinstance(self.model.expr, sp.Expr)):
            
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input
            from tensorflow.keras.layers import Lambda

            
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

    def _shape_handler(self, x):
        return (np.atleast_3d(x) if self._atleast_3d else
                np.atleast_2d(x) if self._atleast_2d else
                x)

    def _call_explainer(self, X):
        explain_func = (self._explainer.GetSmoothedMask if self.smooth else
                        self._explainer.GetMask)

        explanation = [
            [
                explain_func(
                    self._shape_handler(x_i),
                    self._saliency_call_func,
                    call_model_args={'target_class_idx': k,
                                     'orig_shape': x_i.shape},
                    **self._explain_kwargs,
                ).reshape(x_i.shape)
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

        return {'contribs': contribs}

    def _saliency_call_func(self,
                            data,
                            call_model_args=None,
                            expected_keys=None):
        
        import tensorflow as tf

        assert self._tf_model is not None
        if (saliency.CONVOLUTION_LAYER_VALUES in self._expected_keys or
                saliency.CONVOLUTION_OUTPUT_GRADIENTS in self._expected_keys):
            raise NotImplementedError(
                'Convolutional-based salience map methods')

        target_class_idx = call_model_args['target_class_idx']
        orig_shape = call_model_args['orig_shape']
        grad_shape = data.shape  
        data = data.reshape(-1, *orig_shape)
        data = tf.convert_to_tensor(data)

        if saliency.INPUT_OUTPUT_GRADIENTS in expected_keys:
            with tf.GradientTape() as tape:
                tape.watch(data)
                model_out = self._tf_model(data)
                if isinstance(model_out, tuple):
                    _, output_layer = model_out
                else:
                    output_layer = model_out
                if target_class_idx is not None:
                    output_layer = output_layer[:, target_class_idx]
                gradients = tape.gradient(output_layer, data)
                gradients = gradients.numpy().reshape(grad_shape)
                ret = {saliency.INPUT_OUTPUT_GRADIENTS: gradients}
                if saliency.OUTPUT_LAYER_VALUES in expected_keys:
                    assert len(expected_keys) == 2
                    ret[saliency.OUTPUT_LAYER_VALUES] = output_layer.numpy()
                else:
                    assert len(expected_keys) == 1
                return ret
                
                
                
                
                
                
                
        elif saliency.OUTPUT_LAYER_VALUES in expected_keys:
            assert len(expected_keys) == 1

            model_out = self._tf_model(data)
            if isinstance(model_out, tuple):
                _, output = model_out
            else:
                output = model_out
            if target_class_idx is not None:
                output = output[:, target_class_idx]
            return {saliency.OUTPUT_LAYER_VALUES: output.numpy()}

        raise ValueError(f'Unexpected keys: {expected_keys}')
