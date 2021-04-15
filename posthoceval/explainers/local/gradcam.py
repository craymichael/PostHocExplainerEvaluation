import logging

from typing import Union
from typing import Optional

import sympy as sp

from tf_explain.core.grad_cam import GradCAM

from posthoceval.model_generation import AdditiveModel
from posthoceval.explainers._base import BaseExplainer

logger = logging.getLogger(__name__)


def _infer_n_classes(output_shape):
    if len(output_shape) < 2:
        n_classes = 1
    elif len(output_shape) == 2:
        n_classes = output_shape[-1]
    else:
        raise ValueError(f'Unexpected number of dimensions in output of '
                         f'model: {len(output_shape)}')
    return n_classes


def _convert_model_to_keras_model(model: AdditiveModel, data):
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(f'You must have tensorflow installed to use '
                           f'tf-explain explainers')

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Lambda

    if hasattr(model, '_dnn') and isinstance(model._dnn, Model):
        # TODO: AdditiveModel -> TFAdditiveModel
        tf_model = model._dnn
        return model._dnn, None, _infer_n_classes(tf_model.output_shape)

    if hasattr(model, 'expr') and isinstance(model.expr, sp.Expr):
        # TODO: AdditiveModel -> SymbolicAdditiveModel
        expr = model.expr
        symbols = model.symbols

        output_shape = model(data[:1]).shape
        scalar_out = (len(output_shape) < 2)
        n_classes = _infer_n_classes(output_shape)

        in_layer = Input([model.n_features])
        f = sp.lambdify(symbols, expr, 'tensorflow')
        layer = Lambda(
            (lambda x: f(*(x[:, i] for i in range(model.n_features)))[:, None])
            if scalar_out else
            (lambda x: f(*(x[:, i] for i in range(model.n_features)))),
            name='target_layer'
        )
        output = layer(in_layer)

        tf_model = Model(in_layer, output)

        return tf_model, 'target_layer', n_classes

    # Otherwise
    raise TypeError(f'Incompatible model type {type(model)} for tf-explain '
                    f'explainer: {model}')


class GradCAMExplainer(BaseExplainer):

    def __init__(self,
                 model: AdditiveModel,
                 task: str = 'regression',
                 seed: Optional[int] = None,
                 verbose: Union[int, bool] = 1,
                 **explainer_kwargs):
        """"""
        super().__init__(
            model=model,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self._explainer = GradCAM()
        self._tf_model = None
        self._target_layer = None
        self._n_classes = None

    def fit(self, X, y=None):
        if self.verbose > 0:
            logger.info('Fitting GradCAM')
        (self._tf_model,
         self._target_layer,
         self._n_classes) = _convert_model_to_keras_model(self.model, X)

    def predict(self, X):
        pass  # TODO: n/a atm

    def feature_contributions(self, X, return_y=False, as_dict=False):
        if self.verbose > 0:
            logger.info('Fetching GradCAM explanations')

        # TODO so much to do
        explanation = [
            [
                self._explainer.explain(
                    ([x_i], None), self._tf_model,
                    class_index=k,
                    layer_name=self._target_layer,
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
