"""
cnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from posthoceval.expl_utils import standardize_effect
from posthoceval.models.base_dnn import BaseAdditiveDNN
from posthoceval.models.op_util import symbolic_conv2d


class AdditiveCNN(BaseAdditiveDNN):
    def __init__(
            self,
            task='classification',
            input_shape=None,
            n_features=None,
            symbols=None,
            symbol_names=None,
            filters=4,
            kernel_size=(2, 1),
            strides=None,
            padding='SAME',
            activation='relu',
    ):
        super().__init__(
            task=task,
            input_shape=input_shape,
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )
        self._filters = filters
        self._kernel_size = kernel_size
        if strides is None:
            strides = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activation

    def _build_pre_sum_map(self):
        # Lazy-load
        import tensorflow as tf
        from tensorflow.keras.layers import Add
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Dense
        # TODO: try out max pooling...

        self._pre_sum_map = {}

        conv2d_common = dict(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
        )
        # convolve
        l_conv = Conv2D(activation=self._activation, **conv2d_common)
        out = l_conv(self._input_tensor)
        symbols_in = np.asarray(self.symbols).reshape(self.input_shape)
        symbols_out = symbolic_conv2d(
            symbols_in,
            **conv2d_common,
        )
        # flatten
        out = Flatten()(out)
        symbols_out = symbols_out.flatten()
        # dense time
        for i, effect in enumerate(symbols_out):
            # gather effect out
            effect = standardize_effect(effect)
            feats_str = ','.join(map(str, effect))
            out_i = tf.gather(out, [i], axis=1, name=f'gather_{feats_str}')
            # dot product --> num outputs
            xl = Dense(self._n_outputs, activation=None)(out_i)

            # TODO: this is boilerplate...
            xl_prev = self._pre_sum_map.get(effect)
            if xl_prev is not None:
                # add contributions for the same effect together
                xl = Add()([xl_prev, xl])
            self._pre_sum_map[effect] = xl
