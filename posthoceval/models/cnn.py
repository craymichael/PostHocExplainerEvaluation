"""
cnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
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
        from tensorflow.keras.layers import Add

        self._pre_sum_map = {}

        conv2d_common = dict(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
        )

        symbolic_conv2d()

        for xx in xxx:


            feat_symbols = standardize_effect(
                tuple(self.symbols[fi] for fi in term))
            # TODO: this is boilerplate...
            xl_prev = self._pre_sum_map.get(feat_symbols)
            if xl_prev is not None:
                # add contributions for the same effect together
                xl = Add()([xl_prev, xl])
            self._pre_sum_map[feat_symbols] = xl
