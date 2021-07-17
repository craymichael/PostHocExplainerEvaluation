"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Sequence

from posthoceval.expl_utils import standardize_effect
from posthoceval.models.base_dnn import BaseAdditiveDNN


class AdditiveDNN(BaseAdditiveDNN):
    """Additive feedforward DNN"""

    def __init__(
            self,
            terms: Sequence[Sequence[int]],
            task: str = 'regression',
            input_shape=None,
            n_features=None,
            symbols=None,
            symbol_names=None,
            n_units: int = 64,
            activation='relu',
    ):
        """
        Additive feedforward DNN

        :param terms: each term is a sequence of indices of features (symbols)
        :param task: the task, either "classification" or "regression"
        :param input_shape: the shape of a data sample
        :param n_features: the number of features
        :param symbols: sequence of symbols, one for each feature
        :param symbol_names: the name of each feature/symbol
        :param n_units: the base number of units in each feedforward layer, see
            _make_branch for branch architecture
        :param activation: the Keras-compatible activation function (string),
            see _make_branch for branch architecture
        """
        super().__init__(
            task=task,
            input_shape=input_shape,
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )
        self._terms = terms
        self._n_units = n_units
        self._activation = activation

    def _build_pre_sum_map(self):
        # Lazy-load
        import tensorflow as tf
        from tensorflow.keras.layers import Add
        from tensorflow.keras.layers import Flatten

        inp = self._input_tensor
        if len(self.input_shape) > 1:
            inp = Flatten()(inp)
        self._pre_sum_map = {}
        for term in self._terms:  # term features are indices into symbols
            if len(term) == 1:
                feats_str = term[0]
                base_name = f'branch_main/feature_{feats_str}_'
            else:
                feats_str = '_'.join(map(str, term))
                base_name = f'branch_interact/features_{feats_str}_'

            branch = self._make_branch(
                base_name, self._n_units, self._activation)

            xl = tf.gather(inp, term, axis=1, name=f'gather_{feats_str}')
            for layer in branch:
                xl = layer(xl)
            feat_symbols = standardize_effect(
                tuple(self.symbols[fi] for fi in term))
            # TODO: this is boilerplate...
            xl_prev = self._pre_sum_map.get(feat_symbols)
            if xl_prev is not None:
                # add contributions for the same effect together
                xl = Add()([xl_prev, xl])
            self._pre_sum_map[feat_symbols] = xl

    def _make_branch(self, base_name, n_units, activation):
        from tensorflow.keras.layers import Dense

        branch = [
            Dense(n_units, activation=activation,
                  name=base_name + f'l1_d{n_units}'),
            Dense(n_units, activation=activation,
                  name=base_name + f'l2_d{n_units}'),
            Dense(n_units // 2, activation=activation,
                  name=base_name + f'l3_d{n_units // 2}'),
            Dense(self._n_outputs, activation=None,
                  name=base_name + 'linear_d1'),
        ]
        return branch
