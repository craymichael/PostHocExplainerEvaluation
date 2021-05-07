"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from posthoceval.expl_utils import standardize_effect
from posthoceval.models.base_dnn import BaseAdditiveDNN


class AdditiveDNN(BaseAdditiveDNN):
    def __init__(
            self,
            terms,
            task='regression',
            input_shape=None,
            n_features=None,
            symbols=None,
            symbol_names=None,
            n_units=64,
            activation='relu',
    ):
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

            xl = tf.gather(self._input_tensor, term, axis=1,
                           name=f'gather_{feats_str}')
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
