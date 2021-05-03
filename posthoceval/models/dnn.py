"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from posthoceval.models.model import AdditiveModel


class AdditiveDNN(AdditiveModel):
    def __init__(
            self,
            terms,
            task='regression',
            n_features=None,
            symbols=None,
            symbol_names=None,
            n_units=64,
            activation='relu',
    ):
        # Lazy-load
        import tensorflow as tf
        from tensorflow.keras.layers import Add
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        task = task.lower()
        if task != 'regression':  # TODO
            raise NotImplementedError(task)
        self.task = task

        super().__init__(
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )

        # TODO: input_shape? e.g. images...
        x = Input([self.n_features])

        self._pre_sum_map = {}
        outputs = []
        for term in terms:  # term features are indices into symbols
            if len(term) == 1:
                base_name = f'branch_main/feature_{term[0]}_'
            else:
                feats_str = '_'.join(map(str, term))
                base_name = f'branch_interact/features_{feats_str}_'

            branch = self._make_branch(
                base_name, n_units, activation)

            xl = tf.gather(x, term, axis=1, name=str())
            for layer in branch:
                xl = layer(xl)
            outputs.append(xl)
            feat_symbols = tuple(self.symbols[fi] for fi in term)
            self._pre_sum_map[feat_symbols] = xl

        output = Add()(outputs)
        self._dnn = Model(x, output)

    @staticmethod
    def _make_branch(base_name, n_units, activation):
        from tensorflow.keras.layers import Dense

        return [
            Dense(n_units, activation=activation,
                  name=base_name + f'l1_d{n_units}'),
            Dense(n_units, activation=activation,
                  name=base_name + f'l2_d{n_units}'),
            Dense(n_units // 2, activation=activation,
                  name=base_name + f'l3_d{n_units // 2}'),
            Dense(1, activation=None,
                  name=base_name + 'linear_d1'),
        ]

    def plot_model(self,
                   to_file='model.png',
                   show_shapes=False,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=96):
        # lazy load
        from tensorflow.keras.utils import plot_model

        return plot_model(self._dnn,
                          to_file=to_file,
                          show_shapes=show_shapes,
                          show_dtype=show_dtype,
                          show_layer_names=show_layer_names,
                          rankdir=rankdir,
                          expand_nested=expand_nested,
                          dpi=dpi)

    def __call__(self, X):
        ret = self._dnn(X).numpy()
        if ret.ndim == 2 and ret.shape[1] == 1:
            ret = ret.squeeze(axis=1)
        return ret

    def fit(self, X, y, optimizer='rmsprop', loss='mean_squared_error',
            **kwargs):
        # kwargs: epochs, batch_size, shuffle, etc...
        self._dnn.compile(optimizer=optimizer, loss=loss)
        self._dnn.fit(X, y, **kwargs)

        return self

    def feature_contributions(self, X):
        # lazy load
        from tensorflow.keras.models import Model

        all_feats, all_outputs = zip(*self._pre_sum_map.items())

        intermediate_model = Model(inputs=self._dnn.input,
                                   outputs=all_outputs)

        packed_contribs = intermediate_model(X)

        contribs = {
            feats: contrib.numpy().squeeze(axis=1)
            for feats, contrib in zip(all_feats, packed_contribs)
        }

        return contribs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._dnn.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != 'classification':
            raise TypeError(f'predict_proba does not make sense for a '
                            f'{self.task} task')
        return self._dnn(X)
