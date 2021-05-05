"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from sklearn.preprocessing import OneHotEncoder

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
        task = task.lower()
        self.task = task

        super().__init__(
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )
        self._terms = terms
        self._n_units = n_units
        self._activation = activation
        self._dnn = None

        self._y_encoder = self._n_classes = None

    def _build_dnn(self):
        # Lazy-load
        import tensorflow as tf
        from tensorflow.keras.layers import Add
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        # TODO: input_shape? e.g. images...
        x = Input([self.n_features])

        self._pre_sum_map = {}
        outputs = []
        for term in self._terms:  # term features are indices into symbols
            if len(term) == 1:
                base_name = f'branch_main/feature_{term[0]}_'
            else:
                feats_str = '_'.join(map(str, term))
                base_name = f'branch_interact/features_{feats_str}_'

            branch = self._make_branch(
                base_name, self._n_units, self._activation)

            xl = tf.gather(x, term, axis=1, name=str())
            for layer in branch:
                xl = layer(xl)
            outputs.append(xl)
            feat_symbols = tuple(self.symbols[fi] for fi in term)
            self._pre_sum_map[feat_symbols] = xl

        output = Add()(outputs)
        self._dnn = Model(x, output)

    def _make_branch(self, base_name, n_units, activation):
        from tensorflow.keras.layers import Dense

        return [
            Dense(n_units, activation=activation,
                  name=base_name + f'l1_d{n_units}'),
            Dense(n_units, activation=activation,
                  name=base_name + f'l2_d{n_units}'),
            Dense(n_units // 2, activation=activation,
                  name=base_name + f'l3_d{n_units // 2}'),
            Dense(1 if self.task == 'regression' else self._n_classes,
                  activation=None, name=base_name + 'linear_d1'),
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
        if self.task == 'regression' and ret.ndim == 2:
            ret = ret.squeeze(axis=1)
        return ret

    def _standardize_y(self, y):
        y = np.asarray(y)
        if self.task == 'regression':
            assert y.ndim == 1
        else:
            assert self.task == 'classification'
            if y.ndim == 2:
                y = y.squeeze(axis=1)
            if y.ndim == 1:
                self._y_encoder = OneHotEncoder(sparse=False)
                y = self._y_encoder.fit_transform(y.reshape(-1, 1))
            else:
                assert y.ndim == 2
            self._n_classes = y.shape[1]
        return y

    def fit(self, X, y, optimizer='rmsprop', loss=None,
            **kwargs):
        if loss is None:
            from tensorflow.keras.losses import CategoricalCrossentropy
            loss = ('mean_squared_error' if self.task == 'regression' else
                    CategoricalCrossentropy(from_logits=True))
        y = self._standardize_y(y)
        if self._dnn is None:
            self._build_dnn()
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

        packed_contribs = [out.numpy() for out in intermediate_model(X)]

        contribs = [
            {feats: contrib[:, k]
             for feats, contrib in zip(all_feats, packed_contribs)}
            for k in range(self._n_classes if self.task == 'classification'
                           else 1)
        ]
        if self.task == 'regression':
            contribs = contribs[0]

        return contribs

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._dnn.predict(X)
        if self.task == 'classification' and preds.ndim > 1:
            preds = np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != 'classification':
            raise TypeError(f'predict_proba does not make sense for a '
                            f'{self.task} task')
        return self._dnn(X)
