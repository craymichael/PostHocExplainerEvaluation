"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from posthoceval.models.model import AdditiveModel


class DNNRegressor(AdditiveModel):
    def __init__(
            self,
            dnn,
            pre_sum_map,
            n_features=None,
            symbols=None,
            symbol_names=None
    ):
        super().__init__(
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )
        # Lazy-load
        from tensorflow.keras.models import Model

        self._dnn: Model = dnn
        self._pre_sum_map = pre_sum_map

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
        return self._dnn(X)
