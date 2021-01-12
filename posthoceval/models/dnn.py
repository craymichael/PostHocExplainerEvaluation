"""
dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import warnings

from tensorflow.keras.models import Model

from posthoceval.model_generation import AdditiveModel


class DNNRegressor(AdditiveModel):
    def __init__(self, dnn: Model, pre_sum_map, symbols, symbol_names=None):
        # TODO: re-abstract AdditiveModel...this is quite sloppy

        self._dnn = dnn
        self._pre_sum_map = pre_sum_map

        self.symbols = symbols
        if symbol_names is None:
            symbol_names = tuple(s.name if hasattr(s, 'name') else str(s)
                                 for s in symbols)
        else:
            assert len(symbol_names) == len(symbols)

        # for compatibility
        self.symbol_names = symbol_names
        self.n_features = len(symbols)
        self._symbol_map = None
        self.expr = self.backend = None

    def __call__(self, X, backend=None):
        if backend is not None:
            warnings.warn(f'{self.__class__} ignores kwarg "backend" '
                          f'({backend}) - this is N/A here')
        return self._dnn(X)

    def fit(self, X, y, optimizer='rmsprop', loss='mean_squared_error',
            **kwargs):
        # kwargs: epochs, batch_size, shuffle, etc...
        self._dnn.compile(optimizer=optimizer, loss=loss)
        self._dnn.fit(X, y, **kwargs)

        return self

    def feature_contributions(self, X, **kwargs):
        if kwargs:
            warnings.warn(f'Ignoring all kwargs {kwargs} - these are N/A '
                          f'here.')

        all_feats, all_outputs = zip(*self._pre_sum_map.items())

        intermediate_model = Model(inputs=self._dnn.input,
                                   outputs=all_outputs)

        packed_contribs = intermediate_model(X)

        contribs = {
            feats: contrib
            for feats, contrib in zip(all_feats, packed_contribs)
        }

        return contribs

    def predict(self, X):
        return self._dnn.predict(X)
