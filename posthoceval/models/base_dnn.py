"""
base_dnn.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from typing import Optional
from typing import Dict

from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from posthoceval.models.model import AdditiveModel
from posthoceval.utils import prod


class BaseAdditiveDNN(AdditiveModel, metaclass=ABCMeta):
    def __init__(
            self,
            task,
            input_shape=None,
            n_features=None,
            symbols=None,
            symbol_names=None,
    ):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input

        task = task.lower()
        self.task = task

        if input_shape is not None:
            n_features_from_is = prod(input_shape)
            if n_features is None:
                n_features = n_features_from_is
            else:
                assert n_features == n_features_from_is
        super().__init__(
            symbol_names=symbol_names,
            symbols=symbols,
            n_features=n_features,
        )
        if input_shape is None:
            input_shape = [self.n_features]
        self.input_shape = input_shape

        self._input_tensor = Input(self.input_shape)
        self._dnn: Optional[Model] = None
        self._pre_sum_map: Optional[Dict] = None

        self._y_encoder = self._n_outputs = None

    def _build_dnn(self):
        # Lazy-load
        from tensorflow.keras.layers import Add
        from tensorflow.keras.models import Model

        self._build_pre_sum_map()
        outputs = [*self._pre_sum_map.values()]
        output = Add()(outputs)
        self._dnn = Model(self._input_tensor, output)

    @abstractmethod
    def _build_pre_sum_map(self):
        raise NotImplementedError

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
            self._n_outputs = 1
        else:
            assert self.task == 'classification'
            if y.ndim == 2:
                y = y.squeeze(axis=1)
            if y.ndim == 1:
                self._y_encoder = OneHotEncoder(sparse=False)
                y = self._y_encoder.fit_transform(y.reshape(-1, 1))
            else:
                assert y.ndim == 2
            self._n_outputs = y.shape[1]
        return y

    def fit(self, X, y, optimizer='adam', loss=None, **kwargs):
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
            for k in range(self._n_outputs)
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
