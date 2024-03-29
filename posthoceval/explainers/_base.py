from abc import ABC
from abc import abstractmethod

from typing import Union
from typing import Optional
from typing import List
from typing import Dict
from typing import Any
from typing import Tuple

import numpy as np

from posthoceval.expl_utils import standardize_effect
from posthoceval.models.model import AdditiveModel
from posthoceval.datasets.dataset import Dataset
from posthoceval.utils import prod

Contribs = Union[np.ndarray, List[np.ndarray],
                 Dict[Any, np.ndarray], List[Dict[Any, np.ndarray]]]


class _TabularExplainerModel(AdditiveModel):
    """Wrapper for models used in tabular-only explainers"""

    def __init__(
            self,
            model: AdditiveModel,
    ):
        super().__init__(
            symbol_names=model.symbol_names,
            n_features=model.n_features,
            symbols=model.symbols,
        )
        self._model = model
        self._input_shape = None

    def set_input_shape(self, input_shape: Tuple[int, ...]):
        self._input_shape = input_shape

    # noinspection PyPep8Naming
    def _handle_X(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], *self._input_shape)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = self._handle_X(X)
        return self._model(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._handle_X(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._handle_X(X)
        return self._model.predict_proba(X)

    def feature_contributions(self, X: np.ndarray):
        # This should never be called, it does not make sense for an explainer
        #  to do so, and a private attribute should not have been called by
        #  something else...
        raise RuntimeError(
            f'The tabular explainer wrapping {self._model.__class__.__name__} '
            f'should never call the model method feature_contributions!'
        )


class BaseExplainer(ABC):
    """Base explainer class"""
    model: Union[AdditiveModel, _TabularExplainerModel]

    def __init__(self,
                 model: AdditiveModel,
                 tabular: bool,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 verbose: Union[int, bool] = 1):
        """
        Base explainer class


        :param model: the model to explain
        :param tabular: this should be set by subclasses. This flag informs
            that the wrapped or implemented explainer only supports tabular
            data. To support these explainers for non-tabular data, we simply
            flatten the data and restore shapes during execution.
        :param seed: the RNG seed for reproducibility
        :param task: the task, either "classification" or "regression"
        :param verbose: print more messages if True
        """
        task = task.lower()
        # TODO: formalize tasks among all relevant objects in project (e.g.
        #  dataset/model)
        if task not in ['regression', 'classification']:
            raise ValueError(f'Invalid task name: {task}')
        self.task = task

        self._tabular = tabular
        if self._tabular:
            model = _TabularExplainerModel(model)
        self.model = model

        self.verbose = verbose
        self.seed = seed

        # initialized in fit
        self._explainer = None
        self._fitted = False

    @abstractmethod
    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names: Optional[
                List[Union[str, Tuple[str, List[Any]]]]] = None,
    ) -> None:
        raise NotImplementedError

    # noinspection PyPep8Naming
    def fit(
            self,
            X: Union[np.ndarray, Dataset],
            y=None,
            grouped_feature_names: Optional[
                List[Union[str, Tuple[str, List[Any]]]]] = None,
            **kwargs,
    ) -> 'BaseExplainer':
        """
        Fits the explainer given a dataset.

        :param X: The features or Dataset to explain for the given model
        :param y: The labels of the dataset (optional, unused by most
            explainers)
        :param grouped_feature_names: The grouped feature names
        :param kwargs: passed to the subclass implementation of _fit
        :return: self
        """
        if isinstance(X, Dataset):
            assert grouped_feature_names is None, (
                'Cannot provide both grouped_feature_names and a Dataset '
                'object for X')
            # assert y is None, (
            #     'Cannot provide both y and a Dataset object for X')
            grouped_feature_names = X.grouped_feature_names
            # y = X.y
            X = X.X
        if self._tabular:
            if isinstance(self.model, _TabularExplainerModel):
                self.model.set_input_shape(X.shape[1:])
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
        # noinspection PyArgumentList
        self._fit(X=X, y=y, grouped_feature_names=grouped_feature_names,
                  **kwargs)
        self._fitted = True
        return self

    @abstractmethod
    def predict(self, X: Union[np.ndarray, Dataset]) -> np.ndarray:
        """
        Predict for the given data

        :param X: the input data or dataset
        :return: predicted model output
        """
        # TODO: predict abstract --> _predict abstract
        #  predict calls _predict
        #  predict checks self._fitted
        raise NotImplementedError

    @abstractmethod
    def _call_explainer(
            self,
            X: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Contribs: ->
             Union[np.ndarray, List[np.ndarray],
                   Dict[Any, np.ndarray], List[Dict[Any, np.ndarray]]]:
        """
        raise NotImplementedError

    def feature_contributions(
            self,
            X: Union[np.ndarray, Dataset],
            return_y: bool = False,
            return_intercepts: bool = False,
            return_predictions: bool = False,
            as_dict: Optional[bool] = None,
    ):
        """
        Estimate feature contributions for the provided data and model.

        :param X: the input data or Dataset object
        :param return_y: whether to also return the model output
        :param return_intercepts: whether to also return the explainer
            intercepts. This will be None if not applicable to the explainer
        :param return_predictions: whether to also return the explainer
            predictions
        :param as_dict: Whether to return the estimated feature contributions
            as a dict. By default this is inferred by what the explainer
            returns. Returning not as a dict may not always be possible, i.e.,
            if the explainer estimates interaction effects.
        :return: explained feature contributions[, [y, [intercepts,
            [predictions]]]]
        """
        if not self._fitted:
            raise RuntimeError('You must call fit() before calling '
                               'feature_contributions()')

        if isinstance(X, Dataset):
            X = X.X
        orig_shape = X.shape[1:]
        if self._tabular and X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        call_result = self._call_explainer(X)

        contribs = self._handle_as_dict(call_result['contribs'], as_dict,
                                        orig_shape)
        ret = (contribs,)

        if return_y:
            y = call_result.get('y')
            if y is None:
                y = self.model(X)
            ret += (y,)

        if return_intercepts:
            intercepts = call_result.get('intercepts')
            ret += (intercepts,)

        if return_predictions:
            predictions = call_result.get('predictions')
            if predictions is None:
                predictions = self.predict(X)
            ret += (predictions,)

        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _handle_as_dict(
            self,
            contribs: Contribs,
            as_dict: Optional[bool],
            orig_shape: Tuple[int, ...],
    ):
        if self.task == 'regression':
            is_dict = isinstance(contribs, dict)
            if not is_dict:
                contribs_input_shape = contribs.shape[1:]
                assert prod(contribs_input_shape) == prod(orig_shape), (
                    f'{contribs_input_shape} incompatible with {orig_shape}')
                contribs = contribs.reshape(contribs.shape[0], *orig_shape)
        else:
            # hope that children allow us to make this assumption...
            is_dict = isinstance(contribs[0], dict)
            if not is_dict:
                contribs_input_shape = contribs[0].shape[1:]
                assert prod(contribs_input_shape) == prod(orig_shape), (
                    f'{contribs_input_shape} incompatible with {orig_shape}')
                contribs = [
                    contribs_i.reshape(contribs_i.shape[0], *orig_shape)
                    for contribs_i in contribs
                ]
        if is_dict:
            if self.task == 'regression':
                contribs = {standardize_effect(k): v
                            for k, v in contribs.items()}
            else:
                # classification task should return a list of contribs, one for
                #  each class
                contribs = [{standardize_effect(k): v
                             for k, v in contribs_k.items()}
                            for contribs_k in contribs]

        if as_dict is None:  # give whatever child gave us
            return contribs

        symbols = [*map(standardize_effect, self.model.symbols)]

        # Note: this assumes per-feature attribution, which is the only
        #  case where this a non-dict is legal
        if is_dict and not as_dict:
            if self.task == 'regression':
                contribs = self._single_contribs_from_dict(
                    contribs, symbols, orig_shape)
            else:
                contribs = [
                    self._single_contribs_from_dict(
                        contribs_k, symbols, orig_shape)
                    for contribs_k in contribs
                ]
        elif not is_dict and as_dict:
            if self.task == 'regression':
                contribs = self._single_contribs_to_dict(contribs, symbols)
            else:
                contribs = [
                    self._single_contribs_to_dict(contribs_k, symbols)
                    for contribs_k in contribs
                ]
        # else, contribs is already in the requested format

        return contribs

    # noinspection PyMethodMayBeStatic
    def _single_contribs_to_dict(
            self,
            contribs: np.ndarray,
            symbols: List[Tuple[Any, ...]],
    ):
        contribs = contribs.reshape(contribs.shape[0], -1)
        # TODO: n_features in case on nd...?
        n_features = contribs.shape[1]
        assert len(symbols) == n_features
        # Contribs: feature: ndarray[n_samples]
        return dict(zip(symbols, contribs.T))

    # noinspection PyMethodMayBeStatic
    def _single_contribs_from_dict(
            self,
            contribs: Dict[Any, np.ndarray],
            symbols: List[Tuple[Any, ...]],
            orig_shape: Tuple[int, ...],
    ):
        assert not (set(contribs.keys()) - set(symbols))
        assert contribs, 'unsupported when contribs is empty'
        one_contrib = next(iter(contribs.values()))
        n_explained = len(one_contrib)
        dtype = one_contrib.dtype
        # This will raise a KeyError if the contribs are invalid
        # Contribs: ndarray[n_samples x (n_features)]
        return np.asarray([
            contribs.get(sym, np.zeros(n_explained, dtype=dtype))
            for sym in symbols
        ]).T.reshape(n_explained, *orig_shape)
