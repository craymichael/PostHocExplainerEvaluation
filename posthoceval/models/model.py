
from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import List
from typing import Dict
from typing import TypeVar

import string

import numpy as np

T = TypeVar('T')


class AdditiveModel(ABC):
    
    
    

    def __init__(
            self,
            symbol_names: Optional[List[str]] = None,
            n_features: Optional[int] = None,
            symbols: Optional[List[T]] = None,
    ):
        
        
        if symbol_names is None:
            if symbols is None:
                assert n_features is not None
                symbol_names = gen_symbol_names(n_features)
            else:
                symbol_names = [*map(str, symbols)]
        if n_features is None:
            if symbols is not None:
                assert len(symbols) == len(symbol_names)
            n_features = len(symbol_names)
        if symbols is not None:
            assert n_features == len(symbols)
            assert n_features == len(symbol_names)

        self.symbol_names: List[str] = symbol_names
        self.n_features: int = n_features
        
        
        self._symbols: Optional[List[T]] = symbols
        self._symbol_map: Optional[Dict[str, T]] = None

    @property
    def symbols(self) -> List[T]:
        if self._symbols is None:
            self.symbols = self.symbol_names.copy()
        return self._symbols

    @symbols.setter
    def symbols(self, value: List[T]) -> None:
        assert len(value) == self.n_features
        self._symbols = value

    def get_symbol(self, symbol_name: str) -> T:
        if self._symbol_map is None:
            self._symbol_map = dict(zip(self.symbol_names, self.symbols))
        return self._symbol_map[symbol_name]

    @abstractmethod
    def __call__(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  
        raise NotImplementedError

    @abstractmethod
    def feature_contributions(
            self,
            X: np.ndarray,
    ):  
        raise NotImplementedError


def gen_symbol_names(n_features: int, excel_like: bool = False) -> List[str]:
    
    assert n_features >= 1, 'Invalid number of features < 1: %d' % n_features
    if excel_like:
        alphabet = string.ascii_uppercase
        ret = []
        for d in range(1, n_features + 1):
            ret_i = ''
            while d > 0:
                d, m = divmod(d - 1, 26)
                ret_i = alphabet[m] + ret_i
            ret.append(ret_i)
        return ret
    else:
        return [f'x{i}' for i in range(1, n_features + 1)]
