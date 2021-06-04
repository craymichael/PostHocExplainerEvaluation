
import sys
import logging
import importlib

from types import ModuleType

logger = logging.getLogger(__name__)


class LazyLoader(ModuleType):
    __slots__ = 'lib_name', '_mod'

    def __init__(self, lib_name):
        self.lib_name = lib_name
        self._mod = None
        if self.lib_name not in sys.modules:
            sys.modules[self.lib_name] = self

        super().__init__(lib_name)

    @property
    def __spec__(self):
        if self._mod is None:
            return None
        return self._mod.__spec__

    def __getattr__(self, name):
        if self._mod is None:
            logger.info(f'Triggered lazy-loading of {self.lib_name}')
            _ = sys.modules.pop(self.lib_name, None)
            self._mod = importlib.import_module(self.lib_name)
            sys.modules[self.lib_name] = self._mod

        return getattr(self._mod, name)
