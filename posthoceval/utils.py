from typing import Sized
from typing import Iterable
from typing import Union
from typing import Tuple
from typing import Optional
from typing import Any


def as_sized(x: Iterable) -> Union[Tuple, Sized]:
    if not isinstance(x, Sized):
        x = tuple(x)
    return x


def assert_same_size(expected: int,
                     received: int,
                     units: str = 'values',
                     ret=None) -> Optional[Any]:
    if expected != received:
        raise ValueError('Expected %d %s but received %d instead.' %
                         (expected, units, received))
    return ret
