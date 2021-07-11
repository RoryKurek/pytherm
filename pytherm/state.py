from functools import cached_property
from typing import Optional


class State:
    def __init__(self, P: Optional[float] = None, T: Optional[float] = None, v: Optional[float] = None):
        if [P, T, v].count(None) != 1:
            raise ValueError('State object requires exactly two of P, T, v')

    @property
    def P(self):
        return self._P