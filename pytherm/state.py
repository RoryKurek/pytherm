from typing import Optional, Callable
from functools import cached_property
from .eos import EOS


class FluidState:
    def __init__(self, eos: EOS, P: Optional[float] = None, T: Optional[float] = None, v: Optional[float] = None):
        self.eos = eos

        if [P, T, v].count(None) > 1:
            raise ValueError('At least two of P, T, v must be specified')

        if P is not None:
            if P > 0:
                self.P = P
            else:
                raise ValueError('P must have a positive value')
        if T is not None:
            if T > 0:
                self.T = T
            else:
                raise ValueError('T must have a positive value')
        if v is not None:
            if v > 0:
                self.v = v
            else:
                raise ValueError('v must have a positive value')

    @cached_property
    def P(self):
        return self.eos.P(T=self.T, v=self.v)

    @cached_property
    def T(self):
        return self.eos.T(P=self.P, v=self.v)

    @cached_property
    def v(self):
        return self.eos.v(P=self.P, T=self.T)

    @cached_property
    def z(self):
        return self.eos.z(P=self.P, T=self.T, v=self.v)
