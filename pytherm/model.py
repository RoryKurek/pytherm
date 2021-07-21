from .eos import PExplicitEOS
from typing import Optional
from .prop import TDepCorrelation
from .const import R


class FluidModel:
    def __init__(self, eos: PExplicitEOS,
                 cp_ideal: Optional[TDepCorrelation] = None,
                 cv_ideal: Optional[TDepCorrelation] = None):
        self._eos = eos

        if cp_ideal is not None and cv_ideal is None:
            self._cp_ideal = cp_ideal
            self._cv_ideal = lambda T: self._cp_ideal(T) - R
        elif cv_ideal is not None and cp_ideal is None:
            self._cv_ideal = cv_ideal
            self._cp_ideal = lambda T: self._cv_ideal(T) + R
        else:
            raise ValueError('FluidModel requires exactly one of cp_ideal or cv_ideal')

    def P(self, T: float, v: float) -> float:
        return self._eos.P(T, v)

    def T(self, P: float, v: float) -> float:
        return self._eos.T(P, v)

    def v(self, P: float, T: float) -> float:
        return self._eos.v(P, T)

    def z(self, T: float, v: float) -> float:
        return self._eos.z(T, v)

    def cp_ideal(self, T: float) -> float:
        return self._cp_ideal(T)

    def cv_ideal(self, T: float) -> float:
        return self._cv_ideal(T)
