from .eos import PExplicitEOS
from typing import Optional
from .prop import TDepCorrelation


class FluidModel:
    def __init__(self, eos: PExplicitEOS, Cp_ideal: Optional[TDepCorrelation] = None):
        self._eos = eos
        self._Cp_ideal = Cp_ideal

    def P(self, T: float, v: float) -> float:
        return self._eos.P(T, v)

    def T(self, P: float, v: float) -> float:
        return self._eos.T(P, v)

    def v(self, P: float, T: float) -> float:
        return self._eos.v(P, T)

    def z(self, T: float, v: float) -> float:
        return self._eos.z(T, v)

    def Cp_ideal(self, T: float) -> Optional[float]:
        if self._Cp_ideal is not None:
            return self._Cp_ideal(T)
        else:
            return None
