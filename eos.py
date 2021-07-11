from abc import ABC, abstractmethod
from typing import Optional, Callable
from scipy.optimize import root_scalar


R = 8.3144622


class EOS(ABC):
    def z(self, P: Optional[float] = None, T: Optional[float] = None, v: Optional[float] = None):
        if P is None and T is not None and v is not None:
            return self.z_from_Tv(T, v)
        elif P is not None and T is None and v is not None:
            return self.z_from_Pv(P, v)
        elif P is not None and T is not None and v is None:
            return self.z_from_PT(P, T)
        else:
            raise ValueError('z() requires exactly two of P, T, V')

    @abstractmethod
    def z_from_Tv(self, T: float, v: float):
        ...

    @abstractmethod
    def z_from_Pv(self, P: float, v: float):
        ...

    @abstractmethod
    def z_from_PT(self, P: float, T: float):
        ...

    @abstractmethod
    def P(self, T: float, v: float):
        ...

    @abstractmethod
    def T(self, P: float, v: float):
        ...

    @abstractmethod
    def v(self, P: float, T: float):
        ...


class EOSIdeal(EOS):
    def z(self, P: Optional[float] = None, T: Optional[float] = None, v: Optional[float] = None):
        return 1.0

    def z_from_Tv(self, T: float, v: float):
        return 1.0

    def z_from_Pv(self, P: float, v: float):
        return 1.0

    def z_from_PT(self, P: float, T: float):
        return 1.0

    def P(self, T: float, v: float):
        return R * T / v

    def T(self, P: float, v: float):
        return P * v / R

    def v(self, P: float, T: float):
        return R * T / P


class EOSVirial2nd(EOS):
    def __init__(self, B: Callable[[float], float]):
        self._B = B

    def z_from_Tv(self, T: float, v: float):
        return 1 + self._B(T) / v

    def z_from_Pv(self, P: float, v: float):
        return 1 + self._B(self.T(P, v)) / v

    def z_from_PT(self, P: float, T: float):
        return 1 + self._B(T) * P / R / T

    def P(self, T: float, v: float):
        return self.z_from_Tv(T, v) * R * T / v

    def T(self, P: float, v: float):
        def func(T: float):
            return T - P * v / R / (1 + self._B(T))
        return root_scalar(func, x0=P * v / R).root

    def v(self, P: float, T: float):
        return self.z_from_PT(P, T) * R * T / P
