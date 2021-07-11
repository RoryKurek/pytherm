from abc import ABC, abstractmethod
from typing import Optional, Callable
from scipy.optimize import root_scalar


R = 8.3144622


class EOS(ABC):
    @abstractmethod
    def P(self, T: float, v: float):
        ...

    @abstractmethod
    def T(self, P: float, v: float):
        ...

    @abstractmethod
    def v(self, P: float, T: float):
        ...

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


class EOSIdeal(EOS):
    def P(self, T: float, v: float):
        if T > 0 and v > 0:
            return R * T / v
        else:
            raise ValueError('T and v must be greater than zero')

    def T(self, P: float, v: float):
        if P > 0 and v > 0:
            return P * v / R
        else:
            raise ValueError('P and v must be greater than zero')

    def v(self, P: float, T: float):
        if P > 0 and T > 0:
            return R * T / P
        else:
            raise ValueError('P and T must be greater than zero')

    def z_from_Tv(self, T: float, v: float):
        return 1.0

    def z_from_Pv(self, P: float, v: float):
        return 1.0

    def z_from_PT(self, P: float, T: float):
        return 1.0


class EOSVirial2ndOrder(EOS):
    def __init__(self, B: Callable[[float], float]):
        self._B = B

    def P(self, T: float, v: float):
        if T > 0 and v > 0:
            return self.z_from_Tv(T, v) * R * T / v
        else:
            raise ValueError('T and v must be greater than zero')

    def T(self, P: float, v: float):
        if P > 0 and v > 0:
            def func(T: float):
                return T - P * v / R / (1 + self._B(T) / v)
            x0 = P * v / R
            return root_scalar(func, x0=x0, x1=x0+1.0).root
        else:
            raise ValueError('P and v must be greater than zero')

    def v(self, P: float, T: float):
        if P > 0 and T > 0:
            return self.z_from_PT(P, T) * R * T / P
        else:
            raise ValueError('P and T must be greater than zero')

    def z_from_Tv(self, T: float, v: float):
        return 1 + self._B(T) / v

    def z_from_Pv(self, P: float, v: float):
        return 1 + self._B(self.T(P, v)) / v

    def z_from_PT(self, P: float, T: float):
        def func(z: float):
            return z - 1 - self._B(T) / (z * R * T / P)
        return root_scalar(func, x0=1.0, x1=1.05).root
