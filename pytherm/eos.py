from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import root_scalar


R = 8.3144622


class EOS(ABC):
    @abstractmethod
    def P(self, T: float, v: float) -> float:
        ...

    @abstractmethod
    def T(self, P: float, v: float) -> float:
        ...

    @abstractmethod
    def v(self, P: float, T: float) -> float:
        ...

    @abstractmethod
    def z(self, P: float, T: float, v: float) -> float:
        ...


class EOSIdeal(EOS):
    def P(self, T: float, v: float) -> float:
        return R * T / v

    def T(self, P: float, v: float) -> float:
        return P * v / R

    def v(self, P: float, T: float) -> float:
        return R * T / P

    def z(self, P: float, T: float, v: float) -> float:
        return 1.0


class EOSVirial2ndOrder(EOS):
    def __init__(self, B: Callable[[float], float]):
        self._B = B

    def P(self, T: float, v: float) -> float:
        return (1 + self._B(T) / v)* R * T / v

    def T(self, P: float, v: float) -> float:
        def func(T: float):
            return T - P * v / R / (1 + self._B(T) / v)
        x0 = P * v / R
        return root_scalar(func, x0=x0, x1=x0+1.0).root

    def v(self, P: float, T: float) -> float:
        def func(z: float):
            return z - 1 - self._B(T) / (z * R * T / P)
        z = root_scalar(func, x0=1.0, x1=1.05).root
        return z * R * T / P

    def z(self, P: float, T: float, v: float) -> float:
        return 1 + self._B(T) / v


class EOSPurePR(EOS):
    def __init__(self, Pc: float, Tc: float, omega: float):
        self._Pc = Pc
        self._Tc = Tc
        self._omega = omega

        self._alpha_coeff = 0.37464 + 1.54226*omega - 0.26992*omega**2
        self._a_coeff = 0.45724 * R**2 * Tc**2 / Pc
        self._b = 0.0778 * R * Tc / Pc

    def _a(self, T: float):
        Tr = T / self._Tc
        return self._a_coeff * (1 + self._alpha_coeff * (1 - Tr**0.5)) ** 2

    def P(self, T: float, v: float) -> float:
        return R*T/(v-self._b) - self._a(T)/(v*(v+self._b) + self._b*(v-self._b))

    def T(self, P: float, v: float) -> float:
        def func(T):
            return P - R * T / (v - self._b) - self._a(T) / (v * (v + self._b) + self._b * (v - self._b))
        T0 = P * v / R
        return root_scalar(func, x0=T0, x1=T0+1.0).root

    def v(self, P: float, T: float) -> float:
        def func(v):
            return P - R * T / (v - self._b) - self._a(T) / (v * (v + self._b) + self._b * (v - self._b))
        v0 = R * T / P
        return root_scalar(func, x0=v0, x1=v0*1.1).root

    def z(self, P: float, T: float, v: float) -> float:
        return v/(v-self._b) - self._a(T)*v/R/T/(v*(v+self._b) + self._b*(v-self._b))
