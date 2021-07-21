import abc
from dataclasses import dataclass
from math import exp, sinh, cosh
from .const import R


@dataclass
class TDepCorrelation(abc.ABC):
    T_min: float
    T_max: float

    @abc.abstractmethod
    def __call__(self, T: float) -> float:
        ...

@dataclass
class Wagner5Corr(TDepCorrelation):
    """
    Correlation function using the Wagner equation in its
    2.5-5 form, commonly used for saturation pressure.

    .. math::
        P_s = P_c \\times \\text{exp}\\left[\\left(A𝜏 + B𝜏^{1.5} + C𝜏^{2.5} + D𝜏^5\\right) / T_r\\right]

        T_r = \\frac{T}{T_c}

        𝜏 = 1 - T_r
    """
    Pc: float
    Tc: float
    A: float = 0
    B: float = 0
    C: float = 0
    D: float = 0

    def __call__(self, T: float) -> float:
        Tr = T / self.Tc
        tao = 1 - Tr
        return self.Pc * exp((self.A * tao + self.B * tao ** 1.5 +
                              self.C * tao ** 2.5 + self.D * tao ** 5) / Tr)


@dataclass
class Wagner6Corr(TDepCorrelation):
    """
    Correlation function using the Wagner equation in its
    3-6 form, commonly used for saturation pressure.

    .. math::
        P_s = P_c \\times \\text{exp}\\left[\\left(A𝜏 + B𝜏^{1.5} + C𝜏^3 + D𝜏^6\\right) / T_r\\right]

        T_r = \\frac{T}{T_c}

        𝜏 = 1 - T_r
    """
    Pc: float
    Tc: float
    A: float = 0
    B: float = 0
    C: float = 0
    D: float = 0

    def __call__(self, T: float) -> float:
        Tr = T / self.Tc
        tao = 1 - Tr
        return self.Pc * exp((self.A * tao + self.B * tao ** 1.5 +
                              self.C * tao ** 3 + self.D * tao ** 6) / Tr)


@dataclass
class PPDScp0Corr(TDepCorrelation):
    """
    Creates a correlation function using the PPDS equation for isobaric
    ideal gas heat capacity (:math:`c_P^0`).

    .. math::
        c_P^0 = R \\left\\{ B + (C-B)y^2 \\left[1 + (y-1) (D+Ey+Fy^2+Gy^3+Hy^4) \\right] \\right\\}

        y = \\frac{T}{A+T}
    """
    A: float = 0
    B: float = 0
    C: float = 0
    D: float = 0
    E: float = 0
    F: float = 0
    G: float = 0
    H: float = 0

    def __call__(self, T: float) -> float:
        y = T / (self.A+T)
        return R * (self.B + (self.C - self.B)*y**2 *
                    (1 + (y-1) * (self.D + self.E*y + self.F*y**2 +
                                  self.G*y**3 + self.H*y**4)))


@dataclass
class AlyLeeCorr(TDepCorrelation):
    """
    Creates a correlation function based on the Aly-Lee equation,
    commonly used for isobaric ideal gas heat capacity (:math:`c_P^0`).

    .. math:: c_P^0 = A + B\\left(\\frac{C/T}{\\text{sinh}(C/T)}\\right)^2
                        + D\\left(\\frac{E/T}{\\text{cosh}(E/T)}\\right)^2
    """
    A: float = 0
    B: float = 0
    C: float = 0
    D: float = 0
    E: float = 0

    def __call__(self, T: float) -> float:
        return self.A + self.B * (self.C/T / sinh(self.C/T))**2 + \
               self.D * (self.E/T / cosh(self.E/T))**2
