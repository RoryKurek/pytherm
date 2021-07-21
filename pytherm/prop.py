from math import exp, sinh, cosh
from typing import Callable
from .eos import R

TDepCorrelation = Callable[[float], float]


def wagner_5_corr(Pc: float, Tc: float, A: float = 0, B: float = 0, C: float = 0, D: float = 0) -> TDepCorrelation:
    """
    Creates a correlation function using the Wagner equation in its
    2.5-5 form, commonly used for saturation pressure.

    .. math::
        P_s = P_c \\times \\text{exp}\\left[\\left(A𝜏 + B𝜏^{1.5} + C𝜏^{2.5} + D𝜏^5\\right) / T_r\\right]

        T_r = \\frac{T}{T_c}

        𝜏 = 1 - T_r

    Returns:
        A function taking a temperature as input and returning the
        value of the correlation at that temperature.
    """
    def corr(T: float) -> float:
        Tr = T / Tc
        tao = 1 - Tr
        return Pc * exp((A * tao + B * tao**1.5 + C * tao**2.5 + D * tao**5)/Tr)
    return corr


def wagner_6_corr(Pc: float, Tc: float, A: float = 0, B: float = 0, C: float = 0, D: float = 0) -> TDepCorrelation:
    """
    Creates a correlation function using the Wagner equation in its
    3-6 form, commonly used for saturation pressure.

    .. math::
        P_s = P_c \\times \\text{exp}\\left[\\left(A𝜏 + B𝜏^{1.5} + C𝜏^3 + D𝜏^6\\right) / T_r\\right]

        T_r = \\frac{T}{T_c}

        𝜏 = 1 - T_r

    Returns:
        A function taking a temperature as input and returning the
        value of the correlation at that temperature.
    """
    def corr(T: float) -> float:
        Tr = T / Tc
        tao = 1 - Tr
        return Pc * exp((A * tao + B * tao**1.5 + C * tao**3 + D * tao**6)/Tr)
    return corr


def ppds_cp0_corr(A: float = 0, B: float = 0, C: float = 0, D: float = 0,
                 E: float = 0, F: float = 0, G: float = 0, H: float = 0) -> TDepCorrelation:
    """
    Creates a correlation function using the PPDS equation for isobaric
    ideal gas heat capacity (:math:`c_P^0`).

    .. math::
        c_P^0 = R \\left\\{ B + (C-B)y^2 \\left[1 + (y-1) (D+Ey+Fy^2+Gy^3+Hy^4) \\right] \\right\\}

        y = \\frac{T}{A+T}

    Returns:
        A function taking a temperature as input and returning the
        value of the correlation at that temperature.
    """
    def corr(T: float) -> float:
        y = T / (A+T)
        return R * (B + (C - B)*y**2 *(1 + (y-1) * (D + E*y + F*y**2 + G*y**3 + H*y**4)))
    return corr


def aly_lee_corr(A: float = 0, B: float = 0, C: float = 0, D: float = 0,
                 E: float = 0) -> TDepCorrelation:
    """
    Creates a correlation function based on the Aly-Lee equation,
    commonly used for isobaric ideal gas heat capacity (:math:`c_P^0`).

    .. math:: c_P^0 = A + B\\left(\\frac{C/T}{\\text{sinh}(C/T)}\\right)^2
                        + D\\left(\\frac{E/T}{\\text{cosh}(E/T)}\\right)^2

    Returns:
        A function taking a temperature as input and returning the
        value of the correlation at that temperature.
    """
    def corr(T: float) -> float:
        return A + B * (C/T / sinh(C/T))**2 + D * (E/T / cosh(E/T))**2
    return corr
