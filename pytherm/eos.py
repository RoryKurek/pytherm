from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import root_scalar
from scipy.integrate import quad


R = 8.3144622


class EOS(ABC):
    """
    Abstract base class for modeling the relationships between fluid
    properties using a particular equation of state.

    The __init__ methods of concrete classes should take all the
    parameters required for a given equation of state. Instances of the
    class should have no internal state other than those parameters
    (and any related variables/methods). In other words, method calls
    should have no side effects. Essentially, all EOS calculations
    could be implemented as pure functions. However, as the same EOS
    parameters will likely be used many times, it is more convenient to
    instantiate an EOS object with the parameters "baked in". All
    method calls to that instance can then be thought of as function
    calls with those parameters *implicitly* included.

    It important to be careful when writing concrete classes to ensure
    there are no circular dependencies between methods. In general,
    a method of an EOS should not call other methods of the same class.
    """
    @abstractmethod
    def P(self, T: float, v: float) -> float:
        """
        Calculate the pressure of a fluid at the specified temperature
        and specific volume.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            Pressure [Pa]
        """
        ...

    @abstractmethod
    def T(self, P: float, v: float) -> float:
        """
        Calculate the temperature of a fluid at the specified pressure
        and specific volume.

        Args:
            P: Pressure [Pa]
            v: Specific Volume [m^3/mol]

        Returns:
            Temperature [K]
        """
        ...

    @abstractmethod
    def v(self, P: float, T: float) -> float:
        """
        Calculate the specific volume of a fluid at the specified
        pressure and temperature.

        Args:
            P: Pressure [Pa]
            T: Temperature [K]

        Returns:
            Specific Volume [m^3/mol]
        """
        ...


class PExplicitEOS(EOS):
    def T(self, P: float, v: float) -> float:
        T0 = P * v / R
        return root_scalar(lambda T: self.P(T, v) - P, x0=T0, x1=T0 + 1.0).root

    def v(self, P: float, T: float) -> float:
        v0 = R * T / P
        return root_scalar(lambda v: self.P(T, v) - P, x0=v0, x1=v0 * 1.1).root

    def z(self, T: float, v: float) -> float:
        """
        Calculate the compressibility factor of a fluid at the given
        conditions.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            Compressibility Factor [dimensionless]
        """
        return self.P(T, v) * v / R / T

    # First-order P-v-T derivatives

    @abstractmethod
    def dP_dT_v(self, T: float, v: float) -> float:
        """
        First derivative of pressure with respect to temperature at
        constant volume.

        Should be derived from the equation of state itself.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂P/∂T at constant volume [Pa/K]
        """
        ...

    @abstractmethod
    def dP_dv_T(self, T: float, v: float) -> float:
        """
        First derivative of pressure with respect to volume at
        constant temperature.

        Should be derived from the equation of state itself.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂P/∂v at constant temperature [Pa*mol/m^3]
        """
        ...

    def dT_dP_v(self, T: float, v: float) -> float:
        """
        First derivative of temperature with respect to pressure at
        constant volume.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂T/∂P at constant volume [K/Pa]
        """
        return 1.0 / self.dP_dT_v(T, v)

    def dT_dv_P(self, T: float, v: float) -> float:
        """
        First derivative of temperature with respect to volume at
        constant pressure.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂T/∂v at constant pressure [K*mol/m^3]
        """
        return 1.0 / self.dv_dT_P(T, v)

    def dv_dP_T(self, T: float, v: float) -> float:
        """
        First derivative of volume with respect to pressure at
        constant temperature.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂v/∂P at constant temperature [m^3/mol/Pa]
        """
        return 1.0 / self.dP_dv_T(T, v)

    def dv_dT_P(self, T: float, v: float) -> float:
        """
        First derivative of volume with respect to temperature at
        constant pressure.

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂v/∂T at constant pressure [m^3/mol/K]
        """
        return - self.dP_dT_v(T, v) / self.dP_dv_T(T, v)

    # Additional first-order derivatives

    def du_dv_T(self, T: float, v: float) -> float:
        """
        First derivative of internal energy with respect to volume at
        constant temperature. This should be derivable from just the
        equation of state (whereas :math:`(\\frac{du}{dT})_v` requries
        heat capacity data.)

        .. math::
            \\left( \\frac{∂u}{∂v} \\right)_T = T \\left( \\frac{∂P}{∂T} \\right)_v - P

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂u/∂v at constant temperature [J/m^3]
        """
        return T * self.dP_dT_v(T, v) - self.P(T, v)

    def integrate_du_dv_T(self, T: float, v1: float, v2: float) -> float:
        """
        Integral of :math:`(\\frac{du}{dv})_T` between initial and final
        specific volumes. Default implementation uses `scipy.integrate`,
        but can can be overridden with an analytical calculation for
        the integral if desired/practical.

        .. math::
            \\int_{v_1}^{v_2}\\left( \\frac{∂u}{∂v} \\right)_T \\text{d}v =
            \\int_{v_1}^{v_2}\\left[ T \\left( \\frac{∂P}{∂T} \\right)_v - P \\right] \\text{d}v

        Args:
            T: Temperature [K]
            v1: Specific volume at initial state [m^3/mol]
            v2: Specific volume at final state [m^3/mol]
        Returns:
            Change in internal energy [J/mol]
        """
        return quad(lambda v: self.du_dv_T(T, v), v1, v2)[0]

    def dh_dP_T(self, T: float, v: float) -> float:
        """
        First derivative of enthalpy with respect to pressure at
        constant temperature. This should be derivable from just the
        equation of state (whereas :math:`(\\frac{dh}{dT})_P` requries
        heat capacity data.)

        .. math::
            \\left( \\frac{∂h}{∂P} \\right)_T = v - T \\left( \\frac{∂v}{∂T} \\right)_P

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂h/∂P at constant temperature [J/mol/Pa]
        """
        return v - T * self.dv_dT_P(T, v)

    def integrate_dh_dP_T(self, T: float, v1: float, v2: float) -> float:
        """
        Integral of :math:`(\\frac{dh}{dP})_T` between initial and final
        specific volumes. Default implementation uses `scipy.integrate`,
        but can can be overridden with an analytical calculation for
        the integral if desired/practical.

        .. math::
            \\int_{v_1}^{v_2}\\left( \\frac{∂h}{∂P} \\right)_T \\text{d}v =
            \\int_{v_1}^{v_2}\\left[ v - T \\left( \\frac{∂v}{∂T} \\right)_P \\right] \\text{d}v

        Args:
            T: Temperature [K]
            v1: Specific volume at initial state [m^3/mol]
            v2: Specific volume at final state [m^3/mol]
        Returns:
            Change in enthalpy [J/mol]
        """
        return quad(lambda v: self.dh_dP_T(T, v), v1, v2)[0]


class EOSIdeal(EOS):
    """
    Class modeling the ideal gas law.

    .. math:: Pv = RT

    By definition, the compressibility factor (z) for an ideal gas is
    always one.
    """

    def P(self, T: float, v: float) -> float:
        return R * T / v

    def T(self, P: float, v: float) -> float:
        return P * v / R

    def v(self, P: float, T: float) -> float:
        return R * T / P

    def z(self, P: float, T: float, v: float) -> float:
        return 1.0

    def dP_dT_v(self, P: float, T: float, v: float) -> float:
        return R / v


class EOSVirial2ndOrder(EOS):
    """
    Class modeling the virial equation of state, truncated after the
    second term.

    Calculations are based on the Leiden form of the equation:

    .. math::
        PV = zRT

        z = 1 + \\frac{B \\left( T \\right)}{v}
    """
    def __init__(self, B: Callable[[float], float]):
        """
        Initialize the EOS with the desired parameters.

        Args:
            B: A function representing the second virial "coefficient",
                B(T). Takes in a temperature [K] and returns a
                coefficient [dimensionless].
        """
        self._B = B

    def P(self, T: float, v: float) -> float:
        return (1 + self._B(T) / v)* R * T / v

    def T(self, P: float, v: float) -> float:
        def func(T: float):
            return T - P * v / R / (1 + self._B(T) / v)
        x0 = P * v / R
        return root_scalar(func, x0=x0, x1=x0+1.0).root

    def v(self, P: float, T: float) -> float:
        def func(v: float):
            return v - R * T / P * (1 + self._B(T) / v)
        v0 = R * T / P
        return root_scalar(func, x0=v0, x1=v0*1.1).root

    def dP_dT_v(self, P: float, T: float, v: float) -> float:
        return R * T


class PurePREOS(PExplicitEOS):
    """
    Class modeling the Peng-Robinson equation of state for a pure
    (single-component) fluid.

    .. math::
        z = \\frac{v}{v - b} - \\frac{a \\left( T \\right) v}
            {RT \\left[ v \\left( v + b \\right) + b \\left( v - b \\right) \\right]}

        P = \\frac{RT}{v - b} - \\frac{a \\left( T \\right)}
            {v \\left( v + b \\right) + b \\left( v - b \\right)}

        a \\left( T \\right) = C_a α \\left( T \\right)

        α \\left( T \\right) = \\left[1 + C_α \\left( 1 - T_r^{0.5} \\right) \\right]^2

        C_a = 0.45724 \\frac{R^2 T_c^2}{P_c}

        C_α = 0.37464 + 1.54226 ω - 0.26992 ω^2

        b = 0.0778 \\frac{R T_c}{P_c}
    """

    def __init__(self, Pc: float, Tc: float, omega: float):
        """
        Initialize the EOS with the desired parameters.

        Args:
            Pc: Fluid critical pressure [Pa]
            Tc: Fluid critical temperature [K]
            omega: Fluid accentric factor [dimensionless]
        """
        self._Pc = Pc
        self._Tc = Tc
        self._omega = omega

        self._C_alpha = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
        self._C_a = 0.45724 * R ** 2 * Tc ** 2 / Pc
        self._b = 0.0778 * R * Tc / Pc

    def _a(self, T: float):
        Tr = T / self._Tc
        return self._C_a * (1 + self._C_alpha * (1 - Tr ** 0.5)) ** 2

    def _da_dT(self, T: float):
        sqrt_Tr = (T / self._Tc) ** 0.5
        return -self._C_a * self._C_alpha * sqrt_Tr * (1 + self._C_alpha * (1 - sqrt_Tr)) / T

    def P(self, T: float, v: float) -> float:
        return R*T/(v-self._b) - self._a(T)/(v*(v+self._b) + self._b*(v-self._b))

    def dP_dT_v(self, T: float, v: float) -> float:
        """
        First derivative of pressure with respect to temperature at
        constant volume.

        .. math::
            \\left(\\frac{∂P}{∂T}\\right)_v = \\frac{R}{v-b} -
            \\frac{\\left(∂a(T)/∂T\\right)_v}{\\left[ v \\left( v + b \\right) + b \\left( v - b \\right) \\right]^2}

            \\left(∂a(T)/∂T\\right)_v =
            \\frac{C_a C_α T_r^0.5 \\left[1 + C_α \\left(1-T_r^0.5\\right) \\right]}{T}

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂P/∂T at constant volume [Pa/K]
        """
        return R / (v - self._b) - self._da_dT(T) / (v * (v + self._b) + self._b * (v - self._b))

    def dP_dv_T(self, T: float, v: float) -> float:
        """
        First derivative of pressure with respect to specific volume at
        constant temperature.

        .. math::
            \\left(\\frac{∂P}{∂v}\\right)_T = - \\frac{RT}{(v-b)^2} +
            \\frac{2a(T) (v+b)}{b \\left( v + b \\right) + b \\left( v - b \\right)}

        Args:
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            ∂P/∂v at constant temperature [Pa*mol/m^3]
        """
        return -R * T / (v - self._b) ** 2 + \
            2 * self._a(T) * (v+self._b) / (v*(v+self._b) + self._b*(v-self._b)) ** 2

