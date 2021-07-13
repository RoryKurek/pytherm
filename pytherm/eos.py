from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import root_scalar


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

    def z(self, P: float, T: float, v: float) -> float:
        """
        Calculate the compressibility factor of a fluid at the given
        conditions.

        Args:
            P: Pressure [Pa]
            T: Temperature [K]
            v: Specific Volume [m^3/mol]

        Returns:
            Compressibility Factor [dimensionless]
        """
        return P * v / R / T


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


class EOSPurePR(EOS):
    """
    Class modeling the Peng-Robinson equation of state for a pure
    (single-component) fluid.

    .. math::
        z = \\frac{v}{v - b} - \\frac{a \\left( T \\right) v}
            {RT \\left[ v \\left( v + b \\right) + b \\left( v - b \\right) \\right]}

        P = \\frac{RT}{v - b} - \\frac{a \\left( T \\right)}
            {v \\left( v + b \\right) + b \\left( v - b \\right)}

        a \\left( T \\right) = 0.45724 \\frac{R^2 T_c^2}{P_c} α \\left( T \\right)

        α \\left( T \\right) = \\left[ 1 +
            \\left( 0.37464 + 1.54226 ω - 0.26992 ω^2 \\right)
            \\left( 1 - T_r^0.5 \\right) \\right]^2

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
