from abc import ABC, abstractmethod
from math import exp


class TDepCorrelation(ABC):
    @abstractmethod
    def __call__(self, T: float) -> float:
        ...

    @abstractmethod
    @property
    def constants(self) -> dict[str, float]:
        ...

    @abstractmethod
    @property
    def latex_form(self) -> str:
        ...

    @abstractmethod
    @property
    def form_name(self) -> str:
        ...


class Wagner5Eqn(TDepCorrelation):
    def __init__(self, Pc: float, Tc: float, A: float = 0, B: float = 0,
                 C: float = 0, D: float = 0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Pc = Pc
        self.Tc = Tc

    def __call__(self, T: float) -> float:
        Tr = T / self.Tc
        tao = 1 - Tr
        return self.Pc * exp((self.A * tao + self.B * tao**1.5 + self.C * tao**2.5 + self.D * tao**5)/Tr)

    @property
    def constants(self) -> dict[str, float]:
        return {'A': self.A,
                'B': self.B,
                'C': self.C,
                'D': self.D,
                'Pc': self.Pc,
                'Tc': self.Tc}

    @property
    def latex_form(self) -> str:
        return r'\text{ln} \left(\frac{Y}{P_C}\right) = \frac{1}{T_r}' \
               r'\left(A𝜏 + B𝜏^{1.5} + C𝜏^{2.5} + D𝜏^5\right)'

    @property
    def form_name(self) -> str:
        return 'Wagner Equation - 2.5-5 Form'
