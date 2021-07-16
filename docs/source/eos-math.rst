Fluid Model Background
======================

Equations of State
------------------

Equations of state are intended to model a fluid's P-v-T behavior. They can be described in one of two forms:
pressure-explicit (i.e. :math:`P=P(v, T)`) or volume-explicit (i.e. :math:`v=v(P, T)`). In general, this describes
whether it is easier/faster to perform all the state variable calculations in terms of :math:`v` and :math:`T` or
:math:`P` and :math:`T`.

Caloric Properties
------------------

The following equations describe the change in any caloric property between any two states. They rely on an EOS for
P-v-T behavior, and a :math:`c_v` or :math:`c_P` correlation. Equations are based on Table 2.1 from [GKKR2019]_.

.. math::
    Δu &= \int_{T_1}^{T_2} c_v \text{d}T + \int_{v_1}^{v_2}\left[ T \left( \frac{∂P}{∂T} \right)_v - P \right] \text{d}v

    Δh &= \int_{T_1}^{T_2} c_P \text{d}T + \int_{P_1}^{P_2} \left[ v - T \left( \frac{∂v}{∂T} \right)_P \right] \text{d}P

    Δs &= \int_{T_1}^{T_2} \frac{c_P}{T} \text{d}T - \int_{P_1}^{P_2} \left( \frac{∂v}{∂T} \right)_P \text{d}P

    Δg &= - \int_{T_1}^{T_2} s \text{d}T + \int_{P_1}^{P_2} v \text{d}P

    Δa &= - \int_{T_1}^{T_2} s \text{d}T - \int_{v_1}^{v_2} P \text{d}v

Note that it is easier to calculate :math:`Δu` when using a pressure-specific EOS and to calculate :math:`Δh` when
using a volume-specific EOS. Since the two are related via :math:`h = u + Pv`, any conversion between the two should be
relatively simple. When using a pressure-specific EOS, for instance, it makes sense to calculate :math:`Δu` using the
above forumla and then calculating:

.. math::
    h_2 - h_1 &= (u_2 + P_2 v_2) - (u_1 + P_1 v_1)

    Δh &= Δu + (P_2 v_2 - P_1 v_1)


.. [GKKR2019] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen
    Rarey. 2019. *Chemical Thermodynamics for Process Simulation*.
    Weinheim: Wiley.