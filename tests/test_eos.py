import pytest
from scipy.misc import derivative
from pytherm import eos
from pytherm.eos import R
#
#
# class TestEOSIdeal:
#     @pytest.fixture
#     def test_eos(self):
#         return eos.EOSIdeal()
#
#     @pytest.mark.parametrize('T, v', [
#         (450.0, 0.002),
#         (500.0, 0.008),
#         (2000.0, 0.001),
#         (100.0, 0.02),
#     ])
#     def test_P_examples(self, test_eos, T, v):
#         assert test_eos.P(T, v) * v == pytest.approx(R * T)
#
#     @pytest.mark.parametrize('P, v', [
#         (100.0, 0.002),
#         (1000.0, 0.008),
#         (1e6, 0.001),
#         (1e5, 0.02),
#     ])
#     def test_T_examples(self, test_eos, P, v):
#         assert P * v == pytest.approx(R * test_eos.T(P, v))
#
#     @pytest.mark.parametrize('P, T', [
#         (100.0, 450.0),
#         (1000.0, 500.0),
#         (1e6, 100.0),
#         (1e5, 2000.0),
#     ])
#     def test_v_examples(self, test_eos, P, T):
#         assert P * test_eos.v(P, T) == pytest.approx(R * T)
#
#     @pytest.mark.parametrize('P, T, v', [
#         (100.0, 450.0, None),
#         (1000.0, None, 0.008),
#         (None, 100.0, 0.02),
#     ])
#     def test_z_examples(self, test_eos, P, T, v):
#         assert test_eos.z(P=P, T=T, v=v) == 1.0
#
#
# B_val = -0.0001
#
#
# class TestEOSVirial2ndOrder:
#     @pytest.fixture
#     def test_eos(self):
#         B = lambda T: B_val
#         return eos.EOSVirial2ndOrder(B)
#
#     @pytest.mark.parametrize('P, T, v', [
#         (1e7, 2532.049707189404, 0.002),
#         (1e5, 97.43583683361759, 0.008),
#         (1e8, 13363.595676832967, 0.001),
#         (1e6, 2417.534896311491, 0.02),
#     ])
#     def test_P_examples(self, test_eos, P, T, v):
#         assert test_eos.P(T, v) == pytest.approx(P)
#
#     @pytest.mark.parametrize('T, v, z', [
#         (2532.049707189404, 0.002, 0.95),
#         (97.43583683361759, 0.008, 0.9875),
#         (13363.595676832967, 0.001, 0.9),
#     ])
#     def test_P_real_gas_law(self, test_eos, T, v, z):
#         assert test_eos.P(T=T, v=v) == pytest.approx(z * R * T / v)
#
#     @pytest.mark.parametrize('P, T, v', [
#         (1e7, 2532.049707189404, 0.002),
#         (1e5, 97.43583683361759, 0.008),
#         (1e8, 13363.595676832967, 0.001),
#         (1e6, 2417.534896311491, 0.02),
#     ])
#     def test_T_examples(self, test_eos, P, T, v):
#         assert test_eos.T(P, v) == pytest.approx(T)
#
#     @pytest.mark.parametrize('P, v, z', [
#         (1e7, 0.002, 0.95),
#         (1e5, 0.008, 0.9875),
#         (1e8, 0.001, 0.9),
#     ])
#     def test_T_real_gas_law(self, test_eos, P, v, z):
#         assert test_eos.T(P=P, v=v) == pytest.approx(P * v / z / R)
#
#     @pytest.mark.parametrize('P, T, v', [
#         (1e7, 2532.049707189404, 0.002),
#         (1e5, 97.43583683361759, 0.008),
#         (1e8, 13363.595676832967, 0.001),
#         (1e6, 2417.534896311491, 0.02),
#     ])
#     def test_v_examples(self, test_eos, P, T, v):
#         assert test_eos.v(P, T) == pytest.approx(v)
#
#     @pytest.mark.parametrize('P, T, z', [
#         (1e7, 2532.049707189404, 0.95),
#         (1e5, 97.43583683361759, 0.9875),
#         (1e8, 13363.595676832967, 0.9),
#     ])
#     def test_v_real_gas_law(self, test_eos, P, T, z):
#         assert test_eos.v(P=P, T=T) == pytest.approx(z * R * T / P)
#
#     @pytest.mark.parametrize('P, T, v, z', [
#         (1e7, 2532.049707189404, 0.002, 0.95),
#         (1e5, 97.43583683361759, 0.008, 0.9875),
#         (1e8, 13363.595676832967, 0.001, 0.9),
#     ])
#     def test_z_examples(self, test_eos, P, T, v, z):
#         assert test_eos.z(P=P, T=T, v=v) == pytest.approx(z)
#
#     @pytest.mark.parametrize('P, T, v', [
#         (1e7, 2532.049707189404, 0.002),
#         (1e5, 97.43583683361759, 0.008),
#         (1e8, 13363.595676832967, 0.001),
#     ])
#     def test_z_virial_eqn(self, test_eos, P, T, v):
#         assert test_eos.z(P=P, T=T, v=v) == pytest.approx(1 + B_val / v)


class TestEOSPurePR:
    @pytest.mark.parametrize('Pc, Tc, omega, T, a', [
        (22064000.0, 647.096, 0.3443, 493.15, 0.740404951803127),
        (22064000.0, 647.096, 0.3443, 300.0, 0.9809878826615795),
        (22064000.0, 647.096, 0.3, 493.15, 0.7301777410707092),
        (22064000.0, 700.0, 0.3443, 493.15, 0.9128609548797745),
        (25000000.0, 700.0, 0.3443, 493.15, 0.8056545643386938)])
    def test_a_examples(self, Pc, Tc, omega, T, a):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos._a(T) == pytest.approx(a)

    @pytest.mark.parametrize('Pc, Tc, omega, b', [
        (22064000.0, 647.096, 0.3443, 1.897134957540787e-05),
        (22064000.0, 647.096, 0.3, 1.897134957540787e-05),
        (22064000.0, 700.0, 0.3443, 2.0522371800761417e-05),
        (25000000.0, 700.0, 0.3443, 1.8112224456479996e-05)])
    def test_b_examples(self, Pc, Tc, omega, b):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos._b == pytest.approx(b)

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_P_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.P(T, v) == pytest.approx(P), 'P(T, v) should match specified P'

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_v_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.v(P, T) == pytest.approx(v), 'v(P, T) should match specified v'

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_T_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.T(P, v) == pytest.approx(T), 'T(P, v) should match specified T'

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_z_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.z(T, v) == pytest.approx(P*v/R/T), 'z(T, v) should equal Pv/RT'

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dP_dT_v_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dP_dT_v(T=T, v=v) == \
               pytest.approx(derivative(lambda T_est: example_eos.P(T=T_est, v=v), x0=T, dx=T*1e-6))

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dP_dv_T_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dP_dv_T(T=T, v=v) == \
               pytest.approx(derivative(lambda v_est: example_eos.P(T=T, v=v_est), x0=v, dx=v*1e-6))

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dT_dP_v_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dT_dP_v(T=T, v=v) == \
               pytest.approx(derivative(lambda P_est: example_eos.T(P=P_est, v=v), x0=P, dx=P*1e-6))

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dT_dv_P_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dT_dv_P(T=T, v=v) == \
               pytest.approx(derivative(lambda v_est: example_eos.T(P=P, v=v_est), x0=v, dx=v*1e-6))

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dv_dT_P_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dv_dT_P(T=T, v=v) == \
               pytest.approx(derivative(lambda T_est: example_eos.v(P=P, T=T_est), x0=T, dx=T*1e-6))

    @pytest.mark.parametrize('Pc, Tc, omega, P, T, v', [
        (22064000.0, 647.096, 0.3443, 2076800.6734812967, 493.15, 0.0018015),
        (22064000.0, 647.096, 0.3443, 2447532.764830227, 493.15, 0.0015),
        (22064000.0, 647.096, 0.3443, 1879295.7426579897, 400.0, 0.0015),
        (22064000.0, 647.096, 0.3, 1887251.502133773, 400.0, 0.0015),
        (22064000.0, 700.0, 0.3, 1811704.2117260671, 400.0, 0.0015),
        (25000000.0, 700.0, 0.3, 1858087.6312887536, 400.0, 0.0015),
    ])
    def test_dv_dP_T_examples(self, Pc, Tc, omega, P, T, v):
        example_eos = eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)
        assert example_eos.dv_dP_T(T=T, v=v) == \
               pytest.approx(derivative(lambda P_est: example_eos.v(P=P_est, T=T), x0=P, dx=P*1e-6))
