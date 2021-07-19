import pytest
from pytherm import eos
from pytherm.eos import R


class TestEOSIdeal:
    @pytest.fixture
    def test_eos(self):
        return eos.EOSIdeal()

    @pytest.mark.parametrize('T, v', [
        (450.0, 0.002),
        (500.0, 0.008),
        (2000.0, 0.001),
        (100.0, 0.02),
    ])
    def test_P_examples(self, test_eos, T, v):
        assert test_eos.P(T, v) * v == pytest.approx(R * T)

    @pytest.mark.parametrize('P, v', [
        (100.0, 0.002),
        (1000.0, 0.008),
        (1e6, 0.001),
        (1e5, 0.02),
    ])
    def test_T_examples(self, test_eos, P, v):
        assert P * v == pytest.approx(R * test_eos.T(P, v))

    @pytest.mark.parametrize('P, T', [
        (100.0, 450.0),
        (1000.0, 500.0),
        (1e6, 100.0),
        (1e5, 2000.0),
    ])
    def test_v_examples(self, test_eos, P, T):
        assert P * test_eos.v(P, T) == pytest.approx(R * T)

    @pytest.mark.parametrize('P, T, v', [
        (100.0, 450.0, None),
        (1000.0, None, 0.008),
        (None, 100.0, 0.02),
    ])
    def test_z_examples(self, test_eos, P, T, v):
        assert test_eos.z(P=P, T=T, v=v) == 1.0


B_val = -0.0001


class TestEOSVirial2ndOrder:
    @pytest.fixture
    def test_eos(self):
        B = lambda T: B_val
        return eos.EOSVirial2ndOrder(B)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
        (1e6, 2417.534896311491, 0.02),
    ])
    def test_P_examples(self, test_eos, P, T, v):
        assert test_eos.P(T, v) == pytest.approx(P)

    @pytest.mark.parametrize('T, v, z', [
        (2532.049707189404, 0.002, 0.95),
        (97.43583683361759, 0.008, 0.9875),
        (13363.595676832967, 0.001, 0.9),
    ])
    def test_P_real_gas_law(self, test_eos, T, v, z):
        assert test_eos.P(T=T, v=v) == pytest.approx(z * R * T / v)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
        (1e6, 2417.534896311491, 0.02),
    ])
    def test_T_examples(self, test_eos, P, T, v):
        assert test_eos.T(P, v) == pytest.approx(T)

    @pytest.mark.parametrize('P, v, z', [
        (1e7, 0.002, 0.95),
        (1e5, 0.008, 0.9875),
        (1e8, 0.001, 0.9),
    ])
    def test_T_real_gas_law(self, test_eos, P, v, z):
        assert test_eos.T(P=P, v=v) == pytest.approx(P * v / z / R)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
        (1e6, 2417.534896311491, 0.02),
    ])
    def test_v_examples(self, test_eos, P, T, v):
        assert test_eos.v(P, T) == pytest.approx(v)

    @pytest.mark.parametrize('P, T, z', [
        (1e7, 2532.049707189404, 0.95),
        (1e5, 97.43583683361759, 0.9875),
        (1e8, 13363.595676832967, 0.9),
    ])
    def test_v_real_gas_law(self, test_eos, P, T, z):
        assert test_eos.v(P=P, T=T) == pytest.approx(z * R * T / P)

    @pytest.mark.parametrize('P, T, v, z', [
        (1e7, 2532.049707189404, 0.002, 0.95),
        (1e5, 97.43583683361759, 0.008, 0.9875),
        (1e8, 13363.595676832967, 0.001, 0.9),
    ])
    def test_z_examples(self, test_eos, P, T, v, z):
        assert test_eos.z(P=P, T=T, v=v) == pytest.approx(z)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
    ])
    def test_z_virial_eqn(self, test_eos, P, T, v):
        assert test_eos.z(P=P, T=T, v=v) == pytest.approx(1 + B_val / v)


Tc = 647.096
Pc = 2.2064e7
omega = 0.3443


class TestEOSPurePR:
    @pytest.fixture
    def test_eos(self):
        return eos.PurePREOS(Pc=Pc, Tc=Tc, omega=omega)

    @pytest.mark.parametrize('P, T, v', [
        (2076800.6734812967, 493.15, 0.0018015),
    ])
    def test_P_examples(self, test_eos, P, T, v):
        assert test_eos.P(T, v) == pytest.approx(P)

    @pytest.mark.parametrize('P, T, v, z', [
        (2076800.6734812967, 493.15, 0.0018015, 0.912464299928186),
    ])
    def test_z_examples(self, test_eos, P, T, v, z):
        assert test_eos.z(P, T, v) == pytest.approx(z)
