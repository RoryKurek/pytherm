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

    @pytest.mark.parametrize('T, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_P_fails_with_nonpositive_args(self, test_eos, T, v):
        with pytest.raises(ValueError):
            test_eos.P(T, v)

    @pytest.mark.parametrize('P, v', [
        (100.0, 0.002),
        (1000.0, 0.008),
        (1e6, 0.001),
        (1e5, 0.02),
    ])
    def test_T_examples(self, test_eos, P, v):
        assert P * v == pytest.approx(R * test_eos.T(P, v))

    @pytest.mark.parametrize('P, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_T_fails_with_nonpositive_args(self, test_eos, P, v):
        with pytest.raises(ValueError):
            test_eos.T(P, v)

    @pytest.mark.parametrize('P, T', [
        (100.0, 450.0),
        (1000.0, 500.0),
        (1e6, 100.0),
        (1e5, 2000.0),
    ])
    def test_v_examples(self, test_eos, P, T):
        assert P * test_eos.v(P, T) == pytest.approx(R * T)

    @pytest.mark.parametrize('P, T', [
        (0.0, 100.0),
        (-5.0, 500.0),
        (100.0, 0.0),
        (100.0, -5.0),
    ])
    def test_v_fails_with_nonpositive_args(self, test_eos, P, T):
        with pytest.raises(ValueError):
            test_eos.v(P, T)

    @pytest.mark.parametrize('P, T, v', [
        (100.0, 450.0, None),
        (1000.0, None, 0.008),
        (None, 100.0, 0.02),
    ])
    def test_z_examples(self, test_eos, P, T, v):
        assert test_eos.z(P=P, T=T, v=v) == 1.0

    def test_z_from_Tv_example(self, test_eos):
        assert test_eos.z_from_Tv(100.0, 0.008) == 1.0

    def test_z_from_Pv_example(self, test_eos):
        assert test_eos.z_from_Pv(100.0, 0.008) == 1.0

    def test_z_from_PT_example(self, test_eos):
        assert test_eos.z_from_PT(100.0, 450.0) == 1.0


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

    def test_P_real_gas_law(self, test_eos):
        T = 500.0
        v = 0.008
        z = test_eos.z_from_Tv(T=T, v=v)
        assert test_eos.P(T=T, v=v) == pytest.approx(z * R * T / v)

    @pytest.mark.parametrize('T, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_P_fails_with_nonpositive_args(self, test_eos, T, v):
        with pytest.raises(ValueError):
            test_eos.P(T, v)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
        (1e6, 2417.534896311491, 0.02),
    ])
    def test_T_examples(self, test_eos, P, T, v):
        assert test_eos.T(P, v) == pytest.approx(T)

    def test_T_real_gas_law(self, test_eos):
        P = 1e7
        v = 0.008
        z = test_eos.z_from_Pv(P=P, v=v)
        assert test_eos.T(P=P, v=v) == pytest.approx(P * v / z / R)

    @pytest.mark.parametrize('P, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_T_fails_with_nonpositive_args(self, test_eos, P, v):
        with pytest.raises(ValueError):
            test_eos.T(P, v)

    @pytest.mark.parametrize('P, T, v', [
        (1e7, 2532.049707189404, 0.002),
        (1e5, 97.43583683361759, 0.008),
        (1e8, 13363.595676832967, 0.001),
        (1e6, 2417.534896311491, 0.02),
    ])
    def test_v_examples(self, test_eos, P, T, v):
        assert test_eos.v(P, T) == pytest.approx(v)

    def test_v_real_gas_law(self, test_eos):
        P = 1e7
        T = 2000.0
        z = test_eos.z_from_PT(P=P, T=T)
        assert test_eos.v(P=P, T=T) == pytest.approx(z * R * T / P)

    @pytest.mark.parametrize('P, T', [
        (0.0, 100.0),
        (-5.0, 500.0),
        (100.0, 0.0),
        (100.0, -5.0),
    ])
    def test_v_fails_with_nonpositive_args(self, test_eos, P, T):
        with pytest.raises(ValueError):
            test_eos.v(P, T)

    @pytest.mark.parametrize('P, T, v, z', [
        (1e7, 2532.049707189404, None, 0.95),
        (1e5, None, 0.008, 0.9875),
        (None, 13363.595676832967, 0.001, 0.9),
    ])
    def test_z_examples(self, test_eos, P, T, v, z):
        assert test_eos.z(P=P, T=T, v=v) == pytest.approx(z)

    @pytest.mark.parametrize('P, T, v', [
        (100.0, None, None),
        (None, None, 0.008),
        (None, 100.0, None),
        (None, None, None),
        (100.0, 100.0, 0.008),
    ])
    def test_z_fails_with_invalid_num_args(self, test_eos, P, T, v):
        with pytest.raises(ValueError):
            test_eos.z(P=P, T=T, v=v)

    def test_z_from_Tv_example(self, test_eos):
        assert test_eos.z_from_Tv(13363.595676832967, 0.001) == pytest.approx(0.9)

    def test_z_from_Tv_virial_eqn(self, test_eos):
        T = 500.0
        v = 0.002
        assert test_eos.z_from_Tv(T=T, v=v) == pytest.approx(1 + B_val / v)

    @pytest.mark.parametrize('T, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_z_from_Tv_fails_with_nonpositive_args(self, test_eos, T, v):
        with pytest.raises(ValueError):
            test_eos.z_from_Tv(T, v)

    def test_z_from_Pv_example(self, test_eos):
        assert test_eos.z_from_Pv(1e5, 0.008) == pytest.approx(0.9875)

    def test_z_from_Pv_virial_eqn(self, test_eos):
        P = 1e6
        v = 0.002
        assert test_eos.z_from_Pv(P=P, v=v) == pytest.approx(1 + B_val / v)

    @pytest.mark.parametrize('P, v', [
        (0.0, 0.01),
        (-5.0, 0.01),
        (100.0, 0.0),
        (100.0, -0.5),
    ])
    def test_z_from_Pv_fails_with_nonpositive_args(self, test_eos, P, v):
        with pytest.raises(ValueError):
            test_eos.z_from_Pv(P, v)

    def test_z_from_PT_example(self, test_eos):
        assert test_eos.z_from_PT(1e7, 2532.049707189404) == pytest.approx(0.95)

    def test_z_from_PT_virial_eqn(self, test_eos):
        P = 1e6
        T = 500.0
        z = test_eos.z_from_PT(P=P, T=T)
        assert z == pytest.approx(1 + B_val / (z * R * T / P))

    @pytest.mark.parametrize('P, T', [
        (0.0, 100.0),
        (-5.0, 500.0),
        (100.0, 0.0),
        (100.0, -5.0),
    ])
    def test_z_from_PT_fails_with_nonpositive_args(self, test_eos, P, T):
        with pytest.raises(ValueError):
            test_eos.z_from_PT(P, T)


Tc = 647.096
Pc = 2.2064e7
omega = 0.3443


class TestEOSPurePR:
    @pytest.fixture
    def test_eos(self):
        return eos.EOSPurePR(Pc=Pc, Tc=Tc, omega=omega)

    @pytest.mark.parametrize('P, T, v', [
        (2076800.6734812967, 493.15, 0.0018015),
    ])
    def test_P_examples(self, test_eos, P, T, v):
        assert test_eos.P(T, v) == pytest.approx(P)

    @pytest.mark.parametrize('P, T, v, z', [
        (None, 493.15, 0.0018015, 0.912464299928186),
        (2076800.6734812967, None, 0.0018015, 0.912464299928186),
        (2076800.6734812967, 493.15, None, 0.912464299928186),
    ])
    def test_z_examples(self, test_eos, P, T, v, z):
        assert test_eos.z(P, T, v) == pytest.approx(z)

    def test_z_from_Tv_example(self, test_eos):
        assert test_eos.z_from_Tv(493.15, 0.0018015) == pytest.approx(0.912464299928186)