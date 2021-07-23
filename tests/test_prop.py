import pytest
from pytherm.prop import Wagner5Corr


class TestWagner5Corr:
    wagner5cases = 'T_min, T_max, Tc, Pc, A, B, C, D, T, prop', [
        # GKKR data for water, ammonia, HCl, Cl, N
        (274, 647.096, 647.096, 220.64, -7.870154, 1.906774, -2.31033, -2.06339, 393.15, 1.9859),
        (196, 405.5, 405.5, 113.592, -7.303825, 1.649953, -2.021615, -1.960295, 275.15, 4.6237),
        (134, 324.55, 324.55, 82.631, -6.454142, 0.934797, -0.636477, -1.704349, 243.15, 10.8202),
        (174, 416.958, 416.958, 79.911, -6.442452, 1.492841, -1.225096, -2.015398, 273.15, 3.6877),
        (65, 126.192, 126.192, 33.958, -6.12368, 1.26061, -0.760446, -1.794726, 103.15, 9.6198)
    ]

    @pytest.mark.parametrize(*wagner5cases)
    def test_examples(self, T_min, T_max, Tc, Pc, A, B, C, D, T, prop):
        corr = Wagner5Corr(T_min=T_min, T_max=T_max, Pc=Pc, Tc=Tc, A=A, B=B, C=C, D=D)
        assert(corr(T)) == pytest.approx(prop, abs=5e-5)

    def test_T_min_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_max=647.096, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)

    def test_T_max_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_min=274, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)
