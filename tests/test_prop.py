import pytest
from pytherm.prop import *


class TestWagner5Corr:
    wagner5cases = 'substance, T_min, T_max, Tc, Pc, A, B, C, D, T, prop', [
        ('Water', 274, 647.096, 647.096, 220.64, -7.870154, 1.906774, -2.31033, -2.06339, 393.15, 1.9859),
        ('Ammonia', 196, 405.5, 405.5, 113.592, -7.303825, 1.649953, -2.021615, -1.960295, 275.15, 4.6237),
        ('Hydrogen Chloride', 134, 324.55, 324.55, 82.631, -6.454142, 0.934797, -0.636477, -1.704349, 243.15, 10.8202),
        ('Chloride', 174, 416.958, 416.958, 79.911, -6.442452, 1.492841, -1.225096, -2.015398, 273.15, 3.6877),
        ('Nitrogen', 65, 126.192, 126.192, 33.958, -6.12368, 1.26061, -0.760446, -1.794726, 103.15, 9.6198)
    ]

    @pytest.mark.parametrize(*wagner5cases)
    def test_examples(self, substance, T_min, T_max, Tc, Pc, A, B, C, D, T, prop):
        corr = Wagner5Corr(T_min=T_min, T_max=T_max, Pc=Pc, Tc=Tc, A=A, B=B, C=C, D=D)
        assert corr(T) == pytest.approx(prop, rel=5e-5), f'Validate using GKKR data for {substance}'

    def test_T_min_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_max=647.096, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)

    def test_T_max_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_min=274, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)

    def test_Pc_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_max=647.096, T_min=274, Tc=647.096, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)

    def test_Tc_required(self):
        with pytest.raises(TypeError):
            Wagner5Corr(T_max=647.096, T_min=274, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033, D=-2.06339)

    def test_A_defaults_to_zero(self):
        corr = Wagner5Corr(T_max=647.096, T_min=274, Tc=647.096, Pc=220.64, B=1.906774, C=-2.31033, D=-2.06339)
        assert corr.A == 0.0

    def test_B_defaults_to_zero(self):
        corr = Wagner5Corr(T_max=647.096, T_min=274, Tc=647.096, Pc=220.64, A=-7.870154, C=-2.31033, D=-2.06339)
        assert corr.B == 0.0

    def test_C_defaults_to_zero(self):
        corr = Wagner5Corr(T_max=647.096, T_min=274, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, D=-2.06339)
        assert corr.C == 0.0

    def test_D_defaults_to_zero(self):
        corr = Wagner5Corr(T_max=647.096, T_min=274, Tc=647.096, Pc=220.64, A=-7.870154, B=1.906774, C=-2.31033)
        assert corr.D == 0.0

    @pytest.mark.parametrize('T_min, T_max, Tc, Pc, A, B, C, D, T', [
        (274, 647.096, 647.096, 220.64, -7.870154, 1.906774, -2.31033, -2.06339, 700.0),
        (196, 405.5, 405.5, 113.592, -7.303825, 1.649953, -2.021615, -1.960295, 500.0),
        (134, 324.55, 324.55, 82.631, -6.454142, 0.934797, -0.636477, -1.704349, 100.0),
        (174, 416.958, 416.958, 79.911, -6.442452, 1.492841, -1.225096, -2.015398, 150.0),
    ])
    def test_raises_error_when_extrapolating(self, T_min, T_max, Tc, Pc, A, B, C, D, T):
        corr = Wagner5Corr(T_min=T_min, T_max=T_max, Pc=Pc, Tc=Tc, A=A, B=B, C=C, D=D)
        with pytest.raises(ValueError):
            corr(T)


class TestPPDScp_idCorr:
    @pytest.mark.parametrize('substance, A, B, C, D, E, F, G, T_min, T_max, T, prop', [
        ('Ethane', 903.41135, 4.48148, 11.69046, 8.47923, -77.02151, 122.97656,
         -74.05999, 123, 1500, 273.15, 49.669626),
        ('Propane', 1222.85277, 4.63428, 6.17777, -31.84476, -487.58918, 1216.90986,
         -972.09252, 123, 1500, 373.15, 88.546776),
        ('n-Butane', 668.64918, 8.90810, 14.24670, 41.04664, -258.18297, 411.82384,
         -258.68803, 163, 1500, 373.15, 118.0730936),
        ('Isobutane', 2084.48334, 5.07542, 7.06198, -264.30218, -47.27861, 2309.95342,
         -3524.85868, 143, 1223, 373.15, 117.0442988),
        ('n-Pentane', 1074.74180, 8.97762, 11.92509, 31.16797, -592.50351, 1201.64991,
         -830.32720, 183, 1673, 323.15, 128.1473911),
    ])
    def test_examples(self, substance, T_min, T_max, A, B, C, D, E, F, G, T, prop):
        corr = PPDScp_idCorr(T_min=T_min, T_max=T_max, A=A, B=B, C=C, D=D, E=E, F=F, G=G)
        assert corr(T) == pytest.approx(prop, rel=5e-5), f'Validate using GKKR data for {substance}'

    def test_T_min_required(self):
        with pytest.raises(TypeError):
            PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                          E=-77.02151, F=122.97656, G=-74.05999, T_max=1500)

    def test_T_max_required(self):
        with pytest.raises(TypeError):
            PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                          E=-77.02151, F=122.97656, G=-74.05999, T_min=123)

    def test_A_defaults_to_zero(self):
        corr = PPDScp_idCorr(B=4.48148, C=11.69046, D=8.47923, E=-77.02151,
                             F=122.97656, G=-74.05999, T_min=123, T_max=1500)
        assert corr.A == 0.0

    def test_B_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, C=11.69046, D=8.47923, E=-77.02151,
                             F=122.97656, G=-74.05999, T_min=123, T_max=1500)
        assert corr.B == 0.0

    def test_C_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, D=8.47923, E=-77.02151,
                             F=122.97656, G=-74.05999, T_min=123, T_max=1500)
        assert corr.C == 0.0

    def test_D_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, E=-77.02151,
                             F=122.97656, G=-74.05999, T_min=123, T_max=1500)
        assert corr.D == 0.0

    def test_E_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                             F=122.97656, G=-74.05999, T_min=123, T_max=1500)
        assert corr.E == 0.0

    def test_F_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                             E=-77.02151, G=-74.05999, T_min=123, T_max=1500)
        assert corr.F == 0.0

    def test_G_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                             E=-77.02151, F=122.97656, T_min=123, T_max=1500)
        assert corr.G == 0.0

    def test_H_defaults_to_zero(self):
        corr = PPDScp_idCorr(A=903.41135, B=4.48148, C=11.69046, D=8.47923,
                             E=-77.02151, F=122.97656, G=-74.05999, T_min=123,
                             T_max=1500)
        assert corr.H == 0.0

    @pytest.mark.parametrize('A, B, C, D, E, F, G, T_min, T_max, T', [
        (1222.85277, 4.63428, 6.17777, -31.84476, -487.58918, 1216.90986,
         -972.09252, 123, 1500, 100),
        (68.64918, 8.90810, 14.24670, 41.04664, -258.18297, 411.82384,
         -258.68803, 163, 1500, 130),
        (2084.48334, 5.07542, 7.06198, -264.30218, -47.27861, 2309.95342,
         -3524.85868, 143, 1223, 1300),
        (1074.74180, 8.97762, 11.92509, 31.16797, -592.50351, 1201.64991,
         -830.32720, 183, 1673, 1700),
    ])
    def test_raises_error_when_extrapolating(self, T_min, T_max, A, B, C, D, E, F, G, T):
        corr = PPDScp_idCorr(T_min=T_min, T_max=T_max, A=A, B=B, C=C, D=D, E=E, F=F, G=G)
        with pytest.raises(ValueError):
            corr(T)


class TestAlyLeeCorr:
    @pytest.mark.parametrize('substance, A, B, C, D, E, T_min, T_max, T, prop', [
        # GKKR data for water, ammonia, chlorine, nitrogen, oxygen
        ('Water', 33484.75, 9275.30, 1218.48, 20241.42, 2919.59, 278, 1273, 373.15, 34.0645635),
        ('Ammonia', 34083.18, 26087.00, 990.77, 33100.02, 2905.60, 196, 1500, 394.75, 38.4866538),
        ('Chlorine', 29197.65, 8502.80, 405.49, -3253.99, 3892.43, 173, 1123, 273.15, 33.4747226),
        ('Nitrogen', 29108.79, 8526.28, 1678.41, 66784.83, 10672.63, 73, 1773, 313.15, 29.1317586),
        ('Oxygen', 29116.90, 10437.46, 2565.44, 9338.84, 1149.97, 63, 1773, 1200, 35.678885),
    ])
    def test_examples(self, substance, T_min, T_max, A, B, C, D, E, T, prop):
        corr = AlyLeeCorr(T_min=T_min, T_max=T_max, A=A, B=B, C=C, D=D, E=E)
        assert corr(T) == pytest.approx(prop, rel=5e-5), f'Validate using GKKR data for {substance}'

    def test_T_min_required(self):
        with pytest.raises(TypeError):
            AlyLeeCorr(A=33484.75, B=9275.30, C=1218.48, D=20241.42,
                       E=2919.59, T_max=1273)

    def test_T_max_required(self):
        with pytest.raises(TypeError):
            AlyLeeCorr(A=33484.75, B=9275.30, C=1218.48, D=20241.42,
                       E=2919.59, T_min=278)

    def test_A_defaults_to_zero(self):
        corr = AlyLeeCorr(B=9275.30, C=1218.48, D=20241.42, E=2919.59,
                          T_min=278, T_max=1273)
        assert corr.A == 0.0

    def test_B_defaults_to_zero(self):
        corr = AlyLeeCorr(A=33484.75, C=1218.48, D=20241.42, E=2919.59,
                          T_min=278, T_max=1273)
        assert corr.B == 0.0

    def test_C_defaults_to_zero(self):
        corr = AlyLeeCorr(A=33484.75, B=9275.30, D=20241.42, E=2919.59,
                          T_min=278, T_max=1273)
        assert corr.C == 0.0

    def test_D_defaults_to_zero(self):
        corr = AlyLeeCorr(A=33484.75, B=9275.30, C=1218.48, E=2919.59,
                          T_min=278, T_max=1273)
        assert corr.D == 0.0

    def test_E_defaults_to_zero(self):
        corr = AlyLeeCorr(A=33484.75, B=9275.30, C=1218.48, D=20241.42,
                          T_min=278, T_max=1273)
        assert corr.E == 0.0

    @pytest.mark.parametrize('A, B, C, D, E, T_min, T_max, T', [
        (33484.75, 9275.30, 1218.48, 20241.42, 2919.59, 278, 1273, 230),
        (34083.18, 26087.00, 990.77, 33100.02, 2905.60, 196, 1500, 170),
        (29197.65, 8502.80, 405.49, -3253.99, 3892.43, 173, 1123, 1300),
        (29108.79, 8526.28, 1678.41, 66784.83, 10672.63, 73, 1773, 1800),
    ])
    def test_raises_error_when_extrapolating(self, T_min, T_max, A, B, C, D, E, T):
        corr = PPDScp_idCorr(T_min=T_min, T_max=T_max, A=A, B=B, C=C, D=D, E=E)
        with pytest.raises(ValueError):
            corr(T)
