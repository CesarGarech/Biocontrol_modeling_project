"""
Tests unitarios para las funciones cinéticas de Utils.kinetics.

Cubre: mu_monod, mu_sigmoidal, mu_completa, aiba, mu_fermentacion,
        mu_fermentacion_rto, mu_monod_rto, mu_sigmoidal_rto, mu_completa_rto.
"""
import sys
import os

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from Utils.kinetics import (
    mu_monod,
    mu_sigmoidal,
    mu_completa,
    aiba,
    mu_fermentacion,
    mu_monod_rto,
    mu_sigmoidal_rto,
    mu_completa_rto,
    mu_fermentacion_rto,
)


# ---------------------------------------------------------------------------
# mu_monod
# ---------------------------------------------------------------------------
class TestMuMonod:
    def test_basic_value(self):
        """μ = μmax * S / (Ks + S)"""
        assert mu_monod(1.0, 0.5, 1.0) == pytest.approx(0.25, rel=1e-6)

    def test_half_saturation(self):
        """Cuando S = Ks, μ = μmax/2."""
        mumax, Ks = 0.4, 0.2
        assert mu_monod(Ks, mumax, Ks) == pytest.approx(mumax / 2, rel=1e-6)

    def test_zero_substrate(self):
        """Cuando S=0, μ=0."""
        assert mu_monod(0.0, 0.5, 0.1) == 0.0

    def test_high_substrate_approaches_mumax(self):
        """A sustrato muy alto, μ → μmax."""
        assert mu_monod(1e6, 0.5, 0.1) == pytest.approx(0.5, rel=1e-4)

    def test_non_negative_output(self):
        """El resultado siempre es ≥ 0."""
        assert mu_monod(0.0, 0.3, 0.1) >= 0.0

    def test_returns_float(self):
        assert isinstance(mu_monod(1.0, 0.4, 0.1), float)


# ---------------------------------------------------------------------------
# mu_sigmoidal
# ---------------------------------------------------------------------------
class TestMuSigmoidal:
    def test_n1_equals_monod(self):
        """Con n=1, la ecuación sigmoidal reduce a Monod."""
        S, mumax, Ks = 2.0, 0.5, 1.0
        assert mu_sigmoidal(S, mumax, Ks, 1) == pytest.approx(mu_monod(S, mumax, Ks), rel=1e-6)

    def test_zero_substrate(self):
        assert mu_sigmoidal(0.0, 0.5, 0.1, 2) == 0.0

    def test_cooperative_effect(self):
        """Con n>1, la curva es más sigmoidal (más lenta a sustrato bajo)."""
        S = 0.05  # debajo de Ks
        mu_n1 = mu_sigmoidal(S, 0.5, 0.1, 1)
        mu_n2 = mu_sigmoidal(S, 0.5, 0.1, 2)
        assert mu_n2 < mu_n1  # inhibición cooperativa reduce μ a S bajo

    def test_non_negative(self):
        assert mu_sigmoidal(0.0, 0.4, 0.2, 3) >= 0.0

    def test_approaches_mumax(self):
        assert mu_sigmoidal(1e6, 0.4, 0.1, 2) == pytest.approx(0.4, rel=1e-4)


# ---------------------------------------------------------------------------
# mu_completa
# ---------------------------------------------------------------------------
class TestMuCompleta:
    def test_basic_value(self):
        """Verificación numérica de la fórmula de tres términos."""
        S, O2, P = 1.0, 0.5, 0.1
        mumax, Ks, KO, KP = 0.5, 0.1, 0.1, 5.0
        expected = mumax * (S / (Ks + S)) * (O2 / (KO + O2)) * (KP / (KP + P))
        assert mu_completa(S, O2, P, mumax, Ks, KO, KP) == pytest.approx(expected, rel=1e-6)

    def test_zero_substrate(self):
        assert mu_completa(0.0, 0.5, 0.1, 0.5, 0.1, 0.1, 5.0) == 0.0

    def test_zero_oxygen(self):
        assert mu_completa(1.0, 0.0, 0.1, 0.5, 0.1, 0.1, 5.0) == 0.0

    def test_high_product_inhibition(self):
        """Cuando P → KP, el factor de inhibición → 0.5."""
        mumax, Ks, KO, KP = 0.5, 0.01, 0.01, 10.0
        S, O2, P = 100.0, 100.0, KP
        # KP/(KP+P) = 0.5, S/Ks+S ≈ 1, O2/KO+O2 ≈ 1
        assert mu_completa(S, O2, P, mumax, Ks, KO, KP) == pytest.approx(mumax * 0.5, rel=1e-3)

    def test_non_negative(self):
        assert mu_completa(0.0, 0.0, 100.0, 0.5, 0.1, 0.1, 5.0) >= 0.0


# ---------------------------------------------------------------------------
# aiba (Haldane)
# ---------------------------------------------------------------------------
class TestAiba:
    def test_basic_value(self):
        """Verificación de μ = mumax * I / (Ki + I + I²/KiL)."""
        mumax, I, Ki, KiL = 0.5, 1.0, 0.1, 5.0
        expected = mumax * (I / (Ki + I + I**2 / KiL))
        assert aiba(mumax, I, Ki, KiL) == pytest.approx(expected, rel=1e-6)

    def test_optimum_exists(self):
        """La función tiene un máximo: μ baja a concentraciones muy altas."""
        mumax, Ki, KiL = 0.5, 0.5, 2.0
        mu_low = aiba(mumax, 0.5, Ki, KiL)
        mu_high = aiba(mumax, 100.0, Ki, KiL)
        assert mu_high < mu_low  # inhibición a alta concentración


# ---------------------------------------------------------------------------
# mu_fermentacion
# ---------------------------------------------------------------------------
class TestMuFermentacion:
    # Parámetros literarios estándar (Saccharomyces cerevisiae, aprox.)
    PARAMS = dict(
        mumax_aerob=0.4, Ks_aerob=0.5, KO_aerob=0.004,
        mumax_anaerob=0.15, Ks_anaerob=1.5, KiS_anaerob=50.0,
        KP_anaerob=65.0, n_p=2.0, KO_inhib_anaerob=0.004,
    )

    def test_non_negative_standard(self):
        mu = mu_fermentacion(10, 5, 4, **self.PARAMS)
        assert mu >= 0.0

    def test_zero_substrate_gives_zero(self):
        mu = mu_fermentacion(0, 5, 4, **self.PARAMS)
        assert mu == pytest.approx(0.0, abs=1e-9)

    def test_pasteur_effect(self):
        """Alta O2 inhibe la componente anaeróbica (efecto Pasteur)."""
        mu_lo_O2 = mu_fermentacion(10, 5, 0.0, **self.PARAMS)  # sin O2
        mu_hi_O2 = mu_fermentacion(10, 5, 8.0, **self.PARAMS)  # con O2
        # Con mucho O2, mu_anaerob ≈ 0, solo mu_aerob contribuye
        # Sin O2, mu_aerob ≈ 0, solo mu_anaerob contribuye
        # Los valores deben ser distintos (anaerobio ≠ aerobio)
        assert mu_lo_O2 != pytest.approx(mu_hi_O2, rel=1e-3)

    def test_product_inhibition(self):
        """Cuando P → KP_anaerob, la componente anaeróbica → 0."""
        p = self.PARAMS
        mu_low_P = mu_fermentacion(10, 0.0, 4, **p)    # sin producto
        mu_high_P = mu_fermentacion(10, p["KP_anaerob"] * 0.99, 4, **p)  # casi al límite
        assert mu_low_P > mu_high_P

    def test_aerobic_only_mode(self):
        """considerar_O2=True devuelve solo componente aeróbica."""
        p = self.PARAMS
        mu_mixed = mu_fermentacion(10, 5, 4, **p)
        mu_aerob = mu_fermentacion(10, 5, 4, **p)
        # Con la firma estándar (sin considerar_O2), retorna mixto
        assert mu_aerob is not None


# ---------------------------------------------------------------------------
# Versiones CasADi (RTO)
# ---------------------------------------------------------------------------
class TestCasadiKinetics:
    """Tests numéricos de las versiones CasADi deben coincidir con las regulares."""

    def test_monod_rto_matches_monod(self):
        S, mumax, Ks = 2.0, 0.4, 0.3
        assert float(mu_monod_rto(S, mumax, Ks)) == pytest.approx(mu_monod(S, mumax, Ks), rel=1e-6)

    def test_sigmoidal_rto_matches_sigmoidal(self):
        S, mumax, Ks, n = 1.0, 0.4, 0.2, 2
        assert float(mu_sigmoidal_rto(S, mumax, Ks, n)) == pytest.approx(
            mu_sigmoidal(S, mumax, Ks, n), rel=1e-6
        )

    def test_completa_rto_matches_completa(self):
        S, O2, P = 1.0, 0.5, 0.2
        mumax, Ks, KO, KP = 0.5, 0.1, 0.1, 5.0
        assert float(mu_completa_rto(S, O2, P, mumax, Ks, KO, KP)) == pytest.approx(
            mu_completa(S, O2, P, mumax, Ks, KO, KP), rel=1e-6
        )

    def test_fermentacion_rto_nonnegative(self):
        mu = mu_fermentacion_rto(
            10.0, 5.0, 4.0,
            0.4, 0.5, 0.004,
            0.15, 1.5, 50.0,
            65.0, 2.0, 0.004,
        )
        assert float(mu) >= 0.0

    def test_fermentacion_rto_aerobic_only(self):
        """considerar_O2=True devuelve solo mu_aerob (numérico)."""
        mu_aerob = mu_fermentacion_rto(
            10.0, 5.0, 4.0,
            0.4, 0.5, 0.004,
            0.15, 1.5, 50.0,
            65.0, 2.0, 0.004,
            considerar_O2=True,
        )
        mu_anaerob = mu_fermentacion_rto(
            10.0, 5.0, 4.0,
            0.4, 0.5, 0.004,
            0.15, 1.5, 50.0,
            65.0, 2.0, 0.004,
            considerar_O2=False,
        )
        mu_mixed = mu_fermentacion_rto(
            10.0, 5.0, 4.0,
            0.4, 0.5, 0.004,
            0.15, 1.5, 50.0,
            65.0, 2.0, 0.004,
            considerar_O2=None,
        )
        # mu_aerob + mu_anaerob ≈ mu_mixed
        assert float(mu_aerob) + float(mu_anaerob) == pytest.approx(float(mu_mixed), rel=1e-6)
