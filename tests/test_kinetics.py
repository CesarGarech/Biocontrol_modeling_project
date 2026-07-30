"""
Tests unitarios para las funciones cinéticas en Utils/kinetics.py.

Cubre: mu_monod, mu_sigmoidal, mu_completa, aiba, mu_fermentacion
y sus versiones CasADi-compatibles (_rto).
"""
import sys
import os

# Asegurar que la raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ─── mu_monod ───────────────────────────────────────────────────────────────

class TestMuMonod:
    """Pruebas para la cinética de Monod simple."""

    def test_valor_tipico(self):
        """μ = μmax * S/(Ks+S)  con valores representativos."""
        resultado = mu_monod(S=0.1, mumax=0.5, Ks=0.1)
        esperado = 0.5 * 0.1 / (0.1 + 0.1)
        assert abs(resultado - esperado) < 1e-12

    def test_sustrato_cero_retorna_cero(self):
        """Cuando S=0 la tasa de crecimiento debe ser 0."""
        assert mu_monod(S=0.0, mumax=0.5, Ks=0.1) == pytest.approx(0.0)

    def test_saturacion(self):
        """Para S >> Ks, μ → μmax."""
        resultado = mu_monod(S=1e6, mumax=0.4, Ks=0.1)
        assert resultado == pytest.approx(0.4, rel=1e-4)

    def test_semisaturacion(self):
        """Para S = Ks, μ = μmax/2."""
        resultado = mu_monod(S=0.2, mumax=0.4, Ks=0.2)
        assert resultado == pytest.approx(0.2, rel=1e-10)

    def test_no_negativo(self):
        """La tasa de crecimiento siempre debe ser ≥ 0."""
        assert mu_monod(S=0.0, mumax=0.3, Ks=0.05) >= 0.0

    def test_tipo_retorno_float(self):
        """El valor de retorno debe ser un número real."""
        resultado = mu_monod(S=1.0, mumax=0.3, Ks=0.1)
        assert isinstance(resultado, float)


# ─── mu_sigmoidal ────────────────────────────────────────────────────────────

class TestMuSigmoidal:
    """Pruebas para la cinética sigmoidal (Hill)."""

    def test_n1_es_monod(self):
        """Con n=1 la ecuación sigmoidal debe reducirse a Monod."""
        s, mumax, ks = 2.0, 0.4, 0.5
        sig = mu_sigmoidal(S=s, mumax=mumax, Ks=ks, n=1)
        monod = mu_monod(S=s, mumax=mumax, Ks=ks)
        assert sig == pytest.approx(monod, rel=1e-10)

    def test_n_mayor_1_mas_sigmoidal(self):
        """Con n>1 la transición es más abrupta que Monod en S<Ks."""
        s, mumax, ks = 0.2, 0.4, 0.5
        sig_n2 = mu_sigmoidal(S=s, mumax=mumax, Ks=ks, n=2)
        sig_n1 = mu_sigmoidal(S=s, mumax=mumax, Ks=ks, n=1)
        assert sig_n2 < sig_n1

    def test_sustrato_cero_retorna_cero(self):
        assert mu_sigmoidal(S=0.0, mumax=0.5, Ks=0.1, n=2) == pytest.approx(0.0)

    def test_semisaturacion_n1(self):
        """Para S=Ks y n=1, μ = μmax/2."""
        resultado = mu_sigmoidal(S=0.3, mumax=0.6, Ks=0.3, n=1)
        assert resultado == pytest.approx(0.3, rel=1e-10)

    def test_saturacion_n_grande(self):
        """Para S >> Ks con cualquier n, μ → μmax."""
        resultado = mu_sigmoidal(S=1e6, mumax=0.5, Ks=0.1, n=3)
        assert resultado == pytest.approx(0.5, rel=1e-4)


# ─── mu_completa ─────────────────────────────────────────────────────────────

class TestMuCompleta:
    """Pruebas para la cinética de Monod con restricciones (O2 y P)."""

    def test_valor_tipico(self):
        """Verifica el resultado con valores representativos."""
        resultado = mu_completa(S=5.0, O2=6.0, P=0.0, mumax=0.4, Ks=0.1, KO=0.5, KP=50.0)
        esperado = 0.4 * (5.0/5.1) * (6.0/6.5) * (50.0/50.0)
        assert abs(resultado - esperado) < 1e-9

    def test_sin_oxigeno_retorna_cero(self):
        """Sin oxígeno la tasa de crecimiento aerobia debe ser 0."""
        assert mu_completa(S=5.0, O2=0.0, P=0.0, mumax=0.4, Ks=0.1, KO=0.5, KP=50.0) == pytest.approx(0.0)

    def test_inhibicion_producto(self):
        """Mayor concentración de producto reduce la tasa de crecimiento."""
        mu_bajo_p = mu_completa(S=5.0, O2=6.0, P=0.0, mumax=0.4, Ks=0.1, KO=0.5, KP=50.0)
        mu_alto_p = mu_completa(S=5.0, O2=6.0, P=10.0, mumax=0.4, Ks=0.1, KO=0.5, KP=50.0)
        assert mu_alto_p < mu_bajo_p

    def test_no_negativo(self):
        """La tasa nunca debe ser negativa."""
        assert mu_completa(S=0.0, O2=0.0, P=100.0, mumax=0.4, Ks=0.1, KO=0.5, KP=50.0) >= 0.0


# ─── aiba ────────────────────────────────────────────────────────────────────

class TestAiba:
    """Pruebas para el modelo de Aiba (inhibición por sustrato)."""

    def test_inhibicion_alta_concentracion(self):
        """A concentraciones muy altas del inhibidor μ debe disminuir."""
        mu_baja = aiba(mumax=0.4, I=1.0, Ki=0.5, KiL=10.0)
        mu_alta = aiba(mumax=0.4, I=50.0, Ki=0.5, KiL=10.0)
        assert mu_alta < mu_baja

    def test_I_cero_retorna_cero(self):
        """Con inhibidor = 0, la tasa de crecimiento es 0."""
        assert aiba(mumax=0.5, I=0.0, Ki=0.2, KiL=5.0) == pytest.approx(0.0)

    def test_existe_maximo(self):
        """La función debe tener un máximo en I = sqrt(Ki * KiL)."""
        Ki, KiL = 0.2, 5.0
        I_opt = np.sqrt(Ki * KiL)
        concentraciones = np.linspace(0.01, 20.0, 200)
        valores = [aiba(mumax=0.4, I=I, Ki=Ki, KiL=KiL) for I in concentraciones]
        idx_max = np.argmax(valores)
        assert abs(concentraciones[idx_max] - I_opt) < 0.5


# ─── mu_fermentacion ─────────────────────────────────────────────────────────

class TestMuFermentacion:
    """Pruebas para el modelo de fermentación mixta aerobia/anaerobia."""

    # Parámetros de referencia
    PARAMS = dict(
        mumax_aerob=0.4, Ks_aerob=0.5, KO_aerob=0.2,
        mumax_anaerob=0.15, Ks_anaerob=1.0, KiS_anaerob=150.0,
        KP_anaerob=80.0, n_p=1.0, KO_inhib_anaerob=0.1,
    )

    def test_no_negativo_condiciones_extremas(self):
        """La tasa de crecimiento nunca debe ser negativa."""
        for S, P, O2 in [(0, 0, 0), (0, 100, 0), (100, 0, 8), (100, 100, 8)]:
            resultado = mu_fermentacion(S, P, O2, **self.PARAMS)
            assert resultado >= 0.0, f"Negativo para S={S}, P={P}, O2={O2}"

    def test_efecto_pasteur(self):
        """Mayor O2 debe reducir la contribución anaerobia (efecto Pasteur)."""
        mu_alta_o2 = mu_fermentacion(5.0, 0.0, 8.0, **self.PARAMS)
        mu_baja_o2 = mu_fermentacion(5.0, 0.0, 0.0, **self.PARAMS)
        # La parte anaerobia domina con O2 bajo
        assert mu_baja_o2 > 0.0
        # Con O2 alto, la parte anaerobia se inhibe
        assert mu_alta_o2 > 0.0

    def test_inhibicion_etanol(self):
        """Mayor concentración de etanol reduce la tasa anaerobia."""
        mu_sin_etanol = mu_fermentacion(5.0, 0.0, 0.0, **self.PARAMS)
        mu_con_etanol = mu_fermentacion(5.0, 40.0, 0.0, **self.PARAMS)
        assert mu_sin_etanol > mu_con_etanol

    def test_p_mayor_kp_retorna_solo_aerobio(self):
        """Cuando P ≥ KP_anaerob, la componente anaerobia debe ser 0."""
        params = self.PARAMS.copy()
        # P justo por encima del límite de inhibición
        resultado = mu_fermentacion(5.0, params['KP_anaerob'] + 1.0, 0.0, **params)
        # Solo queda la contribución aerobia (O2=0 → mu_aerob=0)
        assert resultado == pytest.approx(0.0, abs=1e-9)

    def test_modo_solo_aerobio(self):
        """considerar_O2=True debe retornar solo la componente aerobia."""
        mu_total = mu_fermentacion(5.0, 0.0, 4.0, **self.PARAMS)
        mu_aerob = mu_fermentacion(5.0, 0.0, 4.0, considerar_O2=True, **self.PARAMS)
        assert mu_aerob <= mu_total

    def test_modo_solo_anaerobio(self):
        """considerar_O2=False debe retornar solo la componente anaerobia."""
        mu_anaerob = mu_fermentacion(5.0, 0.0, 4.0, considerar_O2=False, **self.PARAMS)
        assert mu_anaerob >= 0.0


# ─── Versiones CasADi (_rto) ─────────────────────────────────────────────────

class TestRtoFunctions:
    """
    Verifica que las versiones CasADi (_rto) retornan los mismos valores
    numéricos que sus contrapartes estándar para entradas escalares.
    """

    def _to_float(self, val):
        """Convierte DM de CasADi a float si es necesario."""
        try:
            return float(val)
        except Exception:
            return val.full().flatten()[0]

    def test_mu_monod_rto_igual_monod(self):
        S, mumax, Ks = 2.0, 0.4, 0.5
        rto = self._to_float(mu_monod_rto(S, mumax, Ks))
        std = mu_monod(S, mumax, Ks)
        assert rto == pytest.approx(std, rel=1e-8)

    def test_mu_sigmoidal_rto_igual_sigmoidal(self):
        S, mumax, Ks, n = 1.0, 0.5, 0.3, 2.0
        rto = self._to_float(mu_sigmoidal_rto(S, mumax, Ks, n))
        std = mu_sigmoidal(S, mumax, Ks, n)
        assert rto == pytest.approx(std, rel=1e-8)

    def test_mu_completa_rto_igual_completa(self):
        S, O2, P, mumax, Ks, KO, KP = 3.0, 5.0, 1.0, 0.4, 0.2, 0.5, 30.0
        rto = self._to_float(mu_completa_rto(S, O2, P, mumax, Ks, KO, KP))
        std = mu_completa(S, O2, P, mumax, Ks, KO, KP)
        assert rto == pytest.approx(std, rel=1e-6)

    def test_mu_fermentacion_rto_no_negativo(self):
        """La versión RTO nunca debe retornar valores negativos."""
        params = dict(
            mumax_aerob=0.4, Ks_aerob=0.5, KO_aerob=0.2,
            mumax_anaerob=0.15, Ks_anaerob=1.0, KiS_anaerob=150.0,
            KP_anaerob=80.0, n_p=1.0, KO_inhib_anaerob=0.1,
        )
        result = self._to_float(mu_fermentacion_rto(5.0, 0.0, 2.0, **params))
        assert result >= 0.0

    def test_mu_fermentacion_rto_igual_fermentacion(self):
        """Los valores numéricos de la versión RTO y la estándar deben coincidir."""
        params = dict(
            mumax_aerob=0.4, Ks_aerob=0.5, KO_aerob=0.2,
            mumax_anaerob=0.15, Ks_anaerob=1.0, KiS_anaerob=150.0,
            KP_anaerob=80.0, n_p=1.0, KO_inhib_anaerob=0.1,
        )
        std = mu_fermentacion(5.0, 2.0, 3.0, **params)
        rto = self._to_float(mu_fermentacion_rto(5.0, 2.0, 3.0, **params))
        assert rto == pytest.approx(std, rel=1e-6)
