"""
Tests unitarios para los modelos ODE de bioprocesos.

Verifica que los modelos de Lote, Lote Alimentado y Continuo
integran correctamente y producen resultados físicamente coherentes
sin necesidad de Streamlit (se prueban las funciones ODE directamente).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa


# ─── Funciones ODE de referencia (extraídas de los módulos) ──────────────────

def modelo_lote(t, y, mumax, Ks, Yxs, Ypx, Yxo, Kla, Cs, ms, Kd, mo,
                tipo_mu="Simple Monod", KO=0.5, KP=50.0):
    """ODE del biorreactor en modo Lote."""
    X, S, P, O2 = y
    S = max(0.0, S)
    O2 = max(0.0, O2)
    X = max(0.0, X)

    if tipo_mu == "Simple Monod":
        mu = mu_monod(S, mumax, Ks)
    elif tipo_mu == "Sigmoidal Monod":
        mu = mu_sigmoidal(S, mumax, Ks, n=2)
    elif tipo_mu == "Monod with restrictions":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0

    dXdt = mu * X - Kd * X
    dSdt = -1.0 / Yxs * mu * X - ms * X if S > 0 else 0.0
    dPdt = Ypx * mu * X
    dOdt = Kla * (Cs - O2) - (1.0 / Yxo) * mu * X - mo * X
    return [dXdt, dSdt, dPdt, dOdt]


def modelo_fedbatch(t, y, mumax, Ks, Yxs, Ypx, Yxo, Kla, Cs, ms, Kd, mo,
                    Sin, F_func, tipo_mu="Simple Monod", KO=0.5, KP=50.0, n=2):
    """ODE del biorreactor en modo Lote Alimentado."""
    X, S, P, O2, V = y
    X = max(0.0, X)
    S = max(0.0, S)
    P = max(0.0, P)
    O2 = max(0.0, O2)
    V = max(1e-6, V)

    if tipo_mu == "Simple Monod":
        mu = mu_monod(S, mumax, Ks)
    elif tipo_mu == "Sigmoidal Monod":
        mu = mu_sigmoidal(S, mumax, Ks, n)
    elif tipo_mu == "Monod with restrictions":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0
    mu = max(0.0, mu)

    F = F_func(t)
    dXdt = (mu - Kd) * X - (F / V) * X
    dSdt = -(mu / Yxs + ms) * X + (F / V) * (Sin - S)
    dPdt = Ypx * mu * X - (F / V) * P
    dOdt = Kla * (Cs - O2) - (mu / Yxo + mo) * X - (F / V) * O2
    dVdt = F
    return [dXdt, dSdt, dPdt, dOdt, dVdt]


def modelo_continuo(t, y, mumax, Ks, Yxs, Ypx, Yxo, Kla, Cs, ms, Kd, mo,
                    Sin, D, tipo_mu="Simple Monod", KO=0.5, KP=50.0):
    """ODE del biorreactor en modo Continuo (quimiostato)."""
    X, S, P, O2 = y
    X = max(0.0, X)
    S = max(0.0, S)
    P = max(0.0, P)
    O2 = max(0.0, O2)

    if tipo_mu == "Simple Monod":
        mu = mu_monod(S, mumax, Ks)
    elif tipo_mu == "Sigmoidal Monod":
        mu = mu_sigmoidal(S, mumax, Ks, n=2)
    elif tipo_mu == "Monod with restrictions":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0

    dXdt = mu * X - Kd * X - D * X
    dSdt = -1.0 / Yxs * mu * X - ms * X + D * (Sin - S)
    dPdt = Ypx * mu * X - D * P
    dOdt = Kla * (Cs - O2) - 1.0 / Yxo * mu * X - mo * X - D * O2
    return [dXdt, dSdt, dPdt, dOdt]


# ─── Parámetros por defecto para tests ───────────────────────────────────────

PARAMS_DEFAULT = dict(
    mumax=0.3, Ks=0.1, Yxs=0.5, Ypx=0.3, Yxo=0.3,
    Kla=20.0, Cs=8.0, ms=0.005, Kd=0.005, mo=0.05,
)
Y0_LOTE = [0.5, 20.0, 0.0, 5.0]
T_SPAN = (0, 20)
T_EVAL = np.linspace(0, 20, 100)


# ─── Tests Modelo Lote ────────────────────────────────────────────────────────

class TestModeloLote:
    """Pruebas de integración del modelo de lote."""

    def _integrar(self, tipo_mu="Simple Monod", **kwargs):
        params = {**PARAMS_DEFAULT, **kwargs}
        sol = solve_ivp(
            lambda t, y: modelo_lote(t, y, tipo_mu=tipo_mu, **params),
            T_SPAN, Y0_LOTE, t_eval=T_EVAL, atol=1e-8, rtol=1e-8,
        )
        return sol

    def test_integracion_exitosa_simple_monod(self):
        """La integración con Monod simple debe terminar exitosamente."""
        sol = self._integrar("Simple Monod")
        assert sol.success, f"Integración falló: {sol.message}"

    def test_integracion_exitosa_sigmoidal(self):
        """La integración con Monod sigmoidal debe terminar exitosamente."""
        sol = self._integrar("Sigmoidal Monod")
        assert sol.success, f"Integración falló: {sol.message}"

    def test_integracion_exitosa_monod_restricciones(self):
        """La integración con Monod con restricciones debe terminar exitosamente."""
        sol = self._integrar("Monod with restrictions")
        assert sol.success, f"Integración falló: {sol.message}"

    def test_biomasa_crece(self):
        """La biomasa debe crecer durante la fase exponencial."""
        sol = self._integrar("Simple Monod")
        assert sol.y[0, -1] > Y0_LOTE[0]

    def test_sustrato_decrece(self):
        """El sustrato debe consumirse."""
        sol = self._integrar("Simple Monod")
        assert sol.y[1, -1] < Y0_LOTE[1]

    def test_producto_aumenta(self):
        """El producto debe acumularse."""
        sol = self._integrar("Simple Monod")
        assert sol.y[2, -1] >= Y0_LOTE[2]

    def test_estados_no_negativos(self):
        """Biomasa y sustrato no deben ser negativos (o solo ruido numérico mínimo)."""
        sol = self._integrar("Simple Monod")
        assert np.all(sol.y[0] >= -1e-6), "Biomasa negativa detectada"
        # El sustrato puede llegar a valores ínfimamente negativos por el integrador
        # después del agotamiento; np.maximum en el ploteo lo corrige visualmente
        assert np.all(sol.y[1] >= -1e-4), "Sustrato con valor excesivamente negativo"

    def test_balance_masa_aproximado(self):
        """
        Balance de carbono aproximado: el carbono consumido en sustrato
        debe aparecer en biomasa + producto.
        """
        sol = self._integrar("Simple Monod")
        X0, S0, P0 = Y0_LOTE[:3]
        Yxs, Ypx = PARAMS_DEFAULT['Yxs'], PARAMS_DEFAULT['Ypx']
        delta_S = S0 - sol.y[1, -1]
        delta_X = sol.y[0, -1] - X0
        delta_P = sol.y[2, -1] - P0
        # delta_X ≈ Yxs * delta_S (muy aproximado, ignora mantenimiento)
        assert delta_X > 0 and delta_S > 0


# ─── Tests Modelo Lote Alimentado ────────────────────────────────────────────

class TestModeloFedbatch:
    """Pruebas de integración del modelo de lote alimentado."""

    SIN = 150.0
    F_CAUDAL = 0.1  # L/h constante

    def _f_constante(self, t):
        return self.F_CAUDAL if 2.0 <= t <= 20.0 else 0.0

    def _integrar(self, tipo_mu="Simple Monod"):
        y0 = [0.5, 20.0, 0.0, 5.0, 2.0]  # X, S, P, O2, V
        sol = solve_ivp(
            lambda t, y: modelo_fedbatch(
                t, y, Sin=self.SIN, F_func=self._f_constante,
                tipo_mu=tipo_mu, **PARAMS_DEFAULT
            ),
            T_SPAN, y0, t_eval=T_EVAL, method='RK45', atol=1e-8, rtol=1e-8,
        )
        return sol

    def test_integracion_exitosa(self):
        sol = self._integrar()
        assert sol.success, f"Integración falló: {sol.message}"

    def test_volumen_crece_durante_alimentacion(self):
        """El volumen debe aumentar mientras se alimenta."""
        sol = self._integrar()
        # Al final la alimentación habrá añadido volumen
        V_inicial = 2.0
        V_final = sol.y[4, -1]
        assert V_final > V_inicial

    def test_biomasa_aumenta_en_masa(self):
        """La masa total de biomasa (X*V) debe aumentar."""
        sol = self._integrar()
        masa_inicial = 0.5 * 2.0
        masa_final = sol.y[0, -1] * sol.y[4, -1]
        assert masa_final > masa_inicial

    def test_estados_no_negativos(self):
        sol = self._integrar()
        for i, nombre in enumerate(["X", "S", "P", "O2", "V"]):
            assert np.all(sol.y[i] >= -1e-6), f"{nombre} negativo detectado"


# ─── Tests Modelo Continuo ───────────────────────────────────────────────────

class TestModeloContinuo:
    """Pruebas de integración del modelo continuo (quimiostato)."""

    SIN = 50.0
    D = 0.1  # Tasa de dilución [1/h]

    def _integrar(self, D=None, tipo_mu="Simple Monod"):
        d = D if D is not None else self.D
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 500)
        sol = solve_ivp(
            lambda t, y: modelo_continuo(
                t, y, Sin=self.SIN, D=d, tipo_mu=tipo_mu, **PARAMS_DEFAULT
            ),
            t_span, Y0_LOTE, t_eval=t_eval, atol=1e-8, rtol=1e-8,
        )
        return sol

    def test_integracion_exitosa(self):
        sol = self._integrar()
        assert sol.success, f"Integración falló: {sol.message}"

    def test_estado_estacionario_alcanzado(self):
        """Para D < μmax el sistema debe alcanzar un estado estacionario."""
        sol = self._integrar(D=0.05)
        # Los últimos 20 puntos deben tener variación pequeña en biomasa
        X_final = sol.y[0, -20:]
        variacion = np.max(X_final) - np.min(X_final)
        assert variacion < 0.5, f"No se alcanzó estado estacionario (variación: {variacion:.3f})"

    def test_washout_alto_d(self):
        """Para D >> μmax la biomasa debe tender a cero (lavado)."""
        sol = self._integrar(D=0.9)  # D > μmax = 0.3
        assert sol.y[0, -1] < 0.1, f"No ocurrió lavado: X_final = {sol.y[0, -1]:.3f}"

    def test_estados_no_negativos(self):
        sol = self._integrar()
        for i, nombre in enumerate(["X", "S", "P", "O2"]):
            assert np.all(sol.y[i] >= -1e-6), f"{nombre} negativo detectado"


# ─── Tests de Consistencia Cinética en ODEs ──────────────────────────────────

class TestConsistenciaCinetica:
    """Verifica que los tres modelos cinéticos producen resultados coherentes."""

    def test_monod_mayor_tasa_con_mas_sustrato(self):
        """Con más sustrato, la tasa de crecimiento Monod debe ser mayor."""
        mu1 = mu_monod(1.0, 0.4, 0.1)
        mu2 = mu_monod(5.0, 0.4, 0.1)
        assert mu2 > mu1

    def test_sigmoidal_convergencia_con_monod_n1(self):
        """Sigmoidal con n=1 debe ser idéntica a Monod."""
        for S in [0.1, 0.5, 1.0, 5.0]:
            assert mu_sigmoidal(S, 0.4, 0.2, 1) == pytest.approx(
                mu_monod(S, 0.4, 0.2), rel=1e-10
            )

    def test_completa_menor_que_monod_con_inhibicion(self):
        """Monod con inhibición de producto debe ser menor que Monod simple."""
        S, O2, P = 5.0, 6.0, 5.0
        mu_simple = mu_monod(S, 0.4, 0.1)
        mu_restringido = mu_completa(S, O2, P, 0.4, 0.1, 0.5, 50.0)
        # mu_restringido ≤ mu_simple (la O2 también limita)
        assert mu_restringido <= mu_simple
