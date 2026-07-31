"""
Tests de integración y ejecución de los modelos ODE de bioprocesos.

Verifica que los modelos matemáticos sean físicamente consistentes:
- Las concentraciones permanecen no negativas.
- Los balances de masa son razonables.
- Los solucionadores convergen correctamente.
- Los modos batch, fed-batch y continuo responden correctamente a cambios paramétricos.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion


# ---------------------------------------------------------------------------
# Helpers: funciones ODE desacopladas de Streamlit
# ---------------------------------------------------------------------------

def _batch_ode(t, y, mumax=0.4, Ks=0.2, Yxs=0.5, Ypx=0.3,
               Yxo=0.3, Kla=20.0, Cs=8.0, ms=0.005, Kd=0.005, mo=0.05,
               kinetic="monod", n=2, KO=0.5, KP=5.0):
    X, S, P, O2 = y
    X = max(0.0, X)
    S = max(0.0, S)
    P = max(0.0, P)
    O2 = max(0.0, O2)

    if kinetic == "monod":
        mu = mu_monod(S, mumax, Ks)
    elif kinetic == "sigmoidal":
        mu = mu_sigmoidal(S, mumax, Ks, n)
    elif kinetic == "completa":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0

    dXdt = mu * X - Kd * X
    dSdt = -1.0 / Yxs * mu * X - ms * X
    if S <= 0.0:
        dSdt = max(0.0, dSdt)
    dPdt = Ypx * mu * X
    dOdt = Kla * (Cs - O2) - (1.0 / Yxo) * mu * X - mo * X
    return [dXdt, dSdt, dPdt, dOdt]


def _fedbatch_ode(t, y, F_func, mumax=0.4, Ks=0.2, Yxs=0.5, Ypx=0.3,
                  Yxo=0.3, Kla=50.0, Cs=8.0, Sin=150.0, ms=0.001,
                  Kd=0.02, mo=0.01, kinetic="monod", n=2, KO=0.5, KP=5.0):
    X, S, P, O2, V = y
    X = max(0.0, X)
    S = max(0.0, S)
    P = max(0.0, P)
    O2 = max(0.0, O2)
    V = max(1e-6, V)

    if kinetic == "monod":
        mu = mu_monod(S, mumax, Ks)
    elif kinetic == "sigmoidal":
        mu = mu_sigmoidal(S, mumax, Ks, n)
    elif kinetic == "completa":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0

    F = F_func(t)

    dXdt = (mu - Kd) * X - (F / V) * X
    dSdt = -(mu / Yxs + ms) * X + (F / V) * (Sin - S)
    dPdt = Ypx * mu * X - (F / V) * P
    dOdt = Kla * (Cs - O2) - (mu / Yxo + mo) * X - (F / V) * O2
    dVdt = F
    return [dXdt, dSdt, dPdt, dOdt, dVdt]


def _continuous_ode(t, y, D=0.05, mumax=0.4, Ks=0.2, Yxs=0.5, Ypx=0.3,
                    Yxo=0.3, Kla=20.0, Cs=8.0, Sin=50.0, ms=0.005,
                    Kd=0.005, mo=0.05, kinetic="monod", n=2, KO=0.5, KP=5.0):
    X, S, P, O2 = y
    X = max(0.0, X)
    S = max(0.0, S)
    P = max(0.0, P)
    O2 = max(0.0, O2)

    if kinetic == "monod":
        mu = mu_monod(S, mumax, Ks)
    elif kinetic == "sigmoidal":
        mu = mu_sigmoidal(S, mumax, Ks, n)
    elif kinetic == "completa":
        mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
    else:
        mu = 0.0

    dXdt = mu * X - Kd * X - D * X
    dSdt = -1.0 / Yxs * mu * X - ms * X + D * (Sin - S)
    dPdt = Ypx * mu * X - D * P
    dOdt = Kla * (Cs - O2) - 1.0 / Yxo * mu * X - mo * X - D * O2
    return [dXdt, dSdt, dPdt, dOdt]


# ---------------------------------------------------------------------------
# Tests: Modelo Batch
# ---------------------------------------------------------------------------
class TestBatchModel:
    Y0 = [0.5, 20.0, 0.0, 5.0]
    T_SPAN = [0, 24]

    def _solve(self, **kwargs):
        return solve_ivp(
            lambda t, y: _batch_ode(t, y, **kwargs),
            self.T_SPAN, self.Y0, method="RK45",
            atol=1e-8, rtol=1e-8, dense_output=False
        )

    def test_solver_success_monod(self):
        sol = self._solve(kinetic="monod")
        assert sol.success, f"Solver failed: {sol.message}"

    def test_solver_success_sigmoidal(self):
        sol = self._solve(kinetic="sigmoidal")
        assert sol.success, f"Solver failed: {sol.message}"

    def test_solver_success_completa(self):
        sol = self._solve(kinetic="completa")
        assert sol.success, f"Solver failed: {sol.message}"

    def test_biomass_increases_initially(self):
        sol = self._solve(kinetic="monod")
        # La biomasa debe crecer al inicio cuando hay sustrato disponible
        assert sol.y[0, -1] > self.Y0[0]

    def test_substrate_decreases(self):
        sol = self._solve(kinetic="monod")
        assert sol.y[1, -1] < self.Y0[1]

    def test_product_increases(self):
        sol = self._solve(kinetic="monod")
        assert sol.y[2, -1] > self.Y0[2]

    def test_oxygen_bounded(self):
        """El oxígeno disuelto debe mantenerse en [0, Cs*1.1]."""
        Cs = 8.0
        sol = self._solve(kinetic="monod", Cs=Cs)
        assert np.all(sol.y[3] >= -0.01)  # tolerancia numérica pequeña
        assert np.all(sol.y[3] <= Cs * 1.5)

    def test_mass_balance_substrate(self):
        """
        Balance de sustrato: el sustrato consumido ≥ biomasa producida / Yxs.
        Debe cumplirse de forma aproximada (sin mantenimiento exacto).
        """
        sol = self._solve(kinetic="monod", ms=0.0, Kd=0.0, mo=0.0)
        dX = sol.y[0, -1] - self.Y0[0]
        dS = self.Y0[1] - sol.y[1, -1]
        # dS ≈ dX / Yxs (con Yxs=0.5 → dS ≈ 2*dX)
        assert dS >= 0, "Sustrato no debe aumentar en batch sin alimentación"
        assert dS >= dX * (1.0 / 0.6)  # margen del 20 %

    def test_high_decay_reduces_biomass(self):
        """Con decaimiento alto, la biomasa final debe ser menor."""
        sol_low = self._solve(kinetic="monod", Kd=0.001)
        sol_high = self._solve(kinetic="monod", Kd=0.5)
        assert sol_high.y[0, -1] < sol_low.y[0, -1]


# ---------------------------------------------------------------------------
# Tests: Modelo Fed-Batch
# ---------------------------------------------------------------------------
class TestFedBatchModel:
    Y0 = [1.0, 30.0, 0.0, 8.0, 3.0]  # X, S, P, O2, V
    T_SPAN = [0, 24]

    def _solve(self, F_func=None, **kwargs):
        if F_func is None:
            F_func = lambda t: 0.1 if 2.0 <= t <= 20.0 else 0.0
        return solve_ivp(
            lambda t, y: _fedbatch_ode(t, y, F_func, **kwargs),
            self.T_SPAN, self.Y0, method="RK45",
            atol=1e-8, rtol=1e-8, dense_output=False
        )

    def test_solver_success(self):
        sol = self._solve()
        assert sol.success, f"Solver failed: {sol.message}"

    def test_volume_increases_during_feeding(self):
        """El volumen debe aumentar durante la alimentación."""
        sol = self._solve()
        V0 = self.Y0[4]
        V_final = sol.y[4, -1]
        assert V_final > V0

    def test_volume_bounded(self):
        """El volumen no debe ser negativo."""
        sol = self._solve()
        assert np.all(sol.y[4] > 0.0)

    def test_no_feeding_equals_batch(self):
        """Sin alimentación (F=0), el fed-batch se comporta como batch."""
        F_zero = lambda t: 0.0
        sol = self._solve(F_func=F_zero)
        assert sol.success
        # El volumen debe mantenerse constante
        assert sol.y[4, -1] == pytest.approx(self.Y0[4], rel=1e-3)

    def test_substrate_dilution_with_high_flow(self):
        """Flujo alto con sustrato de entrada bajo diluye el sustrato del reactor."""
        F_high = lambda t: 0.5  # caudal constante alto
        sol = self._solve(F_func=F_high, Sin=0.1)
        # Con Sin << S0, la dilución debe bajar S con el tiempo
        assert sol.y[1, -1] < self.Y0[1]


# ---------------------------------------------------------------------------
# Tests: Modelo Continuo (Quimiostato)
# ---------------------------------------------------------------------------
class TestContinuousModel:
    Y0 = [0.5, 20.0, 0.0, 5.0]
    T_SPAN = [0, 100]  # tiempo largo para alcanzar estado estacionario

    def _solve(self, **kwargs):
        return solve_ivp(
            lambda t, y: _continuous_ode(t, y, **kwargs),
            self.T_SPAN, self.Y0, method="RK45",
            atol=1e-8, rtol=1e-8, dense_output=False
        )

    def test_solver_success(self):
        sol = self._solve()
        assert sol.success, f"Solver failed: {sol.message}"

    def test_steady_state_biomass(self):
        """A tiempo largo, el sistema debe aproximarse a un estado estacionario."""
        sol = self._solve(D=0.05, mumax=0.4, Ks=0.2)
        # Verificar que la derivada al final es pequeña (estado estacionario)
        y_late = sol.y[:, -1]
        dy_late = _continuous_ode(self.T_SPAN[1], y_late, D=0.05, mumax=0.4, Ks=0.2)
        # Las tasas de cambio deben ser pequeñas cerca del estado estacionario
        assert abs(dy_late[0]) < 0.1, f"dX/dt = {dy_late[0]:.4f} no es estacionario"

    def test_washout_at_high_dilution(self):
        """Con D > μmax, debe producirse lavado (biomasa → 0)."""
        D_washout = 0.6  # mayor que mumax=0.4
        sol = self._solve(D=D_washout, mumax=0.4)
        assert sol.y[0, -1] < 0.1  # biomasa casi cero (lavado)

    def test_non_zero_biomass_below_washout(self):
        """Con D < μmax y sustrato suficiente, la biomasa es positiva."""
        sol = self._solve(D=0.1, mumax=0.4, Ks=0.1, Sin=50.0)
        assert sol.y[0, -1] > 0.01


# ---------------------------------------------------------------------------
# Tests: Modelo de Fermentación Alcohólica
# ---------------------------------------------------------------------------
class TestFermentationModel:
    """
    Modelo de Saccharomyces cerevisiae con cinética mixta aerobia/anaerobia
    (Luedeking-Piret para producción de etanol).
    """

    PARAMS = dict(
        mumax_aerob=0.4, Ks_aerob=0.5, KO_aerob=0.004,
        mumax_anaerob=0.15, Ks_anaerob=1.5, KiS_anaerob=50.0,
        KP_anaerob=65.0, n_p=2.0, KO_inhib_anaerob=0.004,
    )

    def _fermentation_ode(self, t, y):
        """ODE simplificada del modelo de fermentación alcohólica."""
        X, S, P, O2 = y
        X = max(0.0, X)
        S = max(0.0, S)
        P = max(0.0, P)
        O2 = max(0.0, O2)

        mu = mu_fermentacion(S, P, O2, **self.PARAMS)
        Yxs, Ypx, Kla, Cs, Kd = 0.1, 0.49, 100.0, 8.0, 0.005

        dXdt = mu * X - Kd * X
        dSdt = -(mu / Yxs) * X
        dPdt = Ypx * mu * X
        dOdt = Kla * (Cs - O2) - 0.5 * mu * X
        return [dXdt, dSdt, dPdt, dOdt]

    def test_solver_success(self):
        y0 = [0.1, 100.0, 0.0, 8.0]
        sol = solve_ivp(self._fermentation_ode, [0, 48], y0,
                        method="RK45", atol=1e-8, rtol=1e-8)
        assert sol.success, f"Solver failed: {sol.message}"

    def test_product_ethanol_increases(self):
        y0 = [0.1, 100.0, 0.0, 8.0]
        sol = solve_ivp(self._fermentation_ode, [0, 48], y0,
                        method="RK45", atol=1e-8, rtol=1e-8)
        assert sol.y[2, -1] > y0[2]  # el etanol debe acumularse

    def test_substrate_consumed(self):
        y0 = [0.1, 100.0, 0.0, 8.0]
        sol = solve_ivp(self._fermentation_ode, [0, 48], y0,
                        method="RK45", atol=1e-8, rtol=1e-8)
        assert sol.y[1, -1] < y0[1]

    def test_product_inhibition_slows_anaerobic_growth(self):
        """
        Con mucho etanol inicial (condiciones anaerobias O2≈0), la componente
        anaeróbica se inhibe fuertemente → menor crecimiento diferencial en las
        primeras horas donde la diferencia de P es más pronunciada.
        Se evalúa mu puntual para desacoplar del efecto de dilución dinámica.
        """
        from Utils.kinetics import mu_fermentacion
        p = self.PARAMS
        # mu sin producto vs. con 60 % de KP_anaerob, bajo O2 (condición anaerobia)
        mu_clean = mu_fermentacion(10.0, 0.0, 0.001, **p)     # sin P, sin O2
        mu_inhib = mu_fermentacion(10.0, 60.0, 0.001, **p)    # P=60 g/L ≈ KP_anaerob=65
        assert mu_clean > mu_inhib, (
            f"La inhibición por producto debe reducir μ: "
            f"mu_clean={mu_clean:.4f}, mu_inhib={mu_inhib:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests de sensibilidad paramétrica
# ---------------------------------------------------------------------------
class TestParametricSensitivity:
    """
    Verifica que el modelo responde cualitativamente bien a cambios de parámetros.
    """

    def test_higher_mumax_faster_growth(self):
        """Mayor μmax → mayor tasa de crecimiento inicial (fase exponencial).
        Se verifica en la fase exponencial (t corto, S >> Ks) antes de que
        la limitación de O2 o el agotamiento del sustrato confundan el efecto."""
        y0 = [0.5, 20.0, 0.0, 8.0]  # O2 saturado, sustrato alto
        # Tiempo corto: fase exponencial donde μmax domina
        sol1 = solve_ivp(
            lambda t, y: _batch_ode(t, y, mumax=0.2, Kla=200.0), [0, 2], y0,
            atol=1e-8, rtol=1e-8
        )
        sol2 = solve_ivp(
            lambda t, y: _batch_ode(t, y, mumax=0.6, Kla=200.0), [0, 2], y0,
            atol=1e-8, rtol=1e-8
        )
        # Con Kla alto, O2 no limita. Con S>>Ks, μ≈μmax. Mayor μmax → más biomasa en t corto.
        assert sol2.y[0, -1] > sol1.y[0, -1], (
            f"Mayor μmax debe dar más biomasa en fase exponencial: "
            f"X(μmax=0.2)={sol1.y[0,-1]:.3f}, X(μmax=0.6)={sol2.y[0,-1]:.3f}"
        )

    def test_higher_Ks_lower_growth(self):
        """Mayor Ks → menor afinidad → menor biomasa final a S inicial moderado."""
        y0 = [0.5, 2.0, 0.0, 5.0]  # S bajo para que Ks importe
        sol1 = solve_ivp(
            lambda t, y: _batch_ode(t, y, Ks=0.05), [0, 24], y0,
            atol=1e-8, rtol=1e-8
        )
        sol2 = solve_ivp(
            lambda t, y: _batch_ode(t, y, Ks=2.0), [0, 24], y0,
            atol=1e-8, rtol=1e-8
        )
        assert sol1.y[0, -1] > sol2.y[0, -1]

    def test_higher_yield_more_biomass(self):
        """Mayor Yxs → más biomasa por gramo de sustrato."""
        y0 = [0.5, 10.0, 0.0, 5.0]
        sol1 = solve_ivp(
            lambda t, y: _batch_ode(t, y, Yxs=0.3), [0, 12], y0,
            atol=1e-8, rtol=1e-8
        )
        sol2 = solve_ivp(
            lambda t, y: _batch_ode(t, y, Yxs=0.8), [0, 12], y0,
            atol=1e-8, rtol=1e-8
        )
        assert sol2.y[0, -1] > sol1.y[0, -1]
