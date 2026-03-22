# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from io import BytesIO
import traceback

PAGE_ID = 'fermentation'

# ---------------------------------------------------------------------------
# Local kinetic function definitions
# ---------------------------------------------------------------------------

def mu_monod(S, mumax, Ks):
    S = max(0.0, S); Ks = max(1e-9, Ks)
    den = Ks + S
    if den < 1e-9: return 0.0
    return mumax * S / den


def mu_sigmoidal(S, mumax, Ks, n):
    S = max(0.0, S); Ks = max(1e-9, Ks); n = max(1e-6, n)
    try:
        S_n = S**n; Ks_n = Ks**n; den = Ks_n + S_n
        if den < 1e-9: return 0.0
        return mumax * S_n / den
    except OverflowError:
        return mumax if S > Ks else 0.0


def mu_completa(S, O2, P, mumax, Ks, KO, KP_gen):
    S = max(0.0, S); O2_mgL = max(0.0, O2); P = max(0.0, P)
    Ks = max(1e-9, Ks); KO_mgL = max(1e-9, KO); KP_gen = max(1e-9, KP_gen)
    term_S = S / (Ks + S)
    term_O2 = O2_mgL / (KO_mgL + O2_mgL)
    term_P = KP_gen / (KP_gen + P)
    return max(0.0, mumax * term_S * term_O2 * term_P)


def mu_fermentacion(S, P, O2, mumax_aerob, Ks_aerob, KO_aerob, mumax_anaerob,
                    Ks_anaerob, KiS_anaerob, KP_anaerob, n_p, KO_inhib_anaerob,
                    considerar_O2=None):
    S = max(0.0, S); P = max(0.0, P); O2_mgL = max(0.0, O2)
    Ks_aerob = max(1e-9, Ks_aerob); KO_aerob_mgL = max(1e-9, KO_aerob)
    Ks_anaerob = max(1e-9, Ks_anaerob); KiS_anaerob = max(1e-9, KiS_anaerob)
    KP_anaerob = max(1e-9, KP_anaerob); n_p = max(1e-6, n_p)
    KO_inhib_anaerob_mgL = max(1e-9, KO_inhib_anaerob)

    den_O2_aer = KO_aerob_mgL + O2_mgL
    term_O2_aer = O2_mgL / den_O2_aer if den_O2_aer > 1e-9 else 0.0
    mu_aer = max(0.0, mumax_aerob * (S / (Ks_aerob + S)) * term_O2_aer)

    den_S_an = Ks_anaerob + S + (S**2 / KiS_anaerob)
    term_S_an = S / max(1e-9, den_S_an)
    term_P_base = max(0.0, 1.0 - (P / KP_anaerob))
    term_P_an = term_P_base**n_p
    den_O2_inhib = KO_inhib_anaerob_mgL + O2_mgL
    term_O2_inhib_an = KO_inhib_anaerob_mgL / den_O2_inhib if den_O2_inhib > 1e-9 else 0.0
    mu_anaer = max(0.0, mumax_anaerob * term_S_an * term_P_an * term_O2_inhib_an)

    if considerar_O2 is True:
        return mu_aer
    elif considerar_O2 is False:
        return mu_anaer
    else:
        return mu_aer + mu_anaer


# ---------------------------------------------------------------------------
# Helper to build a slider row
# ---------------------------------------------------------------------------

def _slider(label, id_, min_, max_, value, step, marks=None):
    return html.Div([
        html.Label(label, style={'fontSize': '0.85rem'}),
        dcc.Slider(
            id=id_, min=min_, max=max_, value=value, step=step,
            tooltip={'placement': 'bottom', 'always_visible': False},
            marks=marks if marks else {},
        ),
    ], className='mb-2')


# ---------------------------------------------------------------------------
# get_params_layout
# ---------------------------------------------------------------------------

def get_params_layout():
    return html.Div([
        # Section 1 – Kinetic Model
        html.H6("1. Kinetic Model and Parameters", className='mt-2 fw-bold'),

        html.Label("Kinetic Model", style={'fontSize': '0.85rem'}),
        dcc.Dropdown(
            id='fermentation-tipo-mu',
            options=[
                {'label': 'Fermentation', 'value': 'Fermentation'},
                {'label': 'Switched Fermentation', 'value': 'Switched Fermentation'},
                {'label': 'Simple Monod', 'value': 'Simple Monod'},
                {'label': 'Sigmoidal Monod', 'value': 'Sigmoidal Monod'},
                {'label': 'Monod with restrictions', 'value': 'Monod with restrictions'},
            ],
            value='Fermentation',
            clearable=False,
        ),
        html.Div(className='mb-2'),

        _slider("Base Ks [g/L]", 'fermentation-ks', 0.01, 10.0, 1.0, 0.1),

        # Conditional: Simple Monod
        html.Div(id='fermentation-simple-monod-params', style={'display': 'none'}, children=[
            _slider("μmax [1/h]", 'fermentation-mumax-simple', 0.1, 1.0, 0.4, 0.05),
        ]),

        # Conditional: Sigmoidal Monod
        html.Div(id='fermentation-sigmoidal-params', style={'display': 'none'}, children=[
            _slider("μmax [1/h]", 'fermentation-mumax-sig', 0.1, 1.0, 0.4, 0.05),
            _slider("Sigmoidal power (n)", 'fermentation-n-sig', 1, 5, 2, 1),
        ]),

        # Conditional: Monod with restrictions
        html.Div(id='fermentation-restrictions-params', style={'display': 'none'}, children=[
            _slider("μmax [1/h]", 'fermentation-mumax-restr', 0.1, 1.0, 0.4, 0.05),
            _slider("KO (O2 restriction) [mg/L]", 'fermentation-ko-restr', 0.01, 5.0, 0.1, 0.01),
            _slider("KP (Generic product inhibition) [g/L]", 'fermentation-kp-gen', 1.0, 100.0, 50.0, 1.0),
        ]),

        # Conditional: Switched Fermentation
        html.Div(id='fermentation-switched-params', style={'display': 'none'}, children=[
            dbc.Alert("Switched model: Uses mu_fermentation with mumax_aero/anaero and optional O2.",
                      color='info', className='p-1 small'),
            _slider("μmax (Aerobic Phase) [1/h]", 'fermentation-mumax-aero-c', 0.1, 1.0, 0.45, 0.05),
            _slider("μmax (Anaerobic Phase) [1/h]", 'fermentation-mumax-anaero-c', 0.05, 0.8, 0.15, 0.05),
            _slider("KiS (Substrate Inhibition) [g/L]", 'fermentation-kis-c', 50.0, 500.0, 150.0, 10.0),
            _slider("KP (Ethanol Inhibition) [g/L]", 'fermentation-kp-c', 20.0, 150.0, 80.0, 5.0),
            _slider("Ethanol Inhib. Exponent (n_p)", 'fermentation-np-c', 0.5, 3.0, 1.0, 0.1),
            _slider("KO (O2 aerobic affinity) [mg/L]", 'fermentation-ko-ferm-c', 0.01, 5.0, 0.1, 0.01),
            _slider("KO_inhib_anaerob (O2 inhibition on anaerobic μ) [mg/L]",
                    'fermentation-ko-inhib-c', 0.01, 5.0, 0.1, 0.01),
        ]),

        # Conditional: Fermentation (mixed)
        html.Div(id='fermentation-fermentation-params', style={'display': 'block'}, children=[
            dbc.Alert("Mixed Model: mu = mu1(aerobic) + mu2(anaerobic).",
                      color='info', className='p-1 small'),
            html.Label("mu1 parameters (Aerobic):", className='fw-semibold small'),
            _slider("μmax_aerob [1/h]", 'fermentation-mumax-aerob-m', 0.1, 1.0, 0.4, 0.05),
            _slider("Ks_aerob [g/L]", 'fermentation-ks-aerob-m', 0.01, 10.0, 0.5, 0.05),
            _slider("KO_aerob (O2 affinity) [mg/L]", 'fermentation-ko-aerob-m', 0.01, 5.0, 0.2, 0.01),
            html.Label("mu2 parameters (Anaerobic/Fermentative):", className='fw-semibold small'),
            _slider("μmax_anaerob [1/h]", 'fermentation-mumax-anaerob-m', 0.05, 0.8, 0.15, 0.05),
            _slider("Ks_anaerob [g/L]", 'fermentation-ks-anaerob-m', 0.1, 20.0, 1.0, 0.1),
            _slider("KiS_anaerob [g/L]", 'fermentation-kis-anaerob-m', 50.0, 500.0, 150.0, 10.0),
            _slider("KP_anaerob (Inhib. Etanol) [g/L]", 'fermentation-kp-anaerob-m', 20.0, 150.0, 80.0, 5.0),
            _slider("Ethanol Inhib. Exponent (n_p)", 'fermentation-np-m', 0.5, 3.0, 1.0, 0.1),
            _slider("KO_inhib_anaerob (Inhib. O2) [mg/L]", 'fermentation-ko-inhib-m', 0.01, 5.0, 0.1, 0.01),
        ]),

        html.Hr(),
        # Section 2 – Stoichiometric / Maintenance
        html.H6("2. Stoichiometric and Maintenance Parameters", className='fw-bold'),
        _slider("Yxs (Biomass/Substrate) [g/g]", 'fermentation-yxs', 0.05, 0.6, 0.1, 0.01),
        _slider("Yps (Ethanol/Substrate) [g/g]", 'fermentation-yps', 0.1, 0.51, 0.45, 0.01),
        _slider("Yxo (Biomass/O2) [gX/gO2]", 'fermentation-yxo', 0.1, 2.0, 0.8, 0.1),
        _slider("α (Associated to growth) [g P / g X]", 'fermentation-alpha', 0.0, 10.0, 4.5, 0.1),
        _slider("β (Not associated to growth) [g P / g X / h]", 'fermentation-beta', 0.0, 1.5, 0.40, 0.01),
        _slider("ms (Substrate Maintenance) [g S / g X / h]", 'fermentation-ms', 0.0, 0.2, 0.02, 0.01),
        _slider("mo (O2 Maintenance) [gO2/gX/h]", 'fermentation-mo', 0.0, 0.1, 0.01, 0.005),
        _slider("Kd (Biomass Decay) [1/h]", 'fermentation-kd', 0.0, 0.1, 0.01, 0.005),
        _slider("KO_inhib_prod (O2 Inhib in Ethanol Prod) [mg/L]",
                'fermentation-ko-inhib-prod', 0.001, 1.0, 0.05, 0.005),

        html.Hr(),
        # Section 3 – Oxygen Transfer
        html.H6("3. Oxygen Transfer", className='fw-bold'),
        _slider("kLa [1/h]", 'fermentation-kla', 10.0, 400.0, 100.0, 10.0),
        _slider("Saturated O2 (Cs) [mg/L]", 'fermentation-cs', 0.01, 15.0, 0.09, 0.01),

        html.Hr(),
        # Section 4 – Feeding and Operation Phases
        html.H6("4. Feeding and Operation Phases", className='fw-bold'),
        _slider("End Initial Batch Phase [h]", 'fermentation-t-batch-fin', 1.0, 30.0, 4.0, 1.0),
        _slider("Start Feeding [h]", 'fermentation-t-alim-ini', 1.0, 48.0, 4.01, 0.5),
        _slider("End Feeding (Start Final Batch) [h]", 'fermentation-t-alim-fin', 2.0, 60.0, 9.0, 1.0),
        _slider("Total Simulation Time [h]", 'fermentation-t-total', 3.0, 100.0, 16.0, 1.0),
        _slider("O2 Objective/Ref Level (Initial Batch Phase) [mg/L]",
                'fermentation-o2-control', 0.01, 15.0, 0.08, 0.01),

        html.Label("Feeding Strategy", style={'fontSize': '0.85rem'}),
        dcc.Dropdown(
            id='fermentation-strat',
            options=[
                {'label': 'Linear', 'value': 'Linear'},
                {'label': 'Exponential', 'value': 'Exponential'},
                {'label': 'Constant', 'value': 'Constant'},
                {'label': 'Step', 'value': 'Step'},
            ],
            value='Linear',
            clearable=False,
        ),
        html.Div(className='mb-2'),

        _slider("Substrate in Feeding (Sin) [g/L]", 'fermentation-sin', 10.0, 700.0, 250.0, 10.0),
        _slider("Base Flow (or Initial) [L/h]", 'fermentation-fbase', 0.01, 5.0, 0.01, 0.01),

        # Conditional: Linear feeding
        html.Div(id='fermentation-linear-feed-params', style={'display': 'block'}, children=[
            _slider("Final Flow (Linear) [L/h]", 'fermentation-ffin-lin', 0.01, 10.0, 0.11, 0.01),
        ]),

        # Conditional: Exponential feeding
        html.Div(id='fermentation-exp-feed-params', style={'display': 'none'}, children=[
            _slider("Constant Growth Exp. (k_exp) [1/h]", 'fermentation-kexp', 0.01, 0.5, 0.1, 0.01),
        ]),

        html.Hr(),
        # Section 5 – Initial Conditions
        html.H6("5. Initial Conditions", className='fw-bold'),
        html.Label("Initial Volume [L]", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-v0', type='number', min=0.1, max=100.0, value=0.25, step=0.05,
                  className='mb-2'),
        html.Label("Initial Biomass [g/L]", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-x0', type='number', min=0.05, max=10.0, value=1.20, step=0.05,
                  className='mb-2'),
        html.Label("Initial Substrate [g/L]", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-s0', type='number', min=10.0, max=200.0, value=20.0, step=1.0,
                  className='mb-2'),
        html.Label("Initial Ethanol [g/L]", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-p0', type='number', min=0.0, max=50.0, value=0.0, step=0.1,
                  className='mb-2'),
        html.Label("Initial O2 [mg/L]", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-o0', type='number', min=0.0, max=15.0, value=0.08, step=0.01,
                  className='mb-2'),

        html.Hr(),
        # Section 6 – Solver
        html.H6("6. Solver Parameters", className='fw-bold'),
        html.Label("Absolute Tolerance (atol)", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-atol', type='number', min=1e-9, max=1e-3, value=1e-6, step=1e-7,
                  className='mb-2'),
        html.Label("Relative Tolerance (rtol)", style={'fontSize': '0.85rem'}),
        dbc.Input(id='fermentation-rtol', type='number', min=1e-9, max=1e-3, value=1e-6, step=1e-7,
                  className='mb-2'),

        html.Hr(),
        dbc.Button("Run Simulation", id='fermentation-run-btn', n_clicks=0,
                   color='success', className='w-100 mb-2'),
        html.Button("Download Simulation Data as Excel", id='fermentation-download-btn',
                    n_clicks=0, className='btn btn-primary w-100 mt-2'),
    ], style={'padding': '0.5rem'})


# ---------------------------------------------------------------------------
# get_content_layout
# ---------------------------------------------------------------------------

def get_content_layout():
    return html.Div([
        html.H2("Fed-Batch Alcoholic Fermentation Simulation"),
        dcc.Markdown(
            "This model simulates an alcoholic fermentation that begins in **batch (aerobic phase)**, "
            "continues in **fed-batch (transition/anaerobic phase)**, and ends in "
            "**batch (anaerobic phase)** to deplete the substrate. Select the kinetic model."
        ),
        html.H4("Simulation results"),
        dcc.Graph(id='fermentation-graph', figure=go.Figure()),
        html.Div(id='fermentation-results-div'),
        dcc.Download(id='fermentation-download'),
        dcc.Store(id='fermentation-sim-data'),
    ])


# ---------------------------------------------------------------------------
# register_callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):

    # ------------------------------------------------------------------
    # Callback 1: toggle kinetic model conditional sections
    # ------------------------------------------------------------------
    @app.callback(
        Output('fermentation-simple-monod-params', 'style'),
        Output('fermentation-sigmoidal-params', 'style'),
        Output('fermentation-restrictions-params', 'style'),
        Output('fermentation-switched-params', 'style'),
        Output('fermentation-fermentation-params', 'style'),
        Input('fermentation-tipo-mu', 'value'),
    )
    def toggle_kinetic_sections(tipo_mu):
        show = {'display': 'block'}
        hide = {'display': 'none'}
        return (
            show if tipo_mu == 'Simple Monod' else hide,
            show if tipo_mu == 'Sigmoidal Monod' else hide,
            show if tipo_mu == 'Monod with restrictions' else hide,
            show if tipo_mu == 'Switched Fermentation' else hide,
            show if tipo_mu == 'Fermentation' else hide,
        )

    # ------------------------------------------------------------------
    # Callback 2: toggle feeding strategy conditional sections
    # ------------------------------------------------------------------
    @app.callback(
        Output('fermentation-linear-feed-params', 'style'),
        Output('fermentation-exp-feed-params', 'style'),
        Input('fermentation-strat', 'value'),
    )
    def toggle_feed_sections(strat):
        show = {'display': 'block'}
        hide = {'display': 'none'}
        return (
            show if strat == 'Linear' else hide,
            show if strat == 'Exponential' else hide,
        )

    # ------------------------------------------------------------------
    # Callback 3: main simulation
    # ------------------------------------------------------------------
    @app.callback(
        Output('fermentation-graph', 'figure'),
        Output('fermentation-results-div', 'children'),
        Output('fermentation-sim-data', 'data'),
        Input('fermentation-run-btn', 'n_clicks'),
        # Kinetic model
        State('fermentation-tipo-mu', 'value'),
        State('fermentation-ks', 'value'),
        # Simple Monod
        State('fermentation-mumax-simple', 'value'),
        # Sigmoidal Monod
        State('fermentation-mumax-sig', 'value'),
        State('fermentation-n-sig', 'value'),
        # Monod with restrictions
        State('fermentation-mumax-restr', 'value'),
        State('fermentation-ko-restr', 'value'),
        State('fermentation-kp-gen', 'value'),
        # Switched Fermentation
        State('fermentation-mumax-aero-c', 'value'),
        State('fermentation-mumax-anaero-c', 'value'),
        State('fermentation-kis-c', 'value'),
        State('fermentation-kp-c', 'value'),
        State('fermentation-np-c', 'value'),
        State('fermentation-ko-ferm-c', 'value'),
        State('fermentation-ko-inhib-c', 'value'),
        # Fermentation (mixed)
        State('fermentation-mumax-aerob-m', 'value'),
        State('fermentation-ks-aerob-m', 'value'),
        State('fermentation-ko-aerob-m', 'value'),
        State('fermentation-mumax-anaerob-m', 'value'),
        State('fermentation-ks-anaerob-m', 'value'),
        State('fermentation-kis-anaerob-m', 'value'),
        State('fermentation-kp-anaerob-m', 'value'),
        State('fermentation-np-m', 'value'),
        State('fermentation-ko-inhib-m', 'value'),
        # Stoichiometric / Maintenance
        State('fermentation-yxs', 'value'),
        State('fermentation-yps', 'value'),
        State('fermentation-yxo', 'value'),
        State('fermentation-alpha', 'value'),
        State('fermentation-beta', 'value'),
        State('fermentation-ms', 'value'),
        State('fermentation-mo', 'value'),
        State('fermentation-kd', 'value'),
        State('fermentation-ko-inhib-prod', 'value'),
        # Oxygen Transfer
        State('fermentation-kla', 'value'),
        State('fermentation-cs', 'value'),
        # Feeding and Operation
        State('fermentation-t-batch-fin', 'value'),
        State('fermentation-t-alim-ini', 'value'),
        State('fermentation-t-alim-fin', 'value'),
        State('fermentation-t-total', 'value'),
        State('fermentation-o2-control', 'value'),
        State('fermentation-strat', 'value'),
        State('fermentation-sin', 'value'),
        State('fermentation-fbase', 'value'),
        State('fermentation-ffin-lin', 'value'),
        State('fermentation-kexp', 'value'),
        # Initial Conditions
        State('fermentation-v0', 'value'),
        State('fermentation-x0', 'value'),
        State('fermentation-s0', 'value'),
        State('fermentation-p0', 'value'),
        State('fermentation-o0', 'value'),
        # Solver
        State('fermentation-atol', 'value'),
        State('fermentation-rtol', 'value'),
        prevent_initial_call=True,
    )
    def run_simulation(
        n_clicks,
        tipo_mu, Ks,
        mumax_simple,
        mumax_sig, n_sig,
        mumax_restr, ko_restr, kp_gen,
        mumax_aero_c, mumax_anaero_c, kis_c, kp_c, np_c, ko_ferm_c, ko_inhib_c,
        mumax_aerob_m, ks_aerob_m, ko_aerob_m,
        mumax_anaerob_m, ks_anaerob_m, kis_anaerob_m, kp_anaerob_m, np_m, ko_inhib_m,
        Yxs, Yps, Yxo, alpha_lp, beta_lp, ms, mo, Kd, ko_inhib_prod,
        Kla, Cs,
        t_batch_fin, t_alim_ini, t_alim_fin, t_total, o2_control,
        estrategia, Sin, F_base, F_fin_lin, k_exp,
        V0, X0, S0, P0, O0,
        atol, rtol,
    ):
        # ---- Provide safe defaults for None values ----
        def _f(v, default): return v if v is not None else default

        tipo_mu    = _f(tipo_mu, 'Fermentation')
        Ks         = _f(Ks, 1.0)
        Yxs        = _f(Yxs, 0.1);   Yps        = _f(Yps, 0.45)
        Yxo        = _f(Yxo, 0.8);   alpha_lp   = _f(alpha_lp, 4.5)
        beta_lp    = _f(beta_lp, 0.40); ms       = _f(ms, 0.02)
        mo         = _f(mo, 0.01);    Kd         = _f(Kd, 0.01)
        ko_inhib_prod = _f(ko_inhib_prod, 0.05)
        Kla        = _f(Kla, 100.0);  Cs         = _f(Cs, 0.09)
        t_batch_fin  = _f(t_batch_fin, 4.0)
        t_alim_ini   = _f(t_alim_ini, 4.01)
        t_alim_fin   = _f(t_alim_fin, 9.0)
        t_total      = _f(t_total, 16.0)
        o2_control   = _f(o2_control, 0.08)
        estrategia   = _f(estrategia, 'Linear')
        Sin          = _f(Sin, 250.0); F_base    = _f(F_base, 0.01)
        F_fin_lin    = _f(F_fin_lin, 0.11); k_exp = _f(k_exp, 0.1)
        V0  = _f(V0,  0.25); X0 = _f(X0, 1.20)
        S0  = _f(S0,  20.0); P0 = _f(P0, 0.0)
        O0  = _f(O0,  0.08)
        atol = _f(atol, 1e-6); rtol = _f(rtol, 1e-6)

        # ---- Build params dict ----
        params = {
            "tipo_mu": tipo_mu, "Ks": Ks,
            "Yxs": Yxs, "Yps": Yps, "Yxo": Yxo,
            "alpha_lp": alpha_lp, "beta_lp": beta_lp,
            "ms": ms, "mo": mo, "Kd": Kd,
            "Kla": Kla, "Cs": Cs, "Sin": Sin,
            "t_batch_inicial_fin": t_batch_fin,
            "KO_inhib_prod": ko_inhib_prod,
        }

        if tipo_mu == 'Simple Monod':
            params["mumax"] = _f(mumax_simple, 0.4)
        elif tipo_mu == 'Sigmoidal Monod':
            params["mumax"] = _f(mumax_sig, 0.4)
            params["n_sig"] = _f(n_sig, 2)
        elif tipo_mu == 'Monod with restrictions':
            params["mumax"]  = _f(mumax_restr, 0.4)
            params["KO"]     = _f(ko_restr, 0.1)
            params["KP_gen"] = _f(kp_gen, 50.0)
        elif tipo_mu == 'Switched Fermentation':
            params["mumax_aero"]      = _f(mumax_aero_c, 0.45)
            params["mumax_anaero"]    = _f(mumax_anaero_c, 0.15)
            params["Ks_aerob"]        = Ks
            params["KO_aerob"]        = _f(ko_ferm_c, 0.1)
            params["Ks_anaerob"]      = Ks
            params["KiS_anaerob"]     = _f(kis_c, 150.0)
            params["KP_anaerob"]      = _f(kp_c, 80.0)
            params["n_p"]             = _f(np_c, 1.0)
            params["KO_inhib_anaerob"]= _f(ko_inhib_c, 0.1)
            params["O2_controlado"]   = o2_control
        elif tipo_mu == 'Fermentation':
            params["mumax_aerob"]     = _f(mumax_aerob_m, 0.4)
            params["Ks_aerob"]        = _f(ks_aerob_m, 0.5)
            params["KO_aerob"]        = _f(ko_aerob_m, 0.2)
            params["mumax_anaerob"]   = _f(mumax_anaerob_m, 0.15)
            params["Ks_anaerob"]      = _f(ks_anaerob_m, 1.0)
            params["KiS_anaerob"]     = _f(kis_anaerob_m, 150.0)
            params["KP_anaerob"]      = _f(kp_anaerob_m, 80.0)
            params["n_p"]             = _f(np_m, 1.0)
            params["KO_inhib_anaerob"]= _f(ko_inhib_m, 0.1)

        # ---- Flow function (closure) ----
        def calcular_flujo(t):
            if t_alim_ini <= t <= t_alim_fin:
                if estrategia == 'Constant':
                    return F_base
                elif estrategia == 'Exponential':
                    try:
                        return min(F_base * np.exp(k_exp * (t - t_alim_ini)), F_base * 100)
                    except OverflowError:
                        return F_base * 100
                elif estrategia == 'Step':
                    t_medio = t_alim_ini + (t_alim_fin - t_alim_ini) / 2
                    return F_base * 2 if t > t_medio else F_base
                elif estrategia == 'Linear':
                    delta_t = t_alim_fin - t_alim_ini
                    if delta_t > 1e-6:
                        slope = (F_fin_lin - F_base) / delta_t
                        return max(0, F_base + slope * (t - t_alim_ini))
                    return F_base
            return 0.0

        # ---- ODE model ----
        def modelo_fermentacion(t, y, params):
            X, S, P, O2, V = y
            X = max(1e-9, X); S = max(0, S); P = max(0, P)
            O2 = max(0, O2); V = max(1e-6, V)

            es_lote_inicial = (t < params["t_batch_inicial_fin"])
            mu = 0.0; mu_aer = 0.0; mu_anaer = 0.0

            if params["tipo_mu"] == "Fermentation":
                mu_aer = mu_fermentacion(S, P, O2,
                    params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"],
                    0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                mu_anaer = mu_fermentacion(S, P, O2,
                    0, 1, float('inf'),
                    params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"],
                    params["KP_anaerob"], params.get("n_p", 1.0),
                    params["KO_inhib_anaerob"], considerar_O2=False)
                mu = mu_aer + mu_anaer
            elif params["tipo_mu"] == "Switched Fermentation":
                if es_lote_inicial:
                    current_O2 = params.get("O2_controlado", O2)
                    mu_aer = mu_fermentacion(S, P, current_O2,
                        params["mumax_aero"], params["Ks_aerob"], params["KO_aerob"],
                        0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                    mu = mu_aer
                else:
                    mu_anaer = mu_fermentacion(S, P, O2,
                        0, 1, float('inf'),
                        params["mumax_anaero"], params["Ks_anaerob"],
                        params["KiS_anaerob"], params["KP_anaerob"],
                        params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
                    mu = mu_anaer
            elif params["tipo_mu"] == "Simple Monod":
                mu = mu_monod(S, params.get("mumax", 0.0), params["Ks"])
                mu_aer = mu
            elif params["tipo_mu"] == "Sigmoidal Monod":
                mu = mu_sigmoidal(S, params.get("mumax", 0.0), params["Ks"],
                                  params.get("n_sig", 1))
                mu_aer = mu
            elif params["tipo_mu"] == "Monod with restrictions":
                mu = mu_completa(S, O2, P, params.get("mumax", 0.0), params["Ks"],
                                 params.get("KO", 0.1), params.get("KP_gen", 50.0))
                mu_aer = mu

            mu = max(0, mu); mu_aer = max(0, mu_aer); mu_anaer = max(0, mu_anaer)
            mu_net = mu - params["Kd"]

            F = calcular_flujo(t)

            qP = params["alpha_lp"] * mu_anaer + params["beta_lp"]
            qP = max(0.0, qP)
            rate_P = qP * X

            consumo_S_X    = (mu_aer / params["Yxs"]) * X if params["Yxs"] > 1e-6 else 0
            consumo_S_P    = (qP / params["Yps"]) * X if params["Yps"] > 1e-6 else 0
            consumo_S_maint = params["ms"] * X
            rate_S = consumo_S_X + consumo_S_P + consumo_S_maint

            consumo_O2_X    = (mu_aer / params["Yxo"]) * X if params["Yxo"] > 1e-6 else 0
            consumo_O2_maint = params["mo"] * X
            OUR_mg = (consumo_O2_X + consumo_O2_maint) * 1000.0

            dXdt = mu_net * X - (F / V) * X
            dSdt = -rate_S + (F / V) * (params["Sin"] - S)
            dPdt = rate_P - (F / V) * P
            dVdt = F

            if es_lote_inicial:
                dOdt = 0.0
            else:
                OTR = params["Kla"] * (params["Cs"] - O2)
                dOdt = OTR - OUR_mg - (F / V) * O2

            return [dXdt, dSdt, dPdt, dOdt, dVdt]

        # ---- Run solver ----
        try:
            y0 = [X0, S0, P0, O0, V0]
            t_span = [0, t_total]
            num_pts = max(500, int(t_total * 25) + 1)
            t_eval = np.linspace(0, t_total, num_pts)

            sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval,
                            method='RK45', atol=atol, rtol=rtol,
                            args=(params,), max_step=0.5)
            if not sol.success:
                sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval,
                                method='BDF', atol=atol, rtol=rtol, args=(params,))
                if not sol.success:
                    err_fig = go.Figure()
                    err_fig.add_annotation(text=f"Integration failed: {sol.message}",
                                           xref='paper', yref='paper', x=0.5, y=0.5,
                                           showarrow=False, font=dict(size=16, color='red'))
                    return err_fig, dbc.Alert(f"Integration failed: {sol.message}", color='danger'), None

            t_arr  = sol.t
            X_arr, S_arr, P_arr, O2_arr, V_arr = sol.y
            X_arr  = np.maximum(X_arr, 0); S_arr = np.maximum(S_arr, 0)
            P_arr  = np.maximum(P_arr, 0); O2_arr = np.maximum(O2_arr, 0)
            V_arr  = np.maximum(V_arr, 1e-6)

            flujo_sim = np.array([calcular_flujo(ti) for ti in t_arr])

            # ---- Post-simulation rate recalculation ----
            mu_sim_list = []; mu_anaer_sim_list = []
            for i in range(len(t_arr)):
                ti = t_arr[i]
                si, pi, o2i = S_arr[i], P_arr[i], O2_arr[i]
                es_lote_i = (ti < params["t_batch_inicial_fin"])
                mu_i = 0.0; mu_aer_i = 0.0; mu_anaer_i = 0.0

                if params["tipo_mu"] == "Fermentation":
                    mu_aer_i = mu_fermentacion(si, pi, o2i,
                        params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"],
                        0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                    mu_anaer_i = mu_fermentacion(si, pi, o2i,
                        0, 1, float('inf'),
                        params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"],
                        params["KP_anaerob"], params.get("n_p", 1.0),
                        params["KO_inhib_anaerob"], considerar_O2=False)
                    mu_i = mu_aer_i + mu_anaer_i
                elif params["tipo_mu"] == "Switched Fermentation":
                    if es_lote_i:
                        cur_O2 = params.get("O2_controlado", o2i)
                        mu_aer_i = mu_fermentacion(si, pi, cur_O2,
                            params["mumax_aero"], params["Ks_aerob"], params["KO_aerob"],
                            0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                        mu_i = mu_aer_i
                    else:
                        mu_anaer_i = mu_fermentacion(si, pi, o2i,
                            0, 1, float('inf'),
                            params["mumax_anaero"], params["Ks_anaerob"],
                            params["KiS_anaerob"], params["KP_anaerob"],
                            params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
                        mu_i = mu_anaer_i
                elif params["tipo_mu"] == "Simple Monod":
                    mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
                    mu_aer_i = mu_i
                elif params["tipo_mu"] == "Sigmoidal Monod":
                    mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"],
                                        params.get("n_sig", 1))
                    mu_aer_i = mu_i
                elif params["tipo_mu"] == "Monod with restrictions":
                    mu_i = mu_completa(si, o2i, pi, params.get("mumax", 0.0),
                                       params["Ks"], params.get("KO", 0.1),
                                       params.get("KP_gen", 50.0))
                    mu_aer_i = mu_i

                mu_i = max(0, mu_i); mu_anaer_i = max(0, mu_anaer_i)
                mu_sim_list.append(mu_i)
                mu_anaer_sim_list.append(mu_anaer_i)

            mu_sim    = np.array(mu_sim_list)
            mu_anaer_sim = np.array(mu_anaer_sim_list)

            # ---- Build 2×3 subplot figure ----
            fig = make_subplots(
                rows=2, cols=3,
                specs=[[{'secondary_y': True}, {}, {}],
                       [{}, {}, {}]],
                subplot_titles=[
                    'Feeding and Volume', 'Biomass (X)', 'Substrate (S)',
                    'Ethanol (P)', 'Dissolved Oxygen (O₂)', 'Specific Growth Rate (μ)',
                ],
            )

            # (1,1) Flow + Volume with secondary y
            fig.add_trace(go.Scatter(x=t_arr, y=flujo_sim, name='Flow [L/h]',
                                     line=dict(color='red')),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t_arr, y=V_arr, name='Volume [L]',
                                     line=dict(color='blue', dash='dash')),
                          row=1, col=1, secondary_y=True)

            # (1,2) Biomass
            fig.add_trace(go.Scatter(x=t_arr, y=X_arr, name='Biomass X [g/L]',
                                     line=dict(color='green')),
                          row=1, col=2)

            # (1,3) Substrate
            fig.add_trace(go.Scatter(x=t_arr, y=S_arr, name='Substrate S [g/L]',
                                     line=dict(color='magenta')),
                          row=1, col=3)

            # (2,1) Ethanol
            fig.add_trace(go.Scatter(x=t_arr, y=P_arr, name='Ethanol P [g/L]',
                                     line=dict(color='black')),
                          row=2, col=1)

            # (2,2) Dissolved O2
            fig.add_trace(go.Scatter(x=t_arr, y=O2_arr, name='O₂ [mg/L]',
                                     line=dict(color='cyan')),
                          row=2, col=2)

            # (2,3) Specific growth rate
            fig.add_trace(go.Scatter(x=t_arr, y=mu_sim, name='μ [1/h]',
                                     line=dict(color='goldenrod')),
                          row=2, col=3)

            # Phase vertical lines
            for xval, color, dash in [
                (t_batch_fin, 'gray', 'dash'),
                (t_alim_ini, 'orange', 'dash'),
                (t_alim_fin, 'purple', 'dash'),
            ]:
                fig.add_vline(x=xval, line_color=color, line_dash=dash, line_width=1.5)

            fig.update_yaxes(title_text='Flow [L/h]', row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text='Volume [L]', row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text='[g/L]', row=1, col=2)
            fig.update_yaxes(title_text='[g/L]', row=1, col=3)
            fig.update_yaxes(title_text='[g/L]', row=2, col=1)
            fig.update_yaxes(title_text='[mg/L]', row=2, col=2)
            fig.update_yaxes(title_text='[1/h]', row=2, col=3)
            fig.update_xaxes(title_text='Time [h]')
            fig.update_layout(height=620, showlegend=False,
                              margin=dict(l=40, r=20, t=60, b=40))

            # ---- Compute metrics ----
            vol_final        = float(V_arr[-1])
            etanol_final     = float(P_arr[-1])
            biomasa_final    = float(X_arr[-1])

            S_inicial_total  = S0 * V0
            S_alimentado     = float(np.trapz(flujo_sim * Sin, t_arr)) if len(t_arr) > 1 else 0
            S_final_total    = float(S_arr[-1] * V_arr[-1])
            S_consumido      = max(1e-9, S_inicial_total + S_alimentado - S_final_total)

            P_inicial_total  = P0 * V0
            etanol_tot_final = etanol_final * vol_final
            etanol_producido = etanol_tot_final - P_inicial_total

            prod_vol  = etanol_producido / vol_final / t_arr[-1] if t_arr[-1] > 0 and vol_final > 1e-6 else 0
            rend_glob = etanol_producido / S_consumido if S_consumido > 1e-9 else 0

            try:
                XV_int = float(np.trapz(X_arr * V_arr, t_arr))
                avg_sp = etanol_producido / XV_int if XV_int > 1e-9 else None
            except Exception:
                avg_sp = None

            try:
                p_max_idx = int(np.argmax(P_arr))
                max_eth_str = f"{P_arr[p_max_idx]:.2f} (t={t_arr[p_max_idx]:.1f} h)"
            except Exception:
                max_eth_str = "N/A"

            metrics_layout = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Final Volume [L]"),
                    html.H4(f"{vol_final:.2f}"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Final Ethanol [g/L]"),
                    html.H4(f"{etanol_final:.2f}"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Final Biomass [g/L]"),
                    html.H4(f"{biomasa_final:.2f}"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Ethanol Volume Productivity [g/L/h]"),
                    html.H4(f"{prod_vol:.3f}"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Global Yield P/S [g/g]"),
                    html.H4(f"{rend_glob:.3f}"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Avg. Specific Ethanol Production [g/gXh]"),
                    html.H4(f"{avg_sp:.4f}" if avg_sp is not None else "N/A"),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Max Ethanol [g/L]"),
                    html.H4(max_eth_str),
                ])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Residual Substrate [g/L]"),
                    html.H4(f"{float(S_arr[-1]):.2f}"),
                ])), width=3),
            ], className='g-2 mt-2')

            # ---- Store data for download ----
            store_data = {
                'Time [h]':                        t_arr.tolist(),
                'Flow [L/h]':                      flujo_sim.tolist(),
                'Volume [L]':                      V_arr.tolist(),
                'Biomass (X) [g/L]':               X_arr.tolist(),
                'Substrate (S) [g/L]':             S_arr.tolist(),
                'Ethanol (P) [g/L]':               P_arr.tolist(),
                'Dissolved Oxygen (O2) [mg/L]':    O2_arr.tolist(),
                'Specific Growth Rate (mu) [1/h]': mu_sim.tolist(),
            }

            return fig, metrics_layout, store_data

        except Exception as e:
            err_fig = go.Figure()
            err_fig.add_annotation(text=f"Error: {e}",
                                   xref='paper', yref='paper', x=0.5, y=0.5,
                                   showarrow=False, font=dict(size=14, color='red'))
            error_div = dbc.Alert([
                html.Strong("Simulation error: "),
                html.Pre(traceback.format_exc(), style={'fontSize': '0.75rem'}),
            ], color='danger')
            return err_fig, error_div, None

    # ------------------------------------------------------------------
    # Callback 4: Excel download
    # ------------------------------------------------------------------
    @app.callback(
        Output('fermentation-download', 'data'),
        Input('fermentation-download-btn', 'n_clicks'),
        State('fermentation-sim-data', 'data'),
        prevent_initial_call=True,
    )
    def download_excel(n_clicks, sim_data):
        if not sim_data:
            return dash.no_update

        df = pd.DataFrame(sim_data)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Simulation Data')
        buffer.seek(0)

        return dcc.send_bytes(buffer.read(), filename='Simulation_data.xlsx')
