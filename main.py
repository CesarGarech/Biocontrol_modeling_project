import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Bioprocess Modeling",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# --- Menu Structure ---
menu_structure = {
    "🏠 Home": None,
    "🔬 Models": ["Batch", "Fed-Batch", "Continuous", "Fermentation"],
    "📈 Sensitivity Analysis": None,
    "🔧 Parameter Adjustment": [
        "Batch Parameter Adjustment",
        "Fed-Batch Parameter Adjustment",
        "Fermentation Parameter Adjustment"
    ],
    "📊 State Estimation": ["EKF", "ANN"],
    "⚙️ Control": {
        "Regulatory": [
            "Identification (pH)", "Temperature", "pH",
            "Oxygen", "Cascade-Oxygen", "On-Off Feeding"
        ],
        "Advanced": [
            "RTO", "RTO Ferm", "NMPC", "LMPC", "EKF-NMPC", "Fuzzy Control"
        ]
    }
}

# --- Import all page modules (with graceful fallback) ---
def _import(module_path):
    try:
        parts = module_path.split('.')
        mod = __import__(module_path, fromlist=[parts[-1]])
        return mod
    except Exception as e:
        print(f"Warning: Could not import {module_path}: {e}")
        return None

home_mod            = _import('Body.home')
lote_mod            = _import('Body.modeling.lote')
lote_alim_mod       = _import('Body.modeling.lote_alimentado')
continuo_mod        = _import('Body.modeling.continuo')
ferm_mod            = _import('Body.modeling.ferm_alcohol')
analysis_mod        = _import('Body.analysis')
aj_lote_mod         = _import('Body.parameter_estimation.ajuste_parametros_lote')
aj_alim_mod         = _import('Body.parameter_estimation.ajuste_parametros_lote_alim')
aj_ferm_mod         = _import('Body.parameter_estimation.ajuste_parametros_ferm')
ekf_mod             = _import('Body.estimation.ekf')
ann_mod             = _import('Body.estimation.ann')
reg_ident_mod       = _import('Body.control.regulatorio.reg_ident')
reg_temp_mod        = _import('Body.control.regulatorio.reg_temp')
reg_ph_mod          = _import('Body.control.regulatorio.reg_ph')
reg_ox_mod          = _import('Body.control.regulatorio.reg_oxigeno')
reg_casc_mod        = _import('Body.control.regulatorio.reg_cascade_oxigen')
reg_feed_mod        = _import('Body.control.regulatorio.reg_feed_onoff')
rto_mod             = _import('Body.control.avanzado.rto')
rto_ferm_mod        = _import('Body.control.avanzado.rto_ferm')
nmpc_mod            = _import('Body.control.avanzado.nmpc')
lmpc_mod            = _import('Body.control.avanzado.lmpc')
ekf_nmpc_mod        = _import('Body.control.avanzado.ekf_nmpc')
fuzzy_mod           = _import('Body.control.avanzado.fuzzy_control')

# --- Map page names to modules ---
PAGE_MODULE_MAP = {
    "🏠 Home":                          home_mod,
    "Batch":                            lote_mod,
    "Fed-Batch":                        lote_alim_mod,
    "Continuous":                       continuo_mod,
    "Fermentation":                     ferm_mod,
    "📈 Sensitivity Analysis":          analysis_mod,
    "Batch Parameter Adjustment":       aj_lote_mod,
    "Fed-Batch Parameter Adjustment":   aj_alim_mod,
    "Fermentation Parameter Adjustment":aj_ferm_mod,
    "EKF":                              ekf_mod,
    "ANN":                              ann_mod,
    "Identification (pH)":              reg_ident_mod,
    "Temperature":                      reg_temp_mod,
    "pH":                               reg_ph_mod,
    "Oxygen":                           reg_ox_mod,
    "Cascade-Oxygen":                   reg_casc_mod,
    "On-Off Feeding":                   reg_feed_mod,
    "RTO":                              rto_mod,
    "RTO Ferm":                         rto_ferm_mod,
    "NMPC":                             nmpc_mod,
    "LMPC":                             lmpc_mod,
    "EKF-NMPC":                         ekf_nmpc_mod,
    "Fuzzy Control":                    fuzzy_mod,
}

# --- Register callbacks from all page modules ---
for page_name, mod in PAGE_MODULE_MAP.items():
    if mod is not None and hasattr(mod, 'register_callbacks'):
        try:
            mod.register_callbacks(app)
        except Exception as e:
            print(f"Warning: Could not register callbacks for '{page_name}': {e}")

# --- Sidebar style ---
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "300px",
    "padding": "20px 15px",
    "backgroundColor": "#2c3e50",
    "overflowY": "auto",
    "zIndex": 1000,
}

CONTENT_STYLE = {
    "marginLeft": "310px",
    "padding": "20px 30px",
    "minHeight": "100vh",
    "backgroundColor": "#f5f6fa",
}

LABEL_STYLE = {"color": "#adb5bd", "fontSize": "12px", "marginTop": "8px", "marginBottom": "2px"}
RADIO_STYLE = {"color": "white", "display": "block", "margin": "4px 0", "fontSize": "14px"}

# Pre-build sub-navigation components (static IDs, conditionally shown)
def build_sidebar():
    ctrl_struct = menu_structure["⚙️ Control"]
    ctrl_type_opts = [{"label": k, "value": k} for k in ctrl_struct.keys()]
    return html.Div([
        html.H5("🧫 Bioprocess Modeling", style={"color": "white", "fontSize": "16px"}),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.3)", "margin": "10px 0"}),

        html.Div("Section", style=LABEL_STYLE),
        dcc.Dropdown(
            id="main-nav-dropdown",
            options=[{"label": k, "value": k} for k in menu_structure.keys()],
            value="🏠 Home",
            clearable=False,
            style={"marginBottom": "8px"},
        ),

        # --- Sub-navs (conditionally shown by callbacks) ---
        html.Div(id="models-subnav", style={"display": "none"}, children=[
            html.Div("Model:", style=LABEL_STYLE),
            dcc.RadioItems(
                id="models-radio",
                options=[{"label": v, "value": v} for v in menu_structure["🔬 Models"]],
                value="Batch",
                labelStyle=RADIO_STYLE,
            ),
        ]),

        html.Div(id="param-adj-subnav", style={"display": "none"}, children=[
            html.Div("Adjustment:", style=LABEL_STYLE),
            dcc.RadioItems(
                id="param-adj-radio",
                options=[{"label": v, "value": v} for v in menu_structure["🔧 Parameter Adjustment"]],
                value="Batch Parameter Adjustment",
                labelStyle=RADIO_STYLE,
            ),
        ]),

        html.Div(id="state-est-subnav", style={"display": "none"}, children=[
            html.Div("Estimator:", style=LABEL_STYLE),
            dcc.RadioItems(
                id="state-est-radio",
                options=[{"label": v, "value": v} for v in menu_structure["📊 State Estimation"]],
                value="EKF",
                labelStyle=RADIO_STYLE,
            ),
        ]),

        html.Div(id="control-subnav", style={"display": "none"}, children=[
            html.Div("Control Type:", style=LABEL_STYLE),
            dcc.Dropdown(
                id="control-type-dropdown",
                options=ctrl_type_opts,
                value="Regulatory",
                clearable=False,
                style={"marginBottom": "6px"},
            ),
            html.Div("Page:", style=LABEL_STYLE),
            dcc.RadioItems(
                id="control-page-radio",
                options=[{"label": v, "value": v} for v in ctrl_struct["Regulatory"]],
                value="Identification (pH)",
                labelStyle=RADIO_STYLE,
            ),
        ]),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.3)", "margin": "10px 0"}),

        # Parameters area injected by selected page
        html.Div(id="sidebar-params"),
    ], style=SIDEBAR_STYLE)


# --- App Layout ---
app.layout = html.Div([
    dcc.Store(id="selected-page-store", data="🏠 Home"),
    build_sidebar(),
    html.Div(id="main-content", style=CONTENT_STYLE),
])


# --- Callback: Show/hide sub-navs and update control page radio options ---
@app.callback(
    Output("models-subnav",   "style"),
    Output("param-adj-subnav","style"),
    Output("state-est-subnav","style"),
    Output("control-subnav",  "style"),
    Input("main-nav-dropdown", "value"),
)
def toggle_subnavs(main_cat):
    show   = {"display": "block"}
    hidden = {"display": "none"}
    return (
        show   if main_cat == "🔬 Models" else hidden,
        show   if main_cat == "🔧 Parameter Adjustment" else hidden,
        show   if main_cat == "📊 State Estimation" else hidden,
        show   if main_cat == "⚙️ Control" else hidden,
    )


@app.callback(
    Output("control-page-radio", "options"),
    Output("control-page-radio", "value"),
    Input("control-type-dropdown", "value"),
)
def update_control_radio(ctrl_type):
    ctrl_struct = menu_structure["⚙️ Control"]
    opts = ctrl_struct.get(ctrl_type, [])
    options = [{"label": v, "value": v} for v in opts]
    value = opts[0] if opts else None
    return options, value


# --- Callback: Derive selected page from all nav inputs ---
@app.callback(
    Output("selected-page-store", "data"),
    Input("main-nav-dropdown",   "value"),
    Input("models-radio",         "value"),
    Input("param-adj-radio",      "value"),
    Input("state-est-radio",      "value"),
    Input("control-page-radio",   "value"),
)
def update_selected_page(main_cat, models_val, param_val, est_val, ctrl_val):
    if main_cat == "🏠 Home":
        return "🏠 Home"
    elif main_cat == "🔬 Models":
        return models_val or "Batch"
    elif main_cat == "📈 Sensitivity Analysis":
        return "📈 Sensitivity Analysis"
    elif main_cat == "🔧 Parameter Adjustment":
        return param_val or "Batch Parameter Adjustment"
    elif main_cat == "📊 State Estimation":
        return est_val or "EKF"
    elif main_cat == "⚙️ Control":
        return ctrl_val or "Identification (pH)"
    return "🏠 Home"


# --- Callback: Render page content + sidebar params ---
def _not_found_layout(page):
    return html.Div([
        dbc.Alert(f"Page '{page}' is not available.", color="warning")
    ])


@app.callback(
    Output("sidebar-params", "children"),
    Output("main-content",   "children"),
    Input("selected-page-store", "data"),
)
def render_page(selected_page):
    mod = PAGE_MODULE_MAP.get(selected_page)
    if mod is None:
        # Fallback to home
        fallback = home_mod
        if fallback and hasattr(fallback, 'get_params_layout') and hasattr(fallback, 'get_content_layout'):
            return fallback.get_params_layout(), fallback.get_content_layout()
        return html.Div(), _not_found_layout(selected_page)

    try:
        params  = mod.get_params_layout()  if hasattr(mod, 'get_params_layout')  else html.Div()
        content = mod.get_content_layout() if hasattr(mod, 'get_content_layout') else _not_found_layout(selected_page)
        return params, content
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return html.Div(), html.Div([
            dbc.Alert(f"Error loading '{selected_page}': {str(e)}", color="danger"),
            html.Pre(err, style={"fontSize": "12px", "color": "gray"}),
        ])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
