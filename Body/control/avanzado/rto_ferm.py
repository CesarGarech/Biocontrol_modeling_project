from dash import html, dcc
import dash_bootstrap_components as dbc

PAGE_ID = 'rto_ferm'

def get_params_layout():
    return html.Div([
        html.H4("Configuration"),
        html.P("This module is currently being converted to Dash. It will be available soon.")
    ])

def get_content_layout():
    return html.Div([
        html.H2("RTO Fermentation - Under Conversion"),
        dbc.Alert(
            "This module is currently being converted to Dash. It will be available soon.",
            color="warning"
        )
    ])

def register_callbacks(app):
    pass
