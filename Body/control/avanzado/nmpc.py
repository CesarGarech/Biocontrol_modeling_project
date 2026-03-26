# nmpc.py - Nonlinear Model Predictive Control (Dash Version)
# TEMPORARILY DISABLED - Module is being converted
from dash import dcc, html
import dash_bootstrap_components as dbc

PAGE_ID = 'nmpc'

def get_params_layout():
    return html.Div([html.P("NMPC module is being converted. Check back soon.")])

def get_content_layout():
    return html.Div([
        html.H2("NMPC - Under Conversion"),
        dbc.Alert("This module is currently being converted to Dash. It will be available soon.", color="warning")
    ])

def register_callbacks(app):
    pass
