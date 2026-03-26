# ann.py - ANN-Based State Estimation (Dash Version)
import numpy as np
import casadi as ca
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import base64
import pickle

PAGE_ID = 'ann'

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("ANN Configuration", className="mb-3"),
        
        # STEP 1: Data Generation
        html.H6("STEP 1: Generate Training Data", className="mt-3"),
        html.Label("Simulation time (h):"),
        dcc.Slider(id=f'{PAGE_ID}-tfinal', min=10, max=100, step=10, value=40,
                   marks={10: '10', 50: '50', 100: '100'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Noise Level:", className="mt-3"),
        dcc.Dropdown(id=f'{PAGE_ID}-noise', options=[
            {'label': 'Low', 'value': 'Low'},
            {'label': 'Medium', 'value': 'Medium'},
            {'label': 'High', 'value': 'High'}
        ], value='Medium', clearable=False, className="mb-2"),
        dbc.Button("Generate Simulation Data", id=f'{PAGE_ID}-btn-generate', color="success", className="w-100 mb-2"),
        dbc.Button("Reset Data", id=f'{PAGE_ID}-btn-reset', color="secondary", className="w-100 mb-3", outline=True),
        
        html.Hr(),
        
        # STEP 2: ANN Configuration
        html.H6("STEP 2: Configure and Train ANN", className="mt-3"),
        html.Label("Validation Set Size (%):"),
        dcc.Slider(id=f'{PAGE_ID}-val_split', min=5, max=50, step=5, value=20,
                   marks={5: '5', 25: '25', 50: '50'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Neurons Layer 1:", className="mt-3"),
        dcc.Input(id=f'{PAGE_ID}-n1', type='number', value=64, min=1, max=256, className="form-control mb-2"),
        html.Label("Neurons Layer 2:"),
        dcc.Input(id=f'{PAGE_ID}-n2', type='number', value=64, min=1, max=256, className="form-control mb-2"),
        html.Label("Activation Function:"),
        dcc.Dropdown(id=f'{PAGE_ID}-activation', options=[
            {'label': 'ReLU', 'value': 'relu'},
            {'label': 'Tanh', 'value': 'tanh'},
            {'label': 'Sigmoid', 'value': 'sigmoid'}
        ], value='relu', clearable=False, className="mb-2"),
        html.Label("Optimizer:"),
        dcc.Dropdown(id=f'{PAGE_ID}-optimizer', options=[
            {'label': 'Adam', 'value': 'adam'},
            {'label': 'RMSprop', 'value': 'rmsprop'},
            {'label': 'SGD', 'value': 'sgd'}
        ], value='adam', clearable=False, className="mb-2"),
        html.Label("Learning Rate:"),
        dcc.Input(id=f'{PAGE_ID}-lr', type='number', value=0.001, min=1e-5, max=1e-1, step=1e-4, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("Loss Function:"),
        dcc.Dropdown(id=f'{PAGE_ID}-loss', options=[
            {'label': 'Mean Squared Error', 'value': 'mean_squared_error'},
            {'label': 'Mean Absolute Error', 'value': 'mean_absolute_error'}
        ], value='mean_squared_error', clearable=False, className="mb-2"),
        html.Label("Epochs:"),
        dcc.Input(id=f'{PAGE_ID}-epochs', type='number', value=200, min=10, max=1000, step=10, className="form-control mb-2"),
        html.Label("Batch Size:"),
        dcc.Dropdown(id=f'{PAGE_ID}-batch', options=[
            {'label': '8', 'value': 8},
            {'label': '16', 'value': 16},
            {'label': '32', 'value': 32},
            {'label': '64', 'value': 64}
        ], value=16, clearable=False, className="mb-3"),
        
        dbc.Button("Train and Run ANN", id=f'{PAGE_ID}-btn-train', color="primary", className="w-100 mt-3")
    ], style={'maxHeight': '80vh', 'overflowY': 'scroll'})

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("Estimation of States and Parameters with Artificial Neural Network (ANN)", className="mb-3"),
        dcc.Markdown("""
        This page demonstrates the use of an ANN as a **soft sensor**. The network is trained to predict 
        the process states (X, S, P) and key kinetic parameters ($\\mu_{max}$, $Y_{X/S}$) using only 
        the available online measurements (DO, pH, T).

        **Workflow:**
        1. **Generate Data:** First, use STEP 1 in the sidebar to run a bioprocess simulation. 
           This creates the dataset for training the ANN.
        2. **Configure & Train ANN:** Once the data is generated, configure the ANN's hyperparameters 
           in STEP 2 and click 'Train and Run ANN' to see the prediction results.
        
        **Mathematical Model:**
        
        **Data Generation (CasADi ODE):**
        
        $$\\frac{dX}{dt} = \\mu X, \\quad \\mu = \\mu_{max} \\frac{S}{K_s + S}$$
        
        $$\\frac{dS}{dt} = -\\frac{1}{Y_{XS}} \\mu X$$
        
        $$\\frac{dP}{dt} = \\alpha \\mu X$$
        
        **Measurement Model:**
        
        $$z = [DO_{sat} - k_{OUR}X, \\, pH_0 - k_{acid}(P - P_{ref}), \\, T] + noise$$
        
        **ANN Architecture:**
        - Input: [DO, pH, T] (3 measurements)
        - Hidden Layer 1: Dense(n1, activation)
        - Hidden Layer 2: Dense(n2, activation)
        - Output: Dense(5, 'linear') → [X, S, P, μmax, Yxs]
        """, mathjax=True),
        html.Hr(),
        
        # Storage for data between steps
        dcc.Store(id=f'{PAGE_ID}-data-store', storage_type='memory'),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("⬅️ First Step: Configure settings in STEP 1 in the sidebar and click 'Generate Simulation Data'.", 
                     color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    # Callback 1: Generate simulation data
    @app.callback(
        [Output(f'{PAGE_ID}-data-store', 'data'),
         Output(f'{PAGE_ID}-output', 'children', allow_duplicate=True)],
        [Input(f'{PAGE_ID}-btn-generate', 'n_clicks'),
         Input(f'{PAGE_ID}-btn-reset', 'n_clicks')],
        [State(f'{PAGE_ID}-tfinal', 'value'),
         State(f'{PAGE_ID}-noise', 'value')],
        prevent_initial_call=True
    )
    def generate_data(n_gen, n_reset, t_final, noise_level):
        from dash import ctx
        
        if ctx.triggered_id == f'{PAGE_ID}-btn-reset':
            return None, dbc.Alert("Data reset. Click 'Generate Simulation Data' to create new training data.", color="warning")
        
        if not n_gen:
            return None, dbc.Alert("Click the generate button.", color="info")
        
        try:
            # Noise configuration
            noise_map = {
                'Low': {'Q_factor': 1e-7, 'R_factor': 1e-3},
                'Medium': {'Q_factor': 1e-5, 'R_factor': 5e-2},
                'High': {'Q_factor': 1e-4, 'R_factor': 2e-1}
            }
            q_factor = noise_map[noise_level]['Q_factor']
            r_factor = noise_map[noise_level]['R_factor']
            Q_sim = np.diag([q_factor, q_factor*1e-3, q_factor, q_factor*0.1, q_factor*0.1])
            R_sim = np.diag([r_factor, r_factor*0.5, r_factor*10])
            
            # Simulation parameters
            dt_sim = 0.1
            n_states, n_meas = 5, 3
            Ks, alpha = 0.1, 0.1
            OD_sat, k_OUR, pH0, P0_meas_ref, k_acid, Tset, k_Temp = 8.0, 0.5, 7.0, 0.0, 0.2, 30, 0.02
            
            # CasADi model
            x_sym = ca.SX.sym('x', n_states)
            X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym)
            mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
            dX = mu_sym * X_sym
            dS = -(1 / Yxs_sym) * dX
            dP = alpha * dX
            x_next_sym = x_sym + dt_sim * ca.vertcat(dX, dS, dP, 0, 0)
            f_func = ca.Function('f', [x_sym], [x_next_sym])
            
            OD_val = OD_sat - k_OUR*X_sym
            pH_val = pH0 - k_acid*(P_sym - P0_meas_ref)
            T_val = Tset + k_Temp*(X_sym*S_sym)
            z_sym = ca.vertcat(OD_val, pH_val, T_val)
            h_func = ca.Function('h', [x_sym], [z_sym])
            
            # Simulate
            time_vec = np.arange(0, t_final + dt_sim, dt_sim)
            N = len(time_vec)
            x_real = np.array([[0.1], [5.0], [0.0], [0.4], [0.5]])
            x_real_hist = np.zeros((n_states, N))
            z_meas_hist = np.zeros((n_meas, N))
            
            for k in range(N):
                x_real_hist[:, k] = x_real.flatten()
                z_noisefree = h_func(x_real).full()
                noise_meas = np.random.multivariate_normal(np.zeros(n_meas), R_sim).reshape(-1, 1)
                z_meas_hist[:, k] = (z_noisefree + noise_meas).flatten()
                if k < N - 1:
                    x_real_next_no_noise = f_func(x_real).full()
                    noise_proc = np.random.multivariate_normal(np.zeros(n_states), Q_sim).reshape(-1, 1)
                    x_real = x_real_next_no_noise + noise_proc
                    x_real[0:3] = np.maximum(x_real[0:3], 0)
            
            # Store data
            data = {
                'time_vec': time_vec.tolist(),
                'x_real_hist': x_real_hist.tolist(),
                'z_meas_hist': z_meas_hist.tolist(),
                't_final': t_final
            }
            
            return data, dbc.Alert(f"✓ STEP 1 COMPLETE: Training data generated (Simulated for {t_final} hours). Now configure ANN in STEP 2.", 
                                  color="success")
            
        except Exception as e:
            return None, dbc.Alert(f"Error generating data: {str(e)}", color="danger")
    
    # Callback 2: Train ANN and display results
    @app.callback(
        Output(f'{PAGE_ID}-output', 'children'),
        Input(f'{PAGE_ID}-btn-train', 'n_clicks'),
        [State(f'{PAGE_ID}-data-store', 'data'),
         State(f'{PAGE_ID}-val_split', 'value'),
         State(f'{PAGE_ID}-n1', 'value'),
         State(f'{PAGE_ID}-n2', 'value'),
         State(f'{PAGE_ID}-activation', 'value'),
         State(f'{PAGE_ID}-optimizer', 'value'),
         State(f'{PAGE_ID}-lr', 'value'),
         State(f'{PAGE_ID}-loss', 'value'),
         State(f'{PAGE_ID}-epochs', 'value'),
         State(f'{PAGE_ID}-batch', 'value')],
        prevent_initial_call=True
    )
    def train_ann(n_clicks, stored_data, val_split, n1, n2, activation, optimizer_name, lr, loss_func, epochs, batch_size):
        if not n_clicks or not stored_data:
            return dbc.Alert("Generate data first in STEP 1.", color="warning")
        
        try:
            # Retrieve data
            time_vec = np.array(stored_data['time_vec'])
            x_real_hist = np.array(stored_data['x_real_hist'])
            z_meas_hist = np.array(stored_data['z_meas_hist'])
            
            # Prepare training data
            X_data = z_meas_hist.T  # Input: measurements
            Y_data = x_real_hist.T  # Output: states + parameters
            
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_data)
            Y_scaled = scaler_Y.fit_transform(Y_data)
            
            val_split_frac = val_split / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, Y_scaled, test_size=val_split_frac, random_state=42)
            
            # Build ANN model
            tf.random.set_seed(42)
            ann_model = Sequential([
                Dense(n1, activation=activation, input_shape=(X_data.shape[1],)),
                Dense(n2, activation=activation),
                Dense(Y_data.shape[1], activation='linear')
            ])
            
            if optimizer_name == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=lr)
            elif optimizer_name == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=lr)
            else:
                optimizer = keras.optimizers.SGD(learning_rate=lr)
            
            ann_model.compile(optimizer=optimizer, loss=loss_func)
            
            # Train model
            history = ann_model.fit(X_train, y_train, epochs=epochs,
                                   batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
            
            # Predict
            Y_pred_scaled = ann_model.predict(X_scaled, verbose=0)
            Y_pred_ann = scaler_Y.inverse_transform(Y_pred_scaled)
            x_ann_hist = Y_pred_ann.T
            
            # Create plots
            fig1 = create_states_plot(time_vec, x_real_hist, x_ann_hist)
            fig2 = create_params_plot(time_vec, x_real_hist, x_ann_hist)
            fig3 = create_measurements_plot(time_vec, z_meas_hist, x_ann_hist)
            fig4 = create_learning_curve(history)
            
            # Model summary
            config_info = {
                'Architecture': f'Input(3) → Dense({n1}, {activation}) → Dense({n2}, {activation}) → Dense(5, linear)',
                'Optimizer': f'{optimizer_name} (lr={lr})',
                'Loss': loss_func,
                'Epochs': epochs,
                'Batch Size': batch_size,
                'Val Split': f'{val_split}%'
            }
            
            return html.Div([
                dbc.Alert("✓ ANN training completed successfully!", color="success", className="mb-3"),
                
                html.H4("Predicted States", className="mt-3"),
                dcc.Graph(figure=fig1),
                
                html.H4("Predicted Parameters", className="mt-3"),
                dcc.Graph(figure=fig2),
                
                html.H4("Measurement Consistency Check", className="mt-3"),
                html.P("Compares noisy measurements (ANN input) with measurements reconstructed from ANN predictions.", 
                      className="text-muted"),
                dcc.Graph(figure=fig3),
                
                html.H4("Learning Curve", className="mt-3"),
                dcc.Graph(figure=fig4),
                
                html.Hr(),
                html.H5("Model Configuration"),
                html.Pre(json.dumps(config_info, indent=2), style={'backgroundColor': '#f8f9fa', 'padding': '10px'})
            ])
            
        except Exception as e:
            return dbc.Alert(f"Error during ANN training: {str(e)}", color="danger")

def create_states_plot(time_vec, x_real_hist, x_ann_hist):
    """Create states comparison plot"""
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Biomass (X)', 'Substrate (S)', 'Product (P)'),
                       vertical_spacing=0.1)
    
    state_labels = ['Biomass', 'Substrate', 'Product']
    colors_real = ['blue', 'blue', 'blue']
    colors_pred = ['green', 'green', 'green']
    
    for i in range(3):
        fig.add_trace(go.Scatter(x=time_vec, y=x_real_hist[i, :], mode='lines', 
                                name=f'{state_labels[i]} Real', line=dict(color=colors_real[i])), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=time_vec, y=x_ann_hist[i, :], mode='lines', 
                                name=f'{state_labels[i]} Predicted', 
                                line=dict(color=colors_pred[i], dash='dash', width=2.5)), row=i+1, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=3, col=1)
    fig.update_yaxes(title_text="g/L")
    fig.update_layout(height=800, title_text="ANN Prediction: Process States", showlegend=True)
    return fig

def create_params_plot(time_vec, x_real_hist, x_ann_hist):
    """Create parameters comparison plot"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=('μmax', 'Y_X/S'),
                       vertical_spacing=0.15)
    
    fig.add_trace(go.Scatter(x=time_vec, y=x_real_hist[3, :], mode='lines', 
                            name='μmax Real', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=x_ann_hist[3, :], mode='lines', 
                            name='μmax Predicted', line=dict(color='green', dash='dash', width=2.5)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time_vec, y=x_real_hist[4, :], mode='lines', 
                            name='Yxs Real', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=x_ann_hist[4, :], mode='lines', 
                            name='Yxs Predicted', line=dict(color='green', dash='dash', width=2.5), showlegend=False), row=2, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_yaxes(title_text="1/h", row=1, col=1)
    fig.update_yaxes(title_text="gX/gS", row=2, col=1)
    fig.update_layout(height=600, title_text="ANN Prediction: Model Parameters")
    return fig

def create_measurements_plot(time_vec, z_meas_hist, x_ann_hist):
    """Create measurement consistency plot"""
    OD_sat, k_OUR, pH0, k_acid, P0_meas_ref, Tset, k_Temp = 8.0, 0.5, 7.0, 0.2, 0.0, 30, 0.02
    X_ann, S_ann, P_ann = x_ann_hist[0, :], x_ann_hist[1, :], x_ann_hist[2, :]
    OD_pred = OD_sat - k_OUR * X_ann
    pH_pred = pH0 - k_acid * (P_ann - P0_meas_ref)
    T_pred = Tset + k_Temp * (X_ann * S_ann)
    
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Dissolved Oxygen', 'pH', 'Temperature'),
                       vertical_spacing=0.1)
    
    fig.add_trace(go.Scatter(x=time_vec, y=z_meas_hist[0, :], mode='markers', 
                            name='DO Measured', marker=dict(color='blue', size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=OD_pred, mode='lines', 
                            name='DO Reconstructed', line=dict(color='red', dash='dash', width=2.5)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time_vec, y=z_meas_hist[1, :], mode='markers', 
                            name='pH Measured', marker=dict(color='blue', size=4), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=pH_pred, mode='lines', 
                            name='pH Reconstructed', line=dict(color='red', dash='dash', width=2.5), showlegend=False), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=time_vec, y=z_meas_hist[2, :], mode='markers', 
                            name='T Measured', marker=dict(color='blue', size=4), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=T_pred, mode='lines', 
                            name='T Reconstructed', line=dict(color='red', dash='dash', width=2.5), showlegend=False), row=3, col=1)
    
    fig.update_xaxes(title_text="Time (h)", row=3, col=1)
    fig.update_yaxes(title_text="mg/L", row=1, col=1)
    fig.update_yaxes(title_text="pH", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_layout(height=800, title_text="Measurements: Actual vs Reconstructed")
    return fig

def create_learning_curve(history):
    """Create learning curve plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'],
                            mode='lines', name='Training Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'],
                            mode='lines', name='Validation Loss', line=dict(color='red')))
    fig.update_layout(title='ANN Learning Curve', xaxis_title='Epoch', yaxis_title='Loss Value',
                     yaxis_type='log', height=500)
    return fig