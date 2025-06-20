import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import StringIO

# We use a session state check to manage the two-step process
if 'ann_data_generated' not in st.session_state:
    st.session_state.ann_data_generated = False

def ann_page():
    """
    Creates the Streamlit page for ANN-based state estimation.
    """
    st.header("Estimation of States and Parameters with Artificial Neural Network (ANN)")
    st.markdown("""
    This page demonstrates the use of an ANN as a **soft sensor**. The network is trained to predict the process states (X, S, P) and key kinetic parameters ($\mu_{max}$, $Y_{X/S}$) using only the available online measurements (DO, pH, T).

    **Workflow:**
    1.  **Generate Data:** First, use the settings in the sidebar to run a bioprocess simulation. This creates the dataset for training the ANN.
    2.  **Configure & Train ANN:** Once the data is generated, configure the ANN's hyperparameters (architecture, optimizer, etc.) and click 'Train and Run ANN' to see the prediction results.
    """)
    st.markdown("---") # Visual separator

    # --- Helper function to reset state ---
    def reset_page_state():
        st.session_state.ann_data_generated = False
        # Clear other potential keys if they exist
        for key in ['ann_config', 'ann_results_generated']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # ==============================================================================
    # SIDEBAR SECTION (CONTROLS ONLY)
    # ==============================================================================
    with st.sidebar.expander("STEP 1: Generate Training Data", expanded=True):
        st.write("Configure the 'real' process simulation to create a dataset.")
        
        t_final_sim = st.slider("Simulation time (h)", 10, 100, 40, key="ann_tf_sim")
        dt_sim = 0.1

        st.markdown("**Process & Measurement Noise**")
        noise_level = st.select_slider(
            "Noise Level (for Q and R)",
            options=['Low', 'Medium', 'High'],
            value='Medium',
            key="ann_noise_level"
        )
        
        noise_map = {
            'Low':    {'Q_factor': 1e-7, 'R_factor': 1e-3},
            'Medium': {'Q_factor': 1e-5, 'R_factor': 5e-2},
            'High':   {'Q_factor': 1e-4, 'R_factor': 2e-1}
        }
        q_factor = noise_map[noise_level]['Q_factor']
        r_factor = noise_map[noise_level]['R_factor']

        Q_sim = np.diag([q_factor, q_factor*1e-3, q_factor, q_factor*0.1, q_factor*0.1])
        R_sim = np.diag([r_factor, r_factor*0.5, r_factor*10])

        if st.button("Generate Simulation Data"):
            run_data_generation(t_final_sim, dt_sim, Q_sim, R_sim)
        
        if st.session_state.ann_data_generated:
            if st.button("Reset and Generate New Data"):
                reset_page_state()

    st.sidebar.markdown("---")
    
    is_data_ready = st.session_state.ann_data_generated
    
    with st.sidebar.expander("STEP 2: Configure and Train ANN", expanded=is_data_ready):
        val_split = st.slider("Validation Set Size (%)", 5, 50, 20, 5, key='ann_split', disabled=not is_data_ready)
        n1 = st.number_input("Neurons in Hidden Layer 1", 1, 256, 64, key='ann_l1', disabled=not is_data_ready)
        n2 = st.number_input("Neurons in Hidden Layer 2", 1, 256, 64, key='ann_l2', disabled=not is_data_ready)
        act_func = st.selectbox("Hidden Layer Activation", ['relu', 'tanh', 'sigmoid'], key='ann_activation', disabled=not is_data_ready)
        optimizer_name = st.selectbox("Optimizer", ['adam', 'rmsprop', 'sgd'], key='ann_optimizer', disabled=not is_data_ready)
        learning_rate = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.1e", key='ann_lr', disabled=not is_data_ready)
        loss_func = st.selectbox("Loss Function", ['mean_squared_error', 'mean_absolute_error'], key='ann_loss', disabled=not is_data_ready)
        epochs = st.number_input("Epochs", 10, 1000, 200, key='ann_epochs', disabled=not is_data_ready)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16, key='ann_batch_size', disabled=not is_data_ready)

        if st.button("Train and Run ANN", disabled=not is_data_ready):
            config = {
                'val_split': val_split / 100.0, 'n1': n1, 'n2': n2,
                'act_func': act_func, 'optimizer_name': optimizer_name, 'learning_rate': learning_rate,
                'loss_func': loss_func, 'epochs': epochs, 'batch_size': batch_size
            }
            st.session_state.ann_config = config
            st.session_state.ann_results_generated = True

    # ==============================================================================
    # MAIN PAGE AREA (STATUS AND RESULTS)
    # ==============================================================================
    if not st.session_state.ann_data_generated:
        st.info("⬅️ **First Step:** Go to the sidebar, configure the simulation settings in STEP 1, and click 'Generate Simulation Data'.")
    else:
        st.success(f"**STEP 1 COMPLETE:** Training data is now available (Simulated for {st.session_state.ann_t_final} hours).")
        if not st.session_state.get('ann_results_generated', False):
             st.info("⬅️ **Next Step:** Configure the ANN in STEP 2 in the sidebar and click 'Train and Run ANN' to see the results.")

    # If the button was clicked and state is set, run the prediction and show results
    if st.session_state.get('ann_results_generated', False):
        run_ann_training_and_prediction(st.session_state.ann_config)

@st.cache_data(show_spinner=False) # Spinner is handled manually
def run_data_generation(_t_final, _dt, _Q, _R):
    """Runs the simulation to generate data for the ANN. Only updates session state."""
    with st.spinner("Running bioprocess simulation..."):
        n_states, n_meas = 5, 3
        Ks, alpha = 0.1, 0.1
        OD_sat, k_OUR, pH0, P0_meas_ref, k_acid, Tset, k_Temp = 8.0, 0.5, 7.0, 0.0, 0.2, 30, 0.02
        x_sym = ca.SX.sym('x', n_states)
        X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym)
        mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
        dX = mu_sym * X_sym
        dS = -(1 / Yxs_sym) * dX
        dP = alpha * dX
        x_next_sym = x_sym + _dt * ca.vertcat(dX, dS, dP, 0, 0)
        f_func = ca.Function('f', [x_sym], [x_next_sym])
        OD_val, pH_val, T_val = OD_sat - k_OUR*X_sym, pH0 - k_acid*(P_sym - P0_meas_ref), Tset + k_Temp*(X_sym*S_sym)
        z_sym = ca.vertcat(OD_val, pH_val, T_val)
        h_func = ca.Function('h', [x_sym], [z_sym])

        time_vec = np.arange(0, _t_final + _dt, _dt)
        N = len(time_vec)
        x_real = np.array([[0.1], [5.0], [0.0], [0.4], [0.5]])
        x_real_hist = np.zeros((n_states, N))
        z_meas_hist = np.zeros((n_meas, N))

        for k in range(N):
            x_real_hist[:, k] = x_real.flatten()
            z_noisefree = h_func(x_real).full()
            noise_meas = np.random.multivariate_normal(np.zeros(n_meas), _R).reshape(-1, 1)
            z_meas_hist[:, k] = (z_noisefree + noise_meas).flatten()
            if k < N - 1:
                x_real_next_no_noise = f_func(x_real).full()
                noise_proc = np.random.multivariate_normal(np.zeros(n_states), _Q).reshape(-1, 1)
                x_real = x_real_next_no_noise + noise_proc
                x_real[0:3] = np.maximum(x_real[0:3], 0)

    st.session_state.ann_data_generated = True
    st.session_state.ann_time_vec = time_vec
    st.session_state.ann_x_real_hist = x_real_hist
    st.session_state.ann_z_meas_hist = z_meas_hist
    st.session_state.ann_t_final = _t_final
    st.rerun()

def run_ann_training_and_prediction(config):
    """Takes generated data and ANN config, then trains and displays results in the main area."""
    with st.spinner("Preparing data, training network, and generating plots..."):
        time_vec = st.session_state.ann_time_vec
        x_real_hist = st.session_state.ann_x_real_hist
        z_meas_hist = st.session_state.ann_z_meas_hist

        X_data, Y_data = z_meas_hist.T, x_real_hist.T
        scaler_X, scaler_Y = StandardScaler(), StandardScaler()
        X_scaled, Y_scaled = scaler_X.fit_transform(X_data), scaler_Y.fit_transform(Y_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, Y_scaled, test_size=config['val_split'], random_state=42)

        tf.random.set_seed(42)
        ann_model = Sequential([
            Input(shape=(X_data.shape[1],)),
            Dense(config['n1'], activation=config['act_func']),
            Dense(config['n2'], activation=config['act_func']),
            Dense(Y_data.shape[1], activation='linear')])
        
        if config['optimizer_name'] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
        elif config['optimizer_name'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
        else: # sgd
            optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate'])
        
        ann_model.compile(optimizer=optimizer, loss=config['loss_func'])
        
        history = ann_model.fit(X_train, y_train, epochs=config['epochs'],
            batch_size=config['batch_size'], validation_data=(X_test, y_test), verbose=0)
        
        Y_pred_scaled = ann_model.predict(X_scaled)
        Y_pred_ann = scaler_Y.inverse_transform(Y_pred_scaled)

    st.subheader("ANN Estimation Results")
    tab1, tab2, tab3 = st.tabs(["📈 Result Plots", "📚 Learning Curve", "📄 Model Summary"])
    with tab1:
        plot_results(time_vec, x_real_hist, Y_pred_ann.T)
    with tab2:
        plot_learning_curve(history)
    with tab3:
        st.markdown("#### Network Architecture")
        summary_stream = StringIO()
        ann_model.summary(print_fn=lambda x: summary_stream.write(x + '\n'))
        st.code(summary_stream.getvalue())
        st.markdown("#### Hyperparameters Used")
        st.json(config)

def plot_results(time_vec, x_real_hist, x_ann_hist):
    """Generates the result plots comparing real vs ANN predicted values."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_states, axs_states = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig_states.suptitle('ANN Prediction: Process States', fontsize=16)
    state_labels = ['Biomass (X)', 'Substrate (S)', 'Product (P)']
    ylabels = ['Biomass (g/L)', 'Substrate (g/L)', 'Product (g/L)']
    for i in range(3):
        axs_states[i].plot(time_vec, x_real_hist[i, :], 'b-', label=f'{state_labels[i]} Real')
        axs_states[i].plot(time_vec, x_ann_hist[i, :], 'g--', lw=2.5, label=f'{state_labels[i]} Predicted (ANN)')
        axs_states[i].set_ylabel(ylabels[i])
        axs_states[i].legend()
    axs_states[2].set_xlabel('Time (h)')
    st.pyplot(fig_states)

    fig_params, axs_params = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig_params.suptitle('ANN Prediction: Model Parameters', fontsize=16)
    param_labels, param_ylabels = [r'$\mu_{max}$', r'$Y_{X/S}$'], [r'$\mu_{max}$ (1/h)', r'$Y_{X/S}$ (gX/gS)']
    for i in range(2):
        idx = i + 3
        axs_params[i].plot(time_vec, x_real_hist[idx, :], 'b-', label=f'{param_labels[i]} Real')
        axs_params[i].plot(time_vec, x_ann_hist[idx, :], 'g--', lw=2.5, label=f'{param_labels[i]} Predicted (ANN)')
        axs_params[i].set_ylabel(param_ylabels[i])
        axs_params[i].legend()
    axs_params[1].set_xlabel('Time (h)')
    st.pyplot(fig_params)

def plot_learning_curve(history):
    """Plots the training and validation loss."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('ANN Learning Curve'), ax.set_xlabel('Epoch'), ax.set_ylabel('Loss Value')
    ax.set_yscale('log'), ax.legend(), ax.grid(True)
    st.pyplot(fig)

if __name__ == '__main__':
    ann_page()