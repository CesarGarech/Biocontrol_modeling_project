"""
Digital Twin — Sub-page 3: Machine Learning Prediction
========================================================
Interactive ML pipeline for predicting ethanol composition in the distillate:
  - Uses data generated from the Data Analysis Step 1
  - Trains and evaluates multiple ML models (Neural Network, Random Forest, 
    Decision Tree, SVR, Gradient Boosting)
  - Compares model performance with visualizations
  - Shows Plant vs Model prediction comparison for the best model
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# Import shared configuration from Simulation/config.py
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# Import DWSIM data generator (with safe fallback)
try:
    from dwsim_data_generator import (  # noqa: E402
        generate_dwsim_data,
        validate_dwsim_installation,
    )
    _DWSIM_GENERATOR_OK = True
except Exception:
    _DWSIM_GENERATOR_OK = False

    def validate_dwsim_installation():  # type: ignore[misc]
        """Stub when DWSIM modules are not available."""
        return False, "DWSIM integration is not available. Please ensure DWSIM is installed and the dwsim_data_generator module is accessible."

    def generate_dwsim_data(n_points, perturbations=None):  # type: ignore[misc]
        """Stub — never called in production when _DWSIM_GENERATOR_OK is False."""
        raise RuntimeError("dwsim_data_generator is not available.")

# Constants
_EPSILON = 1e-9
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP  # Design target: 80 mol% ethanol

# Composition model coefficients (empirical correlation)
_ENERGY_RATIO_COEFF = 0.05     # Impact of Q_cond/Q_reb ratio on separation quality
_FLOW_RATIO_COEFF = 0.03       # Impact of feed/distillate ratio on purity
_COMPOSITION_NOISE_STD = 0.02  # Process variability (±2% standard deviation)

# Composition bounds and normalization
_MIN_ETHANOL_COMPOSITION = 0.70   # Minimum ethanol mol fraction in distillate
_MAX_ETHANOL_COMPOSITION = 0.95   # Maximum ethanol mol fraction in distillate
_Q_RATIO_CENTER = 0.5             # Center point for Q_cond/Q_reb ratio normalization
_Q_RATIO_SCALE = 0.5              # Scale factor for Q ratio normalization

# Training constraints
_MIN_TRAINING_SAMPLES = 10        # Minimum number of samples required for training


# ── Helper: Generate ethanol composition based on process conditions ─────────
def _generate_ethanol_composition(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ethanol composition in distillate based on process variables.
    
    The ethanol composition is influenced by:
    - Feed flow rate (F_feed)
    - Top (distillate) flow rate (F_top)
    - Energy duties (Q_cond, Q_reb)
    - Random noise to simulate process variability
    
    Returns a copy of df with an additional 'Ethanol_Composition' column (mol fraction).
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    
    # Normalize variables to [0, 1] range for composition calculation
    f_feed_norm = (df_out['F_feed_raw'] - df_out['F_feed_raw'].min()) / (df_out['F_feed_raw'].max() - df_out['F_feed_raw'].min() + _EPSILON)
    f_top_norm = (df_out['F_top_raw'] - df_out['F_top_raw'].min()) / (df_out['F_top_raw'].max() - df_out['F_top_raw'].min() + _EPSILON)
    q_ratio_norm = (df_out['Q_cond_raw'] / (df_out['Q_reb_raw'] + _EPSILON) - _Q_RATIO_CENTER) / _Q_RATIO_SCALE
    
    # Physical model: Higher energy ratio → better separation → higher ethanol purity
    # Lower feed/distillate ratio → better separation
    base_composition = (
        _TARGET_ETHANOL_TOP 
        + _ENERGY_RATIO_COEFF * q_ratio_norm 
        - _FLOW_RATIO_COEFF * (f_feed_norm - f_top_norm)
    )
    
    # Add process noise (±2% variability)
    noise = rng.normal(0, _COMPOSITION_NOISE_STD, len(df_out))
    df_out['Ethanol_Composition'] = np.clip(base_composition + noise, _MIN_ETHANOL_COMPOSITION, _MAX_ETHANOL_COMPOSITION)
    
    return df_out


# ── Helper: Build Neural Network model ────────────────────────────────────────
def _build_neural_network(input_dim: int, seed: int = 42) -> keras.Model:
    """Build a simple feedforward neural network for regression."""
    tf.random.set_seed(seed)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ── Helper: Train and evaluate all models ────────────────────────────────────
def _train_and_evaluate_models(
    X_train, X_test, y_train, y_test, 
    selected_models: list, seed: int = 42, epochs: int = 100
) -> dict:
    """
    Train and evaluate selected ML models.
    
    Returns a dictionary with model names as keys and dictionaries containing
    the trained model, predictions, and metrics as values.
    """
    results = {}
    
    # Define all available models
    models_config = {
        'Neural Network': lambda: _build_neural_network(X_train.shape[1], seed) if _TF_AVAILABLE else None,
        'Random Forest': lambda: RandomForestRegressor(n_estimators=100, random_state=seed, max_depth=10),
        'Decision Tree': lambda: DecisionTreeRegressor(random_state=seed, max_depth=8),
        'SVR': lambda: SVR(kernel='rbf', C=100, gamma='scale'),
        'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=100, random_state=seed, max_depth=5, learning_rate=0.1),
    }
    
    for model_name in selected_models:
        if model_name not in models_config:
            continue
            
        model = models_config[model_name]()
        
        if model is None:
            st.warning(f"⚠️ {model_name} is not available (TensorFlow not installed).")
            continue
        
        # Train model
        if model_name == 'Neural Network':
            # Neural network training with reduced verbosity
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=16,
                validation_split=0.2,
                verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
            )
            y_pred = model.predict(X_test, verbose=0).flatten()
            training_history = history.history
        else:
            # Scikit-learn models
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            training_history = None
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'metrics': {
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'RMSE': np.sqrt(mse)
            },
            'training_history': training_history
        }
    
    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _plot_metrics_comparison(results: dict) -> plt.Figure:
    """Create bar charts comparing metrics across all models."""
    metrics_names = ['R²', 'MSE', 'MAE', 'RMSE']
    n_models = len(results)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Machine Learning Models — Performance Comparison",
        fontsize=14, fontweight="bold"
    )
    
    # Prepare data
    model_names = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_models))
    
    for idx, metric in enumerate(metrics_names):
        ax = axs[idx // 2, idx % 2]
        values = [results[name]['metrics'][metric] for name in model_names]
        
        bars = ax.bar(range(n_models), values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9
            )
        
        # For R², highlight the best (highest)
        if metric == 'R²':
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    fig.tight_layout()
    return fig


def _plot_plant_vs_model(y_test, y_pred, model_name: str, timestamps=None) -> plt.Figure:
    """Create comparison plots between actual (plant) and predicted values."""
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Plant vs Model Prediction — {model_name}",
        fontsize=14, fontweight="bold"
    )
    
    # Prepare x-axis (use timestamps if available, otherwise use sample indices)
    if timestamps is not None:
        x_axis = timestamps
        x_label = "Time"
    else:
        x_axis = np.arange(len(y_test))
        x_label = "Sample Index"
    
    # Panel 1: Time series comparison
    axs[0].plot(x_axis, y_test, 'o-', color='blue', label='Plant (Actual)', alpha=0.7, markersize=4)
    axs[0].plot(x_axis, y_pred, 's-', color='red', label='Model (Predicted)', alpha=0.7, markersize=4)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Ethanol Composition (mol fraction)")
    axs[0].set_title("Time Series: Plant vs Model")
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)
    
    # Panel 2: Parity plot (predicted vs actual)
    axs[1].scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display R²
    r2 = r2_score(y_test, y_pred)
    axs[1].text(
        0.05, 0.95, f'R² = {r2:.4f}',
        transform=axs[1].transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    axs[1].set_xlabel("Plant (Actual) Ethanol Composition")
    axs[1].set_ylabel("Model (Predicted) Ethanol Composition")
    axs[1].set_title("Parity Plot: Predicted vs Actual")
    axs[1].legend(loc='best')
    axs[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def _run_dwsim_comparison(test_data: dict, df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Run DWSIM simulations for test set conditions and return predictions.
    
    Parameters
    ----------
    test_data : dict
        Dictionary containing 'idx_test' with test set indices
    df_ml : pd.DataFrame
        Full dataframe with all data including test set
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DWSIM predictions for test set, including ethanol composition
    """
    test_idx = test_data['idx_test']
    df_test = df_ml.iloc[test_idx].copy()
    
    # Extract feed flow perturbations from test set
    # We need to calculate perturbations relative to base feed flow
    perturbations = df_test['F_feed_raw'].values - _cfg.FLOW_FEED_BASE
    
    # Run DWSIM simulations
    df_dwsim = generate_dwsim_data(len(test_idx), perturbations=perturbations)
    
    # Generate ethanol composition for DWSIM data using the same correlation
    # but with DWSIM's more accurate flow predictions
    df_dwsim_composition = _generate_ethanol_composition(df_dwsim, seed=42)
    
    return df_dwsim_composition


def _plot_dwsim_ml_comparison(
    y_test, y_ml_pred, y_dwsim_pred, model_name: str, timestamps=None
) -> plt.Figure:
    """Create comparison plots between synthetic plant, ML model, and DWSIM."""
    fig, axs = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(
        f"Three-Way Comparison: Synthetic Plant vs ML Model vs DWSIM — {model_name}",
        fontsize=14, fontweight="bold"
    )
    
    # Prepare x-axis
    if timestamps is not None:
        x_axis = timestamps
        x_label = "Time"
    else:
        x_axis = np.arange(len(y_test))
        x_label = "Sample Index"
    
    # Panel 1: Time series comparison (all three)
    axs[0].plot(x_axis, y_test, 'o-', color='blue', label='Synthetic Plant', 
                alpha=0.7, markersize=5, linewidth=2)
    axs[0].plot(x_axis, y_ml_pred, 's-', color='red', label=f'ML Model ({model_name})', 
                alpha=0.7, markersize=4, linewidth=1.5)
    axs[0].plot(x_axis, y_dwsim_pred, '^-', color='green', label='DWSIM Simulation', 
                alpha=0.7, markersize=4, linewidth=1.5)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Ethanol Composition (mol fraction)")
    axs[0].set_title("Time Series: Three-Way Comparison")
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)
    
    # Panel 2: ML Model vs Synthetic Plant (parity plot)
    axs[1].scatter(y_test, y_ml_pred, alpha=0.6, s=60, edgecolors='black', 
                   linewidth=0.5, color='red', label='ML Model')
    min_val = min(y_test.min(), y_ml_pred.min())
    max_val = max(y_test.max(), y_ml_pred.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
                label='Perfect Prediction')
    r2_ml = r2_score(y_test, y_ml_pred)
    mae_ml = mean_absolute_error(y_test, y_ml_pred)
    axs[1].text(
        0.05, 0.95, f'ML Model\nR² = {r2_ml:.4f}\nMAE = {mae_ml:.4f}',
        transform=axs[1].transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    axs[1].set_xlabel("Synthetic Plant Ethanol Composition")
    axs[1].set_ylabel("Predicted Ethanol Composition")
    axs[1].set_title("Parity Plot: ML Model vs Synthetic Plant")
    axs[1].legend(loc='lower right')
    axs[1].grid(True, alpha=0.3)
    
    # Panel 3: DWSIM vs Synthetic Plant (parity plot)
    axs[2].scatter(y_test, y_dwsim_pred, alpha=0.6, s=60, edgecolors='black', 
                   linewidth=0.5, color='green', label='DWSIM')
    min_val = min(y_test.min(), y_dwsim_pred.min())
    max_val = max(y_test.max(), y_dwsim_pred.max())
    axs[2].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
                label='Perfect Prediction')
    r2_dwsim = r2_score(y_test, y_dwsim_pred)
    mae_dwsim = mean_absolute_error(y_test, y_dwsim_pred)
    axs[2].text(
        0.05, 0.95, f'DWSIM\nR² = {r2_dwsim:.4f}\nMAE = {mae_dwsim:.4f}',
        transform=axs[2].transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )
    axs[2].set_xlabel("Synthetic Plant Ethanol Composition")
    axs[2].set_ylabel("Predicted Ethanol Composition")
    axs[2].set_title("Parity Plot: DWSIM vs Synthetic Plant")
    axs[2].legend(loc='lower right')
    axs[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


# ── Streamlit page ────────────────────────────────────────────────────────────
def ml_prediction_page():
    """Main page for ML-based ethanol composition prediction."""
    st.header("🤖 Machine Learning Prediction — Ethanol Composition")
    st.markdown("""
    Use **Machine Learning models** to predict the **ethanol composition** in the 
    distillate stream based on process variables from the SCADA data.
    
    **Workflow:**
    1. Load data generated from **Data Analysis → Step 1**
    2. Select ML models to train and evaluate
    3. Compare model performance (R², MSE, MAE, RMSE)
    4. View Plant vs Model predictions for the best-performing model
    """)
    st.markdown("---")
    
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 ML Configuration")
        
        with st.expander("1. Model Selection", expanded=True):
            available_models = ['Neural Network', 'Random Forest', 'Decision Tree', 
                              'SVR', 'Gradient Boosting']
            
            if not _TF_AVAILABLE:
                st.warning("⚠️ TensorFlow not available. Neural Network disabled.")
                # Create a copy to avoid modifying the original list
                available_models = [m for m in available_models if m != 'Neural Network']
            
            selected_models = st.multiselect(
                "Select models to train and compare:",
                available_models,
                default=available_models[:3] if len(available_models) >= 3 else available_models,
                key="ml_models"
            )
        
        with st.expander("2. Training Configuration", expanded=True):
            test_size = st.slider(
                "Test set size (%)", 10, 40, 20, 5,
                key="ml_test_size",
                help="Percentage of data reserved for testing"
            )
            nn_epochs = st.slider(
                "Neural Network epochs", 50, 300, 100, 10,
                key="ml_epochs",
                help="Number of training epochs for Neural Network (only affects NN model)"
            )
            random_state = st.number_input(
                "Random seed", 0, 9999, 42, step=1, format="%d",
                key="ml_seed",
                help="Seed for reproducibility of random operations (0-9999)"
            )
        
        # Check DWSIM availability
        dwsim_ok, dwsim_msg = validate_dwsim_installation()
        
        with st.expander("3. DWSIM Comparison", expanded=False):
            st.markdown("""
            Compare the best ML model with **rigorous DWSIM simulations** 
            using the test set conditions.
            """)
            
            enable_dwsim_comparison = st.checkbox(
                "Enable DWSIM comparison",
                value=False,
                key="ml_enable_dwsim",
                help="Run DWSIM simulations for test set data and compare with ML predictions",
                disabled=not (dwsim_ok and _DWSIM_GENERATOR_OK),
            )
            
            if not (dwsim_ok and _DWSIM_GENERATOR_OK):
                st.info(f"ℹ️ DWSIM not available: {dwsim_msg}")
            else:
                st.success("✅ DWSIM available for comparison")
    
    # ── Check if data is available ───────────────────────────────────────────
    if "da_df_raw" not in st.session_state:
        st.warning("""
        ⚠️ **No data available!**
        
        Please go to **Data Analysis** → **Step 1** and generate SCADA data first.
        The ML models will use that data for training.
        """)
        return
    
    # ── Load and prepare data ─────────────────────────────────────────────────
    st.subheader("📊 Data Preparation")
    
    if st.button("🔄 Prepare Training Data", key="btn_prepare"):
        with st.spinner("Preparing data for ML training..."):
            # Get raw data from Data Analysis Step 1
            df_raw = st.session_state["da_df_raw"].copy()
            
            # Generate ethanol composition (target variable)
            df_ml = _generate_ethanol_composition(df_raw, seed=random_state)
            
            # Store in session state
            st.session_state["ml_df"] = df_ml
            st.success(f"✅ Data prepared — {len(df_ml)} samples with ethanol composition generated.")
    
    if "ml_df" in st.session_state:
        df_ml = st.session_state["ml_df"]
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df_ml))
        col2.metric("Features", 5, help="F_feed, F_top, F_bottom, Q_cond, Q_reb")
        col3.metric("Target Variable", "Ethanol Composition", help="Mol fraction in distillate")
        
        # Show sample data
        with st.expander("📋 View Sample Data"):
            display_cols = ['Timestamp', 'F_feed_raw', 'F_top_raw', 'F_bottom_raw', 
                          'Q_cond_raw', 'Q_reb_raw', 'Ethanol_Composition']
            st.dataframe(
                df_ml[display_cols].head(10).style.format({
                    'F_feed_raw': '{:.1f}',
                    'F_top_raw': '{:.1f}',
                    'F_bottom_raw': '{:.1f}',
                    'Q_cond_raw': '{:.2f}',
                    'Q_reb_raw': '{:.2f}',
                    'Ethanol_Composition': '{:.4f}',
                }),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # ── Model Training ────────────────────────────────────────────────────
        st.subheader("🎯 Model Training & Evaluation")
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar.")
        else:
            if st.button("🚀 Train Models", key="btn_train"):
                if len(df_ml) < _MIN_TRAINING_SAMPLES:
                    st.error(f"❌ Not enough data points. Please generate at least {_MIN_TRAINING_SAMPLES} samples in Step 1.")
                else:
                    with st.spinner(f"Training {len(selected_models)} models..."):
                        # Prepare features and target
                        feature_cols = ['F_feed_raw', 'F_top_raw', 'F_bottom_raw', 
                                      'Q_cond_raw', 'Q_reb_raw']
                        X = df_ml[feature_cols].values
                        y = df_ml['Ethanol_Composition'].values
                        
                        # Split data and get indices
                        splitter = ShuffleSplit(n_splits=1, test_size=test_size/100, random_state=random_state)
                        train_idx, test_idx = next(splitter.split(X))
                        
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Standardize features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train and evaluate models
                        results = _train_and_evaluate_models(
                            X_train_scaled, X_test_scaled,
                            y_train, y_test,
                            selected_models, random_state, nn_epochs
                        )
                        
                        # Store results
                        st.session_state["ml_results"] = results
                        st.session_state["ml_test_data"] = {
                            'y_test': y_test,
                            'idx_test': test_idx,
                            'timestamps': df_ml.iloc[test_idx]['Timestamp'].values
                        }
                        st.session_state["ml_scaler"] = scaler
                        st.session_state["ml_feature_cols"] = feature_cols
                        
                    st.success(f"✅ Successfully trained {len(results)} models!")
            
            # ── Display Results ───────────────────────────────────────────────
            if "ml_results" in st.session_state:
                results = st.session_state["ml_results"]
                test_data = st.session_state["ml_test_data"]
                
                st.markdown("---")
                st.subheader("📈 Performance Comparison")
                
                # Metrics table
                st.markdown("**Model Performance Metrics:**")
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'R²': [results[m]['metrics']['R²'] for m in results.keys()],
                    'MSE': [results[m]['metrics']['MSE'] for m in results.keys()],
                    'MAE': [results[m]['metrics']['MAE'] for m in results.keys()],
                    'RMSE': [results[m]['metrics']['RMSE'] for m in results.keys()],
                })
                
                # Highlight best model (highest R²)
                best_model_idx = metrics_df['R²'].idxmax()
                best_model_name = metrics_df.loc[best_model_idx, 'Model']
                
                st.dataframe(
                    metrics_df.style.format({
                        'R²': '{:.4f}',
                        'MSE': '{:.6f}',
                        'MAE': '{:.4f}',
                        'RMSE': '{:.4f}',
                    }).highlight_max(subset=['R²'], color='lightgreen')
                     .highlight_min(subset=['MSE', 'MAE', 'RMSE'], color='lightgreen'),
                    use_container_width=True
                )
                
                st.info(f"🏆 **Best Model:** {best_model_name} (R² = {metrics_df.loc[best_model_idx, 'R²']:.4f})")
                
                # Metrics comparison plot
                fig_metrics = _plot_metrics_comparison(results)
                st.pyplot(fig_metrics)
                plt.close(fig_metrics)
                
                st.markdown("---")
                st.subheader("🔍 Plant vs Model Prediction")
                
                # Model selection for detailed view
                selected_model_detail = st.selectbox(
                    "Select model for detailed comparison:",
                    options=list(results.keys()),
                    index=list(results.keys()).index(best_model_name),
                    key="ml_detail_model"
                )
                
                # Plant vs Model plot
                y_test = test_data['y_test']
                y_pred = results[selected_model_detail]['predictions']
                timestamps = test_data['timestamps']
                
                fig_comparison = _plot_plant_vs_model(
                    y_test, y_pred, selected_model_detail, timestamps
                )
                st.pyplot(fig_comparison)
                plt.close(fig_comparison)
                
                # Additional metrics for selected model
                with st.expander("📊 Detailed Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = results[selected_model_detail]['metrics']
                    col1.metric("R² Score", f"{metrics['R²']:.4f}")
                    col2.metric("MSE", f"{metrics['MSE']:.6f}")
                    col3.metric("MAE", f"{metrics['MAE']:.4f}")
                    col4.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    
                    # Error distribution
                    errors = y_test - y_pred
                    st.markdown("**Prediction Error Statistics:**")
                    error_stats = pd.DataFrame({
                        'Metric': ['Mean Error', 'Std Dev', 'Min Error', 'Max Error'],
                        'Value': [errors.mean(), errors.std(), errors.min(), errors.max()]
                    })
                    st.dataframe(
                        error_stats.style.format({'Value': '{:.4f}'}),
                        use_container_width=True
                    )
                
                # ── DWSIM Comparison Section ─────────────────────────────────
                if enable_dwsim_comparison:
                    st.markdown("---")
                    st.subheader("🔬 DWSIM vs ML Model Comparison")
                    
                    st.markdown("""
                    Compare the **best ML model** with **rigorous DWSIM simulations** 
                    using the test set conditions.
                    """)
                    
                    if st.button("⚗️ Run DWSIM Comparison", key="btn_dwsim_compare"):
                        use_dwsim = enable_dwsim_comparison and dwsim_ok and _DWSIM_GENERATOR_OK
                        
                        if not use_dwsim:
                            st.error("❌ DWSIM is not available for comparison.")
                        else:
                            with st.spinner("Running DWSIM simulations for test set..."):
                                try:
                                    # Run DWSIM simulations
                                    df_dwsim = _run_dwsim_comparison(test_data, df_ml)
                                    
                                    # Store DWSIM results
                                    st.session_state["ml_dwsim_results"] = {
                                        'predictions': df_dwsim['Ethanol_Composition'].values,
                                        'df': df_dwsim
                                    }
                                    
                                    st.success(f"✅ DWSIM simulations completed for {len(test_data['y_test'])} test points!")
                                    
                                except Exception as exc:
                                    st.error(f"❌ DWSIM simulation failed: {exc}")
                                    st.session_state.pop("ml_dwsim_results", None)
                    
                    # Display DWSIM comparison results
                    if "ml_dwsim_results" in st.session_state:
                        dwsim_results = st.session_state["ml_dwsim_results"]
                        y_dwsim_pred = dwsim_results['predictions']
                        
                        # Get best model predictions
                        y_test = test_data['y_test']
                        y_ml_pred = results[best_model_name]['predictions']
                        timestamps = test_data['timestamps']
                        
                        # Calculate metrics for DWSIM
                        r2_ml = r2_score(y_test, y_ml_pred)
                        mae_ml = mean_absolute_error(y_test, y_ml_pred)
                        rmse_ml = np.sqrt(mean_squared_error(y_test, y_ml_pred))
                        
                        r2_dwsim = r2_score(y_test, y_dwsim_pred)
                        mae_dwsim = mean_absolute_error(y_test, y_dwsim_pred)
                        rmse_dwsim = np.sqrt(mean_squared_error(y_test, y_dwsim_pred))
                        
                        # Comparison metrics table
                        st.markdown("**Performance Comparison:**")
                        comparison_df = pd.DataFrame({
                            'Model': [best_model_name, 'DWSIM'],
                            'R²': [r2_ml, r2_dwsim],
                            'MAE': [mae_ml, mae_dwsim],
                            'RMSE': [rmse_ml, rmse_dwsim]
                        })
                        
                        st.dataframe(
                            comparison_df.style.format({
                                'R²': '{:.4f}',
                                'MAE': '{:.4f}',
                                'RMSE': '{:.4f}',
                            }).highlight_max(subset=['R²'], color='lightgreen')
                             .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'),
                            use_container_width=True
                        )
                        
                        # Three-way comparison plot
                        fig_dwsim_comparison = _plot_dwsim_ml_comparison(
                            y_test, y_ml_pred, y_dwsim_pred, best_model_name, timestamps
                        )
                        st.pyplot(fig_dwsim_comparison)
                        plt.close(fig_dwsim_comparison)
                        
                        # Additional comparison insights
                        with st.expander("📈 Comparison Insights"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**{best_model_name} (ML Model)**")
                                st.metric("R² Score", f"{r2_ml:.4f}")
                                st.metric("MAE", f"{mae_ml:.4f}")
                                st.metric("RMSE", f"{rmse_ml:.4f}")
                            
                            with col2:
                                st.markdown("**DWSIM Simulation**")
                                st.metric("R² Score", f"{r2_dwsim:.4f}")
                                st.metric("MAE", f"{mae_dwsim:.4f}")
                                st.metric("RMSE", f"{rmse_dwsim:.4f}")
                            
                            # Winner determination
                            if r2_ml > r2_dwsim:
                                st.success(f"🏆 **{best_model_name}** achieves better R² score!")
                            elif r2_dwsim > r2_ml:
                                st.success("🏆 **DWSIM** achieves better R² score!")
                            else:
                                st.info("Both models show similar R² performance.")
                    else:
                        st.info("👆 Press **⚗️ Run DWSIM Comparison** to compare with DWSIM simulations.")
                
            else:
                st.info("👆 Press **🚀 Train Models** to see results and comparisons.")
    else:
        st.info("👆 Press **🔄 Prepare Training Data** to continue.")
