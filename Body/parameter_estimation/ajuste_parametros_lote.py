import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution, basinhopping
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t, zscore
import openpyxl
import seaborn as sns
import traceback
import warnings

#==========================================================================
# HELPER FUNCTIONS FOR ROBUST OPTIMIZATION
#==========================================================================

def detect_outliers_zscore(y_data, threshold=3.0):
    """
    Detect outliers in experimental data using Z-score method.
    
    Parameters
    ----------
    y_data : np.ndarray
        Experimental data array (can contain NaN values)
    threshold : float
        Z-score threshold for outlier detection (default: 3.0)
    
    Returns
    -------
    outlier_mask : np.ndarray
        Boolean mask where True indicates an outlier
    """
    valid_mask = ~np.isnan(y_data)
    if np.sum(valid_mask) < 4:  # Need at least 4 points
        return np.zeros_like(y_data, dtype=bool)
    
    valid_data = y_data[valid_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_scores = np.abs(zscore(valid_data, nan_policy='omit'))
    
    outlier_mask_valid = z_scores > threshold
    outlier_mask = np.zeros_like(y_data, dtype=bool)
    outlier_mask[valid_mask] = outlier_mask_valid
    return outlier_mask

def scale_parameters(params, bounds):
    """Scale parameters to [0, 1] range for better optimization."""
    scaled = np.zeros_like(params)
    for i, (p, (lb, ub)) in enumerate(zip(params, bounds)):
        if ub - lb > 1e-10:
            scaled[i] = (p - lb) / (ub - lb)
        else:
            scaled[i] = 0.5
    return scaled

def unscale_parameters(scaled_params, bounds):
    """Unscale parameters from [0, 1] range back to original range."""
    params = np.zeros_like(scaled_params)
    for i, (sp, (lb, ub)) in enumerate(zip(scaled_params, bounds)):
        params[i] = lb + sp * (ub - lb)
    return params

def validate_parameter_bounds(param_names, initial_guess, bounds):
    """Validate that initial guess is within bounds and bounds are reasonable."""
    warnings_list = []
    for i, (name, guess, (lb, ub)) in enumerate(zip(param_names, initial_guess, bounds)):
        if guess < lb or guess > ub:
            warnings_list.append(f"⚠️ {name}: Initial guess {guess:.4f} outside bounds [{lb:.4f}, {ub:.4f}]")
        if ub - lb < 1e-6:
            warnings_list.append(f"⚠️ {name}: Bounds very tight [{lb:.4f}, {ub:.4f}]")
        if (ub / lb) > 1000 and lb > 1e-6:
            warnings_list.append(f"ℹ️ {name}: Very wide bounds ratio ({ub/lb:.0f}x)")
    return warnings_list

def assess_parameter_identifiability(jacobian, param_names, threshold=1e-3):
    """Assess parameter identifiability using condition number and sensitivity analysis."""
    diagnostics = {
        'identifiable': True,
        'condition_number': np.nan,
        'low_sensitivity_params': [],
        'high_correlation_pairs': []
    }
    
    if jacobian is None or np.isnan(jacobian).any():
        diagnostics['identifiable'] = False
        return diagnostics
    
    try:
        # Check sensitivity (column norms)
        col_norms = np.linalg.norm(jacobian, axis=0)
        max_norm = np.max(col_norms)
        
        for i, (name, norm) in enumerate(zip(param_names, col_norms)):
            if norm < 1e-12:
                diagnostics['low_sensitivity_params'].append(name)
            elif max_norm > 1e-12 and norm / max_norm < threshold:
                if name not in diagnostics['low_sensitivity_params']:
                    diagnostics['low_sensitivity_params'].append(name)
        
        # Compute condition number
        jtj = jacobian.T @ jacobian
        jtj_det = np.linalg.det(jtj)
        if abs(jtj_det) < 1e-20:
            diagnostics['condition_number'] = np.inf
            diagnostics['identifiable'] = False
        else:
            cond_num = np.linalg.cond(jtj)
            diagnostics['condition_number'] = cond_num
            if cond_num > 1e10 or np.isinf(cond_num):
                diagnostics['identifiable'] = False
        
        # Check correlations
        if not np.isinf(diagnostics['condition_number']):
            try:
                cov_matrix = np.linalg.pinv(jtj)
                std_devs = np.sqrt(np.diag(cov_matrix))
                if np.all(std_devs > 1e-15):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                    n_params = len(param_names)
                    for i in range(n_params):
                        for j in range(i+1, n_params):
                            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > 0.95:
                                diagnostics['high_correlation_pairs'].append(
                                    (param_names[i], param_names[j], corr_matrix[i, j])
                                )
            except:
                pass
    except Exception:
        diagnostics['identifiable'] = False
        diagnostics['condition_number'] = np.inf
    
    return diagnostics

#==========================================================================

def ajuste_parametros_page():
    st.header("🔧 Kinetic Parameter Adjustment")

    # 1. Carga de datos experimentales
    with st.expander("📤 Load Experimental Data", expanded=True):
        uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
        if uploaded_file:
            df_exp = pd.read_excel(uploaded_file, engine='openpyxl')
            st.write("Data Preview:")
            st.dataframe(df_exp.head())

            # Validar formato
            required_cols = ['time', 'biomass', 'substrate', 'product']
            if not all(col in df_exp.columns for col in required_cols):
                st.error(f"The file must contain the column names: {', '.join(required_cols)}")
                st.stop()

            t_exp = df_exp['time'].values
            y_exp = df_exp[['biomass', 'substrate', 'product']].values.T

    # 2. Configuración del ajuste
    with st.sidebar:
        st.subheader("⚙️ Adjustment Settings")

        # Parámetros a ajustar
        st.markdown("### Parameters to Optimize")
        mumax_guess = st.number_input("initial μmax [1/h]", 0.01, 2.0, 0.5)
        Ks_guess = st.number_input("initial Ks [g/L]", 0.01, 5.0, 0.2)
        Yxs_guess = st.number_input("initial Yxs [g/g]", 0.1, 1.0, 0.5)
        Kd_guess = st.number_input("initial Kd [1/h]", 0.0, 0.5, 0.01)
        Ypx_guess = st.number_input("initial Ypx [g/g]", 0.1, 1.0, 0.3)

        # Condiciones iniciales
        st.markdown("### Initial Conditions")
        X0_fit = st.number_input("Initial Biomasa [g/L]", 0.1, 10.0, df_exp['biomass'].iloc[0] if uploaded_file else 1.0)
        S0_fit = st.number_input("Initial Sustrato [g/L]", 0.1, 100.0, df_exp['substrate'].iloc[0] if uploaded_file else 20.0)
        P0_fit = st.number_input("Initial Producto [g/L]", 0.0, 50.0, df_exp['product'].iloc[0] if uploaded_file else 0.0)
        O0_fit = st.number_input("Initial O2 [mg/L]", 0.0, 10.0, 8.0)
        atol = st.sidebar.number_input("Absolute Tolerance (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.sidebar.number_input("Relative Tolerance (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

        # Opciones de optimización
        st.markdown("### Optimization Options")
        metodo = st.selectbox("Optimization Method",
                            ['hybrid', 'L-BFGS-B', 'Nelder-Mead', 'differential_evolution', 'basinhopping'],
                            help="hybrid = Global search + Local refinement (recommended)")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_iter = st.number_input("Maximum Iterations", 10, 1000, 500)
            use_param_scaling = st.checkbox("Use Parameter Scaling", value=True,
                                           help="Scales parameters to [0,1] for better convergence")
        with col_opt2:
            n_restarts = st.number_input("Random Restarts", 0, 5, 0,
                                        help="Number of restarts with perturbed initial guess")
            detect_outliers = st.checkbox("Detect and Exclude Outliers", value=True,
                                         help="Uses Z-score method (threshold=3.0)")

    # 3. Multi-stage optimization function
    def multi_stage_optimization(objetivo_func, initial_guess, bounds, args_tuple, method='auto', 
                                 max_iter=500, use_scaling=True, n_restarts=0):
        """Multi-stage optimization: global search followed by local refinement."""
        t_exp, y_exp, outlier_masks = args_tuple
        
        # Build extended args with scaling support
        extended_args = (t_exp, y_exp, bounds, use_scaling, outlier_masks)
        
        # Scale initial guess if using scaling
        if use_scaling:
            initial_guess_opt = scale_parameters(np.array(initial_guess), bounds)
            bounds_opt = [(0, 1)] * len(bounds)
        else:
            initial_guess_opt = np.array(initial_guess)
            bounds_opt = bounds
        
        best_result = None
        best_fun = np.inf
        
        # Strategy selection
        if method == 'auto':
            method = 'hybrid'
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objetivo_func, bounds_opt, args=extended_args,
                    maxiter=max_iter, tol=1e-6, updating='deferred', 
                    workers=-1, seed=42, init='latinhypercube', atol=0, 
                    strategy='best1bin', popsize=15
                )
                best_result = result
                
            elif method == 'basinhopping':
                minimizer_kwargs = {
                    "method": "L-BFGS-B", 
                    "args": extended_args, 
                    "bounds": bounds_opt,
                    "options": {'maxiter': max_iter // 10, 'ftol': 1e-8}
                }
                result = basinhopping(
                    objetivo_func, initial_guess_opt, 
                    minimizer_kwargs=minimizer_kwargs,
                    niter=max(10, max_iter // 50), seed=42
                )
                best_result = result
                
            elif method == 'hybrid':
                st.info("🔄 Stage 1/2: Global search with Differential Evolution...")
                result_global = differential_evolution(
                    objetivo_func, bounds_opt, args=extended_args,
                    maxiter=min(100, max_iter // 3), tol=1e-5, 
                    updating='deferred', workers=-1, seed=42, 
                    init='latinhypercube', atol=0, strategy='best1bin', popsize=10
                )
                
                st.info("🎯 Stage 2/2: Local refinement with L-BFGS-B...")
                result_local = minimize(
                    objetivo_func, result_global.x, args=extended_args,
                    method='L-BFGS-B', bounds=bounds_opt,
                    options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-8}
                )
                
                best_result = result_local if result_local.fun < result_global.fun else result_global
                best_fun = best_result.fun
                
            else:
                minimizer_kwargs = {
                    "args": extended_args, 
                    "method": method,
                    "options": {'maxiter': max_iter, 'disp': False}
                }
                
                if method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                    minimizer_kwargs['bounds'] = bounds_opt
                    minimizer_kwargs['options'].update({'ftol': 1e-8, 'gtol': 1e-7})
                elif method == 'Nelder-Mead':
                    minimizer_kwargs['options'].update({'xatol': 1e-6, 'fatol': 1e-8})
                
                result = minimize(objetivo_func, initial_guess_opt, **minimizer_kwargs)
                best_result = result
                best_fun = result.fun
            
            # Random restarts
            if n_restarts > 0 and method not in ['differential_evolution', 'hybrid']:
                st.info(f"🔄 Performing {n_restarts} random restarts...")
                for i in range(n_restarts):
                    perturbation = np.random.uniform(-0.2, 0.2, len(initial_guess_opt))
                    perturbed_guess = np.clip(initial_guess_opt + perturbation, 
                                             [b[0] for b in bounds_opt], 
                                             [b[1] for b in bounds_opt])
                    
                    minimizer_kwargs_restart = minimizer_kwargs.copy()
                    if method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                        minimizer_kwargs_restart['options'] = {'maxiter': max_iter // 2, 'ftol': 1e-8}
                    
                    try:
                        result_restart = minimize(objetivo_func, perturbed_guess, **minimizer_kwargs_restart)
                        if result_restart.fun < best_fun:
                            best_result = result_restart
                            best_fun = result_restart.fun
                            st.success(f"✅ Restart {i+1}/{n_restarts}: Better solution (obj={best_fun:.6f})")
                    except:
                        pass
            
            # Unscale parameters
            if use_scaling and best_result is not None:
                best_result.x = unscale_parameters(best_result.x, bounds)
            
            return best_result
            
        except Exception as e:
            st.error(f"Multi-stage optimization error: {e}")
            class FailedResult:
                def __init__(self):
                    self.success = False
                    self.x = initial_guess
                    self.fun = 1e20
                    self.message = str(e)
            return FailedResult()

    # 4. Función objetivo y modelo
    def modelo_ode(t, y, params):
        X, S, P, O2 = y  # Desempaquetar variables del vector de estado
        mumax, Ks, Yxs, Kd, Ypx = params

        mu = mumax * S / (Ks + S)

        dXdt = mu * X - Kd * X
        dSdt = - (mu/Yxs) * X
        dPdt = Ypx * mu * X
        dO2dt = 0  # Simplificado para ejemplo

        return [dXdt, dSdt, dPdt, dO2dt]

    # 3.1 Función para calcular Jacobiano

    def compute_jacobian(params_opt, t_exp, y_exp, X0_fit, S0_fit, P0_fit, O0_fit):
        delta = 1e-6  # Perturbación pequeña
        jac = []

        # Simulación nominal
        sol_nominal = solve_ivp(modelo_ode, [0, t_exp[-1]],
                            [X0_fit, S0_fit, P0_fit, O0_fit],
                            args=(params_opt,),
                            t_eval=t_exp, atol=atol, rtol=rtol)
        y_nominal = np.vstack([sol_nominal.y[0], sol_nominal.y[1], sol_nominal.y[2]])

        # Calcular derivadas numéricas
        for i in range(len(params_opt)):
            params_perturbed = np.array(params_opt, dtype=float)
            params_perturbed[i] += delta

            sol_perturbed = solve_ivp(modelo_ode, [0, t_exp[-1]],
                                    [X0_fit, S0_fit, P0_fit, O0_fit],
                                    args=(params_perturbed,),
                                    t_eval=t_exp, atol=atol, rtol=rtol)

            y_perturbed = np.vstack([sol_perturbed.y[0], sol_perturbed.y[1], sol_perturbed.y[2]])
            derivative = (y_perturbed - y_nominal) / delta
            jac.append(derivative.flatten())  # Aplanar para todas las variables

        return np.array(jac).T  # Formato correcto (n_observaciones × n_parámetros)

    def objetivo(params_trial, t_exp, y_exp, bounds=None, use_scaling=False, outlier_masks=None):
        """Enhanced objective function with scaling and outlier detection support."""
        try:
            # Unscale parameters if scaling is enabled
            if use_scaling and bounds is not None:
                params = unscale_parameters(params_trial, bounds)
            else:
                params = params_trial
            
            # Integrate model
            sol = solve_ivp(modelo_ode,
                        [0, t_exp[-1]],
                        [X0_fit, S0_fit, P0_fit, O0_fit],
                        args=(params,),
                        t_eval=t_exp, atol=atol, rtol=rtol)

            if sol.status != 0:
                return 1e10
            
            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])
            
            # Apply outlier masks if provided
            residuals = y_pred - y_exp
            if outlier_masks is not None:
                for i in range(3):
                    residuals[i, outlier_masks[i]] = 0
            
            # Calculate weighted RMSE with scaling
            rmse_total = 0
            for i in range(3):
                valid_mask = ~np.isnan(y_exp[i])
                if outlier_masks is not None:
                    valid_mask = valid_mask & (~outlier_masks[i])
                
                if np.sum(valid_mask) > 0:
                    scale_factor = max(np.abs(y_exp[i][valid_mask]).max(), 1e-6)
                    rmse_var = np.sqrt(np.mean((residuals[i, valid_mask] / scale_factor)**2))
                    rmse_total += rmse_var
            
            return rmse_total
            
        except Exception as e:
            return 1e10

    # 4. Ejecutar ajuste
    if uploaded_file and st.button("🚀 Run Adjustment"):
        with st.spinner("Optimizing parameters..."):
            bounds = [(0.01, 2), (0.01, 5), (0.1, 1), (0, 0.5), (0.1, 1)]
            initial_guess = [mumax_guess, Ks_guess, Yxs_guess, Kd_guess, Ypx_guess]
            param_names = ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx']
            
            # Validate bounds
            warnings_list = validate_parameter_bounds(param_names, initial_guess, bounds)
            if warnings_list:
                with st.expander("⚠️ Parameter Bounds Validation", expanded=False):
                    for warning in warnings_list[:5]:
                        st.warning(warning)
            
            # Detect outliers if enabled
            outlier_masks = None
            if detect_outliers:
                outlier_masks = []
                outlier_counts = []
                var_names = ['Biomass', 'Substrate', 'Product']
                for i, var_name in enumerate(var_names):
                    mask = detect_outliers_zscore(y_exp[i], threshold=3.0)
                    outlier_masks.append(mask)
                    n_outliers = np.sum(mask)
                    outlier_counts.append(n_outliers)
                    if n_outliers > 0:
                        st.info(f"🔍 {var_name}: {n_outliers} outliers detected")
                
                if sum(outlier_counts) == 0:
                    st.success("✅ No outliers detected in data")
                    outlier_masks = None
            
            # Build args tuple
            args_tuple = (t_exp, y_exp, outlier_masks)
            
            # Run optimization
            result = multi_stage_optimization(
                objetivo, initial_guess, bounds, args_tuple,
                method=metodo, max_iter=max_iter, 
                use_scaling=use_param_scaling, n_restarts=n_restarts
            )
            
            # Convergence diagnostics
            with st.expander("🔍 Optimization Diagnostics", expanded=False):
                st.write("**Convergence Status:**")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if result.success:
                        st.success("• Success: ✅ Yes")
                    else:
                        st.warning("• Success: ⚠️ No")
                    st.write(f"• Iterations: {result.get('nit', result.get('nfev', 'N/A'))}")
                with col_d2:
                    st.write(f"• Message: {result.message if hasattr(result, 'message') else 'N/A'}")
                
                st.write("**Objective Function:**")
                st.write(f"  • Final value: {result.fun:.6e}")
                
                # Check for parameters near bounds
                params_near_bounds = []
                for i, (name, val) in enumerate(zip(param_names, result.x)):
                    lb, ub = bounds[i]
                    if abs(val - lb) / (ub - lb) < 0.05:
                        params_near_bounds.append(f"{name} (near lower)")
                    elif abs(val - ub) / (ub - lb) < 0.05:
                        params_near_bounds.append(f"{name} (near upper)")
                if params_near_bounds:
                    st.warning(f"⚠️ Parameters near bounds: {', '.join(params_near_bounds[:3])}")
                    st.caption("Consider adjusting bounds if physically reasonable")

            # Resultados del ajuste
            st.subheader("📊 Adjustment Results")
            params_opt = result.x
            st.write(f"**Final RMSE:** {result.fun:.4f}")

            # Tabla de parámetros
            parametros = pd.DataFrame({
                'Parameter': param_names,
                'Value': params_opt,
                'Units': ['1/h', 'g/L', 'g/g', '1/h', 'g/g']
            })
            st.dataframe(parametros.style.format({'Value': '{:.4f}'}))

            # 5. Análisis estadístico
            st.subheader("📈 Statistical Analysis")

            # Predicción final
            sol = solve_ivp(modelo_ode, [0, t_exp[-1]],
                           [X0_fit, S0_fit, P0_fit, O0_fit],
                           args=(params_opt,),
                           t_eval=t_exp, atol=atol, rtol=rtol)

            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])

            # Calcular métricas
            metricas = {
                'Variable': ['Biomass', 'Substrate', 'Product'],
                'R²': [r2_score(y_exp[i], y_pred[i]) for i in range(3)],
                'RMSE': [np.sqrt(mean_squared_error(y_exp[i], y_pred[i])) for i in range(3)]
            }
            st.dataframe(pd.DataFrame(metricas))

            # 6. Gráficos comparativos
            fig, ax = plt.subplots(3, 1, figsize=(10, 12))
            variables = ['Biomass', 'Substrate', 'Product']
            unidades = ['g/L', 'g/L', 'g/L']

            for i in range(3):
                ax[i].plot(t_exp, y_exp[i], 'o', label='Experimental')
                ax[i].plot(sol.t, sol.y[i], '--', label='Model')
                ax[i].set_title(f"{variables[i]} ({unidades[i]})")
                ax[i].legend()
                ax[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            # 7. Intervalos de confianza (metodología numérica)
            with st.spinner("Calculating confidence intervals..."):
                # Calcular residuos
                residuals = y_exp - y_pred
                residuals_flat = residuals.flatten()  # Aplanar para todas las variables

                # Calcular Jacobiano numérico
                jac = compute_jacobian(params_opt, t_exp, y_exp, X0_fit, S0_fit, P0_fit, O0_fit)

                # Parameter identifiability analysis
                st.subheader("🔬 Parameter Identifiability Analysis")
                identif_diag = assess_parameter_identifiability(jac, param_names, threshold=1e-3)
                
                col_id1, col_id2 = st.columns(2)
                with col_id1:
                    if identif_diag['identifiable']:
                        st.success("✅ Parameters are identifiable")
                    else:
                        st.error("⚠️ Poor parameter identifiability detected")
                    
                    cond_num = identif_diag['condition_number']
                    if not np.isnan(cond_num) and not np.isinf(cond_num):
                        st.metric("Condition Number", f"{cond_num:.2e}")
                        if cond_num < 1e6:
                            st.caption("🟢 Well-conditioned")
                        elif cond_num < 1e10:
                            st.caption("🟡 Moderately conditioned")
                        else:
                            st.caption("🔴 Ill-conditioned - parameters may be difficult to identify")
                    elif np.isinf(cond_num):
                        st.metric("Condition Number", "inf")
                        st.caption("🔴 Singular matrix - some parameters cannot be identified")
                    else:
                        st.metric("Condition Number", "N/A")
                
                with col_id2:
                    if identif_diag['low_sensitivity_params']:
                        st.warning("⚠️ Low sensitivity parameters:")
                        for param in identif_diag['low_sensitivity_params'][:3]:
                            st.caption(f"  • {param}")
                        st.caption("Consider fixing these parameters")
                    
                    if identif_diag['high_correlation_pairs']:
                        st.warning("⚠️ Highly correlated parameters:")
                        for p1, p2, corr in identif_diag['high_correlation_pairs'][:2]:
                            st.caption(f"  • {p1} ↔ {p2} (r={corr:.3f})")
                        st.caption("These parameters may be redundant")
                
                # Add actionable recommendations when identifiability is poor
                if not identif_diag['identifiable'] or np.isinf(identif_diag['condition_number']):
                    with st.expander("💡 How to Fix Identifiability Issues", expanded=True):
                        st.markdown("""
                        **Recommendations:**
                        
                        1. **Fix low-sensitivity parameters**: Set them to literature values
                           - Use fixed values instead of optimizing them
                        
                        2. **Collect more diverse data**:
                           - More time points help identify dynamics
                           - Ensure wide range of concentrations
                        
                        3. **Check parameter bounds**:
                           - Narrow bounds based on literature or prior knowledge
                           - Too wide bounds can cause poor identifiability
                        
                        4. **Increase data quality**:
                           - Reduce measurement noise
                           - Add replicate measurements
                        """)
                        
                        if identif_diag['low_sensitivity_params']:
                            low_params_str = ', '.join(identif_diag['low_sensitivity_params'][:5])
                            st.info(f"🎯 **Quick Fix**: Try re-running with these parameters fixed: {low_params_str}")

                # Calcular matriz de covarianza con estabilidad numérica
                try:
                    cov_matrix = np.linalg.pinv(jac.T @ jac) * (residuals_flat @ residuals_flat) / (len(residuals_flat) - len(params_opt))
                    std_errors = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    std_errors = np.full(len(params_opt), np.nan)

                # Calcular intervalos de confianza
                t_val = t.ppf(0.975, df=len(residuals_flat) - len(params_opt))
                intervals = t_val * std_errors

                # Mostrar resultados
                parametros['Interval ±'] = intervals
                parametros['95% CI - Lower bound'] = parametros['Value'] - intervals
                parametros['95% CI - Upper bound'] = parametros['Value'] + intervals

                st.write("Confidence Intervals (95%):")
                st.dataframe(parametros.style.format({
                    'Value': '{:.4f}',
                    'Interval ±': '{:.4f}',
                    '95% CI - Lower bound': '{:.4f}',
                    '95% CI - Upper bound': '{:.4f}'
                }))

            # Gráficos de Intervalos de Confianza
            # -------------------------
            st.subheader("📐 Parameter Confidence Intervals")

            fig_ci, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(parametros))
            ax.barh(y_pos, parametros['Value'], xerr=parametros['Interval ±'],
                align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(parametros['Parameter'])
            ax.invert_yaxis()
            ax.set_xlabel('Parameter Value')
            ax.set_title('95% Confidence Intervals')
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_ci)

        # Histogramas de Residuales
        st.subheader("📉 Error Distribution")

        fig_hist, axs = plt.subplots(1, 3, figsize=(15, 5))
        variables = ['Biomass', 'Substrate', 'Product']
        colors = ['#2ca02c', '#9467bd', '#d62728']

        for i, (var, color) in enumerate(zip(variables, colors)):
            residuals = y_exp[i] - y_pred[i]
            sns.histplot(residuals, kde=True, color=color, ax=axs[i])
            axs[i].set_title(f'Residuals {var}')
            axs[i].set_xlabel('Error (Experimental - Model)')
            axs[i].set_ylabel('Frecuence')
            axs[i].axvline(0, color='k', linestyle='--')
            axs[i].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_hist)

        # Gráficos de Correlación de Parámetros
        st.subheader("📌 Parameter Correlation Matrix")

        fig_corr, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = pd.DataFrame(jac, columns=parametros['Parameter']).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation between Parameters')
        st.pyplot(fig_corr)

    elif not uploaded_file:
        st.warning("⏳ Please upload a data file to begin the adjustment")

if __name__ == '__main__':
    ajuste_parametros_page()