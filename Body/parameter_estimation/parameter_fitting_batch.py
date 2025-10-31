import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
import openpyxl
import seaborn as sns

def parameter_fitting_page():
    st.header("🔧 Kinetic Parameter Adjustment")

    # 1. Load experimental data
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

    # 2. Adjustment configuration
    with st.sidebar:
        st.subheader("⚙️ Adjustment Settings")

        # Parameters to adjust
        st.markdown("### Parameters to Optimize")
        mumax_guess = st.number_input("initial μmax [1/h]", 0.01, 2.0, 0.5)
        Ks_guess = st.number_input("initial Ks [g/L]", 0.01, 5.0, 0.2)
        Yxs_guess = st.number_input("initial Yxs [g/g]", 0.1, 1.0, 0.5)
        Kd_guess = st.number_input("initial Kd [1/h]", 0.0, 0.5, 0.01)
        Ypx_guess = st.number_input("initial Ypx [g/g]", 0.1, 1.0, 0.3)

        # Initial conditions
        st.markdown("### Initial Conditions")
        X0_fit = st.number_input("Initial Biomass [g/L]", 0.1, 10.0, df_exp['biomass'].iloc[0] if uploaded_file else 1.0)
        S0_fit = st.number_input("Initial Substrate [g/L]", 0.1, 100.0, df_exp['substrate'].iloc[0] if uploaded_file else 20.0)
        P0_fit = st.number_input("Initial Product [g/L]", 0.0, 50.0, df_exp['product'].iloc[0] if uploaded_file else 0.0)
        O0_fit = st.number_input("Initial O2 [mg/L]", 0.0, 10.0, 8.0)
        atol = st.sidebar.number_input("Absolute Tolerance (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.sidebar.number_input("Relative Tolerance (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

        # Optimization options
        st.markdown("### Optimization Options")
        optimization_method = st.selectbox("Optimization Method",
                            ['L-BFGS-B', 'Nelder-Mead', 'differential_evolution'])
        max_iter = st.number_input("Maximum Iterations", 10, 1000, 100)

    # 3. Objective function and model
    def ode_model(t, y, params):
        X, S, P, O2 = y  # Unpack state vector variables
        mumax, Ks, Yxs, Kd, Ypx = params

        mu = mumax * S / (Ks + S)

        dXdt = mu * X - Kd * X
        dSdt = - (mu/Yxs) * X
        dPdt = Ypx * mu * X
        dO2dt = 0  # Simplified for example

        return [dXdt, dSdt, dPdt, dO2dt]

    # 3.1 Function to calculate Jacobian

    def compute_jacobian(params_opt, t_exp, y_exp, X0_fit, S0_fit, P0_fit, O0_fit):
        delta = 1e-6  # Small perturbation
        jac = []

        # Nominal simulation
        sol_nominal = solve_ivp(ode_model, [0, t_exp[-1]],
                            [X0_fit, S0_fit, P0_fit, O0_fit],
                            args=(params_opt,),
                            t_eval=t_exp, atol=atol, rtol=rtol)
        y_nominal = np.vstack([sol_nominal.y[0], sol_nominal.y[1], sol_nominal.y[2]])

        # Calculate numerical derivatives
        for i in range(len(params_opt)):
            params_perturbed = np.array(params_opt, dtype=float)
            params_perturbed[i] += delta

            sol_perturbed = solve_ivp(ode_model, [0, t_exp[-1]],
                                    [X0_fit, S0_fit, P0_fit, O0_fit],
                                    args=(params_perturbed,),
                                    t_eval=t_exp, atol=atol, rtol=rtol)

            y_perturbed = np.vstack([sol_perturbed.y[0], sol_perturbed.y[1], sol_perturbed.y[2]])
            derivative = (y_perturbed - y_nominal) / delta
            jac.append(derivative.flatten())  # Flatten for all variables

        return np.array(jac).T  # Correct format (n_observations × n_parameters)

    def objective(params, t_exp, y_exp):
        try:
            sol = solve_ivp(ode_model,
                        [0, t_exp[-1]],
                        [X0_fit, S0_fit, P0_fit, O0_fit],  # Initial state vector
                        args=(params,),  # Pass parameters correctly
                        t_eval=t_exp, atol=atol, rtol=rtol)

            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])
            rmse = np.sqrt(np.nanmean((y_pred - y_exp)**2))
            return rmse
        except Exception as e:
            st.error(f"Integration Error: {str(e)}")
            return 1e6

    # 4. Execute adjustment
    if uploaded_file and st.button("🚀 Run Adjustment"):
        with st.spinner("Optimizing parameters..."):
            bounds = [(0.01, 2), (0.01, 5), (0.1, 1), (0, 0.5), (0.1, 1)]
            initial_guess = [mumax_guess, Ks_guess, Yxs_guess, Kd_guess, Ypx_guess]

            if optimization_method == 'differential_evolution':
                result = differential_evolution(objective, bounds, args=(t_exp, y_exp))
            else:
                result = minimize(objective, initial_guess, args=(t_exp, y_exp),
                                method=optimization_method, bounds=bounds,
                                options={'maxiter': max_iter})

            # Adjustment results
            st.subheader("📊 Adjustment Results")
            params_opt = result.x
            st.write(f"** Final RMSE:** {result.fun:.4f}")

            # Parameters table
            parameters = pd.DataFrame({
                'Parameter': ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx'],
                'Value': params_opt,
                'Units': ['1/h', 'g/L', 'g/g', '1/h', 'g/g']
            })
            st.dataframe(parameters.style.format({'Value': '{:.4f}'}))

            # 5. Statistical analysis
            st.subheader("📈 Statistical Analysis")

            # Final prediction
            sol = solve_ivp(ode_model, [0, t_exp[-1]],
                           [X0_fit, S0_fit, P0_fit, O0_fit],
                           args=(params_opt,),
                           t_eval=t_exp, atol=atol, rtol=rtol)

            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])

            # Calculate metrics
            metrics = {
                'Variable': ['Biomass', 'Substrate', 'Product'],
                'R²': [r2_score(y_exp[i], y_pred[i]) for i in range(3)],
                'RMSE': [np.sqrt(mean_squared_error(y_exp[i], y_pred[i])) for i in range(3)]
            }
            st.dataframe(pd.DataFrame(metrics))

            # 6. Comparative plots
            fig, ax = plt.subplots(3, 1, figsize=(10, 12))
            variables = ['Biomass', 'Substrate', 'Product']
            units = ['g/L', 'g/L', 'g/L']

            for i in range(3):
                ax[i].plot(t_exp, y_exp[i], 'o', label='Experimental')
                ax[i].plot(sol.t, sol.y[i], '--', label='Model')
                ax[i].set_title(f"{variables[i]} ({units[i]})")
                ax[i].legend()
                ax[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            # 7. Confidence intervals (numerical methodology)
            with st.spinner("Calculating confidence intervals..."):
                # Calculate residuals
                residuals = y_exp - y_pred
                residuals_flat = residuals.flatten()  # Flatten for all variables

                # Calculate numerical Jacobian
                jac = compute_jacobian(params_opt, t_exp, y_exp, X0_fit, S0_fit, P0_fit, O0_fit)

                # Calculate covariance matrix with numerical stability
                try:
                    cov_matrix = np.linalg.pinv(jac.T @ jac) * (residuals_flat @ residuals_flat) / (len(residuals_flat) - len(params_opt))
                    std_errors = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    std_errors = np.full(len(params_opt), np.nan)

                # Calculate confidence intervals
                t_val = t.ppf(0.975, df=len(residuals_flat) - len(params_opt))
                intervals = t_val * std_errors

                # Show results
                parameters['Interval ±'] = intervals
                parameters['95% CI - Lower bound'] = parameters['Value'] - intervals
                parameters['95% CI - Upper bound'] = parameters['Value'] + intervals

                st.write("Confidence Intervals (95%):")
                st.dataframe(parameters.style.format({
                    'Value': '{:.4f}',
                    'Interval ±': '{:.4f}',
                    '95% CI - Lower bound': '{:.4f}',
                    '95% CI - Upper bound': '{:.4f}'
                }))

            # Confidence Interval Plots
            # -------------------------
            st.subheader("📐 Parameter Confidence Intervals")

            fig_ci, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(parameters))
            ax.barh(y_pos, parameters['Value'], xerr=parameters['Interval ±'],
                align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(parameters['Parameter'])
            ax.invert_yaxis()
            ax.set_xlabel('Parameter Value')
            ax.set_title('95% Confidence Intervals')
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_ci)

        # Residual Histograms
        st.subheader("📉 Error Distribution")

        fig_hist, axs = plt.subplots(1, 3, figsize=(15, 5))
        variables = ['Biomass', 'Substrate', 'Product']
        colors = ['#2ca02c', '#9467bd', '#d62728']

        for i, (var, color) in enumerate(zip(variables, colors)):
            residuals = y_exp[i] - y_pred[i]
            sns.histplot(residuals, kde=True, color=color, ax=axs[i])
            axs[i].set_title(f'Residuals {var}')
            axs[i].set_xlabel('Error (Experimental - Model)')
            axs[i].set_ylabel('Frequency')
            axs[i].axvline(0, color='k', linestyle='--')
            axs[i].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_hist)

        # Parameter Correlation Plots
        st.subheader("📌 Parameter Correlation Matrix")

        fig_corr, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = pd.DataFrame(jac, columns=parameters['Parameter']).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation between Parameters')
        st.pyplot(fig_corr)

    elif not uploaded_file:
        st.warning("⏳ Please upload a data file to begin the adjustment")

if __name__ == '__main__':
    parameter_fitting_page()