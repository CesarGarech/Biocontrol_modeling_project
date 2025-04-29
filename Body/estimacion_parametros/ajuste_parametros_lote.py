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

def ajuste_parametros_page():
    st.header("🔧 Ajuste de Parámetros Cinéticos")

    # 1. Carga de datos experimentales
    with st.expander("📤 Cargar Datos Experimentales", expanded=True):
        uploaded_file = st.file_uploader("Subir archivo Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            df_exp = pd.read_excel(uploaded_file, engine='openpyxl')
            st.write("Vista previa de datos:")
            st.dataframe(df_exp.head())

            # Validar formato
            required_cols = ['tiempo', 'biomasa', 'sustrato', 'producto']
            if not all(col in df_exp.columns for col in required_cols):
                st.error(f"El archivo debe contener los nombres enlas columnas: {', '.join(required_cols)}")
                st.stop()

            t_exp = df_exp['tiempo'].values
            y_exp = df_exp[['biomasa', 'sustrato', 'producto']].values.T

    # 2. Configuración del ajuste
    with st.sidebar:
        st.subheader("⚙️ Configuración del Ajuste")

        # Parámetros a ajustar
        st.markdown("### Parámetros a Optimizar")
        mumax_guess = st.number_input("μmax inicial [1/h]", 0.01, 2.0, 0.5)
        Ks_guess = st.number_input("Ks inicial [g/L]", 0.01, 5.0, 0.2)
        Yxs_guess = st.number_input("Yxs inicial [g/g]", 0.1, 1.0, 0.5)
        Kd_guess = st.number_input("Kd inicial [1/h]", 0.0, 0.5, 0.01)
        Ypx_guess = st.number_input("Ypx inicial [g/g]", 0.1, 1.0, 0.3)

        # Condiciones iniciales
        st.markdown("### Condiciones Iniciales")
        X0_fit = st.number_input("Biomasa inicial [g/L]", 0.1, 10.0, df_exp['biomasa'].iloc[0] if uploaded_file else 1.0)
        S0_fit = st.number_input("Sustrato inicial [g/L]", 0.1, 100.0, df_exp['sustrato'].iloc[0] if uploaded_file else 20.0)
        P0_fit = st.number_input("Producto inicial [g/L]", 0.0, 50.0, df_exp['producto'].iloc[0] if uploaded_file else 0.0)
        O0_fit = st.number_input("O2 inicial [mg/L]", 0.0, 10.0, 8.0)
        atol = st.sidebar.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.sidebar.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

        # Opciones de optimización
        st.markdown("### Opciones de Optimización")
        metodo = st.selectbox("Método de optimización",
                            ['L-BFGS-B', 'Nelder-Mead', 'differential_evolution'])
        max_iter = st.number_input("Iteraciones máximas", 10, 1000, 100)

    # 3. Función objetivo y modelo
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

    def objetivo(params, t_exp, y_exp):
        try:
            sol = solve_ivp(modelo_ode,
                        [0, t_exp[-1]],
                        [X0_fit, S0_fit, P0_fit, O0_fit],  # Vector de estado inicial
                        args=(params,),  # Pasar parámetros correctamente
                        t_eval=t_exp, atol=atol, rtol=rtol)

            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])
            rmse = np.sqrt(np.nanmean((y_pred - y_exp)**2))
            return rmse
        except Exception as e:
            st.error(f"Error en integración: {str(e)}")
            return 1e6

    # 4. Ejecutar ajuste
    if uploaded_file and st.button("🚀 Ejecutar Ajuste"):
        with st.spinner("Optimizando parámetros..."):
            bounds = [(0.01, 2), (0.01, 5), (0.1, 1), (0, 0.5), (0.1, 1)]
            initial_guess = [mumax_guess, Ks_guess, Yxs_guess, Kd_guess, Ypx_guess]

            if metodo == 'differential_evolution':
                result = differential_evolution(objetivo, bounds, args=(t_exp, y_exp))
            else:
                result = minimize(objetivo, initial_guess, args=(t_exp, y_exp),
                                method=metodo, bounds=bounds,
                                options={'maxiter': max_iter})

            # result = minimize(objetivo, initial_guess, args=(t_exp, y_exp),
            #                  method=metodo, bounds=bounds,
            #                  options={'maxiter': max_iter})

            # Resultados del ajuste
            st.subheader("📊 Resultados del Ajuste")
            params_opt = result.x
            st.write(f"**RMSE final:** {result.fun:.4f}")

            # Tabla de parámetros
            parametros = pd.DataFrame({
                'Parámetro': ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx'],
                'Valor': params_opt,
                'Unidades': ['1/h', 'g/L', 'g/g', '1/h', 'g/g']
            })
            st.dataframe(parametros.style.format({'Valor': '{:.4f}'}))

            # 5. Análisis estadístico
            st.subheader("📈 Análisis Estadístico")

            # Predicción final
            sol = solve_ivp(modelo_ode, [0, t_exp[-1]],
                           [X0_fit, S0_fit, P0_fit, O0_fit],
                           args=(params_opt,),
                           t_eval=t_exp, atol=atol, rtol=rtol)

            y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])

            # Calcular métricas
            metricas = {
                'Variable': ['Biomasa', 'Sustrato', 'Producto'],
                'R²': [r2_score(y_exp[i], y_pred[i]) for i in range(3)],
                'RMSE': [np.sqrt(mean_squared_error(y_exp[i], y_pred[i])) for i in range(3)]
            }
            st.dataframe(pd.DataFrame(metricas))

            # 6. Gráficos comparativos
            fig, ax = plt.subplots(3, 1, figsize=(10, 12))
            variables = ['Biomasa', 'Sustrato', 'Producto']
            unidades = ['g/L', 'g/L', 'g/L']

            for i in range(3):
                ax[i].plot(t_exp, y_exp[i], 'o', label='Experimental')
                ax[i].plot(sol.t, sol.y[i], '--', label='Modelo')
                ax[i].set_title(f"{variables[i]} ({unidades[i]})")
                ax[i].legend()
                ax[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            # 7. Intervalos de confianza (metodología numérica)
            with st.spinner("Calculando intervalos de confianza..."):
                # Calcular residuos
                residuals = y_exp - y_pred
                residuals_flat = residuals.flatten()  # Aplanar para todas las variables

                # Calcular Jacobiano numérico
                jac = compute_jacobian(params_opt, t_exp, y_exp, X0_fit, S0_fit, P0_fit, O0_fit)

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
                parametros['Intervalo ±'] = intervals
                parametros['IC 95% Inferior'] = parametros['Valor'] - intervals
                parametros['IC 95% Superior'] = parametros['Valor'] + intervals

                st.write("Intervalos de confianza (95%):")
                st.dataframe(parametros.style.format({
                    'Valor': '{:.4f}',
                    'Intervalo ±': '{:.4f}',
                    'IC 95% Inferior': '{:.4f}',
                    'IC 95% Superior': '{:.4f}'
                }))

            # Gráficos de Intervalos de Confianza
            # -------------------------
            st.subheader("📐 Intervalos de Confianza de Parámetros")

            fig_ci, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(parametros))
            ax.barh(y_pos, parametros['Valor'], xerr=parametros['Intervalo ±'],
                align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(parametros['Parámetro'])
            ax.invert_yaxis()
            ax.set_xlabel('Valor del Parámetro')
            ax.set_title('Intervalos de Confianza al 95%')
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_ci)

        # Histogramas de Residuales
        st.subheader("📉 Distribución de Errores")

        fig_hist, axs = plt.subplots(1, 3, figsize=(15, 5))
        variables = ['Biomasa', 'Sustrato', 'Producto']
        colors = ['#2ca02c', '#9467bd', '#d62728']

        for i, (var, color) in enumerate(zip(variables, colors)):
            residuals = y_exp[i] - y_pred[i]
            sns.histplot(residuals, kde=True, color=color, ax=axs[i])
            axs[i].set_title(f'Residuales {var}')
            axs[i].set_xlabel('Error (Experimental - Modelo)')
            axs[i].set_ylabel('Frecuencia')
            axs[i].axvline(0, color='k', linestyle='--')
            axs[i].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_hist)

        # Gráficos de Correlación de Parámetros
        st.subheader("📌 Matriz de Correlación de Parámetros")

        fig_corr, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = pd.DataFrame(jac, columns=parametros['Parámetro']).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlación entre Parámetros')
        st.pyplot(fig_corr)

    elif not uploaded_file:
        st.warning("⏳ Por favor suba un archivo de datos para comenzar el ajuste")

if __name__ == '__main__':
    ajuste_parametros_page()