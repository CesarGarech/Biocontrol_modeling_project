import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns

def analysis_page():
    st.header("📈 Sensitivity Analysis - Batch Model")

    with st.sidebar:
        st.subheader("⚙️ Analysis Configuration")

        # 1. Analyzed parameter
        parametro = st.selectbox("Key parameter",
                               ["μ_max", "K_s", "Yxs", "Kd"])

        # 2. Variation range
        rango = st.slider("Percentage variation",
                         -50, 200, (0, 100),
                         help="% de change from base value")

        # 3. Number of simulations
        n_sim = st.slider("Number of simulations", 2, 50, 5)

        # 4. Base parameters
        st.subheader("🔬 Base parameters")
        mumax_base = st.number_input("μ_max base [1/h]", 0.1, 2.0, 0.5)
        Ks_base = st.number_input("K_s base [g/L]", 0.01, 5.0, 0.2)
        Yxs_base = st.number_input("Yxs base [g/g]", 0.1, 1.0, 0.5)
        Kd_base = st.number_input("Kd base [1/h]", 0.0, 0.5, 0.01)

        # 5. Fixed parameters
        st.subheader("🔧 Fixed parameters")
        Ypx = st.number_input("Ypx [g/g]", 0.1, 1.0, 0.3)
        Kla = st.number_input("kLa [1/h]", 0.1, 100.0, 20.0)
        Cs = st.number_input("Saturated oxygen [mg/L]", 0.1, 10.0, 8.0)
        mo = st.number_input("O2 maintenance [g/g/h]", 0.0, 0.5, 0.05)

        # 6. Initial conditions
        st.subheader("🎚 Initial conditions")
        X0 = st.number_input("Initial Biomass [g/L]", 0.1, 10.0, 1.0)
        S0 = st.number_input("Initial Substrate [g/L]", 0.1, 100.0, 20.0)
        P0 = st.number_input("Initial Product [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("Initial O2 [mg/L]", 0.0, 10.0, 5.0)
        y0 = [X0, S0, P0, O0]

        # 7. Temporary configuration
        st.subheader("⏳ Simulation Time")
        t_final = st.slider("Duration [h]", 1, 100, 24)
        t_eval = np.linspace(0, t_final, 100)

    if st.button("🚀 Run Analysis"):
        with st.spinner(f"Performing {n_sim} simulations..."):
            valores = np.linspace(1 + rango[0]/100, 1 + rango[1]/100, n_sim)

            # Set figure
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            colores = plt.cm.viridis(np.linspace(0, 1, n_sim))

            # Save results
            resultados = []

            def modelo_lote_b(t, y, mumax, Ks, Yxs, Kd, Ypx, Kla, Cs, mo):
                X, S, P, O2 = y

                # Calculate mu according to the selected model (assuming simple Monod for the analysis)
                mu = mumax * S / (Ks + S)

                dXdt = mu * X - Kd * X
                dSdt = (-mu/Yxs) * X - 0  # ms is assumed to be zero for simplicity
                dPdt = Ypx * mu * X
                dOdt = Kla * (Cs - O2) - (mu/Yxs) * X - mo * X

                return [dXdt, dSdt, dPdt, dOdt]

            for i, factor in enumerate(valores):
                # Calculate variable parameter
                if parametro == "μ_max":
                    params = [mumax_base*factor, Ks_base, Yxs_base, Kd_base]
                elif parametro == "K_s":
                    params = [mumax_base, Ks_base*factor, Yxs_base, Kd_base]
                elif parametro == "Yxs":
                    params = [mumax_base, Ks_base, Yxs_base*factor, Kd_base]
                else:
                    params = [mumax_base, Ks_base, Yxs_base, Kd_base*factor]

                # Model Simulation
                sol = solve_ivp(modelo_lote_b, [0, t_final], y0,
                              args=(*params, Ypx, Kla, Cs, mo),
                              t_eval=t_eval)

                # Save results
                resultados.append({
                    'Variation (%)': (factor - 1)*100,
                    'Parameter Value': factor,
                    'Max Biomass': sol.y[0].max(),
                    'Min Substrate': sol.y[1].min(),
                    'Max Product': sol.y[2].max(),
                    'Peak Time': sol.t[np.argmax(sol.y[0])]
                })

                # Graph results
                for j, ax in enumerate(axs):
                    ax.plot(sol.t, sol.y[j], color=colores[i], alpha=0.7)

            # Set graphs
            variables = ['Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]']
            for ax, var in zip(axs, variables):
                ax.set_title(var, fontsize=12, pad=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, t_final)
            axs[-1].set_xlabel("Time [h]", fontsize=10)

            # Color bar
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                      norm=plt.Normalize(valores.min(), valores.max()))
            cbar = fig.colorbar(sm, ax=axs, location='right', pad=0.02)
            cbar.set_label(f'Factor de {parametro}', rotation=270, labelpad=20)

            st.pyplot(fig)

            # Numericos Results
            st.subheader("📊 Quantitative Results")
            df = pd.DataFrame(resultados)
            st.dataframe(df.style
                        .format({'Variation (%)': '{:.1f}%',
                                'Parameter Value': '{:.2f}×',
                                'Max Biomass': '{:.2f}',
                                'Min Substrate': '{:.2f}',
                                'Max Product': '{:.2f}',
                                'Peak Time': '{:.1f}h'})
                        .background_gradient(cmap='viridis'))

            # Sensitivity Analysis
            st.subheader("📐 Global Sensitivity")
            sensibilidad = df[['Max Biomass', 'Min Substrate', 'Max Product']].std() / df.mean()

            fig2, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(sensibilidad.index, sensibilidad.values,
                         color=['#4c72b0', '#55a868', '#c44e52'])

            # Add values
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

            ax.set_title("Variation Coefficient (σ/μ)")
            ax.set_ylabel("Relative Sensitivity")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig2)

if __name__ == '__main__':
    analysis_page()