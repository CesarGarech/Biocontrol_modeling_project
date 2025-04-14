import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns

def analysis_page():
    st.header("📈 Análisis de Sensibilidad - Modelo Lote")

    with st.sidebar:
        st.subheader("⚙️ Configuración del Análisis")

        # 1. Parámetro a analizar
        parametro = st.selectbox("Parámetro clave",
                               ["μ_max", "K_s", "Yxs", "Kd"])

        # 2. Rango de variación
        rango = st.slider("Variación porcentual",
                         -50, 200, (0, 100),
                         help="% de cambio respecto al valor base")

        # 3. Número de simulaciones
        n_sim = st.slider("Número de simulaciones", 2, 50, 5)

        # 4. Parámetros base
        st.subheader("🔬 Parámetros Base")
        mumax_base = st.number_input("μ_max base [1/h]", 0.1, 2.0, 0.5)
        Ks_base = st.number_input("K_s base [g/L]", 0.01, 5.0, 0.2)
        Yxs_base = st.number_input("Yxs base [g/g]", 0.1, 1.0, 0.5)
        Kd_base = st.number_input("Kd base [1/h]", 0.0, 0.5, 0.01)

        # 5. Parámetros fijos
        st.subheader("🔧 Parámetros Fijos")
        Ypx = st.number_input("Ypx [g/g]", 0.1, 1.0, 0.3)
        Kla = st.number_input("kLa [1/h]", 0.1, 100.0, 20.0)
        Cs = st.number_input("Oxígeno saturado [mg/L]", 0.1, 10.0, 8.0)
        mo = st.number_input("Mantenimiento O2 [g/g/h]", 0.0, 0.5, 0.05)

        # 6. Condiciones iniciales
        st.subheader("🎚 Condiciones Iniciales")
        X0 = st.number_input("Biomasa inicial [g/L]", 0.1, 10.0, 1.0)
        S0 = st.number_input("Sustrato inicial [g/L]", 0.1, 100.0, 20.0)
        P0 = st.number_input("Producto inicial [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("O2 inicial [mg/L]", 0.0, 10.0, 5.0)
        y0 = [X0, S0, P0, O0]

        # 7. Configuración temporal
        st.subheader("⏳ Tiempo de Simulación")
        t_final = st.slider("Duración [h]", 1, 100, 24)
        t_eval = np.linspace(0, t_final, 100)

    if st.button("🚀 Ejecutar Análisis"):
        with st.spinner(f"Realizando {n_sim} simulaciones..."):
            valores = np.linspace(1 + rango[0]/100, 1 + rango[1]/100, n_sim)

            # Configurar figura
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            colores = plt.cm.viridis(np.linspace(0, 1, n_sim))

            # Almacenar resultados
            resultados = []

            def modelo_lote_b(t, y, mumax, Ks, Yxs, Kd, Ypx, Kla, Cs, mo):
                X, S, P, O2 = y

                # Calcular mu según modelo seleccionado (asumiendo Monod simple para el análisis)
                mu = mumax * S / (Ks + S)

                dXdt = mu * X - Kd * X
                dSdt = (-mu/Yxs) * X - 0  # ms se asume cero para simplificar
                dPdt = Ypx * mu * X
                dOdt = Kla * (Cs - O2) - (mu/Yxs) * X - mo * X

                return [dXdt, dSdt, dPdt, dOdt]

            for i, factor in enumerate(valores):
                # Calcular parámetro variable
                if parametro == "μ_max":
                    params = [mumax_base*factor, Ks_base, Yxs_base, Kd_base]
                elif parametro == "K_s":
                    params = [mumax_base, Ks_base*factor, Yxs_base, Kd_base]
                elif parametro == "Yxs":
                    params = [mumax_base, Ks_base, Yxs_base*factor, Kd_base]
                else:
                    params = [mumax_base, Ks_base, Yxs_base, Kd_base*factor]

                # Simular modelo
                sol = solve_ivp(modelo_lote_b, [0, t_final], y0,
                              args=(*params, Ypx, Kla, Cs, mo),
                              t_eval=t_eval)

                # Almacenar resultados
                resultados.append({
                    'Variación (%)': (factor - 1)*100,
                    'Valor Parametro': factor,
                    'Biomasa Máx': sol.y[0].max(),
                    'Sustrato Mín': sol.y[1].min(),
                    'Producto Máx': sol.y[2].max(),
                    'Tiempo Pico': sol.t[np.argmax(sol.y[0])]
                })

                # Graficar resultados
                for j, ax in enumerate(axs):
                    ax.plot(sol.t, sol.y[j], color=colores[i], alpha=0.7)

            # Configurar gráficos
            variables = ['Biomasa [g/L]', 'Sustrato [g/L]', 'Producto [g/L]']
            for ax, var in zip(axs, variables):
                ax.set_title(var, fontsize=12, pad=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, t_final)
            axs[-1].set_xlabel("Tiempo [h]", fontsize=10)

            # Barra de color
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                      norm=plt.Normalize(valores.min(), valores.max()))
            cbar = fig.colorbar(sm, ax=axs, location='right', pad=0.02)
            cbar.set_label(f'Factor de {parametro}', rotation=270, labelpad=20)

            st.pyplot(fig)

            # Resultados numéricos
            st.subheader("📊 Resultados Cuantitativos")
            df = pd.DataFrame(resultados)
            st.dataframe(df.style
                        .format({'Variación (%)': '{:.1f}%',
                                'Valor Parametro': '{:.2f}×',
                                'Biomasa Máx': '{:.2f}',
                                'Sustrato Mín': '{:.2f}',
                                'Producto Máx': '{:.2f}',
                                'Tiempo Pico': '{:.1f}h'})
                        .background_gradient(cmap='viridis'))

            # Análisis de sensibilidad
            st.subheader("📐 Sensibilidad Global")
            sensibilidad = df[['Biomasa Máx', 'Sustrato Mín', 'Producto Máx']].std() / df.mean()

            fig2, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(sensibilidad.index, sensibilidad.values,
                         color=['#4c72b0', '#55a868', '#c44e52'])

            # Añadir valores
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

            ax.set_title("Coeficiente de Variación (σ/μ)")
            ax.set_ylabel("Sensibilidad Relativa")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig2)

if __name__ == '__main__':
    analysis_page()