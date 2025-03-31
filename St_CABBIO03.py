import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
import openpyxl
import seaborn as sns
from casadi import *
from qpsolvers import solve_qp

# -------------------------
# Página principal (Home)
# -------------------------
st.set_page_config(page_title="Modelado de Bioprocesos", layout="wide")

menu = st.sidebar.selectbox("Seleccione una opción", ["Home", "Lote", "Lote Alimentado", "Continuo", "Análisis de Sensibilidad", "Ajuste de Parámetros", "Control RTO"])

if menu == "Home":
    st.title("Modelado de Bioprocesos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("imagenes/Batch.png", caption="### Reactor Batch", use_container_width=True)
    with col2:
        st.image("imagenes/fed_batch.png", caption="### Reactor Fed-Batch", use_container_width=True)
    with col3:
        st.image("imagenes/continous.png", caption="### Reactor Continuo", use_container_width=True)

    st.markdown("""
    ## Fundamento Teórico
    El modelado de bioprocesos permite describir matemáticamente la evolución de las variables de interés en un biorreactor. A continuación se presentan las ecuaciones diferenciales generales para los tres modos de operación considerados:
    
    ### 🔹 Modo Lote
    - No hay entrada ni salida de materia durante el proceso.

    dX/dt = μ(S, O2, P) * X - Kd * X  
    dS/dt = - (1/Yxs) * μ(S, O2, P) * X - ms * X  
    dP/dt = Ypx * μ(S, O2, P) * X  
    dO2/dt = Kla * (Cs - O2) - (1/Yxo) * μ(S, O2, P) * X - mo * X

    ### 🔹 Modo Lote Alimentado
    - Se agrega alimentación al biorreactor sin retirar producto, y el volumen varía en el tiempo.

    dX/dt = μ(S, O2, P) * X - Kd * X - (F/V) * X  
    dS/dt = - (1/Yxs) * μ(S, O2, P) * X - ms * X + (F/V) * (Sin - S)  
    dP/dt = Ypx * μ(S, O2, P) * X - (F/V) * P  
    dO2/dt = Kla * (Cs - O2) - (1/Yxo) * μ(S, O2, P) * X - mo * X - (F/V) * O2

    ### 🔹 Modo Continuo (chemostato)
    - Hay entrada y salida continua de fluido, el volumen se mantiene constante.

    dX/dt = μ(S, O2, P) * X - Kd * X - D * X  
    dS/dt = - (1/Yxs) * μ(S, O2, P) * X - ms * X + D * (Sin - S)  
    dP/dt = Ypx * μ(S, O2, P) * X - D * P  
    dO2/dt = Kla * (Cs - O2) - (1/Yxo) * μ(S, O2, P) * X - mo * X - D * O2

    Donde:  
    - μ: velocidad específica de crecimiento (Monod y variantes)  
    - Yxs: rendimiento biomasa/sustrato  
    - Ypx: rendimiento producto/biomasa  
    - Yxo: rendimiento biomasa/oxígeno  
    - Kla: coeficiente de transferencia de oxígeno  
    - ms, mo: coeficientes de mantenimiento  
    - Kd: tasa de decaimiento celular
    - F: flujo de alimentación  
    - D: tasa de dilución (D = F/V)  
    - Sin: concentración de sustrato en el alimentado  
    - Cs: concentración de oxígeno a saturación
    """)

# -------------------------
# Funciones de cinética
# -------------------------
def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)

def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * S**n / (Ks**n + S**n)

def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    return mumax * S / (Ks + S) * O2 / (KO + O2) * KP / (KP + P)

# -------------------------
# Dinámicas para cada modo
# -------------------------
# (Se incluirán en las secciones correspondientes de 'Lote', 'Lote Alimentado' y 'Continuo')

# -------------------------
# Página Lote
# -------------------------
if menu == "Lote":
    st.header("Modo de operación: Lote")
    st.sidebar.subheader("Parámetros del modelo")

    # Parámetros generales
    mumax = st.sidebar.slider("μmax", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider("Ks", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Oxígeno saturado (Cs)", 0.1, 10.0, 8.0)
    V = st.sidebar.slider("Volumen del biorreactor (L)", 0.5, 10.0, 2.0)
    ms = st.sidebar.slider("Mantenimiento (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decaimiento (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("Mantenimiento O2 (mo)", 0.0, 0.5, 0.05)

    # Iniciales
    X0 = st.sidebar.number_input("Biomasa inicial (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Sustrato inicial (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Producto inicial (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("O2 disuelto inicial (mg/L)", 0.0, 10.0, 5.0)

    # Tipo de cinética
    tipo_mu = st.sidebar.selectbox("Tipo de cinética", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])

    # Tiempo de simulación
    t_final = st.sidebar.slider("Tiempo final (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)

    # Tolerancias
    atol = st.sidebar.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def modelo_lote(t, y):
        X, S, P, O2 = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n=2)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)

        dXdt = mu * X - Kd * X
        dSdt = -1/Yxs * mu * X - ms * X
        dPdt = Ypx * mu * X
        dOdt = Kla * (Cs - O2) - (1/Yxo) * mu * X - mo * X
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(modelo_lote, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

    # Gráficas
    st.subheader("Resultados de simulación")
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='Biomasa (X)')
    ax.plot(sol.t, sol.y[1], label='Sustrato (S)')
    ax.plot(sol.t, sol.y[2], label='Producto (P)')
    ax.plot(sol.t, sol.y[3], label='Oxígeno disuelto (O2)')
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Concentración (g/L o mg/L)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -------------------------
# Página Lote Alimentado
# -------------------------
if menu == "Lote Alimentado":
    st.header("Modo de operación: Lote Alimentado")

    with st.sidebar:
        st.subheader("Parámetros del Modelo")
        tipo_mu = st.selectbox("Modelo Cinético", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])
        mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4)
        Ks = st.slider("Ks [g/L]", 0.01, 2.0, 0.5)

        if tipo_mu == "Monod sigmoidal":
            n = st.slider("Exponente sigmoidal (n)", 1, 5, 2)
        elif tipo_mu == "Monod con restricciones":
            KO = st.slider("Constante saturación O2 [mg/L]", 0.1, 5.0, 0.5)
            KP = st.slider("Constante inhibición producto [g/L]", 0.1, 10.0, 0.5)

        Yxs = st.slider("Yxs [g/g]", 0.1, 1.0, 0.6)
        Ypx = st.slider("Ypx [g/g]", 0.0, 1.0, 0.3)
        Yxo = st.slider("Yxo [g/g]", 0.1, 1.0, 0.2)
        Kla = st.slider("kLa [1/h]", 1.0, 200.0, 50.0)
        Cs = st.slider("O2 Saturado [mg/L]", 5.0, 15.0, 8.0)
        Sin = st.slider("Sustrato en Alimentado [g/L]", 50.0, 300.0, 150.0)
        ms = st.slider("Mantenimiento S [g/g/h]", 0.0, 0.1, 0.001)
        Kd = st.slider("Decaimiento X [g/g/h]", 0.0, 0.1, 0.02)
        mo = st.slider("Mantenimiento O2 [g/g/h]", 0.0, 0.1, 0.01)

        st.subheader("Estrategia de Alimentación")
        estrategia = st.selectbox("Tipo", ["Constante", "Exponencial", "Escalon"])
        F_base = st.slider("Flujo Base [L/h]", 0.01, 5.0, 0.5)
        t_alim_inicio = st.slider("Inicio Alimentación [h]", 0, 24, 2)
        t_alim_fin = st.slider("Fin Alimentación [h]", 5, 48, 24)

        st.subheader("Condiciones Iniciales")
        V0 = st.number_input("Volumen Inicial [L]", 1.0, 10.0, 3.0)
        X0 = st.number_input("Biomasa Inicial [g/L]", 0.1, 10.0, 1.0)
        S0 = st.number_input("Sustrato Inicial [g/L]", 0.1, 100.0, 30.0)
        P0 = st.number_input("Producto Inicial [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("O2 Inicial [mg/L]", 0.0, 10.0, 8.0)

        t_final = st.slider("Tiempo de Simulación [h]", 10, 100, 48)
        atol = st.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def calcular_flujo(t):
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constante":
                return F_base
            elif estrategia == "Exponencial":
                return F_base * np.exp(0.15 * (t - t_alim_inicio))
            elif estrategia == "Escalon":
                return F_base * 2 if t > (t_alim_inicio + t_alim_fin)/2 else F_base
        return 0.0

    def modelo_fedbatch(t, y):
        X, S, P, O2, V = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)

        F = calcular_flujo(t)
        dXdt = mu*X - Kd*X/V -  (F/V)*X
        dSdt = (-mu/Yxs - ms)*X + (F/V)*(Sin - S)
        dPdt = Ypx*mu*X - (F/V)*P
        dOdt = Kla*(Cs - O2) - (mu/Yxo + mo)*X - (F/V)*O2
        dVdt = F
        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    y0 = [X0, S0, P0, O0, V0]
    t_eval = np.linspace(0, t_final, 300)
    sol = solve_ivp(modelo_fedbatch, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)
    flujo_sim = [calcular_flujo(t) for t in sol.t]

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    ax1.plot(sol.t, flujo_sim, 'r-', label='Flujo de Alimentación')
    ax1.set_ylabel('Flujo [L/h]', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1b = ax1.twinx()
    ax1b.plot(sol.t, sol.y[4], 'b--', label='Volumen')
    ax1b.set_ylabel('Volumen [L]', color='b')
    ax1b.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Perfil de Alimentación y Volumen')
    ax1.grid(True)

    ax2.plot(sol.t, sol.y[0], 'g-')
    ax2.set_title('Biomasa [g/L]')
    ax2.grid(True)

    ax3.plot(sol.t, sol.y[1], 'm-')
    ax3.set_title('Sustrato [g/L]')
    ax3.grid(True)

    ax4.plot(sol.t, sol.y[2], 'k-')
    ax4.set_title('Producto [g/L]')
    ax4.grid(True)

    ax5.plot(sol.t, sol.y[3], 'c-')
    ax5.set_title('Oxígeno Disuelto [mg/L]')
    ax5.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# -------------------------
# Página Continuo (Chemostato)
# -------------------------
if menu == "Continuo":
    st.header("Modo de operación: Continuo (Chemostato)")
    st.sidebar.subheader("Parámetros del modelo")

    mumax = st.sidebar.slider("μmax", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider("Ks", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Oxígeno saturado (Cs)", 0.1, 10.0, 8.0)
    ms = st.sidebar.slider("Mantenimiento (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decaimiento (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("Mantenimiento O2 (mo)", 0.0, 0.5, 0.05)
    Sin = st.sidebar.slider("Sustrato en alimentación (Sin)", 0.0, 100.0, 50.0)
    D = st.sidebar.slider("Tasa de dilución D (1/h)", 0.0, 1.0, 0.01)

    X0 = st.sidebar.number_input("Biomasa inicial (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Sustrato inicial (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Producto inicial (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("O2 disuelto inicial (mg/L)", 0.0, 10.0, 5.0)

    tipo_mu = st.sidebar.selectbox("Tipo de cinética", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])
    n_sigmoidal = st.sidebar.slider("Valor de n (Monod sigmoidal)", 0.0, 5.0, 2.0)

    t_final = st.sidebar.slider("Tiempo final (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)
    atol = st.sidebar.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def modelo_continuo(t, y):
        X, S, P, O2 = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n=n_sigmoidal)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)

        dXdt = mu * X - Kd * X - D * X
        dSdt = -1/Yxs * mu * X - ms * X + D * (Sin - S)
        dPdt = Ypx * mu * X - D * P
        dOdt = Kla * (Cs - O2) - 1/Yxo * mu * X - mo * X - D * O2
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(modelo_continuo, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

    st.subheader("Resultados de simulación")
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='Biomasa (X)')
    ax.plot(sol.t, sol.y[1], label='Sustrato (S)')
    ax.plot(sol.t, sol.y[2], label='Producto (P)')
    ax.plot(sol.t, sol.y[3], label='Oxígeno disuelto (O2)')
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Concentración (g/L o mg/L)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
# -------------------------
# Página Análisis de Sensibilidad
# -------------------------
if menu == "Análisis de Sensibilidad":
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

# -------------------------
# Página Ajuste de Parámetros
# -------------------------
if menu == "Ajuste de Parámetros":
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

# -------------------------
# Página Control RTO 
# -------------------------
if menu == "Control RTO":
    st.header("🧠 Control RTO - Optimización del perfil de alimentación")

    with st.sidebar:
        st.subheader("📌 Parámetros del modelo")
        mu_max = st.number_input("μmax [1/h]", value=0.6, min_value=0.01)
        Ks = st.number_input("Ks [g/L]", value=0.2, min_value=0.01)
        Ko = st.number_input("Ko [g/L]", value=0.01, min_value=0.001)
        Yxs = st.number_input("Yxs [g/g]", value=0.5, min_value=0.1, max_value=1.0)
        Yxo = st.number_input("Yxo [g/g]", value=0.1, min_value=0.01, max_value=1.0)
        Yps = st.number_input("Yps [g/g]", value=0.3, min_value=0.1, max_value=1.0)
        kLa = st.number_input("kLa [1/h]", value=180.0, min_value=0.0)
        O_sat = st.number_input("Oxígeno saturado [g/L]", value=0.18, min_value=0.01)
        Sf = st.number_input("Concentración del alimentado Sf [g/L]", value=350.0)
        V_max = st.number_input("Volumen máximo del reactor [L]", value=10.0)

        st.subheader("🎚 Condiciones Iniciales")
        X0 = st.number_input("X0 (Biomasa) [g/L]", value=1.0)
        S0 = st.number_input("S0 (Sustrato) [g/L]", value=20.0)
        P0 = st.number_input("P0 (Producto) [g/L]", value=0.0)
        O0 = st.number_input("O0 (Oxígeno) [g/L]", value=0.08)
        V0 = st.number_input("V0 (Volumen inicial) [L]", value=2.0)

        st.subheader("⏳ Configuración temporal")
        t_batch = st.number_input("Tiempo de lote (t_batch) [h]", value=5, min_value=0)
        t_total = st.number_input("Tiempo total del proceso [h]", value=24, min_value=t_batch+1)

        st.subheader("🔧 Restricciones de operación")
        F_min = st.number_input("Flujo mínimo [L/h]", value=0.01, min_value=0.0)
        F_max = st.number_input("Flujo máximo [L/h]", value=0.3, min_value=F_min)
        S_max = st.number_input("Sustrato máximo permitido [g/L]", value=30.0)

        st.subheader("🔬 Selección del modelo cinético")
        tipo_mu = st.selectbox("Modelo cinético", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])

    n_seg = int(t_total - t_batch)
    dt = 1.0

    if st.button("🚀 Ejecutar Optimización RTO"):
        st.info("Optimizando perfil de alimentación...")
        progress_bar = st.progress(0)

        try:
            opti = ca.Opti()
            F = opti.variable(n_seg)
            X = ca.MX(X0); S = ca.MX(S0); P = ca.MX(P0); O = ca.MX(O0); V = ca.MX(V0)
            J = 0

            for i in range(n_seg):
                F_i = F[i]
                opti.subject_to(F_i >= F_min)
                opti.subject_to(F_i <= F_max)

                for _ in range(4):
                    mu = {
                        "Monod simple": mu_max * S / (Ks + S),
                        "Monod sigmoidal": mu_max * (S**2) / (Ks**2 + S**2),
                        "Monod con restricciones": mu_max * S / (Ks + S) * O / (Ko + O)
                    }[tipo_mu]

                    D = ca.if_else(V > 0, F_i / V, 0)
                    dX = mu * X - D * X
                    dS = -mu * X / Yxs + D * (Sf - S)
                    dP = Yps * mu * X - D * P
                    dO = kLa * (O_sat - O) - mu * X / Yxo
                    dV = ca.if_else(V < V_max, F_i, 0.0)

                    X += dX * dt / 4
                    S += dS * dt / 4
                    P += dP * dt / 4
                    O += dO * dt / 4
                    V += dV * dt / 4

                opti.subject_to(S <= S_max)
                J -= P * V
                progress_bar.progress((i+1)/n_seg)

            opti.minimize(J)
            opti.solver=nlpsol('solver', 'ipopt', nlp, opts)
            sol = opti.solve()

            F_opt = sol.value(F)
            st.success("Optimización completada ✅")

            def simulate(F_profile):
                X, S, P, O, V = X0, S0, P0, O0, V0
                ts, Xs, Ss, Ps, Os, Vs, Fs = [0], [X0], [S0], [P0], [O0], [V0], [0]
                for i in range(n_seg):
                    F_val = F_profile[i] if i < len(F_profile) else 0
                    for _ in range(4):
                        mu = {
                            "Monod simple": mu_max * S / (Ks + S),
                            "Monod sigmoidal": mu_max * (S**2) / (Ks**2 + S**2),
                            "Monod con restricciones": mu_max * S / (Ks + S) * O / (Ko + O)
                        }[tipo_mu]
                        D = F_val / V if V > 0 else 0
                        dX = mu*X - D*X
                        dS = -mu*X/Yxs + D*(Sf - S)
                        dP = Yps*mu*X - D*P
                        dO = kLa * (O_sat - O) - mu * X / Yxo
                        dV = F_val if V < V_max else 0
                        X += dX * dt/4
                        S += dS * dt/4
                        P += dP * dt/4
                        O += dO * dt/4
                        V += dV * dt/4
                    ts.append(t_batch + i + 1)
                    Xs.append(X)
                    Ss.append(S)
                    Ps.append(P)
                    Os.append(O)
                    Vs.append(V)
                    Fs.append(F_val)
                return ts, Xs, Ss, Ps, Os, Vs, Fs

            ts, Xs, Ss, Ps, Os, Vs, Fs = simulate(F_opt)

            st.subheader("📊 Resultados del perfil óptimo")
            fig, ax = plt.subplots(3, 2, figsize=(14, 12))
            ax[0,0].step(ts[1:], Fs[1:], where='post', color='darkred')
            ax[0,0].set_title("Perfil Óptimo de Flujo")
            ax[0,0].set_ylabel("Flujo [L/h]")
            ax[0,0].grid(True)
            variables = [
                (Xs, 'Biomasa [g/L]', 'blue'),
                (Ss, 'Sustrato [g/L]', 'green'),
                (Ps, 'Producto [g/L]', 'purple'),
                (Os, 'Oxígeno [g/L]', 'orange'),
                (Vs, 'Volumen [L]', 'brown')
            ]
            for i, (data, title, color) in enumerate(variables, 1):
                row = i // 2
                col = i % 2
                ax[row,col].plot(ts, data, color=color)
                ax[row,col].set_title(title)
                ax[row,col].grid(True)
                ax[row,col].set_xlabel("Tiempo [h]")
            plt.tight_layout()
            st.pyplot(fig)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Producto total acumulado", f"{Ps[-1]*Vs[-1]:.2f} g")
                st.metric("Rendimiento Producto/Sustrato", f"{(Ps[-1]*Vs[-1])/(Sf*(Vs[-1]-V0)):.3f} g/g")
            with col2:
                st.metric("Tiempo óptimo de alimentación", f"{n_seg} h")
                st.metric("Volumen final", f"{Vs[-1]:.2f} L")
        except Exception as e:
            st.error(f"Error en la optimización: {str(e)}")
            st.stop()

# -------------------------
# Ejecución principal
# -------------------------
if __name__ == "__main__":
    pass