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
import casadi as ca
# from qpsolvers import solve_qp

# -------------------------
# Página principal (Home)
# -------------------------
st.set_page_config(page_title="Modelado de Bioprocesos", layout="wide")

menu = st.sidebar.selectbox("Seleccione una opción", ["Home", "Lote", "Lote Alimentado", "Continuo", "Análisis de Sensibilidad", "Ajuste de Parámetros", "Estimacion de estados", "Control RTO", "Control NMPC"])

if menu == "Home":
    st.title("Modelado de Bioprocesos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/Batch.png", caption="### Reactor Batch", use_container_width=True)
    with col2:
        st.image("images/fed_batch.png", caption="### Reactor Fed-Batch", use_container_width=True)
    with col3:
        st.image("images/continous.png", caption="### Reactor Continuo", use_container_width=True)

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

def modelo_lote_b(t, y, mumax, Ks, Yxs, Kd, Ypx, Kla, Cs, mo):
    X, S, P, O2 = y
    
    # Calcular mu según modelo seleccionado (asumiendo Monod simple para el análisis)
    mu = mumax * S / (Ks + S)
    
    dXdt = mu * X - Kd * X
    dSdt = (-mu/Yxs) * X - 0  # ms se asume cero para simplificar
    dPdt = Ypx * mu * X
    dOdt = Kla * (Cs - O2) - (mu/Yxs) * X - mo * X
    
    return [dXdt, dSdt, dPdt, dOdt]
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

# ------------------------------------
# ----- PÁGINA ESTIMACION DE ESTADOS -----
# ------------------------------------
if menu == "Estimacion de estados":
    st.header("Estimación de Estados y Parámetros con Filtro de Kalman Extendido (EKF)")
    st.markdown("""
    Esta sección simula un bioproceso batch y utiliza un EKF para estimar las concentraciones
    de Biomasa (X), Sustrato (S), Producto (P), y dos parámetros cinéticos
    ($\mu_{max}$, $Y_{X/S}$) a partir de mediciones simuladas y ruidosas de
    Oxígeno Disuelto (OD), pH y Temperatura (T).

    **Puedes ajustar:**
    * Las **condiciones iniciales** que supone el EKF (la "conjetura inicial").
    * La **incertidumbre inicial** sobre esa conjetura (matriz $P_0$).
    * Los niveles de **ruido** asumidos por el filtro para el proceso ($Q$) y las mediciones ($R$).
    Observa cómo estos ajustes afectan la capacidad del EKF para seguir los valores reales.
    """)

    st.sidebar.subheader("Parámetros del EKF y Simulación")

    # --- Parámetros Fijos del Modelo "Real" (No ajustables por usuario aquí) ---
    #     (Podrían ponerse en un expander si se quiere verlos)
    mu_max_real = 0.4     # (1/h)
    Yxs_real    = 0.5     # (gX/gS)
    Ks          = 0.1     # (g/L)
    alpha       = 0.1     # (gP/gX) - Relacionado con Ypx
    OD_sat      = 8.0     # mg/L
    k_OUR       = 0.5     # mgO2/(L*gX)
    pH0         = 7.0
    P0_meas_ref = 0.0     # P de referencia para cálculo de pH
    k_acid      = 0.2
    Tset        = 30      # (°C)
    k_Temp      = 0.02

    # --- Parámetros Ajustables por el Usuario ---
    t_final_ekf = st.sidebar.slider("Tiempo final (h)", 5, 50, 20, key="ekf_tf")
    dt_ekf      = 0.1 # Fijo para esta simulación EKF

    st.sidebar.markdown("**Condiciones Iniciales del EKF**")
    X0_est  = st.sidebar.number_input("X inicial estimada (g/L)", 0.01, 5.0, 0.05, format="%.2f", key="ekf_x0e")
    S0_est  = st.sidebar.number_input("S inicial estimada (g/L)", 0.1, 50.0, 4.5, format="%.1f", key="ekf_s0e")
    P0_est  = st.sidebar.number_input("P inicial estimada (g/L)", 0.0, 10.0, 0.1, format="%.2f", key="ekf_p0e")
    mu0_est = st.sidebar.number_input("μmax inicial estimada (1/h)", 0.1, 1.0, 0.35, format="%.2f", key="ekf_mu0e")
    yxs0_est= st.sidebar.number_input("Yxs inicial estimada (g/g)", 0.1, 1.0, 0.55, format="%.2f", key="ekf_yxs0e")

    st.sidebar.markdown("**Incertidumbre Inicial $P_0$ (Diagonales)**")
    p0_X   = st.sidebar.number_input("P0 - X", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0x")
    p0_S   = st.sidebar.number_input("P0 - S", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0s")
    p0_P   = st.sidebar.number_input("P0 - P", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0p")
    p0_mu  = st.sidebar.number_input("P0 - μmax", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0mu")
    p0_yxs = st.sidebar.number_input("P0 - Yxs", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0yxs")

    st.sidebar.markdown("**Ruido de Proceso $Q$ (Diagonales)**")
    q_X   = st.sidebar.number_input("Q - X", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_qx")
    q_S   = st.sidebar.number_input("Q - S", 1e-10, 1e-2, 1e-8, format="%.2e", key="ekf_qs")
    q_P   = st.sidebar.number_input("Q - P", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_qp")
    q_mu  = st.sidebar.number_input("Q - μmax", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_qmu")
    q_yxs = st.sidebar.number_input("Q - Yxs", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_qyxs")

    st.sidebar.markdown("**Ruido de Medición $R$ (Diagonales)**")
    r_OD = st.sidebar.number_input("R - OD", 1e-4, 1.0, 0.05, format="%.4f", key="ekf_rod")
    r_pH = st.sidebar.number_input("R - pH", 1e-4, 1.0, 0.02, format="%.4f", key="ekf_rph")
    r_T  = st.sidebar.number_input("R - Temp", 1e-2, 5.0, 0.5, format="%.2f", key="ekf_rtemp")

    # Botón para ejecutar la simulación
    run_ekf = st.sidebar.button("Ejecutar Simulación EKF")

    # --- Definiciones CasADi (Fuera del botón para no redefinir) ---
    n_states_ekf = 5
    n_meas_ekf   = 3
    x_sym_ekf = ca.SX.sym('x', n_states_ekf)
    X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym_ekf)

    mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
    dX = mu_sym * X_sym
    dS = - (1 / Yxs_sym) * dX
    dP = alpha * dX
    dMu_max = 0
    dYxs = 0
    x_next_sym = x_sym_ekf + dt_ekf * ca.vertcat(dX, dS, dP, dMu_max, dYxs)
    f_func_ekf = ca.Function('f', [x_sym_ekf], [x_next_sym], ['x_k'], ['x_k_plus_1'])

    OD_val_sym = OD_sat - k_OUR * X_sym
    pH_val_sym = pH0 - k_acid * (P_sym - P0_meas_ref)
    T_val_sym  = Tset + k_Temp * (X_sym * S_sym)
    z_sym_ekf = ca.vertcat(OD_val_sym, pH_val_sym, T_val_sym)
    h_func_ekf = ca.Function('h', [x_sym_ekf], [z_sym_ekf], ['x'], ['z'])

    F_sym_ekf = ca.jacobian(x_next_sym, x_sym_ekf)
    H_sym_ekf = ca.jacobian(z_sym_ekf, x_sym_ekf)
    F_func_ekf = ca.Function('F', [x_sym_ekf], [F_sym_ekf], ['x'], ['Fk'])
    H_func_ekf = ca.Function('H', [x_sym_ekf], [H_sym_ekf], ['x'], ['Hk'])
    # --- Fin Definiciones CasADi ---


    if run_ekf:
        st.subheader("Resultados de la Estimación EKF")

        # --- Preparación basada en Inputs del Usuario ---
        time_vec_ekf = np.arange(0, t_final_ekf + dt_ekf, dt_ekf)
        N_ekf = len(time_vec_ekf)

        # Covarianzas de ruido desde sliders
        Q_ekf = np.diag([q_X, q_S, q_P, q_mu, q_yxs])
        R_ekf = np.diag([r_OD, r_pH, r_T])

        # Condiciones iniciales "reales" (fijas en este ejemplo)
        X0_real = 0.1
        S0_real = 5.0
        P0_real = 0.0
        x_real_ekf = np.array([[X0_real], [S0_real], [P0_real], [mu_max_real], [Yxs_real]])

        # Estimación inicial EKF desde sliders
        x_est_ekf = np.array([[X0_est], [S0_est], [P0_est], [mu0_est], [yxs0_est]])
        P_est_ekf = np.diag([p0_X, p0_S, p0_P, p0_mu, p0_yxs])

        # Arrays para guardar resultados
        x_real_hist = np.zeros((n_states_ekf, N_ekf))
        x_est_hist  = np.zeros((n_states_ekf, N_ekf))
        z_meas_hist = np.zeros((n_meas_ekf, N_ekf))

        # --- Bucle de Simulación EKF ---
        for k in range(N_ekf):
            # Guardar valores actuales
            x_real_hist[:, k] = x_real_ekf.flatten()
            x_est_hist[:, k]  = x_est_ekf.flatten()

            # (A) Generar medición "real"
            z_noisefree_dm = h_func_ekf(x_real_ekf)
            z_noisefree = z_noisefree_dm.full()
            noise_meas = np.random.multivariate_normal(np.zeros(n_meas_ekf), R_ekf).reshape(-1, 1)
            z_k = z_noisefree + noise_meas
            z_meas_hist[:, k] = z_k.flatten()

            if k < N_ekf - 1:
                # (B) Predicción EKF
                x_pred_dm = f_func_ekf(x_est_ekf)
                x_pred = x_pred_dm.full()
                Fk_dm = F_func_ekf(x_est_ekf)
                Fk = Fk_dm.full()
                P_pred = Fk @ P_est_ekf @ Fk.T + Q_ekf

                # (C) Corrección EKF
                Hk_dm = H_func_ekf(x_pred)
                Hk = Hk_dm.full()
                h_pred_dm = h_func_ekf(x_pred)
                h_pred = h_pred_dm.full()
                Sk = Hk @ P_pred @ Hk.T + R_ekf
                Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk) # Usar pinv
                y_k = z_k - h_pred
                x_upd = x_pred + Kk @ y_k
                P_upd = (np.eye(n_states_ekf) - Kk @ Hk) @ P_pred

                # Actualizar
                x_est_ekf = x_upd
                 # Forzar no-negatividad en estados estimados si se desea
                x_est_ekf[0:3] = np.maximum(x_est_ekf[0:3], 0)
                # Forzar positividad en parámetros estimados si se desea (cuidado con mu_max=0)
                x_est_ekf[3:] = np.maximum(x_est_ekf[3:], 1e-6)

                P_est_ekf = P_upd

                # (D) Avance del proceso real
                x_real_next_no_noise_dm = f_func_ekf(x_real_ekf)
                x_real_next_no_noise = x_real_next_no_noise_dm.full()
                noise_proc = np.random.multivariate_normal(np.zeros(n_states_ekf), Q_ekf).reshape(-1, 1)
                x_real_ekf = x_real_next_no_noise + noise_proc
                # Forzar no-negatividad estados reales físicos
                x_real_ekf[0:3] = np.maximum(x_real_ekf[0:3], 0)


        # --- Gráficas de Resultados EKF ---
        # (Usando los arrays x_real_hist, x_est_hist, z_meas_hist)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Figura 1: Estados y Mediciones
        fig1_ekf, axs1_ekf = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        fig1_ekf.suptitle('Estimación de Estados y Mediciones (EKF)', fontsize=14)

        # Biomasa
        axs1_ekf[0, 0].plot(time_vec_ekf, x_real_hist[0, :], 'b-', label='X real')
        axs1_ekf[0, 0].plot(time_vec_ekf, x_est_hist[0, :], 'r--', label='X estimada')
        axs1_ekf[0, 0].set_ylabel('Biomasa (g/L)')
        axs1_ekf[0, 0].legend()
        axs1_ekf[0, 0].grid(True)

        # Medición OD
        axs1_ekf[0, 1].plot(time_vec_ekf, z_meas_hist[0, :], 'k.-', markersize=3, linewidth=1, label='OD medido')
        axs1_ekf[0, 1].set_ylabel('OD (mg/L)')
        axs1_ekf[0, 1].set_title('Medición OD')
        axs1_ekf[0, 1].legend()
        axs1_ekf[0, 1].grid(True)

        # Sustrato
        axs1_ekf[1, 0].plot(time_vec_ekf, x_real_hist[1, :], 'b-', label='S real')
        axs1_ekf[1, 0].plot(time_vec_ekf, x_est_hist[1, :], 'r--', label='S estimada')
        axs1_ekf[1, 0].set_ylabel('Sustrato (g/L)')
        axs1_ekf[1, 0].legend()
        axs1_ekf[1, 0].grid(True)

        # Medición pH
        axs1_ekf[1, 1].plot(time_vec_ekf, z_meas_hist[1, :], 'k.-', markersize=3, linewidth=1, label='pH medido')
        axs1_ekf[1, 1].set_ylabel('pH')
        axs1_ekf[1, 1].set_title('Medición pH')
        axs1_ekf[1, 1].legend()
        axs1_ekf[1, 1].grid(True)

        # Producto
        axs1_ekf[2, 0].plot(time_vec_ekf, x_real_hist[2, :], 'b-', label='P real')
        axs1_ekf[2, 0].plot(time_vec_ekf, x_est_hist[2, :], 'r--', label='P estimada')
        axs1_ekf[2, 0].set_xlabel('Tiempo (h)')
        axs1_ekf[2, 0].set_ylabel('Producto (g/L)')
        axs1_ekf[2, 0].legend()
        axs1_ekf[2, 0].grid(True)

        # Medición Temperatura
        axs1_ekf[2, 1].plot(time_vec_ekf, z_meas_hist[2, :], 'k.-', markersize=3, linewidth=1, label='T medida')
        axs1_ekf[2, 1].set_xlabel('Tiempo (h)')
        axs1_ekf[2, 1].set_ylabel('Temperatura (°C)')
        axs1_ekf[2, 1].set_title('Medición Temperatura')
        axs1_ekf[2, 1].legend()
        axs1_ekf[2, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        st.pyplot(fig1_ekf)


        # Figura 2: Parámetros Estimados
        fig2_ekf, axs2_ekf = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        fig2_ekf.suptitle('Estimación de Parámetros (EKF)', fontsize=14)

        # mu_max
        axs2_ekf[0].plot(time_vec_ekf, x_real_hist[3, :], 'b-', label=r'$\mu_{max}$ real')
        axs2_ekf[0].plot(time_vec_ekf, x_est_hist[3, :], 'r--', label=r'$\mu_{max}$ estimada')
        axs2_ekf[0].set_ylabel(r'$\mu_{max}$ (1/h)')
        axs2_ekf[0].legend()
        axs2_ekf[0].grid(True)

        # Yxs
        axs2_ekf[1].plot(time_vec_ekf, x_real_hist[4, :], 'b-', label=r'$Y_{X/S}$ real')
        axs2_ekf[1].plot(time_vec_ekf, x_est_hist[4, :], 'r--', label=r'$Y_{X/S}$ estimada')
        axs2_ekf[1].set_xlabel('Tiempo (h)')
        axs2_ekf[1].set_ylabel(r'$Y_{X/S}$ (gX/gS)')
        axs2_ekf[1].legend()
        axs2_ekf[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        st.pyplot(fig2_ekf)

        st.write("Valores Finales:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Reales**")
            st.metric("X", f"{x_real_ekf[0,0]:.3f} g/L")
            st.metric("S", f"{x_real_ekf[1,0]:.3f} g/L")
            st.metric("P", f"{x_real_ekf[2,0]:.3f} g/L")
            st.metric("μmax", f"{x_real_ekf[3,0]:.3f} 1/h")
            st.metric("Yxs", f"{x_real_ekf[4,0]:.3f} g/g")
        with col2:
            st.write("**Estimados**")
            st.metric("X est.", f"{x_est_ekf[0,0]:.3f} g/L")
            st.metric("S est.", f"{x_est_ekf[1,0]:.3f} g/L")
            st.metric("P est.", f"{x_est_ekf[2,0]:.3f} g/L")
            st.metric("μmax est.", f"{x_est_ekf[3,0]:.3f} 1/h")
            st.metric("Yxs est.", f"{x_est_ekf[4,0]:.3f} g/g")

    else:
        st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Ejecutar Simulación EKF'.")


# ====================================================
# 2) Coeficientes de colocación Radau (d=2)
# ====================================================
def radau_coefficients(d):
    """
    Retorna C_mat (shape (d+1, d)) y D_vec (shape d+1)
    para la colocación de Radau IIA con grado d=2.
    Estos valores son los correctos para Radau IIA order 3.
    """
    if d == 2:
        C_mat = np.array([
            [-2.0,   2.0],
            [ 1.5,  -4.5],
            [ 0.5,   2.5]
        ])
        D_vec = np.array([0.0, 0.0, 1.0])
        return C_mat, D_vec
    else:
        raise NotImplementedError("Solo implementado para d=2.")

# ====================================================
# 2) Coeficientes de colocación Radau (d=2)
# ====================================================
def radau_coefficients(d):
    """
    Retorna C_mat (shape (d+1, d)) y D_vec (shape d+1)
    para la colocación de Radau IIA con grado d=2.
    Estos valores son los correctos para Radau IIA order 3.
    """
    if d == 2:
        C_mat = np.array([
            [-2.0,    2.0],
            [ 1.5,   -4.5],
            [ 0.5,    2.5]
        ])
        D_vec = np.array([0.0, 0.0, 1.0])
        return C_mat, D_vec
    else:
        raise NotImplementedError("Solo implementado para d=2.")

# -------------------------
# Página Control RTO
# -------------------------
# menu = st.sidebar.radio("Menú", ["Control RTO"])

if menu == "Control RTO":
    st.header("🧠 Control RTO - Optimización del perfil de alimentación")

    with st.sidebar:
        st.subheader("📌 Parámetros del modelo")
        mu_max = st.number_input("μmax [1/h]", value=0.6, min_value=0.01)
        Ks = st.number_input("Ks [g/L]", value=0.2, min_value=0.01)
        Ko = st.number_input("KO [g/L]", value=0.01, min_value=0.001)
        KP = st.number_input("KP [g/L]", value=0.1, min_value=0.001)
        Yxs = st.number_input("Yxs [g/g]", value=0.5, min_value=0.1, max_value=1.0)
        Yxo = st.number_input("Yxo [g/g]", value=0.1, min_value=0.01, max_value=1.0)
        Yps = st.number_input("Yps [g/g]", value=0.3, min_value=0.1, max_value=1.0)
        Sf_input = st.number_input("Concentración del alimentado Sf [g/L]", value=500.0)
        V_max_input = st.number_input("Volumen máximo del reactor [L]", value=2.0)

        st.subheader("🎚 Condiciones Iniciales")
        X0 = st.number_input("X0 (Biomasa) [g/L]", value=1.0)
        S0 = st.number_input("S0 (Sustrato) [g/L]", value=20.0)
        P0 = st.number_input("P0 (Producto) [g/L]", value=0.0)
        O0 = st.number_input("O0 (Oxígeno) [g/L]", value=0.08)
        V0 = st.number_input("V0 (Volumen inicial) [L]", value=0.2)

        st.subheader("⏳ Configuración temporal")
        t_batch = st.number_input("Tiempo de lote (t_batch) [h]", value=5.0, min_value=0.0)
        t_total = st.number_input("Tiempo total del proceso [h]", value=24.0, min_value=t_batch + 1.0)

        st.subheader("🔧 Restricciones de operación")
        F_min = st.number_input("Flujo mínimo [L/h]", value=0.0, min_value=0.0)
        F_max = st.number_input("Flujo máximo [L/h]", value=0.3, min_value=F_min)
        S_max = st.number_input("Sustrato máximo permitido [g/L]", value=30.0)

        st.subheader("🔬 Selección del modelo cinético")
        kinetic_model = st.selectbox("Modelo cinético", ["Monod", "Sigmoidal", "Completa"])
        if kinetic_model == "Sigmoidal":
            n_sigmoidal = st.number_input("n (para Monod Sigmoidal)", value=2.0, min_value=1.0)

    if st.button("🚀 Ejecutar Optimización RTO"):
        st.info("Optimizando perfil de alimentación...")

        try:
            # ====================================================
            # 1) Definición de la función ODE BIO
            # ====================================================
            def odefun(x, u):
                """
                Ecuaciones diferenciales Fed-Batch con O=constante.
                - Divisiones con fmax(V, epsilon) para evitar 1/0.
                """
                # Parámetros
                mu_max_local = mu_max
                Ks_local = Ks
                Ko_local = Ko
                KP_local = KP
                Yxs_local = Yxs
                Yxo_local = Yxo
                Yps_local = Yps
                Sf_local = Sf_input
                V_max_local = V_max_input

                # Extraer variables de estado (evitar desempacado iterativo)
                X_ = x[0]
                S_ = x[1]
                P_ = x[2]
                O_ = x[3]
                V_ = x[4]

                # Tasa de crecimiento
                if kinetic_model == "Monod":
                    mu = mu_monod(S_, mu_max_local, Ks_local) * (O_ / (Ko_local + O_)) # Assuming oxygen dependence
                elif kinetic_model == "Sigmoidal":
                    mu = mu_sigmoidal(S_, mu_max_local, Ks_local, n_sigmoidal) * (O_ / (Ko_local + O_)) # Assuming oxygen dependence
                elif kinetic_model == "Completa":
                    mu = mu_completa(S_, O_, P_, mu_max_local, Ks_local, Ko_local, KP_local)
                else:
                    raise ValueError("Modelo cinético no seleccionado correctamente.")

                # Tasa de dilución
                D = u / V_

                dX = mu * X_ - D * X_
                dS = -mu * X_ / Yxs_local + D * (Sf_local - S_)
                dP = Yps_local * mu * X_ - D * P_
                dO = 0.0   # asumiendo oxígeno constante
                dV = u

                return ca.vertcat(dX, dS, dP, dO, dV)

            # ====================================================
            # 3) Parámetros del proceso y condiciones iniciales
            # ====================================================
            n_fb_intervals = int((t_total - t_batch))
            dt_fb = (t_total - t_batch) / n_fb_intervals if n_fb_intervals > 0 else 0.0

            # ====================================================
            # 4) Fase BATCH con F=0 (integración)
            # ====================================================
            x_sym = ca.MX.sym("x", 5)
            u_sym = ca.MX.sym("u")
            ode_expr = odefun(x_sym, u_sym)

            batch_integrator = ca.integrator(
                "batch_integrator", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": t_batch}
            )

            x0_np = np.array([X0, S0, P0, O0, V0])
            res_batch = batch_integrator(x0=x0_np, p=0.0)
            x_after_batch = np.array(res_batch['xf']).flatten()
            st.info(f"[INFO] Estado tras fase batch: {x_after_batch}")

            # ====================================================
            # 5) Formulación de la fase Fed-Batch con colocación
            # ====================================================
            opti = ca.Opti()

            d = 2
            C_radau, D_radau = radau_coefficients(d)
            nx = 5

            # Variables de estado y control
            X_col = []
            F_col = []

            for k in range(n_fb_intervals):
                row_states = []
                for j in range(d + 1):
                    if (k == 0 and j == 0):
                        # Fijar el estado inicial del primer intervalo
                        # con un "parameter" (no es variable)
                        xk0_param = opti.parameter(nx)
                        opti.set_value(xk0_param, x_after_batch)
                        row_states.append(xk0_param)
                    else:
                        # variable
                        xk_j = opti.variable(nx)
                        row_states.append(xk_j)
                        # Restricciones
                        # no-negatividad:
                        opti.subject_to(xk_j >= 0)
                        # S <= S_max
                        opti.subject_to(xk_j[1] <= S_max)
                        # V <= V_max
                        opti.subject_to(xk_j[4] <= V_max_input)
                X_col.append(row_states)

                # Variable de control en cada intervalo
                Fk = opti.variable()
                F_col.append(Fk)
                opti.subject_to(Fk >= F_min)
                opti.subject_to(Fk <= F_max)

            # ====================================================
            # 6) Ecuaciones de Colocación
            # ====================================================
            h = dt_fb
            for k in range(n_fb_intervals):
                for j in range(1, d + 1):
                    # xp_j = sum_{m=0..d} C_radau[m, j-1]* X_col[k][m]
                    xp_j = 0
                    for m in range(d + 1):
                        xp_j += C_radau[m, j - 1] * X_col[k][m]

                    # f(Xk_j, Fk)
                    fkj = odefun(X_col[k][j], F_col[k])
                    # Restricción => h*f - xp_j = 0
                    coll_eq = h * fkj - xp_j
                    opti.subject_to(coll_eq == 0)

                # Continuidad al final del subintervalo
                Xk_end = 0
                for m in range(d + 1):
                    Xk_end += D_radau[m] * X_col[k][m]

                if k < n_fb_intervals - 1:
                    # Xk_end = X_{k+1}[0]
                    for i_ in range(nx):
                        opti.subject_to(Xk_end[i_] == X_col[k + 1][0][i_])

            # Estado final global => X_final
            X_final = X_col[-1][-1]

            P_final = X_final[2]
            V_final = X_final[4]

            # ====================================================
            # 7) Función objetivo => maximizar (P_final*V_final)
            # ====================================================
            opti.minimize(-(P_final * V_final))

            # ====================================================
            # 8) Guesses iniciales (importante para evitar NaNs)
            # ====================================================
            for k in range(n_fb_intervals):
                opti.set_initial(F_col[k], 0.1)
                for j in range(d + 1):
                    # Si no es el primer "parameter"
                    if not (k == 0 and j == 0):
                        # Como guess, usemos el estado final de batch (o algo similar)
                        opti.set_initial(X_col[k][j], x_after_batch)

            # ====================================================
            # 9) Configurar y resolver
            # ====================================================
            p_opts = {}
            s_opts = {
                "max_iter": 2000,
                "print_level": 0,
                "sb": 'yes',
                "mu_strategy": "adaptive"
            }
            opti.solver("ipopt", p_opts, s_opts)

            try:
                sol = opti.solve()
                st.success("[INFO] ¡Solución encontrada!")
            except RuntimeError as e:
                st.error(f"[ERROR] No se encontró solución: {e}")
                try:
                    # Mostrar infeasibilidades
                    opti.debug.show_infeasibilities()
                except:
                    pass
                st.stop()

            F_opt = [sol.value(fk) for fk in F_col]
            X_fin_val = sol.value(X_final)
            P_fin_val = X_fin_val[2]
            V_fin_val = X_fin_val[4]

            st.info(f"Flujo óptimo de alimentación (F_opt): {F_opt}")
            st.info(f"Estado final del reactor: {X_fin_val}")
            st.info(f"Concentración final de Producto (P_final): {P_fin_val:.4f} g/L")
            st.info(f"Volumen final del reactor (V_final): {V_fin_val:.4f} L")
            st.info(f"Producto total final: {(P_fin_val * V_fin_val):.4f} g")

            # ====================================================
            # 10) Reconstruir y graficar trayectoria
            #     (batch + fed-batch)
            # ====================================================
            # a) Fase batch: con dt pequeño
            N_batch_plot = 50
            t_batch_plot = np.linspace(0, t_batch, N_batch_plot)
            dt_b = t_batch_plot[1] - t_batch_plot[0]

            batch_plot_int = ca.integrator(
                "batch_plot_int", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": dt_b}
            )

            xbatch_traj = [x0_np]
            xk_ = x0_np.copy()
            for _ in range(N_batch_plot - 1):
                res_ = batch_plot_int(x0=xk_, p=0.0)
                xk_ = np.array(res_["xf"]).flatten()
                xbatch_traj.append(xk_)
            xbatch_traj = np.array(xbatch_traj)

            # b) Fase fed-batch: integrando de t_batch a t_total con dt fino
            t_fb_plot = np.linspace(t_batch, t_total, 400)
            dt_fb_plot = t_fb_plot[1] - t_fb_plot[0]

            fb_plot_int = ca.integrator(
                "fb_plot_int", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": dt_fb_plot}
            )

            xfb_traj = []
            xk_ = xbatch_traj[-1].copy()
            for i, t_ in enumerate(t_fb_plot):
                xfb_traj.append(xk_)
                if i == len(t_fb_plot) - 1:
                    break
                # Determinar en qué subintervalo k estamos
                kk_ = int((t_ - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, kk_)
                kk_ = min(n_fb_intervals - 1, kk_)
                # Tomar F correspondiente
                F_now = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                # Apagar F si V>=Vmax
                if xk_[4] >= V_max_input:
                    F_now = 0.0
                # Integrar
                res_ = fb_plot_int(x0=xk_, p=F_now)
                xk_ = np.array(res_["xf"]).flatten()

            xfb_traj = np.array(xfb_traj)

            # Unimos
            t_full = np.concatenate([t_batch_plot, t_fb_plot])
            x_full = np.vstack([xbatch_traj, xfb_traj])

            X_full = x_full[:, 0]
            S_full = x_full[:, 1]
            P_full = x_full[:, 2]
            O_full = x_full[:, 3]
            V_full = x_full[:, 4]

            # Construir F para graficar
            F_batch_plot = np.zeros_like(t_batch_plot)
            F_fb_plot = []
            for i, tt in enumerate(t_fb_plot):
                kk_ = int((tt - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, kk_)
                kk_ = min(n_fb_intervals - 1, kk_)
                valF = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                if xfb_traj[i, 4] >= V_max_input:
                    valF = 0.0
                F_fb_plot.append(valF)
            F_fb_plot = np.array(F_fb_plot)

            F_plot = np.concatenate([F_batch_plot, F_fb_plot])

            # ====================================================
            # 11) Gráficas
            # ====================================================
            fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            axs = axs.ravel()

            # F
            axs[0].plot(t_full, F_plot, linewidth=2)
            axs[0].set_title("Flujo de alimentación F(t)")
            axs[0].set_xlabel("Tiempo (h)")
            axs[0].set_ylabel("F (L/h)")
            axs[0].grid(True)

            # X
            axs[1].plot(t_full, X_full, linewidth=2)
            axs[1].set_title("Biomasa X(t)")
            axs[1].set_xlabel("Tiempo (h)")
            axs[1].set_ylabel("X (g/L)")
            axs[1].grid(True)

            # S
            axs[2].plot(t_full, S_full, linewidth=2)
            axs[2].axhline(S_max, color='r', linestyle='--', label="S_max")
            axs[2].set_title("Sustrato S(t)")
            axs[2].set_xlabel("Tiempo (h)")
            axs[2].set_ylabel("S (g/L)")
            axs[2].legend()
            axs[2].grid(True)

            # P
            axs[3].plot(t_full, P_full, linewidth=2)
            axs[3].set_title("Producto P(t)")
            axs[3].set_xlabel("Tiempo (h)")
            axs[3].set_ylabel("P (g/L)")
            axs[3].grid(True)

            # O
            axs[4].plot(t_full, O_full, linewidth=2)
            axs[4].set_title("Oxígeno disuelto O(t) (constante)")
            axs[4].set_xlabel("Tiempo (h)")
            axs[4].set_ylabel("O (g/L)")
            axs[4].grid(True)

            # V
            axs[5].plot(t_full, V_full, linewidth=2)
            axs[5].axhline(V_max_input, color='r', linestyle='--', label="V_max")
            axs[5].set_title("Volumen V(t)")
            axs[5].set_xlabel("Tiempo (h)")
            axs[5].set_ylabel("V (L)")
            axs[5].legend()
            axs[5].grid(True)

            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Producto total acumulado", f"{P_fin_val * V_fin_val:.2f} g")
                s_in_total = Sf_input * (V_fin_val - V0)
                rend = (P_fin_val * V_fin_val) / s_in_total if s_in_total > 1e-9 else 0
                st.metric("Rendimiento Producto/Sustrato", f"{rend:.3f} g/g")
            with col2:
                st.metric("Tiempo total del proceso", f"{t_total:.2f} h")
                st.metric("Volumen final", f"{V_fin_val:.2f} L")

        except Exception as e:
            st.error(f"Error en la optimización: {str(e)}")
            st.stop()

# -------------------------
# Página Control NMPC
# -------------------------

# --- Definición del menú (asumiendo que 'menu' está definido en otro lugar) ---
# if 'menu' not in st.session_state:
#     st.session_state['menu'] = "Control NMPC" # Valor por defecto si no está definido
# menu = st.session_state['menu']

if menu == "Control NMPC":
    st.header("Control No Lineal Predictivo (NMPC) del Biorreactor")

    # ---------------------------------------------------
    # 1. Modelo del Biorreactor (Simbólico con CasADi)
    # ---------------------------------------------------
    def get_bioreactor_model(params_input=None):
        """Define el modelo simbólico del biorreactor usando CasADi."""
        # Parámetros del modelo (ejemplo - ¡ajusta a tu sistema!)
        default_params = {
            'mu_max': 0.4,     # Tasa máx crecimiento (1/h)
            'K_S': 0.05,       # Constante de Monod (g/L)
            'Y_XS': 0.5,       # Rendimiento Biomasa/Substrato (g/g)
            'Y_QX': 15000,     # Rendimiento Calor/Biomasa (J/g) - ¡Ajustar!
            'S_in': 10.0,      # Concentración de sustrato entrante (g/L)
            'V': 1.0,          # Volumen del reactor (L)
            'rho': 1000.0,     # Densidad del medio (kg/m^3 -> g/L)
            'Cp': 4184.0,      # Capacidad calorífica (J/(kg*K) -> J/(g*K))
            'T_in': 298.15,    # Temperatura de entrada (K)
            'F_const': 0.0,    # Flujo constante adicional (L/h) - si existe
        }

        if params_input:
            params = params_input
        else:
            params = default_params

        # Variables simbólicas
        X = ca.MX.sym('X')  # Concentración de biomasa (g/L)
        S = ca.MX.sym('S')  # Concentración de sustrato (g/L)
        T = ca.MX.sym('T')  # Temperatura del reactor (K)
        x = ca.vertcat(X, S, T) # x es puramente simbólico (vertcat de MX.sym)

        # Definir 'u' como un símbolo base de 2 elementos
        u = ca.MX.sym('u', 2)
        # u[0] representará F_S (L/h)
        # u[1] representará Q_j / 3600.0 (Carga térmica en J/h) - ¡Comentario indica J/h pero división sugiere Watts!

        # --- Ecuaciones del modelo ---
        mu = params['mu_max'] * S / (params['K_S'] + S + 1e-9) # Tasa crecimiento Monod (+epsilon)

        # Extraer F_S simbólico de u
        F_S = u[0]
        F_total = F_S + params['F_const']
        D = F_total / params['V'] # Tasa de dilución (1/h)

        # Balances de materia
        dX_dt = mu * X - D * X
        dS_dt = D * (params['S_in'] - S) - (mu / params['Y_XS']) * X

        # Balance de energía
        Q_gen = params['Y_QX'] * mu * X * params['V'] # Calor generado (J/h)
        Q_flow = F_total * params['rho'] * params['Cp'] * (params['T_in'] - T) # Calor por flujo (J/h)
        # Asumimos que u[1] es directamente el calor añadido/quitado por la chaqueta en J/h
        # ¡¡¡ POSIBLE INCONSISTENCIA DE UNIDADES: u[1] parece estar en Watts (J/s) por la división !!!
        # Si u[1] está en Watts, Q_rem debería ser - u[1] * 3600.0 para estar en J/h
        Q_rem = - u[1] * 3600.0 # Calor removido (J/h) - Corregido asumiendo u[1] es Q_j/3600 (Watts)

        dT_dt = (Q_gen + Q_rem + Q_flow) / (params['rho'] * params['Cp'] * params['V']) # (K/h)

        # Vector de derivadas
        dx = ca.vertcat(dX_dt, dS_dt, dT_dt)

        # Variables controladas (salidas)
        c = ca.vertcat(X, T)

        # Crear funciones CasADi
        # Ahora los inputs [x, u] son ambos símbolos base o vertcat de símbolos base.
        model_ode = ca.Function('model_ode', [x, u], [dx], ['x', 'u'], ['dx'])
        output_func = ca.Function('output_func', [x], [c], ['x'], ['c'])

        # Devuelve el símbolo 'u' que ahora es MX.sym('u', 2)
        return model_ode, output_func, x, u, c, dx, params

    # ---------------------------------------------------
    # 2. Clase NMPC (Copiada directamente)
    # ---------------------------------------------------
    class NMPCBioreactor:
        def __init__(self, dt, N, M, Q, W, model_ode, output_func, x_sym, u_sym, c_sym, params,
                     lbx, ubx, lbu, ubu, lbdu, ubdu, m=3, pol='legendre'):
            """
            Inicializa el controlador NMPC.
            Args:
                dt: Tiempo de muestreo (h)
                N: Horizonte de predicción
                M: Horizonte de control
                Q: Matriz de peso para error de salida (CV)
                W: Matriz de peso para movimiento de entrada (MV)
                model_ode: Función CasADi para las EDOs dx = f(x, u)
                output_func: Función CasADi para las salidas c = h(x)
                x_sym, u_sym, c_sym: Variables simbólicas CasADi
                params: Diccionario de parámetros del modelo
                lbx, ubx: Límites inferiores/superiores para estados x
                lbu, ubu: Límites inferiores/superiores para entradas u
                lbdu, ubdu: Límites inferiores/superiores para tasa de cambio de entradas du
                m: Grado del polinomio de colocación
                pol: Tipo de polinomio ('legendre' o 'radau')
            """
            self.dt = dt
            self.N = N
            self.M = M
            self.Q = np.diag(Q) # Asegurar que sean matrices diagonales
            self.W = np.diag(W)
            self.model_ode = model_ode
            self.output_func = output_func
            self.params = params
            self.nx = x_sym.shape[0]
            self.nu = u_sym.shape[0]
            self.nc = c_sym.shape[0]
            self.lbx = lbx
            self.ubx = ubx
            self.lbu = lbu
            self.ubu = ubu
            self.lbdu = lbdu
            self.ubdu = ubdu
            self.m = m
            self.pol = pol

            # --- Preparar Colocación Ortogonal ---
            # Puntos de colocación (tau_root[0] = 0)
            self.tau_root = np.append(0, ca.collocation_points(self.m, self.pol))

            # Matriz de coeficientes para derivadas en puntos de colocación
            self.C = np.zeros((self.m + 1, self.m + 1))
            # Matriz de coeficientes para la integral (estado final)
            self.D = np.zeros(self.m + 1)

            # Construir C y D
            for j in range(self.m + 1):
                # Construir polinomio de Lagrange j
                p = np.poly1d([1])
                for r in range(self.m + 1):
                    if r != j:
                        p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j] - self.tau_root[r] + 1e-10) # Evitar div por cero

                # Evaluar la derivada del polinomio en los puntos tau
                p_der = np.polyder(p)
                for i in range(self.m + 1):
                    self.C[j, i] = np.polyval(p_der, self.tau_root[i])

                # Evaluar la integral del polinomio de 0 a 1
                p_int = np.polyint(p)
                self.D[j] = np.polyval(p_int, 1.0)

            # --- Construir el NLP ---
            self._build_nlp()

        def _build_nlp(self):
            """Construye el problema NLP de optimización."""
            # Crear una instancia de la función de colocación una vez
            # Variables simbólicas para un paso de integración
            Xk_step = ca.MX.sym('Xk_step', self.nx)
            Xc_step = ca.MX.sym('Xc_step', self.nx, self.m) # Estados en puntos interiores k,1...k,m
            Uk_step = ca.MX.sym('Uk_step', self.nu)

            # Calcular ecuaciones de colocación y estado final para un paso
            X_all_coll_step = ca.horzcat(Xk_step, Xc_step) # Estados en k,0(=Xk_step), k,1, ..., k,m
            ode_at_coll_step = []
            for j in range(1, self.m + 1): # Calcular ODE en puntos interiores 1..m
                ode_at_coll_step.append(self.model_ode(X_all_coll_step[:, j], Uk_step))
            # ode_at_coll_step es ahora una lista de m vectores columna [nx x 1]

            # Restricciones de colocación: Xk_coll_j - (Xk + dt * sum(Aij*f(Xk_coll_i, uk))) = 0
            # Usaremos una forma diferente: dx/dt = f(x,u) => dt*f(x_coll, u) = sum(Cij*x_coll_j)
            # donde x_coll incluye Xk_step
            coll_eqs_step = []
            for j in range(1, self.m + 1): # Para cada punto de colocación interior j = 1..m
                xp_coll_j = 0 # Suma Polinomios Derivados * Estados Colocacion
                for r in range(self.m + 1):
                    xp_coll_j += self.C[r, j] * X_all_coll_step[:, r]
                # Ecuación: Derivada estimada = dt * Modelo evaluado en el punto
                coll_eqs_step.append(xp_coll_j - (self.dt * ode_at_coll_step[j-1])) # ode_at_coll_step[j-1] es f(X_coll_j, Uk)

            # Estado al final del intervalo (usando coeficientes D)
            Xk_end_step = Xk_step # Empezar con Xk
            for j in range(1, self.m + 1): # Sumar contribuciones de los puntos interiores
                Xk_end_step += self.dt * self.D[j] * ode_at_coll_step[j-1]

            # Crear la función CasADi para un paso
            self.F_coll = ca.Function('F_coll', [Xk_step, Xc_step, Uk_step],
                                        [Xk_end_step, ca.vertcat(*coll_eqs_step)],
                                        ['Xk_step', 'Xc_step', 'Uk_step'], ['Xk_end', 'coll_eqs'])

            # --- Variables de decisión del NMPC ---
            self.w = []        # Vector de variables de decisión
            self.w0_init = [] # Estimación inicial (basada en parámetros)
            self.lbw = []      # Límite inferior
            self.ubw = []      # Límite superior

            self.g = []        # Vector de restricciones
            self.lbg = []      # Límite inferior
            self.ubg = []      # Límite superior

            # Parámetros del NLP (estado inicial, setpoints, entrada anterior)
            self.x0_sym = ca.MX.sym('x0', self.nx)
            self.sp_sym = ca.MX.sym('sp', self.nc, self.N)
            self.uprev_sym = ca.MX.sym('uprev', self.nu)
            p_nlp = ca.vertcat(self.x0_sym, ca.vec(self.sp_sym), self.uprev_sym)

            # Función de costo
            J = 0
            Uk_prev = self.uprev_sym # Entrada anterior para calcular delta_u

            # El estado al inicio del primer intervalo es el PARÁMETRO x0_sym
            Xk_iter = self.x0_sym

            # Bucle sobre el horizonte de predicción N
            for k in range(self.N):
                # Variables de entrada Uk
                Uk_k = ca.MX.sym(f'U_{k}', self.nu)
                self.w.append(Uk_k)
                # Inicializar entradas con u_previous
                self.w0_init.extend(np.zeros(self.nu)) # Placeholder, se usará uprev_sym numérico después
                self.lbw.extend(self.lbu)
                self.ubw.extend(self.ubu)

                # Restricciones delta_u
                delta_u = Uk_k - Uk_prev
                self.g.append(delta_u)
                self.lbg.extend(self.lbdu)
                self.ubg.extend(self.ubdu)

                # Restricciones de horizonte de control M
                if k >= self.M:
                    self.g.append(Uk_k - Uk_prev_control_horizon) # Uk = U_{M-1}
                    self.lbg.extend([-1e-9] * self.nu) # Permitir tolerancia numérica
                    self.ubg.extend([+1e-9] * self.nu)

                # Variables de estado de colocación Xc_k (solo puntos interiores 1..m)
                Xc_k = ca.MX.sym(f'Xc_{k}', self.nx, self.m)
                self.w.append(ca.vec(Xc_k)) # Vectorizar para añadir a w
                # Inicializar estados de colocación con x0_sym
                self.w0_init.extend(np.zeros(self.nx * self.m)) # Placeholder
                self.lbw.extend(self.lbx * self.m)
                self.ubw.extend(self.ubx * self.m)

                # Aplicar el paso de colocación usando Xk_iter (que es x0_sym para k=0)
                Xk_end_k, coll_eqs_k = self.F_coll(Xk_iter, Xc_k, Uk_k)

                # Añadir restricciones de colocación
                self.g.append(coll_eqs_k)
                self.lbg.extend([-1e-9] * self.nx * self.m) # Igualdad con tolerancia numérica (m ecuaciones)
                self.ubg.extend([+1e-9] * self.nx * self.m)

                # Variable para el estado al final del intervalo X_{k+1}
                Xk_next = ca.MX.sym(f'X_{k+1}', self.nx)
                self.w.append(Xk_next)
                # Inicializar Xk_next con x0_sym
                self.w0_init.extend(np.zeros(self.nx)) # Placeholder
                self.lbw.extend(self.lbx)
                self.ubw.extend(self.ubx)

                # Restricción de continuidad (disparo)
                self.g.append(Xk_end_k - Xk_next)
                self.lbg.extend([-1e-9] * self.nx) # Permitir pequeña tolerancia numérica
                self.ubg.extend([+1e-9] * self.nx)

                # Calcular costo del paso k
                Ck = self.output_func(Xk_next) # Salida al final del intervalo k+1
                sp_k = self.sp_sym[:, k]      # Setpoint en el paso k
                J += (Ck - sp_k).T @ self.Q @ (Ck - sp_k) # Costo de salida
                J += delta_u.T @ self.W @ delta_u          # Costo de movimiento de entrada

                # Actualizar para el siguiente paso
                Xk_iter = Xk_next # Ahora Xk_iter es la VARIABLE DE DECISIÓN X_{k+1}
                Uk_prev = Uk_k
                if k == self.M - 1:
                    Uk_prev_control_horizon = Uk_k # Guardar la última entrada del horizonte M

            # --- Crear el solver NLP ---
            nlp_dict = {
                'f': J,
                'x': ca.vertcat(*self.w), # Variables: U0, Xc0, X1, U1, Xc1, X2, ...
                'g': ca.vertcat(*self.g), # Restricciones
                'p': p_nlp                 # Parámetros: x0, sp, uprev
            }

            # Opciones del solver (IPOPT)
            opts = {
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.max_iter': 150,
                'ipopt.tol': 1e-6,
                'ipopt.acceptable_tol': 1e-5,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-9,
                'ipopt.warm_start_mult_bound_push': 1e-9,
                # 'ipopt.mu_strategy': 'adaptive',
                # 'ipopt.hessian_approximation': 'limited-memory'
            }

            self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

            # Guardar índices para extraer resultados fácilmente
            self._prepare_indices()

            # Inicializar w0 (vector de ceros con la dimensión correcta)
            # Se usará numéricamente en solve() para la primera iteración
            self.w0 = np.zeros(ca.vertcat(*self.w).shape[0]) # CORRECCIÓN AQUÍ


        def _prepare_indices(self):
            """Calcula los índices de inicio para cada tipo de variable en w."""
            self.indices = {'X': [], 'U': [], 'Xc': []}
            offset = 0
            # X0 NO está en w
            for k in range(self.N):
                # Uk
                self.indices['U'].append(offset)
                offset += self.nu
                # Xc_k (m*nx variables)
                self.indices['Xc'].append(offset)
                offset += self.nx * self.m
                # X_{k+1}
                self.indices['X'].append(offset) # Este índice corresponde a X_{k+1}
                offset += self.nx
            self.dim_w = offset


        def solve(self, x_current, sp_trajectory, u_previous):
            """
            Resuelve el problema NMPC para un estado inicial y trayectoria de setpoint.
            Args:
                x_current: Estado actual del sistema (vector numpy)
                sp_trajectory: Trayectoria de setpoints para el horizonte N (numpy array [nc x N])
                u_previous: Acción de control aplicada en el paso anterior (vector numpy)
            Returns:
                u_optimal: La primera acción de control óptima a aplicar (vector numpy)
                sol_stats: Estadísticas de la solución del solver
                predicted_x: Trayectoria de estados predicha (numpy array [nx x N+1])
                predicted_u: Secuencia de entradas predicha (numpy array [nu x N])
            """
            # Establecer valores de parámetros
            p_val = np.concatenate([x_current, sp_trajectory.flatten('F'), u_previous])

            # Usar la última solución como punto de partida si está disponible y tiene la dimensión correcta
            current_w0 = self.w0 if len(self.w0) == ca.vertcat(*self.w).shape[0] else np.zeros(ca.vertcat(*self.w).shape[0])

            # Si es la primera llamada, intentar inicializar w0 de forma más inteligente
            if np.all(current_w0 == 0):
                w0_guess = []
                x_guess = x_current
                u_guess = u_previous
                for k in range(self.N):
                    w0_guess.extend(u_guess) # U_k
                    # Inicializar Xc con el estado actual repetido
                    w0_guess.extend(np.tile(x_guess, self.m)) # Xc_k
                    # Predecir X_{k+1} (simplificado - podrías integrar un paso)
                    # dx_guess = self.model_ode(x_guess, u_guess).full().flatten()
                    # x_guess = x_guess + dx_guess * self.dt
                    w0_guess.extend(x_guess) # X_{k+1}
                if len(w0_guess) == ca.vertcat(*self.w).shape[0]:
                    current_w0 = np.array(w0_guess)
                #else:
                    # print(f"Warning: Dimension mismatch in w0 guess: {len(w0_guess)} vs {self.solver.n_x()}")
                    # pass # Keep current_w0 as zeros


            # Resolver el NLP
            try:
                sol = self.solver(
                    x0=current_w0,     # Estimación inicial (warm start)
                    lbx=self.lbw,      # Usar límites originales definidos
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=p_val
                )
            except RuntimeError as e:
                print(f"Error en la llamada al solver: {e}")
                # Intentar resolver sin warm start (x0=[0]) como último recurso
                try:
                    print("Intentando resolver sin warm start...")
                    sol = self.solver(
                        x0=np.zeros(ca.vertcat(*self.w).shape[0]), # Intentar con ceros
                        lbx=self.lbw, ubx=self.ubx,
                        lbg=self.lbg, ubg=self.ubg,
                        p=p_val
                    )
                except RuntimeError as e2:
                    print(f"Segundo error en la llamada al solver: {e2}")
                    return u_previous, {'success': False, 'return_status': 'SolverError'}, None, None


            # Extraer la solución
            w_opt = sol['x'].full().flatten()
            sol_stats = self.solver.stats()

            # Verificar éxito del solver
            if not sol_stats['success']:
                # Imprimir más detalles si está disponible
                print(f"¡ADVERTENCIA: El solver NMPC no convergió! Estado: {sol_stats.get('return_status', 'Desconocido')}")
                # print(f"Iteraciones: {sol_stats.get('iter_count', 'N/A')}")
                # Considerar resetear w0 si falla repetidamente
                # self.w0 = np.zeros(self.solver.n_x())
                return u_previous, sol_stats, None, None

            # Actualizar w0 para warm start para la PRÓXIMA llamada
            self.w0 = w_opt

            # --- Extraer la secuencia de control óptima y predicciones ---
            u_optimal_sequence = np.zeros((self.nu, self.N))
            x_predicted_sequence = np.zeros((self.nx, self.N + 1))

            # El primer estado de la predicción es el estado actual real
            x_predicted_sequence[:, 0] = x_current

            # Extraer el resto de la trayectoria de w_opt
            for k in range(self.N):
                u_optimal_sequence[:, k] = w_opt[self.indices['U'][k] : self.indices['U'][k] + self.nu]
                # self.indices['X'][k] apunta al índice de inicio de X_{k+1} en w_opt
                x_predicted_sequence[:, k+1] = w_opt[self.indices['X'][k] : self.indices['X'][k] + self.nx]

            # Devolver la primera acción de control
            u_apply = u_optimal_sequence[:, 0]

            return u_apply, sol_stats, x_predicted_sequence, u_optimal_sequence

    # ---------------------------------------------------
    # 3. Simulación del Sistema (Planta) (Copiada directamente)
    # ---------------------------------------------------
    def simulate_plant(x0, u, dt, model_ode):
        """Simula un paso de la planta real (integrando el modelo)."""
        # Asegurar que u esté en formato correcto para la EDO
        # u[0] es F_S (L/h), u[1] es Q_j/3600.0 (J/h)
        u_val = u

        # Define la función de EDO para solve_ivp
        def ode_sys(t, x):
            # Llama a la función CasADi y convierte a numpy array plano
            return model_ode(x, u_val).full().flatten()

        # Integra
        t_span = [0, dt]
        sol = solve_ivp(ode_sys, t_span, x0, method='RK45', rtol=1e-5, atol=1e-8) # Puedes usar 'BDF' para sistemas stiff

        # Devuelve el estado al final del intervalo
        return sol.y[:, -1]

    # --- Sidebar para configuración ---
    st.sidebar.header("Configuración del Modelo")
    mu_max_input = st.sidebar.number_input("mu_max", value=0.4)
    K_S_input = st.sidebar.number_input("K_S", value=0.05)
    Y_XS_input = st.sidebar.number_input("Y_XS", value=0.5)
    Y_QX_input = st.sidebar.number_input("Y_QX", value=15000)
    S_in_input = st.sidebar.number_input("S_in", value=10.0)
    V_input = st.sidebar.number_input("V", value=1.0)
    rho_input = st.sidebar.number_input("rho", value=1000.0)
    Cp_input = st.sidebar.number_input("Cp", value=4184.0)
    T_in_input = st.sidebar.number_input("T_in", value=298.15)
    F_const_input = st.sidebar.number_input("F_const", value=0.0)

    st.sidebar.header("Configuración NMPC")
    N_input = st.sidebar.number_input("Horizonte de Predicción (N)", min_value=1, value=10)
    M_input = st.sidebar.number_input("Horizonte de Control (M)", min_value=1, value=4)
    dt_nmpc_input = st.sidebar.number_input("Tiempo de Muestreo NMPC (dt)", min_value=0.01, value=0.1)
    simulation_time = st.sidebar.number_input("Tiempo Total de Simulación", min_value=1.0, value=24.0)

    st.sidebar.header("Límites de Entradas")
    max_FS = st.sidebar.number_input("Máximo Flujo Sustrato (F_S)", value=1.5)
    max_Qj_div_3600 = st.sidebar.number_input("Máxima Carga Térmica (Q_j/3600)", value=10000.0)

    st.sidebar.header("Pesos NMPC")
    Q_X_weight = st.sidebar.number_input("Peso Q para Biomasa (X)", value=1.0)
    Q_T_weight = st.sidebar.number_input("Peso Q para Temperatura (T)", value=0.01)
    W_FS_weight = st.sidebar.number_input("Peso W para Flujo Sustrato (F_S)", value=0.1)
    W_Qj_weight = st.sidebar.number_input("Peso W para Carga Térmica (Q_j/3600)", value=1e-8)

    st.sidebar.header("Condiciones Iniciales y Setpoints")
    initial_X = st.sidebar.number_input("Biomasa Inicial (X0)", value=1.5)
    initial_S = st.sidebar.number_input("Substrato Inicial (S0)", value=9.0)
    initial_T = st.sidebar.number_input("Temperatura Inicial (T0)", value=305.0)
    setpoint_X_t5 = st.sidebar.number_input("Setpoint Biomasa (X) en t=5h", value=2.0)
    setpoint_T_t5 = st.sidebar.number_input("Setpoint Temperatura (T) en t=5h", value=308.0)
    setpoint_X_t12 = st.sidebar.number_input("Setpoint Biomasa (X) en t=12h", value=1.0)
    setpoint_T_t12 = st.sidebar.number_input("Setpoint Temperatura (T) en t=12h", value=303.0)

    start_simulation = st.button("Iniciar Simulación NMPC")

    if start_simulation:
        # --- Configuración de la simulación ---
        t_final = simulation_time
        dt_nmpc = dt_nmpc_input
        n_steps = int(t_final / dt_nmpc)

        # Parámetros del modelo definidos por el usuario
        user_params = {
            'mu_max': mu_max_input,
            'K_S': K_S_input,
            'Y_XS': Y_XS_input,
            'Y_QX': Y_QX_input,
            'S_in': S_in_input,
            'V': V_input,
            'rho': rho_input,
            'Cp': Cp_input,
            'T_in': T_in_input,
            'F_const': F_const_input,
        }

        # Obtener modelo con parámetros definidos por el usuario
        model_ode, output_func, x_sym, u_sym, c_sym, dx_sym, params = get_bioreactor_model(user_params)
        nx = x_sym.shape[0]
        nu = u_sym.shape[0]
        nc = c_sym.shape[0]

        # Parámetros NMPC definidos por el usuario
        N = N_input
        M = M_input
        Q_weights = [Q_X_weight, Q_T_weight]
        W_weights = [W_FS_weight, W_Qj_weight]

        # Límites (Ejemplo - ¡AJUSTAR si es necesario!)
        lbx = [0.0, 0.0, 290.0]    # [X_min, S_min, T_min(K)]
        ubx = [5.0, 10.0, 315.0]   # [X_max, S_max, T_max(K)]
        lbu = [0.0, -max_Qj_div_3600] # [F_S_min, Q_j_min/3600]
        ubu = [max_FS,   max_Qj_div_3600] # [F_S_max, Q_j_max/3600]
        lbdu = [-0.1, -5000.0]
        ubdu = [ 0.1,   5000.0]

        # Crear instancia NMPC
        nmpc = NMPCBioreactor(dt_nmpc, N, M, Q_weights, W_weights, model_ode, output_func,
                                x_sym, u_sym, c_sym, params,
                                lbx, ubx, lbu, ubu, lbdu, ubdu)

        # --- Simulación ---
        x_current = np.array([initial_X, initial_S, initial_T]) # Estado inicial
        u_previous = np.array([0.1, 0.0]) # Entrada inicial
        current_setpoint = np.array([initial_X, initial_T]) # Iniciar con valores iniciales

        t_history = np.linspace(0, t_final, n_steps + 1)
        x_history = np.zeros((nx, n_steps + 1))
        u_history = np.zeros((nu, n_steps))
        c_history = np.zeros((nc, n_steps + 1))
        sp_history = np.zeros((nc, n_steps + 1))

        x_history[:, 0] = x_current
        c_history[:, 0] = output_func(x_current).full().flatten()
        sp_history[:, 0] = current_setpoint

        progress_bar = st.progress(0)
        status_text = st.empty()

        for k in range(n_steps):
            t_current = k * dt_nmpc

            # Cambios de setpoint
            if abs(t_current - 5.0) < 1e-9 or (t_current < 5.0 and abs((k + 1) * dt_nmpc - 5.0) < 1e-9):
                st.write(f"Cambiando setpoint en t=5h a: Biomasa={setpoint_X_t5}, Temperatura={setpoint_T_t5}")
                current_setpoint = np.array([setpoint_X_t5, setpoint_T_t5])
            elif abs(t_current - 12.0) < 1e-9 or (t_current < 12.0 and abs((k + 1) * dt_nmpc - 12.0) < 1e-9):
                st.write(f"Cambiando setpoint en t=12h a: Biomasa={setpoint_X_t12}, Temperatura={setpoint_T_t12}")
                current_setpoint = np.array([setpoint_X_t12, setpoint_T_t12])

            sp_traj = np.tile(current_setpoint, (N, 1)).T

            u_optimal, stats, _, _ = nmpc.solve(x_current, sp_traj, u_previous)

            if not stats['success']:
                u_apply = u_previous
            else:
                u_apply = u_optimal
                u_apply = np.clip(u_apply, lbu, ubu)
                delta_u_applied = u_apply - u_previous
                delta_u_applied = np.clip(delta_u_applied, lbdu, ubdu)
                u_apply = u_previous + delta_u_applied
                u_apply = np.clip(u_apply, lbu, ubu)

            x_next = simulate_plant(x_current, u_apply, dt_nmpc, model_ode)

            x_current = x_next
            u_previous = u_apply

            x_history[:, k+1] = x_current
            u_history[:, k] = u_apply
            c_history[:, k+1] = output_func(x_current).full().flatten()
            sp_history[:, k+1] = current_setpoint

            progress = (k + 1) / n_steps
            progress_bar.progress(progress)
            status_text.text(f"Simulación en progreso: {progress * 100:.2f}%")

        status_text.text("Simulación completada.")

        # --- Graficar Resultados ---
        st.subheader("Resultados de la Simulación NMPC")
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle("Simulación NMPC Biorreactor", fontsize=16)

        # Biomasa
        axs[0, 0].plot(t_history, x_history[0, :], label='Biomasa (X)')
        axs[0, 0].plot(t_history, sp_history[0, :], 'r--', label='Setpoint X')
        axs[0, 0].set_ylabel('Biomasa (g/L)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Substrato
        axs[1, 0].plot(t_history, x_history[1, :], label='Substrato (S)')
        axs[1, 0].set_ylabel('Substrato (g/L)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Temperatura
        axs[2, 0].plot(t_history, x_history[2, :], label='Temperatura (T)')
        axs[2, 0].plot(t_history, sp_history[1, :], 'r--', label='Setpoint T')
        axs[2, 0].set_ylabel('Temperatura (K)')
        axs[2, 0].set_xlabel('Tiempo (h)')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Flujo Sustrato (F_S)
        axs[0, 1].step(t_history[:-1], u_history[0, :], where='post', label='Flujo Sustrato (F_S)')
        axs[0, 1].set_ylabel('F_S (L/h)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(bottom=min(lbu[0] - 0.01, np.min(u_history[0,:]) - 0.01),
                         top=max(ubu[0] + 0.01, np.max(u_history[0,:]) + 0.01))

        # Carga Térmica (Q_j)
        axs[1, 1].step(t_history[:-1], u_history[1, :] * 3600.0, where='post', label='Carga Térmica (Q_j)')
        axs[1, 1].set_ylabel('Q_j (W)')
        axs[1, 1].set_xlabel('Tiempo (h)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim(bottom=min(lbu[1]*3600 - 100, np.min(u_history[1,:]*3600) - 100),
                         top=max(ubu[1]*3600 + 100, np.max(u_history[1,:]*3600) + 100))

        fig.delaxes(axs[2, 1]) # Eliminar el subplot vacío

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)


# -------------------------
# Ejecución principal
# -------------------------
if __name__ == "__main__":
    pass