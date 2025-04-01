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
# Ejecución principal
# -------------------------
if __name__ == "__main__":
    pass