import streamlit as st

def home_page():
    st.title("Bioprocess Modeling and Control")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/Batch.png", use_container_width=True)
        st.caption("**Figure 1:** Batch Reactor")
    with col2:
        st.image("images/fed_batch.png", use_container_width=True)
        st.caption("**Figure 2:** Fed-Batch Reactor")
    with col3:
        st.image("images/continous.png", use_container_width=True)
        st.caption("**Figure 3:** Continuous Reactor")

    st.header("General Theoretical Basis") # Adjusted title
    st.markdown("""
        Bioprocess modeling allows to describe mathematically the evolution of the variables of interest 
        (biomass concentration, substrate, product, dissolved oxygen, etc.) in a bioreactor. 
        The general material balances for the three main modes of operation are presented below,
        assuming a perfect mixing of the three main modes of operation. 

        El modelado de bioprocesos permite describir matemáticamente la evolución de las variables de interés 
        (concentración de biomasa, sustrato, producto, oxígeno disuelto, etc.) en un biorreactor. 
        A continuación se presentan los balances de materia generales para los tres modos de operación 
        principales, asumiendo un mezclado perfecto.
        """)

    st.markdown("---") # Visual separator

    st.subheader("🔹 Batch Mode")
    st.markdown("""
        In this mode, there is no input or output of matter once the process has started. The volume $V$ is constant.
        The material balances are:

        En este modo, no hay entrada ni salida de materia una vez iniciado el proceso. El volumen $V$ es constante. 
        Los balances de materia son:
        """)
    st.latex(r"""
    \frac{dX}{dt} = \mu(S, O_2, P) \cdot X - k_d \cdot X 
    """)
    st.latex(r"""
    \frac{dS}{dt} = - \frac{1}{Y_{XS}} \cdot \mu(S, O_2, P) \cdot X - m_s \cdot X
    """)
    st.latex(r"""
    \frac{dP}{dt} = q_P \cdot X \quad \text{(Using specific rate } q_P) 
    """) # Changed to use general qP
    st.latex(r"""
    \frac{dO_2}{dt} = k_L a \cdot (C_{O_2}^* - O_2) - OUR \quad \text{(Using OUR: Oxygen Uptake Rate)}
    """) # Changed to use general OUR
    
    st.markdown(r"""
        **Donde:**
        * $X$: Biomass concentration ($g/L$)
        * $S$: Limiting substrate concentration ($g/L$)
        * $P$: Product concentration ($g/L$)
        * $O_2$: Dissolved oxygen concentration ($mg/L$)
        * $\mu(S, O_2, P)$: Specific growth rate ($h^{-1}$)
        * $q_P$: Specific product formation rate ($g \cdot g^{-1} \cdot h^{-1}$)
        * $OUR$: Oxygen uptake rate ($mg \cdot L^{-1} \cdot h^{-1}$), usually $OUR = (\frac{\mu}{Y_{XO}} + m_o) \cdot X \cdot 1000$
        * $k_d$: Specific cell death or decay rate ($h^{-1}$)
        * $Y_{XS}$: Biomass/substrate yield coefficient ($g \cdot g^{-1}$)
        * $Y_{XO}$: Biomass/oxygen yield coefficient ($g_X \cdot g_{O2}^{-1}$)
        * $m_s$: Maintenance coefficient for substrate ($g_S \cdot g_X^{-1} \cdot h^{-1}$)
        * $m_o$: Maintenance coefficient for oxygen ($g_{O2} \cdot g_X^{-1} \cdot h^{-1}$)
        * $k_L a$: Volumetric oxygen transfer coefficient ($h^{-1}$)
        * $C_{O_2}^*$: Dissolved oxygen saturation concentration ($mg/L$)
        * $t$: Time ($h$)

        **Modelos Comunes de $\mu$ (Velocidad Específica de Crecimiento):**
        Las variantes comunes de $\mu$ consideradas son:
        1.  **Monod simple:** $\mu = \mu_{max} \frac{S}{K_S + S}$
        2.  **Monod sigmoidal (Hill):** $\mu = \mu_{max} \frac{S^n}{K_S^n + S^n}$
        3.  **Monod con inhibición por sustrato:** $\mu = \mu_{max} \frac{S}{K_S + S + S^2/K_I}$
        4.  **Monod con inhibición por producto:** $\mu = \mu_{max} \frac{S}{K_S + S} \left(1 - \frac{P}{P_{crit}}\right)^m$
        5.  **Monod con limitación por oxígeno:** $\mu = \mu_{max} \frac{S}{K_S + S} \frac{O_2}{K_O + O_2}$
        6.  **Monod con múltiples interacciones:** $\mu = \mu_{max} \frac{S}{K_S + S} \frac{O_2}{K_O + O_2} \frac{K_P}{K_P + P}$ 
            *(Nota: La forma exacta depende del sistema)*

        * $\mu_{max}$: Máxima velocidad específica de crecimiento ($h^{-1}$)
        * $K_S, K_O, K_P, K_I, P_{crit}$: Constantes de afinidad o inhibición (unidades de concentración)
        * $n, m$: Exponentes (adimensionales)
        
        **Modelos Comunes de $q_P$ (Formación de Producto):**
        1.  **Asociado al crecimiento:** $q_P = Y_{PX} \cdot \mu$
        2.  **No asociado al crecimiento:** $q_P = \beta$ (constante)
        3.  **Mixto (Luedeking-Piret):** $q_P = \alpha \cdot \mu + \beta$
        
        * $Y_{PX}$: Coeficiente de rendimiento producto/biomasa ($g \cdot g^{-1}$)
        * $\alpha$: Constante asociada al crecimiento ($g_P \cdot g_X^{-1}$)
        * $\beta$: Constante no asociada al crecimiento ($g_P \cdot g_X^{-1} \cdot h^{-1}$)
        """)

    st.markdown("---")

    st.subheader("🔹 Modo Lote Alimentado (Fed-Batch)")
    st.markdown(r"""
        Se agrega alimentación ($F$) con concentración de sustrato $S_{in}$ al biorreactor. No se retira líquido, por lo que el volumen $V$ varía en el tiempo. El balance de volumen es $\frac{dV}{dt} = F$. 
        Los balances de materia incorporan términos de dilución y adición:
        """)
    st.latex(r"""
    \frac{dX}{dt} = \mu \cdot X - k_d \cdot X - \frac{F}{V} \cdot X
    """)
    st.latex(r"""
    \frac{dS}{dt} = - (\frac{\mu}{Y_{XS}} + m_s) \cdot X + \frac{F}{V} (S_{in} - S) \quad \text{(Asumiendo } q_P \text{ no consume } S \text{ directamente, o } Y_{PS} \text{ está implícito en } q_P \text{)}
    """) # Se puede refinar si qP usa Yps
    st.latex(r"""
    \frac{dP}{dt} = q_P \cdot X - \frac{F}{V} \cdot P
    """)
    st.latex(r"""
    \frac{dO_2}{dt} = k_L a \cdot (C_{O_2}^* - O_2) - OUR - \frac{F}{V} \cdot O_2
    """)
    st.markdown(r"""
        **Nuevas Variables:**
        * $V$: Volumen del cultivo en el reactor ($L$)
        * $F$: Flujo de alimentación ($L \cdot h^{-1}$)
        * $S_{in}$: Concentración de sustrato en la alimentación ($g/L$)
        """)

    st.markdown("---")

    st.subheader("🔹 Modo Continuo (Quimiostato)")
    st.markdown(r"""
        Hay entrada de alimentación ($F$, $S_{in}$) y salida de caldo de cultivo a la misma tasa $F$. El volumen $V$ se mantiene constante. Se define la tasa de dilución $D = F/V$.
        Los balances son:
        """)
    st.latex(r"""
    \frac{dX}{dt} = \mu \cdot X - k_d \cdot X - D \cdot X
    """)
    st.latex(r"""
    \frac{dS}{dt} = - (\frac{\mu}{Y_{XS}} + m_s) \cdot X + D (S_{in} - S)
    """)
    st.latex(r"""
    \frac{dP}{dt} = q_P \cdot X - D \cdot P
    """)
    st.latex(r"""
    \frac{dO_2}{dt} = k_L a \cdot (C_{O_2}^* - O_2) - OUR - D \cdot O_2
    """)
    st.markdown(r"""
        **Nueva Variable:**
        * $D$: Tasa de dilución ($h^{-1}$), definida como $D = F/V$.
        """)
    
    st.markdown("---") # Separador antes de la nueva sección

    # >>> INICIO NUEVA SECCIÓN: FERMENTACIÓN ALCOHÓLICA <<<
    st.subheader("✳️ Ejemplo Específico: Fermentación Alcohólica (Levadura)")
    st.markdown(r"""
        Un ejemplo clásico en bioprocesos es la producción de etanol ($P$) utilizando levaduras (ej. *Saccharomyces cerevisiae*, $X$) que consumen azúcares (ej. glucosa, $S$). Este proceso a menudo se opera en fases para optimizar tanto el crecimiento celular inicial como la producción de etanol posterior:
        1.  **Fase Lote Inicial (Aeróbica):** Se opera en modo lote con aireación para promover un rápido crecimiento de la biomasa. El nivel de oxígeno disuelto ($O_2$) se mantiene bajo pero presente para favorecer la respiración.
        2.  **Fase de Alimentación (Fed-Batch, Anaeróbica/Microaeróbica):** Se alimenta sustrato concentrado ($S_{in}$) para mantener una alta densidad celular y evitar la represión catabólica (efecto Crabtree), mientras se limita o elimina el suministro de oxígeno para inducir la vía fermentativa (producción de etanol).
        3.  **Fase Lote Final (Anaeróbica):** Se detiene la alimentación y se permite que la levadura consuma el sustrato restante en condiciones anaeróbicas.

        Para modelar este comportamiento complejo, se requieren cinéticas que capturen los efectos del sustrato, el producto (etanol, que es inhibidor) y el oxígeno.
        """)

    st.markdown("**Cinética Mixta Aerobia/Anaerobia (Modelo 'Fermentación' de `ferm_alcohol.py`)**")
    st.markdown(r"""
        Este modelo asume que la tasa de crecimiento total ($\mu_{total}$) es la suma de una componente aeróbica ($\mu_{aerobia}$) y una componente anaeróbica/fermentativa ($\mu_{anaerobia}$):
        """)
    st.latex(r"\mu_{total} = \mu_{aerobia} + \mu_{anaerobia}")
    st.markdown("Componente Aeróbica (favorecida por $O_2$, limitada por $S$):")
    st.latex(r"""
    \mu_{aerobia} = \mu_{max, aerob} \left( \frac{S}{K_{S, aerob} + S} \right) \left( \frac{O_2}{K_{O, aerob} + O_2} \right)
    """)
    st.markdown("Componente Anaerobia/Fermentativa (inhibida por $S$, $P$ y $O_2$):")
    st.latex(r"""
    \mu_{anaerobia} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( 1 - \frac{P}{K_{P, anaerob}} \right)^{n_p} \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
    """)
    st.markdown(r"""
        **Parámetros Específicos de esta Cinética:**
        * $\mu_{max, aerob}, \mu_{max, anaerob}$: Máx. $\mu$ para vía aerobia y anaerobia ($h^{-1}$)
        * $K_{S, aerob}, K_{S, anaerob}$: Constantes de afinidad por sustrato ($g/L$)
        * $K_{O, aerob}$: Constante de afinidad por oxígeno para crecimiento aerobio ($mg/L$)
        * $K_{iS, anaerob}$: Constante de inhibición por sustrato para vía anaerobia ($g/L$)
        * $K_{P, anaerob}$: Constante de inhibición por producto (etanol) ($g/L$) - concentración crítica a la que cesa el crecimiento/producción anaerobia.
        * $n_p$: Exponente de inhibición por producto (adimensional)
        * $K_{O, inhib}$: Constante de inhibición por oxígeno para la vía anaerobia ($mg/L$) - indica la sensibilidad de la fermentación a la presencia de $O_2$.
        """)

    st.markdown("**Formación de Producto (Etanol) - Modelo Luedeking-Piret**")
    st.markdown(r"""
        La tasa específica de producción de etanol ($q_P$) se modela frecuentemente con la ecuación de Luedeking-Piret, que incluye términos asociados y no asociados al crecimiento:
        """)
    st.latex(r"q_P = \alpha \cdot \mu_{total} + \beta")
    st.markdown(r"""
        **Parámetros:**
        * $q_P$: Tasa específica de producción de etanol ($g_P \cdot g_X^{-1} \cdot h^{-1}$)
        * $\alpha$: Coeficiente asociado al crecimiento ($g_P \cdot g_X^{-1}$)
        * $\beta$: Coeficiente no asociado al crecimiento ($g_P \cdot g_X^{-1} \cdot h^{-1}$)

        Este modelo permite que el etanol se produzca tanto cuando las células crecen activamente ($\alpha \mu > 0$) como cuando el crecimiento es bajo o nulo pero las células están metabólicamente activas ($\beta > 0$).
        """)
    # >>> FIN NUEVA SECCIÓN <<<

    st.markdown("---") # Separador después de la nueva sección

    st.header("Técnicas Avanzadas de Análisis y Control")
    
    st.subheader("🔹 Análisis de Sensibilidad")
    st.markdown(r"""
        Evalúa cómo la incertidumbre o variaciones en los parámetros del modelo ($\theta$, como $\mu_{max}, K_S, Y_{XS}$, etc.) afectan las salidas del modelo (las variables de estado $X, S, P, O_2$). 
        Permite identificar los parámetros más influyentes, crucial para la optimización y el diseño experimental. 
        Una métrica común es el coeficiente de sensibilidad normalizado:
        $S_{ij} = \frac{\partial y_i / y_i}{\partial \theta_j / \theta_j} = \frac{\partial \ln y_i}{\partial \ln \theta_j}$
        donde $y_i$ es una salida y $\theta_j$ es un parámetro.
        """)

    # ... (Resto de las secciones de Ajuste de Parámetros, EKF, RTO, NMPC sin cambios) ...
    
    st.markdown("---")

    st.subheader("🔹 Ajuste de Parámetros (Estimación)")
    st.markdown(r"""
        Proceso de encontrar los valores de los parámetros del modelo ($\theta$) que mejor describen un conjunto de datos experimentales ($y_{exp}$). Se minimiza una función objetivo $J(\theta)$ que mide la discrepancia entre las predicciones del modelo ($y_{model}$) y los datos.
        El problema de optimización es:
        """)
    st.latex(r"""
    \hat{\theta} = \arg \min_{\theta} J(\theta) 
    """)
    st.markdown(r"""
        Una función objetivo común es la suma de errores cuadráticos ponderados:
        $J(\theta) = \sum_{k=1}^{N} \sum_{i=1}^{M} w_{ik} (y_{i,exp}(t_k) - y_{i,model}(t_k, \theta))^2$
        donde $N$ es el número de puntos de muestreo, $M$ el número de variables medidas, y $w_{ik}$ son pesos.
        Se usan algoritmos de optimización (Levenberg-Marquardt, SQP, genéticos, etc.).
        """)

    st.markdown("---")

    st.subheader("🔹 Filtro de Kalman Extendido (EKF)")
    st.markdown(r"""
        Algoritmo recursivo para estimar el estado de sistemas dinámicos no lineales en presencia de ruido. Utiliza un modelo del sistema y mediciones ruidosas para obtener una estimación óptima (en el sentido de mínima varianza) del estado. Esencial cuando no todas las variables de estado (e.g., biomasa) se pueden medir directamente online.

        **Modelo del sistema (discreto):**
        """)
    st.latex(r"x_{k+1} = f(x_k, u_k) + w_k \quad \text{(Ecuación de proceso)}")
    st.latex(r"z_k = h(x_k) + v_k \quad \text{(Ecuación de medida)}")
    st.markdown(r"""
        **Donde:**
        * $x_k$: Estado del sistema en el instante $k$
        * $u_k$: Entrada de control en el instante $k$
        * $z_k$: Medición en el instante $k$
        * $w_k$: Ruido del proceso (Gaussiano, media cero, covarianza $Q$)
        * $v_k$: Ruido de medición (Gaussiano, media cero, covarianza $R$)
        * $f$: Función de transición de estado (no lineal)
        * $h$: Función de medición (no lineal)

        **Etapas del EKF:**

        1.  **Predicción:**
            * Estado predicho: $\hat{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k)$
            * Covarianza del error predicha: $P_{k+1|k} = F_k P_{k|k} F_k^T + Q$

        2.  **Actualización (Corrección):**
            * Ganancia de Kalman: $K_{k+1} = P_{k+1|k} H_{k+1}^T (H_{k+1} P_{k+1|k} H_{k+1}^T + R)^{-1}$
            * Estado actualizado: $\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} + K_{k+1} (z_{k+1} - h(\hat{x}_{k+1|k}))$
            * Covarianza del error actualizada: $P_{k+1|k+1} = (I - K_{k+1} H_{k+1}) P_{k+1|k}$

        **Donde:**
        * $F_k = \frac{\partial f}{\partial x} \Big|_{\hat{x}_{k|k}, u_k}$ es el Jacobiano de $f$ respecto a $x$.
        * $H_{k+1} = \frac{\partial h}{\partial x} \Big|_{\hat{x}_{k+1|k}}$ es el Jacobiano de $h$ respecto a $x$.
        """)

    st.markdown("---")

    st.subheader("🔹 Control RTO (Real-Time Optimization)")
    st.markdown(r"""
        Estrategia de control de alto nivel que optimiza una función objetivo económica (ej. maximizar beneficio, minimizar coste) ajustando los setpoints de los controladores reguladores o directamente las variables manipuladas, basándose en un modelo del proceso y mediciones actuales. Opera en una escala de tiempo más lenta que el control regulatorio.

        **Problema de Optimización:**
        """)
    st.latex(r"""
    \max_{u_{opt}} \quad \Phi(x_{ss}, u_{opt}, p)
    """)
    st.markdown(r"""
        **Sujeto a:**
        """)
    st.latex(r"""
    f(x_{ss}, u_{opt}, p) = 0 \quad \text{(Modelo en estado estacionario)}
    """)
    st.latex(r"""
    g(x_{ss}, u_{opt}, p) \le 0 \quad \text{(Restricciones de operación)}
    """)
    st.latex(r"""
    u_{min} \le u_{opt} \le u_{max} \quad \text{(Límites en variables manipuladas)}
    """)
    st.markdown(r"""
        **Donde:**
        * $\Phi$: Función objetivo económica.
        * $x_{ss}$: Estado estacionario del proceso.
        * $u_{opt}$: Variables manipuladas óptimas (setpoints).
        * $p$: Parámetros del modelo (pueden ser actualizados).
        * $f$: Ecuaciones del modelo en estado estacionario.
        * $g$: Restricciones (calidad, seguridad, operativas).

        Se resuelve periódicamente (ej. cada pocas horas) para encontrar los $u_{opt}$ óptimos.
        """)

    st.markdown("---")

    st.subheader("🔹 Control NMPC (Nonlinear Model Predictive Control)")
    st.markdown(r"""
        Técnica de control avanzado que utiliza un modelo dinámico no lineal del proceso para predecir su comportamiento futuro sobre un horizonte de predicción ($N_p$) y calcular una secuencia óptima de acciones de control futuras ($\Delta U$) sobre un horizonte de control ($N_c \le N_p$). Minimiza una función objetivo que penaliza desviaciones del setpoint y el esfuerzo de control, sujeto a restricciones.

        **Problema de Optimización (resuelto en cada instante $k$):**
        """)
    st.latex(r"""
    \min_{\Delta U_k} J = \sum_{j=1}^{N_p} ||\hat{y}_{k+j|k} - y_{sp, k+j}||^2_Q + \sum_{j=0}^{N_c-1} ||\Delta u_{k+j|k}||^2_R 
    """)
    st.markdown(r"""
        **Sujeto a:**
        """)
    st.latex(r"""
    \hat{x}_{k+j+1|k} = f(\hat{x}_{k+j|k}, u_{k+j|k}) \quad \text{(Modelo de predicción)}
    """)
    st.latex(r"""
    \hat{y}_{k+j|k} = h(\hat{x}_{k+j|k}) \quad \text{(Salidas predichas)}
    """)
    st.latex(r"""
    u_{min} \le u_{k+j|k} \le u_{max} \quad \text{(Restricciones de entrada)}
    """)
    st.latex(r"""
    \Delta u_{min} \le \Delta u_{k+j|k} \le \Delta u_{max} \quad \text{(Restricciones de cambio de entrada)}
    """)
    st.latex(r"""
    y_{min} \le \hat{y}_{k+j|k} \le y_{max} \quad \text{(Restricciones de salida)}
    """)
    st.markdown(r"""
        **Donde:**
        * $\Delta U_k = [\Delta u_{k|k}, ..., \Delta u_{k+N_c-1|k}]^T$: Secuencia de cambios de control a optimizar.
        * $u_{k+j|k} = u_{k+j-1|k} + \Delta u_{k+j|k}$: Control aplicado.
        * $\hat{x}_{k+j|k}, \hat{y}_{k+j|k}$: Estado y salida predichos en el instante $k+j$ basados en información hasta $k$.
        * $y_{sp, k+j}$: Setpoint (trayectoria futura si es necesario).
        * $Q, R$: Matrices de ponderación.

        Solo se aplica el primer cambio de control calculado ($\Delta u_{k|k}$), se mide el estado actual, y se repite la optimización en el siguiente instante (principio de horizonte deslizante).
        """)


# Para poder ejecutar esta página individualmente si es necesario
if __name__ == "__main__":
    import os
    if not os.path.exists("images"):
        os.makedirs("images")
        dummy_files = ["images/Batch.png", "images/fed_batch.png", "images/continous.png"]
        for f_path in dummy_files:
            if not os.path.exists(f_path):
                with open(f_path, 'w') as fp:
                    pass 
                    
    home_page()