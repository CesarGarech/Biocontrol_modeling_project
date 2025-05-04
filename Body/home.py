import streamlit as st

def home_page():
    st.title("Modelado y Control de Bioprocesos")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/Batch.png", use_container_width=True)
        st.caption("**Figura 1:** Reactor Batch (Discontinuo)")
    with col2:
        st.image("images/fed_batch.png", use_container_width=True)
        st.caption("**Figura 2:** Reactor Fed-Batch (Lote Alimentado)")
    with col3:
        st.image("images/continous.png", use_container_width=True)
        st.caption("**Figura 3:** Reactor Continuo (Quimiostato)")

    st.header("Fundamento Teórico General") # Título ajustado
    st.markdown("""
        El modelado de bioprocesos permite describir matemáticamente la evolución de las variables de interés
        (concentración de biomasa, sustrato, producto, oxígeno disuelto, etc.) en un biorreactor.
        A continuación se presentan los balances de materia generales para los tres modos de operación
        principales, asumiendo un mezclado perfecto.
        """)

    st.markdown("---") # Separador visual

    st.subheader("🔹 Modo Lote (Batch)")
    st.markdown("""
        En este modo, no hay entrada ni salida de materia una vez iniciado el proceso. El volumen $V$ es constante.
        Los balances de materia son:
        """)
    st.latex(r"""
    \frac{dX}{dt} = \mu(S, O_2, P) \cdot X - k_d \cdot X
    """)
    st.latex(r"""
    \frac{dS}{dt} = - \frac{1}{Y_{XS}} \cdot \mu(S, O_2, P) \cdot X - m_s \cdot X - \frac{1}{Y_{PS}} q_P \cdot X
    """) # <<< MODIFICADO: Incluye consumo por producto vía Yps >>>
    st.latex(r"""
    \frac{dP}{dt} = q_P \cdot X \quad \text{(Usando tasa específica } q_P)
    """)
    st.latex(r"""
    \frac{dO_2}{dt} = k_L a \cdot (C_{O_2}^* - O_2) - OUR \quad \text{(Usando OUR: Oxygen Uptake Rate)}
    """)

    st.markdown(r"""
        **Donde:**
        * $X$: Concentración de biomasa ($g/L$)
        * $S$: Concentración de sustrato limitante ($g/L$)
        * $P$: Concentración de producto ($g/L$)
        * $O_2$: Concentración de oxígeno disuelto ($mg/L$)
        * $\mu(S, O_2, P)$: Velocidad específica de crecimiento ($h^{-1}$)
        * $q_P$: Tasa específica de formación de producto ($g_P \cdot g_X^{-1} \cdot h^{-1}$)
        * $OUR$: Tasa de consumo de oxígeno ($mg \cdot L^{-1} \cdot h^{-1}$), usualmente $OUR = q_O \cdot X = (\frac{\mu}{Y_{XO}} + m_o) \cdot X \cdot 1000$
        * $k_d$: Tasa específica de muerte o decaimiento celular ($h^{-1}$)
        * $Y_{XS}$: Coeficiente de rendimiento biomasa/sustrato ($g_X \cdot g_S^{-1}$)
        * $Y_{PS}$: Coeficiente de rendimiento producto/sustrato ($g_P \cdot g_S^{-1}$) # <<< NUEVO >>>
        * $Y_{XO}$: Coeficiente de rendimiento biomasa/oxígeno ($g_X \cdot g_{O2}^{-1}$)
        * $m_s$: Coeficiente de mantenimiento para sustrato ($g_S \cdot g_X^{-1} \cdot h^{-1}$)
        * $m_o$: Coeficiente de mantenimiento para oxígeno ($g_{O2} \cdot g_X^{-1} \cdot h^{-1}$)
        * $k_L a$: Coeficiente volumétrico de transferencia de oxígeno ($h^{-1}$)
        * $C_{O_2}^*$: Concentración de saturación de oxígeno disuelto ($mg/L$)
        * $t$: Tiempo ($h$)

        **Modelos Comunes de $\mu$ (Velocidad Específica de Crecimiento):**
        Las variantes comunes de $\mu$ consideradas son:
        1.  **Monod simple:** $\mu = \mu_{max} \frac{S}{K_S + S}$
        2.  **Monod sigmoidal (Hill):** $\mu = \mu_{max} \frac{S^n}{K_S^n + S^n}$
        3.  **Monod con inhibición por sustrato (Haldane):** $\mu = \mu_{max} \frac{S}{K_S + S + S^2/K_{iS}}$ # <<< Nombre añadido >>>
        4.  **Monod con inhibición por producto:** $\mu = \mu_{max} \frac{S}{K_S + S} \left(1 - \frac{P}{P_{crit}}\right)^m$ (u otras formas)
        5.  **Monod con limitación por oxígeno:** $\mu = \mu_{max} \frac{S}{K_S + S} \frac{O_2}{K_O + O_2}$
        6.  **Monod con múltiples interacciones:** $\mu = \mu_{max} \frac{S}{K_S + S} \frac{O_2}{K_O + O_2} \frac{K_P}{K_P + P}$
            *(Nota: La forma exacta depende del sistema)*

        * $\mu_{max}$: Máxima velocidad específica de crecimiento ($h^{-1}$)
        * $K_S, K_O, K_P, K_{iS}, P_{crit}$: Constantes de afinidad o inhibición (unidades de concentración) # <<< K_iS añadido >>>
        * $n, m$: Exponentes (adimensionales)

        **Modelos Comunes de $q_P$ (Formación de Producto):**
        1.  **Asociado al crecimiento:** $q_P = Y_{PX} \cdot \mu$
        2.  **No asociado al crecimiento:** $q_P = \beta$ (constante)
        3.  **Mixto (Luedeking-Piret):** $q_P = \alpha \cdot \mu + \beta$

        * $Y_{PX}$: Coeficiente de rendimiento producto/biomasa ($g_P \cdot g_X^{-1}$)
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
    \frac{dS}{dt} = - (\frac{\mu}{Y_{XS}} + m_s) \cdot X - \frac{q_P}{Y_{PS}} \cdot X + \frac{F}{V} (S_{in} - S)
    """) # <<< MODIFICADO: Incluye consumo por producto explícitamente >>>
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
    \frac{dS}{dt} = - (\frac{\mu}{Y_{XS}} + m_s) \cdot X - \frac{q_P}{Y_{PS}} \cdot X + D (S_{in} - S)
    """) # <<< MODIFICADO: Incluye consumo por producto explícitamente >>>
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

    # >>> INICIO SECCIÓN MODIFICADA: FERMENTACIÓN ALCOHÓLICA <<<
    st.subheader("✳️ Ejemplo Específico: Fermentación Alcohólica (Levadura)")
    st.markdown(r"""
        Un ejemplo clásico en bioprocesos es la producción de etanol ($P$) utilizando levaduras (ej. *Saccharomyces cerevisiae*, $X$) que consumen azúcares (ej. glucosa, $S$). Este proceso a menudo se opera en fases para optimizar tanto el crecimiento celular inicial como la producción de etanol posterior:
        1.  **Fase Lote Inicial (Aeróbica):** Se opera en modo lote con aireación para promover un rápido crecimiento de la biomasa. El nivel de oxígeno disuelto ($O_2$) se mantiene relativamente bajo pero presente para favorecer la respiración, maximizando la eficiencia energética para el crecimiento celular (alto $Y_{XS}$).
        2.  **Fase de Alimentación (Fed-Batch, Anaeróbica/Microaeróbica):** Se alimenta sustrato concentrado ($S_{in}$) para mantener una alta densidad celular y evitar la represión catabólica por altas concentraciones de glucosa (efecto Crabtree, aunque no modelado explícitamente aquí), mientras se limita o elimina el suministro de oxígeno ($O_2 \approx 0$) para inducir la vía fermentativa y maximizar la producción de etanol (efecto Pasteur).
        3.  **Fase Lote Final (Anaeróbica):** Se detiene la alimentación y se permite que la levadura consuma el sustrato restante en condiciones anaeróbicas, completando la producción de etanol.

        Para modelar este comportamiento complejo, donde el metabolismo cambia drásticamente según la disponibilidad de oxígeno, se requieren cinéticas que capturen los efectos del sustrato, el producto (etanol, que es inhibidor) y, crucialmente, el oxígeno como modulador metabólico.
        """)

    st.markdown("**Cinética Mixta Aerobia/Anaerobia (Modelo 'Fermentación' de `ferm_alcohol.py`)**")
    st.markdown(r"""
        Este modelo intenta capturar la plasticidad metabólica de la levadura asumiendo que la tasa de crecimiento observada ($\mu_{total}$) resulta de la contribución simultánea de una ruta respiratoria (aeróbica) y una ruta fermentativa (anaeróbica), cuya predominancia depende principalmente de la concentración de oxígeno disuelto ($O_2$).

        La tasa de crecimiento total se formula como la suma de ambas componentes:
        """)
    st.latex(r"\mu_{total} = \mu_{aerobia} + \mu_{anaerobia}")

    st.markdown(r"""
        **1. Componente Aeróbica ($\mu_{aerobia}$):** Representa el crecimiento vía respiración.
        """)
    st.latex(r"""
    \mu_{aerobia} = \mu_{max, aerob} \underbrace{\left( \frac{S}{K_{S, aerob} + S} \right)}_{\text{Limitación por Sustrato}} \underbrace{\left( \frac{O_2}{K_{O, aerob} + O_2} \right)}_{\text{Dependencia del Oxígeno}}
    """)
    st.markdown(r"""
        * El término de sustrato es un Monod simple: el crecimiento aumenta con $S$ hasta saturarse.
        * El término de oxígeno también es tipo Monod: el oxígeno actúa como un cosustrato esencial para la respiración. $K_{O, aerob}$ es la constante de afinidad; un valor bajo indica que la respiración puede operar eficientemente incluso a bajas concentraciones de $O_2$.
        * Esta componente domina cuando $O_2$ es suficientemente alto ($O_2 \gg K_{O, aerob}$ y $O_2 \gg K_{O, inhib}$).

        **2. Componente Anaerobia/Fermentativa ($\mu_{anaerobia}$):** Representa el crecimiento (menos eficiente) acoplado a la fermentación alcohólica.
        """)
    # <<< MODIFICADO: Se usará la forma con KP como concentración crítica y exponente n_p >>>
    st.latex(r"""
    \mu_{anaerobia} = \mu_{max, anaerob} \underbrace{\left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right)}_{\text{Inhibición por Sustrato (Haldane)}} \underbrace{\left( \frac{K_{P, anaerob}^{n_p}}{K_{P, anaerob}^{n_p} + P^{n_p}} \right)}_{\text{Inhibición por Producto (Etanol)}} \underbrace{\left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)}_{\text{Inhibición por Oxígeno (Pasteur)}}
    """)
    # Forma alternativa de inhibición por producto (comentada):
    # st.latex(r"""
    # \mu_{anaerobia} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( 1 - \frac{P}{P_{crit, anaerob}} \right)^{n_p} \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
    # """)
    st.markdown(r"""
        * El término de sustrato (Haldane) incluye inhibición ($K_{iS, anaerob}$): concentraciones muy altas de azúcar pueden inhibir la vía fermentativa.
        * El término de producto modela la inhibición por etanol ($P$). $K_{P, anaerob}$ representa la concentración de etanol a la que la tasa se reduce a la mitad (si $n_p=1$) o una constante relacionada con la concentración crítica que detiene la fermentación. El exponente $n_p$ ajusta la severidad de la inhibición.
        * El término de **inhibición por oxígeno** (Monod inverso) es clave para el **efecto Pasteur**: la presencia de oxígeno ($O_2$) suprime activamente la vía fermentativa. $K_{O, inhib}$ es la constante de inhibición; un valor bajo significa que incluso trazas de oxígeno pueden inhibir significativamente la fermentación. Este término es opuesto en efecto al término de dependencia del oxígeno en $\mu_{aerobia}$.
        * Esta componente domina cuando $O_2$ es muy bajo o cero ($O_2 \ll K_{O, inhib}$), siempre que $S$ y $P$ no sean excesivamente inhibidores.

        **Interacción y Parámetros:**
        La combinación $\mu_{total} = \mu_{aerobia} + \mu_{anaerobia}$ permite una transición suave entre metabolismo predominantemente respiratorio (alto $O_2$) y fermentativo (bajo $O_2$). Los parámetros clave que gobiernan esta transición son $K_{O, aerob}$ (afinidad por $O_2$ para respirar) y $K_{O, inhib}$ (sensibilidad de la fermentación a la inhibición por $O_2$). Los valores relativos de $\mu_{max, aerob}$ y $\mu_{max, anaerob}$ reflejan la máxima capacidad de crecimiento en cada condición.
        """)

    st.markdown("**Formación de Producto (Etanol) - Modelo Luedeking-Piret con Inhibición por Oxígeno**")
    st.markdown(r"""
        Si bien el crecimiento $\mu_{total}$ ya refleja la influencia del oxígeno, la tasa específica de producción de etanol ($q_P$) también puede ser directamente afectada por $O_2$. Comúnmente se usa el modelo de Luedeking-Piret como base:
        """)
    st.latex(r"q_{P, base} = \alpha \cdot \mu_{total} + \beta")
    st.markdown(r"""
        Sin embargo, para reflejar más fielmente que la producción de etanol es principalmente un proceso anaeróbico, se introduce un término adicional de inhibición por oxígeno directamente sobre $q_P$, similar al usado en $\mu_{anaerobia}$ pero con su propia constante ($K_{O,P}$):
        """)
    st.latex(r"""
    q_P = q_{P, base} \cdot \underbrace{\left( \frac{K_{O,P}}{K_{O,P} + O_2} \right)}_{\text{Inhibición directa por } O_2} = (\alpha \cdot \mu_{total} + \beta) \left( \frac{K_{O,P}}{K_{O,P} + O_2} \right)
    """)
    st.markdown(r"""
        **Parámetros:**
        * $q_P$: Tasa específica de producción de etanol efectiva ($g_P \cdot g_X^{-1} \cdot h^{-1}$)
        * $\alpha$: Coeficiente asociado al crecimiento ($g_P \cdot g_X^{-1}$)
        * $\beta$: Coeficiente no asociado al crecimiento ($g_P \cdot g_X^{-1} \cdot h^{-1}$)
        * $K_{O,P}$: Constante de inhibición por oxígeno específica para la producción de etanol ($mg/L$). Un valor bajo indica que la producción de etanol se suprime rápidamente con la presencia de $O_2$.

        Esta formulación asegura que, incluso si hay crecimiento residual en presencia de oxígeno ($\mu_{total} > 0$), la producción de etanol ($q_P$) se reduce significativamente, acoplando la formación del producto principal a las condiciones anaeróbicas deseadas.
        """)
    # >>> FIN SECCIÓN MODIFICADA <<<

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

    # ... (Resto de las secciones sin cambios) ...

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
    # Crear carpeta images si no existe y archivos dummy para evitar errores
    if not os.path.exists("images"):
        os.makedirs("images")
    dummy_files = ["images/Batch.png", "images/fed_batch.png", "images/continous.png"]
    for f_path in dummy_files:
        if not os.path.exists(f_path):
            try:
                # Intentar crear un archivo vacío
                with open(f_path, 'w') as fp:
                   pass # Archivo creado
                print(f"Archivo dummy creado: {f_path}")
            except Exception as e:
                print(f"No se pudo crear el archivo dummy {f_path}: {e}")
                # Podrías intentar crear una imagen placeholder real si tienes Pillow
                try:
                    from PIL import Image
                    img = Image.new('RGB', (60, 30), color = 'red')
                    img.save(f_path)
                    print(f"Imagen placeholder creada: {f_path}")
                except ImportError:
                    print("PIL no encontrado, no se puede crear imagen placeholder.")
                except Exception as e_img:
                     print(f"Error creando imagen placeholder {f_path}: {e_img}")


    home_page()