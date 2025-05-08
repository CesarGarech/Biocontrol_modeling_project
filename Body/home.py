# home_page.py
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

    st.markdown("""
        Bienvenido al simulador interactivo para modelado y control de bioprocesos.
        Esta herramienta permite explorar diferentes modos de operación de reactores,
        cinéticas de crecimiento microbiano y estrategias de control avanzadas.
        """)

    st.markdown("---") # Separador visual

    # ========= NUEVA SECCIÓN: MODELOS CINÉTICOS IMPLEMENTADOS =========
    st.header("🔬 Modelos Cinéticos Implementados")
    st.markdown("""
        El corazón del modelado de bioprocesos reside en describir matemáticamente las tasas a las que ocurren las reacciones biológicas clave: crecimiento celular, consumo de sustrato y formación de producto. A continuación se detallan los modelos cinéticos implementados en este simulador, particularmente enfocados en la fermentación alcohólica como caso de estudio.
        """)

    st.subheader("Velocidad Específica de Crecimiento ($\mu$)")
    st.markdown(r"""
        La velocidad específica de crecimiento ($\mu$, unidad $h^{-1}$) describe qué tan rápido aumenta la biomasa por unidad de biomasa existente. Depende de factores como la concentración de sustrato ($S$), producto ($P$) y oxígeno disuelto ($O_2$). Los modelos implementados son:
        """)

    with st.expander("1. Monod Simple"):
        st.markdown("El modelo más básico, asume que el crecimiento solo está limitado por un único sustrato ($S$).")
        st.latex(r"""
        \mu = \mu_{max} \frac{S}{K_S + S}
        """)
        st.markdown(r"""
            * $\mu_{max}$: Máxima velocidad específica de crecimiento ($h^{-1}$).
            * $K_S$: Constante de afinidad por el sustrato (concentración de $S$ a la que $\mu = \mu_{max}/2$, unidades $g/L$).
            """)

    with st.expander("2. Monod Sigmoidal (Hill)"):
        st.markdown("Introduce una respuesta de tipo umbral o cooperativa al sustrato, útil para modelar fenómenos de inducción o cinéticas más complejas.")
        st.latex(r"""
        \mu = \mu_{max} \frac{S^n}{K_S^n + S^n}
        """)
        st.markdown(r"""
            * $n$: Coeficiente de Hill (adimensional), ajusta la forma sigmoidal. $n > 1$ indica cooperatividad.
            """)

    with st.expander("3. Monod con Restricciones (S, O2, P)"):
        st.markdown("Modela el crecimiento limitado simultáneamente por sustrato y oxígeno, e inhibido por producto.")
        st.latex(r"""
        \mu = \mu_{max} \left(\frac{S}{K_S + S}\right) \left(\frac{O_2}{K_O + O_2}\right) \left(\frac{K_P}{K_P + P}\right)
        """)
        st.markdown(r"""
            * $K_O$: Constante de afinidad por el oxígeno ($g/L$).
            * $K_P$: Constante de inhibición por producto ($g/L$). *Nota: Esta forma es una inhibición no competitiva simple.*
            """)

    with st.expander("4. Fermentación (Mixta Aerobia/Anaerobia)"):
        st.markdown(r"""
            Modela la coexistencia de rutas metabólicas. La tasa total es la suma de una componente aerobia y una anaerobia/fermentativa, moduladas por $S$, $P$ y $O_2$.
            $\mu_{total} = \mu_{aerobia} + \mu_{anaerobia}$
            """)
        st.markdown("**Componente Aeróbica:**")
        st.latex(r"""
        \mu_{aerobia} = \mu_{max, aerob} \left( \frac{S}{K_{S, aerob} + S} \right) \left( \frac{O_2}{K_{O, aerob} + O_2} \right)
        """)
        st.markdown("**Componente Anaerobia/Fermentativa:**")
        # Usamos la forma implementada en el código más reciente
        st.latex(r"""
        \mu_{anaerobia} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( 1 - \frac{P}{K_{P, anaerob}} \right)^{n_p} \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
        """)
        # Forma alternativa con KP^n (menos usada en el código final pero común):
        # st.latex(r"""
        # \mu_{anaerobia} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( \frac{K_{P, anaerob}^{n_p}}{K_{P, anaerob}^{n_p} + P^{n_p}} \right) \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
        # """)
        st.markdown(r"""
            **Parámetros Específicos:**
            * $\mu_{max, aerob}, \mu_{max, anaerob}$: Tasas máximas de crecimiento ($h^{-1}$).
            * $K_{S, aerob}, K_{S, anaerob}$: Constantes de afinidad por sustrato ($g/L$).
            * $K_{O, aerob}$: Constante de afinidad por $O_2$ para $\mu_{aerobia}$ ($g/L$).
            * $K_{iS, anaerob}$: Constante de inhibición por sustrato para $\mu_{anaerobia}$ ($g/L$).
            * $K_{P, anaerob}$: Constante de inhibición por producto (etanol) para $\mu_{anaerobia}$ ($g/L$). Representa la concentración crítica de $P$ que detiene el crecimiento anaerobio.
            * $n_p$: Exponente de inhibición por producto (adimensional).
            * $K_{O, inhib}$: Constante de inhibición por $O_2$ para $\mu_{anaerobia}$ (efecto Pasteur sobre crecimiento, $g/L$).
            """)

    with st.expander("5. Fermentación Conmutada"):
        st.markdown(r"""
            Simula un cambio metabólico discreto. Utiliza **solo** $\mu_{aerobia}$ (ecuación vista arriba) durante la Fase 1 (aerobia) y **solo** $\mu_{anaerobia}$ (ecuación vista arriba) durante las Fases 2 y 3 (anaerobias). Requiere definir los parámetros para ambas componentes, pero solo una está activa en cada fase.
            """)

    st.subheader("Tasa Específica de Formación de Producto ($q_P$)")
    st.markdown(r"""
        La tasa específica de formación de producto ($q_P$, unidad $g_P \cdot g_X^{-1} \cdot h^{-1}$), específicamente para el etanol en este contexto, se modela usando la ecuación de Luedeking-Piret modificada para incluir inhibición directa por oxígeno. Esto refleja que la producción de etanol es predominantemente anaeróbica.
        """)
    st.latex(r"""
<<<<<<< HEAD
    q_P = (\alpha \cdot \mu + \beta) \left( \frac{K_{O,P}}{K_{O,P} + O_2} \right)
    """)
    st.markdown(r"""
        * $\mu$: Es la tasa específica de crecimiento calculada por el modelo cinético seleccionado ($\mu_{total}$ si es Mixta/Conmutada).
        * $\alpha$: Coeficiente de formación de producto asociado al crecimiento ($g_P \cdot g_X^{-1}$).
        * $\beta$: Coeficiente de formación de producto no asociado al crecimiento ($g_P \cdot g_X^{-1} \cdot h^{-1}$).
        * $K_{O,P}$: Constante de inhibición por oxígeno sobre la *producción* de etanol ($g/L$). Un valor bajo indica fuerte supresión de la producción de $P$ por $O_2$.
=======
    q_P = (\alpha \cdot \mu_{anaerob} + \beta) 
    """)
    st.markdown(r"""
        * $\mu_{anaerob}$: Es la tasa específica de crecimiento calculada por el modelo cinético seleccionado ($\mu_{anaerob}$ si es Mixta/Conmutada).
        * $\alpha$: Coeficiente de formación de producto asociado al crecimiento ($g_P \cdot g_X^{-1}$).
        * $\beta$: Coeficiente de formación de producto no asociado al crecimiento ($g_P \cdot g_X^{-1} \cdot h^{-1}$).
>>>>>>> f74154cd1f51673d9828c4c543de3f3857db625a
        """)

    st.markdown("---") # Separador visual

    # ========= SECCIÓN MOVIDA: BALANCES GENERALES =========
    st.header("📊 Fundamento Teórico: Balances de Materia") # Título ajustado
    st.markdown("""
        El modelado de bioprocesos permite describir matemáticamente la evolución de las variables de interés
        (concentración de biomasa, sustrato, producto, oxígeno disuelto, etc.) en un biorreactor.
        A continuación se presentan los balances de materia generales para los tres modos de operación
        principales, asumiendo un mezclado perfecto. Las tasas $\mu$ y $q_P$ corresponden a los modelos cinéticos descritos anteriormente.
        """)

    st.subheader("🔹 Modo Lote (Batch)")
    st.markdown("""
        No hay entrada ni salida de materia una vez iniciado el proceso ($F=0$). El volumen $V$ es constante.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a_1 \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X""")
    st.markdown(r"""*Nota: $OUR = (\frac{\mu}{Y_{XO}} + m_o) X$. El $k_L a$ puede variar según la fase.*""")


    st.subheader("🔹 Modo Lote Alimentado (Fed-Batch)")
    st.markdown(r"""
        Se agrega alimentación ($F$) con concentración $S_{in}$. El volumen $V$ varía: $\frac{dV}{dt} = F$.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X - \frac{F}{V} \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X + \frac{F}{V} (S_{in} - S)""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X - \frac{F}{V} \cdot P""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X - \frac{F}{V} \cdot O_2""")


    st.subheader("🔹 Modo Continuo (Quimiostato)")
    st.markdown(r"""
        Entrada y salida de flujo $F$ a la misma tasa. Volumen $V$ constante. Tasa de dilución $D = F/V$.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X - D \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X + D (S_{in} - S)""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X - D \cdot P""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X - D \cdot O_2""")

    st.markdown(r"""
        **Parámetros Generales en Balances:**
        * $X, S, P, O_2, V, F, S_{in}$: Variables ya definidas.
        * $\mu, q_P$: Tasas específicas (definidas por los modelos cinéticos).
        * $k_d$: Tasa específica de decaimiento celular ($h^{-1}$).
        * $Y_{XS}$: Rendimiento biomasa/sustrato ($g_X / g_S$).
        * $Y_{PS}$: Rendimiento producto/sustrato ($g_P / g_S$), usado para calcular consumo de $S$ para $P$.
        * $Y_{XO}$: Rendimiento biomasa/oxígeno ($g_X / g_{O2}$).
        * $m_s$: Mantenimiento basado en sustrato ($g_S \cdot g_X^{-1} \cdot h^{-1}$).
        * $m_o$: Mantenimiento basado en oxígeno ($g_{O2} \cdot g_X^{-1} \cdot h^{-1}$).
        * $k_L a$: Coeficiente de transferencia de oxígeno ($h^{-1}$, puede ser $k_{L}a_1$ o $k_{L}a_2$).
        * $C_{S}^*$: Concentración de saturación de $O_2$ ($g/L$).
        * $D$: Tasa de dilución ($h^{-1}$).
        * $t$: Tiempo ($h$).
        """)

    st.markdown("---") # Separador antes de las técnicas avanzadas

    # ========= SECCIÓN MOVIDA: TÉCNICAS AVANZADAS =========
    st.header("⚙️ Técnicas Avanzadas de Análisis y Control")

    # (Las subsecciones de Sensibilidad, Ajuste, EKF, RTO, NMPC no se modifican)
    st.subheader("🔹 Análisis de Sensibilidad")
    st.markdown(r"""
        Evalúa cómo la incertidumbre o variaciones en los parámetros del modelo ($\theta$, como $\mu_{max}, K_S, Y_{XS}$, etc.) afectan las salidas del modelo (las variables de estado $X, S, P, O_2$).
        Permite identificar los parámetros más influyentes, crucial para la optimización y el diseño experimental.
        Una métrica común es el coeficiente de sensibilidad normalizado:
        $S_{ij} = \frac{\partial y_i / y_i}{\partial \theta_j / \theta_j} = \frac{\partial \ln y_i}{\partial \ln \theta_j}$
        donde $y_i$ es una salida y $\theta_j$ es un parámetro.
        """)
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
        * $x_k, u_k, z_k$: Estado, entrada y medición en $k$.
        * $w_k, v_k$: Ruido de proceso ($Q$) y medición ($R$).
        * $f, h$: Funciones no lineales.
        **Etapas del EKF:**
        1.  **Predicción:** $\hat{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k)$, $P_{k+1|k} = F_k P_{k|k} F_k^T + Q$.
        2.  **Actualización:** $K_{k+1} = P_{k+1|k} H_{k+1}^T (H_{k+1} P_{k+1|k} H_{k+1}^T + R)^{-1}$, $\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} + K_{k+1} (z_{k+1} - h(\hat{x}_{k+1|k}))$, $P_{k+1|k+1} = (I - K_{k+1} H_{k+1}) P_{k+1|k}$.
        * $F_k, H_{k+1}$: Jacobianas de $f$ y $h$.
        """)
    st.markdown("---")
    st.subheader("🔹 Control RTO (Real-Time Optimization)")
    st.markdown(r"""
        Estrategia que optimiza una función objetivo económica ajustando setpoints o variables manipuladas, basándose en un modelo (a menudo estacionario) y mediciones. Opera a escala de tiempo lenta.
        **Problema:** $\max_{u_{opt}} \Phi(x_{ss}, u_{opt}, p)$ sujeto a $f(x_{ss}, u_{opt}, p) = 0$, $g(x_{ss}, u_{opt}, p) \le 0$, $u_{min} \le u_{opt} \le u_{max}$.
        """)
    st.markdown("---")
    st.subheader("🔹 Control NMPC (Nonlinear Model Predictive Control)")
    st.markdown(r"""
        Utiliza un modelo dinámico no lineal para predecir el futuro ($N_p$) y calcular acciones de control óptimas ($\Delta U$ sobre $N_c$) minimizando una función objetivo $J$ sujeta a restricciones. Aplica solo la primera acción y repite.
        **Problema:** $\min_{\Delta U_k} J = \sum_{j=1}^{N_p} ||\hat{y}_{k+j|k} - y_{sp, k+j}||^2_Q + \sum_{j=0}^{N_c-1} ||\Delta u_{k+j|k}||^2_R$ sujeto al modelo dinámico y restricciones en $u, \Delta u, y$.
        """)

# Para poder ejecutar esta página individualmente si es necesario
if __name__ == "__main__":
    import os
    # (Código para crear imágenes dummy sin cambios)
    if not os.path.exists("images"): os.makedirs("images")
    dummy_files = ["images/Batch.png", "images/fed_batch.png", "images/continous.png"]
    for f_path in dummy_files:
        if not os.path.exists(f_path):
            try:
                with open(f_path, 'w') as fp: pass
                print(f"Archivo dummy creado: {f_path}")
            except Exception as e:
                print(f"No se pudo crear archivo dummy {f_path}: {e}")
                try:
                    from PIL import Image
                    img = Image.new('RGB', (60, 30), color = 'red'); img.save(f_path)
                    print(f"Imagen placeholder creada: {f_path}")
                except ImportError: print("PIL no encontrado, no se puede crear imagen.")
                except Exception as e_img: print(f"Error creando imagen {f_path}: {e_img}")
    home_page()