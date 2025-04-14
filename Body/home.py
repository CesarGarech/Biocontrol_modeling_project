import streamlit as st

def home_page():
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

        Donde:
        - μ: velocidad específica de crecimiento (Monod y variantes)
        - Yxs: rendimiento biomasa/sustrato
        - Ypx: rendimiento producto/biomasa
        - Yxo: rendimiento biomasa/oxígeno
        - Kla: coeficiente de transferencia de oxígeno
        - ms, mo: coeficientes de mantenimiento
        - Kd: tasa de decaimiento celular

        Las variantes de μ (velocidad específica de crecimiento) consideradas en este código son:

        1.  **Monod simple:** μ = μmax * S / (Ks + S)
        2.  **Monod sigmoidal:** μ = μmax * S^n / (Ks^n + S^n)
        3.  **Monod con restricciones:** μ = μmax * S / (Ks + S) * O2 / (KO + O2) * KP / (KP + P)

        ### 🔹 Modo Lote Alimentado
        - Se agrega alimentación al biorreactor sin retirar producto, y el volumen varía en el tiempo.

        dX/dt = μ(S, O2, P) * X - Kd * X - (F/V) * X
        dS/dt = - (1/Yxs) * μ(S, O2, P) * X - ms * X + (F/V) * (Sin - S)
        dP/dt = Ypx * μ(S, O2, P) * X - (F/V) * P
        dO2/dt = Kla * (Cs - O2) - (1/Yxo) * μ(S, O2, P) * X - mo * X - (F/V) * O2

        Donde:
        - F: flujo de alimentación
        - Sin: concentración de sustrato en el alimentado

        ### 🔹 Modo Continuo (chemostato)
        - Hay entrada y salida continua de fluido, el volumen se mantiene constante.

        dX/dt = μ(S, O2, P) * X - Kd * X - D * X
        dS/dt = - (1/Yxs) * μ(S, O2, P) * X - ms * X + D * (Sin - S)
        dP/dt = Ypx * μ(S, O2, P) * X - D * P
        dO2/dt = Kla * (Cs - O2) - (1/Yxo) * μ(S, O2, P) * X - mo * X - D * O2

        Donde:
        - D: tasa de dilución (D = F/V)

        ---

        ### 🔹 Análisis de Sensibilidad
        El análisis de sensibilidad es una técnica utilizada para evaluar cómo la incertidumbre en las entradas de un modelo matemático o sistema afecta su salida. En el contexto de los bioprocesos, permite identificar qué parámetros del modelo tienen el mayor impacto en las variables del proceso, lo que es crucial para la optimización y control del sistema.

        ### 🔹 Ajuste de Parámetros
        El ajuste de parámetros, o estimación de parámetros, es el proceso de encontrar los valores de los parámetros de un modelo que mejor se ajustan a un conjunto de datos experimentales. Esto se logra minimizando una función objetivo que cuantifica la discrepancia entre las predicciones del modelo y las mediciones reales.
        Siendo el vector de parámetros a estimar,
        y la función objetivo a minimizar, el problema general se define como:

        argmin (\\theta) J(\\theta)

        donde J es la función objetivo (e.g., suma de errores cuadrados). Métodos de optimización como el algoritmo de Levenberg-Marquardt o algoritmos genéticos se utilizan comúnmente para este propósito.

        ### 🔹 Filtro de Kalman Extendido (EKF)
        El Filtro de Kalman Extendido (EKF) es una versión del filtro de Kalman que se usa para estimar el estado de sistemas no lineales.
        Dado un sistema no lineal:

        x(k+1) = f(x(k), u(k), w(k))
        z(k) = h(x(k), v(k))

        Donde:
        - x(k): estado del sistema en el instante k.
        - u(k): entrada de control en el instante k.
        - z(k): medición del sistema en el instante k.
        - w(k) y v(k): ruido del proceso y ruido de medición, respectivamente.

        Las ecuaciones del EKF son:

        Predicción del estado: x̂(k+1|k) = f(x̂(k|k), u(k), 0)
        Predicción de la covarianza del error: P(k+1|k) = F(k)P(k|k)F(k)T + Q
        Ganancia de Kalman: K(k+1) = P(k+1|k)H(k+1)T [H(k+1)P(k+1|k)H(k+1)T + R]^-1
        Actualización del estado: x̂(k+1|k+1) = x̂(k+1|k) + K(k+1)[z(k+1) - h(x̂(k+1|k), 0)]
        Actualización de la covarianza del error: P(k+1|k+1) = [I - K(k+1)H(k+1)]P(k+1|k)

        Donde:
        - F(k) y H(k) son las matrices Jacobianas de f y h, respectivamente, evaluadas en las estimaciones anteriores.
        - Q y R son las matrices de covarianza del ruido del proceso y del ruido de medición, respectivamente.

        ### 🔹 Control RTO (Real-Time Optimization)
        El Control de Optimización en Tiempo Real (RTO) es una estrategia de control avanzada que optimiza el desempeño de un proceso en tiempo real. En bioprocesos, se utiliza para ajustar las variables de operación para maximizar la producción o minimizar los costos.
        El problema de optimización se formula como:

        max (u) J(x, u)
        sujeto a:
        ẋ = f(x, u, p)
        g(x, u) ≤ 0

        Donde:
        - J: Función objetivo a maximizar o minimizar.
        - x: Vector de estados del proceso.
        - u: Vector de variables manipuladas.
        - p: Vector de parámetros.
        - ẋ = f(x, u, p): Modelo dinámico del proceso (ecuaciones diferenciales).
        - g(x, u) ≤ 0: Restricciones del proceso.

        Los modelos de colocación son usados para la optimización.

        ### 🔹 Control NMPC (Nonlinear Model Predictive Control)
        El Control Predictivo No Lineal (NMPC) es una técnica de control avanzada que utiliza un modelo dinámico del proceso para predecir su comportamiento futuro y optimizar las acciones de control en un horizonte de tiempo.
        La formulación general del problema NMPC es similar al RTO, pero se resuelve repetidamente en cada paso de tiempo:

        min (U) J(X, U, x(k), SP)
        sujeto a:
        Ẋ = F(X, U, P)
        G(X, U) ≤ 0
        U ∈ U

        Donde:
        - U: Secuencia de controles predichos [u(k), u(k+1), ..., u(k+N-1)].
        - X: Trayectoria de estados predicha.
        - x(k): Estado actual del proceso.
        - SP: Setpoint (valor deseado de la salida).
        - F: Modelo dinámico del proceso.
        - G: Restricciones.
        - U: Conjunto de restricciones en las entradas de control.

        En cada paso de tiempo k, el NMPC resuelve este problema, aplica el primer control u(k), y repite el proceso en el siguiente paso.
        """)

if __name__ == "__main__":
    home_page()