# Reporte de Revisión Técnica — Biocontrol Modeling Project

**Fecha:** 2026-07-31  
**Revisado por:** Agente de análisis de ingeniería  
**Idioma del código:** inglés/español (mixto)

---

## Resumen Ejecutivo

El proyecto es una aplicación Streamlit para modelado, simulación y control de bioprocesos.
Cubre modelos cinéticos, simulación de biorreactores (lote, fed-batch, continuo, fermentación
alcohólica), estimación de parámetros, estimación de estados (EKF, ANN), y un completo
sistema de control (regulatorio y avanzado: PID, NMPC, LMPC, RTO, fuzzy).

Se ejecutaron **53 tests unitarios y de ejecución** (100 % aprobados tras las correcciones
documentadas abajo). A continuación se detallan los problemas encontrados por categoría.

---

## 1. Bugs Críticos (afectan resultados de simulación)

### 1.1 `Body/modeling/lote.py` — Rama "Sigmoidal Monod" con `mu` indefinida

**Severidad:** 🔴 Alta  
**Descripción:** En el callback `modelo_lote`, la rama `elif tipo_mu == "Monod sigmoidal":`
solo ejecutaba `if S<=0: S=0` sin calcular `mu`. Si el usuario seleccionaba esta opción
(que además tiene nombre distinto al del selectbox "Sigmoidal Monod"), la variable `mu`
quedaba indefinida y se usaba en las ecuaciones siguientes, causando un `NameError` o
resultados erróneos silenciosos.  
**Corrección aplicada:** Se eliminó la rama muerta ("Monod sigmoidal") y se unificó el
cálculo de `mu` para "Sigmoidal Monod" garantizando que siempre se compute antes de las ODEs.

### 1.2 `Body/modeling/lote.py` y `Body/modeling/continuo.py` — KO y KP hardcodeados

**Severidad:** 🔴 Alta  
**Descripción:** En ambos archivos, el modelo "Monod with restrictions" usaba
`KO=0.5, KP=0.5` como constantes fijas, ignorando cualquier ajuste del usuario.  
**Corrección aplicada:** Se añadieron sliders en la barra lateral para KO y KP cuando
se selecciona "Monod with restrictions".

### 1.3 `Body/modeling/lote_alimentado.py` — Redefinición local de cinéticas con fórmula incorrecta

**Severidad:** 🔴 Alta  
**Descripción:** El archivo redefine localmente `mu_monod`, `mu_sigmoidal` y `mu_completa`
en lugar de importarlas desde `Utils.kinetics`. La versión local de `mu_completa` usaba la
fórmula incorrecta `(1 - P/KP)` para la inhibición por producto, en lugar de la correcta
`KP / (KP + P)` (inhibición no competitiva), lo que produce valores **negativos** cuando
`P > KP`, llevando a `mu < 0` (bloqueado a 0 por `max`) y pérdida silenciosa de inhibición.  
**Corrección aplicada:** Se eliminaron las redefiniciones locales y se importa desde
`Utils.kinetics`.

### 1.4 `Body/modeling/lote_alimentado.py` — Fórmula LaTeX incorrecta mostrada al usuario

**Severidad:** 🟡 Media  
**Descripción:** La ecuación mostrada era `\mu = ... \cdot (1-\frac{K_P}{P})` que es
matemáticamente incorrecta y confunde al usuario. La fórmula correcta es
`\frac{K_P}{K_P + P}` (inhibición no competitiva tipo Monod).  
**Corrección aplicada:** Se actualizó la expresión LaTeX.

### 1.5 `main.py` — `os.chdir` duplicado

**Severidad:** 🟢 Baja  
**Descripción:** `os.chdir(script_dir)` se llamaba dos veces seguidas (líneas 11 y 20),
la segunda llamada es redundante y confunde al lector.  
**Corrección aplicada:** Se eliminó la segunda llamada duplicada junto con el comentario
duplicado.

---

## 2. Problemas de Ingeniería de Modelado y Control

### 2.1 Modelo Batch — Validación de estado de solución no implementada

**Descripción:** `lote.py` y `continuo.py` no verifican `sol.success` después de `solve_ivp`.
`lote_alimentado.py` sí lo hace correctamente.  
**Recomendación:** Agregar `if not sol.success: st.error(...)` en todos los modelos.

### 2.2 Modelo Continuo — No calcula ni muestra el estado estacionario analítico

**Descripción:** Para el quimiostato, el estado estacionario de Monod tiene solución
analítica conocida: `S* = Ks·D / (μmax - D)`, `X* = Yxs·(Sin - S*)`. La app no muestra
estos valores de referencia para validar la simulación.  
**Recomendación:** Añadir un expander con el estado estacionario analítico.

### 2.3 Ajuste de Parámetros Batch — Modelo ODE simplificado ignora O2

**Descripción:** `ajuste_parametros_lote.py` usa `dO2dt = 0` (simplificación explícita).
Si los datos experimentales incluyen mediciones de O2 o si las reacciones son aeróbicas,
el ajuste de parámetros queda desacoplado del oxígeno, subestimando el error de ajuste.  
**Recomendación:** Incorporar la ecuación de O2 completa en el modelo de ajuste.

### 2.4 EKF — Jacobiano calculado por diferencias finitas

**Descripción:** El EKF calcula el Jacobiano numéricamente (diferencias finitas).
Para sistemas con cinética no lineal y rangos muy dispares de parámetros (e.g., μmax vs Yxs),
las diferencias finitas con δ=1e-6 fijo pueden ser imprecisas.  
**Recomendación (avanzado):** Usar diferenciación automática (CasADi) para el Jacobiano
analítico, o al menos escalar δ proporcionalmente al valor del parámetro.

### 2.5 NMPC/RTO — No se verifica la factibilidad del problema NLP

**Descripción:** Los módulos NMPC y RTO llaman al solver IPOPT pero no documentan ni
manejan el caso de infactibilidad más allá de un `try/except` genérico. Un problema
infactible (por restricciones conflictivas) devuelve `nan` en el perfil de control.  
**Recomendación:** Verificar `opti.return_status()` y mostrar un mensaje de advertencia
explícito cuando el solver no converge.

### 2.6 Análisis de Sensibilidad — Solo cubre modelo Batch

**Descripción:** El módulo `analysis.py` está limitado al modo batch con Monod simple.
Los demás modos de operación (fed-batch, continuo) y cinéticas no son analizables.  
**Recomendación:** Extender el análisis de sensibilidad a fed-batch y continuo.

---

## 3. Calidad de Código

| Archivo | Problema | Corrección |
|---------|----------|------------|
| `main.py` | `os.chdir` duplicado | Eliminado |
| `lote_alimentado.py` | Redefine 3 funciones cinéticas localmente | Importar de `Utils.kinetics` |
| `ferm_alcohol.py` | Redefine 4 funciones cinéticas localmente | Importar de `Utils.kinetics` |
| `lote.py` | Rama muerta `"Monod sigmoidal"` | Eliminada |
| `lote.py`, `continuo.py` | KO/KP hardcodeados en "Monod with restrictions" | Sliders añadidos |

---

## 4. Tests Implementados

Se crearon los siguientes archivos de test:

### `tests/test_kinetics.py` — 28 tests unitarios
Cubre todas las funciones cinéticas:
- `mu_monod`: valor puntual, saturación, límites
- `mu_sigmoidal`: reducción a Monod con n=1, efecto cooperativo, límites
- `mu_completa`: fórmula de tres términos, inhibición por producto
- `aiba`: modelo Haldane, existencia de óptimo
- `mu_fermentacion`: efecto Pasteur, inhibición por etanol, signo
- Versiones CasADi: consistencia con versiones Python (tolerancia 1e-6)

### `tests/test_models.py` — 25 tests de integración y sensibilidad
Cubre comportamiento físico de los modelos ODE:
- **Batch**: convergencia del solucionador, biomasa crece, sustrato decrece, O2 acotado, balance de masa, efecto de decaimiento
- **Fed-batch**: volumen aumenta, masa positiva, sin alimentación = batch, dilución con Sin bajo
- **Continuo**: estado estacionario, lavado a D > μmax, biomasa positiva a D < μmax
- **Fermentación**: producción de etanol, consumo de sustrato, inhibición anaeróbica
- **Sensibilidad**: mayor μmax → mayor crecimiento inicial, mayor Ks → menor afinidad, mayor Yxs → más biomasa

---

## 5. Evaluación Conceptual de Ingeniería

### Modelado ✅ / ⚠️
- Las cinéticas implementadas (Monod, Hill, Haldane, mixta aeróbica/anaeróbica) son
  correctas y están bien documentadas con referencias bibliográficas.
- El modelo de fermentación alcohólica con efecto Pasteur es apropiado para
  *Saccharomyces cerevisiae*.
- **Punto a mejorar:** La ecuación diferencial de O2 en el batch usa un modelo de
  transferencia de masa de primer orden (kLa·(Cs-O2)) que es el estándar de la industria.
  Sin embargo, no modela el efecto de la agitación sobre kLa, que es relevante en
  operación real.

### Estimación de Parámetros ✅
- Se usan métodos clásicos (L-BFGS-B, Nelder-Mead, evolución diferencial) de `scipy.optimize`.
- Se incluye análisis estadístico del ajuste (R², RMSE, intervalos de confianza).
- **Punto a mejorar:** El Jacobiano numérico en el cálculo de intervalos de confianza
  puede ser inestable para parámetros muy cercanos a sus límites.

### Estimación de Estados (EKF, ANN) ✅
- El EKF está implementado correctamente con matrices Q y R configurables.
- La ANN como soft-sensor es un enfoque moderno y válido para estimación de estados
  no medibles.
- **Punto a mejorar:** El EKF no implementa corrección de covarianza por cuadrados UDU
  (numericamente más estable para sistemas con rangos de estado muy dispares).

### Control Regulatorio ✅
- PID con split-range para pH es técnicamente correcto.
- Control en cascada para oxígeno es apropiado para biorreactores industriales.
- **Punto a mejorar:** Los modelos de proceso usan funciones de transferencia de primer
  orden (FOPDT), lo que puede ser impreciso para procesos biológicos de alta no linealidad.

### Control Avanzado ✅
- NMPC y LMPC están implementados con CasADi/IPOPT, que es el estándar industrial.
- RTO para optimización del perfil de alimentación es un enfoque correcto.
- El control fuzzy con lógica Mamdani es válido para sistemas con incertidumbre.
- **Punto a mejorar:** El NMPC no implementa estabilidad terminal (conjunto terminal
  invariante o costo terminal), lo que puede reducir robustez con horizontes cortos.

---

## 6. Resumen de Correcciones Implementadas

| # | Archivo | Tipo | Descripción |
|---|---------|------|-------------|
| 1 | `tests/test_kinetics.py` | Nuevo | 28 tests unitarios de funciones cinéticas |
| 2 | `tests/test_models.py` | Nuevo | 25 tests de integración de modelos ODE |
| 3 | `main.py` | Fix | Eliminar `os.chdir` duplicado |
| 4 | `Body/modeling/lote.py` | Fix | Corregir rama "Sigmoidal Monod" y añadir sliders KO/KP |
| 5 | `Body/modeling/lote_alimentado.py` | Fix | Importar cinéticas desde Utils, corregir LaTeX |
| 6 | `Body/modeling/continuo.py` | Fix | Añadir sliders KO/KP para "Monod with restrictions" |
