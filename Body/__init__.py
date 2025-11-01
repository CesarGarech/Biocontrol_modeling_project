"""
Body package for bioprocess modeling and control.

This package contains modules for bioprocess simulation, analysis, and control,
implementing theoretical concepts from:

**Bioprocess Engineering:**
- Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals.
- Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts.

**Process Control:**
- Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic Process Control.
- Camacho, E. F., & Bordons, C. (2007). Model Predictive Control.

**State Estimation:**
- Bastin, G., & Dochain, D. (1990). On-line Estimation and Adaptive Control of Bioreactors.

Modules:
- modeling: Different bioreactor operation modes (batch, fed-batch, continuous, fermentation)
- parameter_estimation: Fitting kinetic parameters to experimental data
- estimation: State estimation (EKF and ANN implementations)
- control: Regulatory (PID) and advanced (RTO, NMPC) control strategies
- analysis: Sensitivity analysis
"""

__version__ = "1.0.0"
