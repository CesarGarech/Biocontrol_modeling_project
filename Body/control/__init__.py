"""
Control module for process control strategies.

Implements classical and advanced control techniques for bioprocess applications:

**Regulatory (PID) Control:**
Classical feedback control for temperature, pH, dissolved oxygen, and other
process variables (Smith & Corripio, 2005; Seborg et al., 2016).

**Advanced Control:**
- Real-Time Optimization (RTO): Economic optimization with steady-state models (Marlin, 2000; Biegler, 2010)
- Nonlinear Model Predictive Control (NMPC): Constraint-based dynamic optimization (Camacho & Bordons, 2007; Rawlings et al., 2017)
- Linear Model Predictive Control (LMPC): MPC with linearized models

Sub-packages:
- regulatorio: Regulatory (PID) control implementations
- avanzado: Advanced control implementations (RTO, NMPC, LMPC)

References:
- Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic Process Control (3rd ed.). John Wiley & Sons.
- Seborg, D. E., et al. (2016). Process Dynamics and Control (4th ed.). John Wiley & Sons.
- Camacho, E. F., & Bordons, C. (2007). Model Predictive Control (2nd ed.). Springer-Verlag.
- Rawlings, J. B., et al. (2017). Model Predictive Control: Theory, Computation, and Design (2nd ed.). Nob Hill Publishing.
- Marlin, T. E. (2000). Process Control: Designing Processes and Control Systems for Dynamic Performance (2nd ed.). McGraw-Hill.
- Biegler, L. T. (2010). Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM.
"""
