"""
Advanced control module.

Implements model-based optimization and predictive control strategies for
bioprocesses using nonlinear programming and optimal control techniques.

**Real-Time Optimization (RTO):**
Economic optimization using steady-state models to find optimal setpoints for
regulatory controllers. Maximizes productivity, product quality, or economic
objectives subject to operational constraints.

**Nonlinear Model Predictive Control (NMPC):**
Advanced control strategy that uses a nonlinear dynamic model to predict future
behavior and computes optimal control actions over a receding horizon. Handles
multivariable interactions, constraints, and nonlinear dynamics explicitly.

**Linear Model Predictive Control (LMPC):**
MPC implementation using linearized models for computational efficiency while
maintaining constraint handling capabilities.

Implementation:
- CasADi framework for symbolic differentiation and optimization (Andersson et al., 2019)
- IPOPT solver for large-scale nonlinear programming
- Multiple shooting methods for dynamic optimization

Available modules:
- rto: Real-Time Optimization for general bioprocesses
- rto_ferm: Real-Time Optimization for fermentation processes
- nmpc: Nonlinear Model Predictive Control
- lmpc: Linear Model Predictive Control

References:
- Camacho, E. F., & Bordons, C. (2007). Model Predictive Control (2nd ed.). Springer-Verlag.
- Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design (2nd ed.). Nob Hill Publishing.
- Biegler, L. T. (2010). Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM.
- Marlin, T. E. (2000). Process Control: Designing Processes and Control Systems for Dynamic Performance (2nd ed.). McGraw-Hill.
- Andersson, J. A. E., et al. (2019). "CasADi: a software framework for nonlinear optimization and optimal control." 
  Mathematical Programming Computation, 11(1), 1-36.
"""
