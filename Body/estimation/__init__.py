"""
State estimation module.

Implements online state estimation algorithms for bioprocesses where not all
state variables can be measured directly (e.g., biomass concentration, substrate).

**Extended Kalman Filter (EKF):**
Recursive state estimation algorithm for nonlinear dynamic systems with noise.
Linearizes the system around current estimates using Jacobian matrices.
Essential for real-time monitoring and control of bioprocesses.

**Artificial Neural Networks (ANN):**
Data-driven state estimation using neural networks trained on historical data.
Useful when mechanistic models are unavailable or incomplete.

Applications:
- Real-time biomass concentration estimation
- Soft-sensor development for unmeasured variables
- Simultaneous state and parameter estimation
- Model validation and fault detection

Available modules:
- ekf: Extended Kalman Filter implementation using CasADi
- ann: Artificial Neural Network implementation using TensorFlow

References:
- Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory. Academic Press.
- Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches. John Wiley & Sons.
- Bastin, G., & Dochain, D. (1990). On-line Estimation and Adaptive Control of Bioreactors. Elsevier.
"""
