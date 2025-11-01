"""
Modeling module for different bioreactor operation modes.

Implements mass balance equations for bioprocess simulation based on fundamental
conservation principles (Bailey & Ollis, 1986; Shuler & Kargi, 2002).

Reactor Configurations:
- Batch: No inlet/outlet flows, constant volume
- Fed-batch: Substrate feeding with volume increase
- Continuous (chemostat): Constant volume with equal inlet/outlet flows
- Fermentation: Specialized models for alcoholic fermentation with aerobic/anaerobic metabolism

The ODEs are solved numerically using scipy.integrate.solve_ivp with adaptive
Runge-Kutta methods (Press et al., 2007).

Available modules:
- lote: Batch reactor modeling
- lote_alimentado: Fed-batch reactor modeling
- continuo: Continuous reactor (chemostat) modeling
- ferm_alcohol: Alcoholic fermentation modeling

References:
- Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals. McGraw-Hill.
- Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall.
- Press, W. H., et al. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.
"""
