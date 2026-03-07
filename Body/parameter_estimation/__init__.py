"""
Parameter estimation module for fitting kinetic model parameters.

Implements nonlinear parameter estimation algorithms to determine kinetic parameters
(μmax, Ks, yields, etc.) from experimental data. Uses optimization algorithms to
minimize the difference between model predictions and experimental measurements.

Optimization Problem:
    min J(θ) = Σ[(y_exp - y_model(θ))²]
    
where θ are the parameters to be estimated.

Methods:
- Nonlinear least squares optimization
- Trust region methods (Levenberg-Marquardt algorithm)
- Sequential Quadratic Programming (SQP)
- Statistical analysis of results (R², RMSE, confidence intervals)

Statistical Metrics:
- Coefficient of determination (R²): Goodness of fit
- Root Mean Square Error (RMSE): Average prediction error  
- Parameter confidence intervals: Uncertainty quantification
- Correlation matrix: Parameter interdependencies
- Residual analysis: Model adequacy assessment

Available modules:
- ajuste_parametros_lote: Batch parameter fitting
- ajuste_parametros_lote_alim: Fed-batch parameter fitting
- ajuste_parametros_ferm: Fermentation parameter fitting

Each module loads experimental data from Excel files, performs optimization,
and provides statistical analysis of the fitted parameters.

References:
- Bard, Y. (1974). Nonlinear Parameter Estimation. Academic Press.
- Beck, J. V., & Arnold, K. J. (1977). Parameter Estimation in Engineering and Science. John Wiley & Sons.
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
- Seber, G. A. F., & Wild, C. J. (2003). Nonlinear Regression. John Wiley & Sons.
"""
