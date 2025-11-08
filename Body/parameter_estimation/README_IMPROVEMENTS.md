# Parameter Estimation Improvements

## Overview

The `ajuste_parametros_ferm.py` module has been enhanced with robust optimization techniques to improve parameter estimation reliability and accuracy for fermentation models.

## New Features

### 1. Multi-Stage Optimization

The new `multi_stage_optimization()` function provides several optimization strategies:

- **Hybrid Method (Recommended)**: Combines global search with differential evolution followed by local refinement with L-BFGS-B
- **Basinhopping**: Uses basin-hopping algorithm to escape local minima
- **Differential Evolution**: Pure global optimization approach
- **Single-Stage Methods**: Traditional L-BFGS-B or Nelder-Mead

**Benefits:**
- More robust convergence
- Better handling of complex objective function landscapes
- Reduced sensitivity to initial parameter guess

### 2. Parameter Scaling

Parameters are automatically scaled to [0, 1] range during optimization:

```python
scaled = scale_parameters(params, bounds)
unscaled = unscale_parameters(scaled_params, bounds)
```

**Benefits:**
- Better convergence when parameters have different magnitudes
- More stable optimization algorithms
- Improved numerical conditioning

### 3. Outlier Detection

Automatic detection and exclusion of outliers using Z-score method (threshold = 3.0):

```python
outliers = detect_outliers_zscore(data, threshold=3.0)
```

**Benefits:**
- Reduces impact of measurement errors
- Improves fit quality
- More robust parameter estimates

### 4. Parameter Bounds Validation

Validates initial guesses and parameter bounds before optimization:

```python
warnings = validate_parameter_bounds(names, initial_guess, bounds)
```

**Checks:**
- Initial guess within bounds
- Bounds not too tight (< 1e-6)
- Bounds not excessively wide (> 1000x ratio)

### 5. Parameter Identifiability Analysis

Assesses whether parameters can be uniquely determined from the data:

```python
diagnostics = assess_parameter_identifiability(jacobian, param_names)
```

**Metrics:**
- **Condition Number**: Measures numerical stability of parameter estimation
  - < 1e6: Well-conditioned ✅
  - 1e6 - 1e10: Moderately conditioned ⚠️
  - > 1e10: Ill-conditioned ❌

- **Low Sensitivity Parameters**: Parameters with minimal impact on model output
- **High Correlation**: Parameter pairs with correlation > 0.95

**Recommendations:**
- Fix parameters with low sensitivity
- Consider removing one parameter from highly correlated pairs
- Collect more diverse experimental data for ill-conditioned problems

### 6. Random Restarts

Performs multiple optimizations with perturbed initial guesses:

**Benefits:**
- Increased confidence in global optimum
- Better exploration of parameter space
- More robust results

### 7. Enhanced Convergence Diagnostics

Provides detailed optimization feedback:
- Success status
- Number of iterations
- Convergence message
- Parameters near bounds
- Final objective value

## Usage Guide

### Basic Usage

1. **Load experimental data** (.xlsx file with columns: time, biomass, substrate, product, oxygen)

2. **Configure fixed parameters** (operational conditions)

3. **Select optimization method**:
   - `hybrid`: Best for most cases (global + local)
   - `differential_evolution`: When you need extensive global search
   - `basinhopping`: For highly multi-modal problems
   - `L-BFGS-B`: Fast local optimization (if good initial guess)

4. **Enable recommended options**:
   - ✅ Use Parameter Scaling
   - ✅ Detect and Exclude Outliers
   - Random Restarts: 1-3 (for additional robustness)

5. **Run optimization**

### Interpreting Results

#### Optimization Diagnostics
- Check convergence status (should be ✅)
- Review parameters near bounds (may need wider bounds)
- Verify reasonable number of iterations

#### Parameter Identifiability
- **Well-conditioned**: Parameters are identifiable ✅
- **Low sensitivity params**: Consider fixing these values
- **High correlation**: May indicate redundant parameters

#### Statistical Analysis
- **R² values**: Should be > 0.90 for good fit
- **RMSE values**: Lower is better, check units
- **Confidence intervals**: Narrow intervals indicate precise estimates
- **Residuals**: Should be randomly distributed around zero

## Best Practices

### 1. Data Quality
- Minimum 8-10 time points
- Data should span different process phases
- Remove obvious measurement errors manually

### 2. Initial Guess
- Use physically reasonable values
- Check literature for typical parameter ranges
- Run sensitivity analysis if uncertain

### 3. Bounds Selection
- Set realistic bounds based on biology/physics
- Don't make bounds too tight (restricts search)
- Don't make bounds too wide (slows convergence)

### 4. Model Selection
- Start with simpler models (e.g., Simple Monod)
- Add complexity only if needed
- More parameters require more data

### 5. Troubleshooting

**Problem**: Optimization doesn't converge
- Solution: Try `hybrid` method, increase max iterations, adjust bounds

**Problem**: Parameters hit bounds
- Solution: Expand bounds if physically reasonable, or fix parameter

**Problem**: Large confidence intervals
- Solution: More/better data, simpler model, fix some parameters

**Problem**: High parameter correlation
- Solution: Fix one of correlated parameters, redesign experiments

**Problem**: Poor fit despite convergence
- Solution: Wrong model structure, consider different kinetic model

## Technical Details

### Objective Function

The objective function minimizes weighted, scaled sum of squared errors:

```
SSE = Σ w_i * [(y_pred_i - y_exp_i) / scale_i]²
```

Where:
- `w_i`: Weight for variable i (biomass, substrate, product, oxygen)
- `scale_i`: Maximum absolute value of experimental data for variable i

**Benefits of scaling:**
- Equal importance to all variables despite different magnitudes
- Numerical stability
- Better optimization convergence

### Convergence Criteria

- **Relative tolerance**: 1e-6 (default)
- **Absolute tolerance**: 1e-9 (default)
- **Maximum iterations**: 500 (default)
- **Gradient tolerance**: 1e-7 (L-BFGS-B)

### Computational Complexity

- **L-BFGS-B**: O(n_iter × n_eval) - Fast
- **Nelder-Mead**: O(n_iter × n_eval) - Moderate
- **Differential Evolution**: O(popsize × n_iter × n_eval) - Slow but robust
- **Hybrid**: O(DE_iter + LBFGS_iter) - Balanced

Typical run times (4-5 parameters):
- L-BFGS-B: 10-30 seconds
- Differential Evolution: 1-3 minutes
- Hybrid: 1-2 minutes

## References

1. **Optimization Algorithms**:
   - Nocedal & Wright (2006). "Numerical Optimization"
   - Storn & Price (1997). "Differential Evolution"

2. **Parameter Estimation**:
   - Bard (1974). "Nonlinear Parameter Estimation"
   - Beck & Arnold (1977). "Parameter Estimation in Engineering"

3. **Identifiability**:
   - Brun et al. (2001). "Practical identifiability analysis of large environmental simulation models"
   - Walter & Pronzato (1997). "Identification of Parametric Models from Experimental Data"

## Changelog

### Version 2.0 (Current)
- Added multi-stage optimization
- Implemented parameter scaling
- Added outlier detection
- Enhanced identifiability analysis
- Improved convergence diagnostics

### Version 1.0 (Original)
- Basic optimization with L-BFGS-B and Nelder-Mead
- Simple objective function
- Basic confidence intervals
