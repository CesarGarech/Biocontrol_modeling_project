# Troubleshooting Guide: Poor Model Fit (Negative R²)

## Problem
You're getting negative R² values, which means the model predictions are worse than just using the mean of your data.

**Your R² Values:**
- Biomass: -3.58
- Substrate: -48.24
- Product: -2.52
- Oxygen: -36,493.76 (extremely poor)

## Root Cause Analysis

### 1. Model Selection Issue
The **Fermentation model** is complex with 12+ parameters including:
- Aerobic pathway parameters (mumax_aerob, Ks_aerob, KO_aerob)
- Anaerobic pathway parameters (mumax_anaerob, Ks_anaerob, KiS_anaerob)
- Product inhibition parameters (KP_anaerob, n_p)
- Yield coefficients and maintenance terms

**If your data doesn't have both aerobic AND anaerobic phases, this model won't work.**

### 2. Common Mistakes

#### A. Wrong Kinetic Model for Your Data
- **Fermentation model**: Requires data with BOTH aerobic (O2 > 0) and anaerobic (O2 ≈ 0) phases
- **Simple Monod**: For aerobic-only or simple substrate-limited growth
- **Monod with restrictions**: For aerobic growth with oxygen limitation
- **Switched Fermentation**: For sequential batch (aerobic) → fed-batch (anaerobic)

#### B. Fixed Parameters Don't Match Experiment
Common mismatches:
- **Batch end time** doesn't match when feeding started
- **Feeding times** (start/end) don't match your experiment
- **kLa value**: Using example value (120) instead of your actual value
- **Initial volume V0**: Not matching your actual starting volume

#### C. Initial Conditions Mismatch
First row of your data MUST match exactly:
```
time=0: X0, S0, P0, O2_0 from your data
```

## Step-by-Step Solution

### Step 1: Identify Your Process Type

**Check your oxygen profile:**
1. Load your data in Excel
2. Look at the "oxygen" column

**If oxygen is:**
- **Constant > 0** → Use **Simple Monod** (batch) or **Monod with restrictions** (fed-batch)
- **Starts high, goes to ~0** → Use **Monod with restrictions** 
- **Alternates high/low** → Use **Fermentation** model
- **Always ~0** → Don't use Fermentation aerobic parameters

### Step 2: Choose Correct Model

| Your Data | Correct Model | Parameters to Optimize |
|-----------|---------------|------------------------|
| Batch, aerobic (O2 > 0) | Simple Monod | mumax, Ks, Yxs, Yps, Yxo |
| Fed-batch, aerobic | Monod with restrictions | mumax, Ks, Yxs, Yps, Yxo, Kla |
| Batch → Fed-batch | Switched Fermentation | Use considerar_O2 flag |
| Mixed aerobic/anaerobic | Fermentation | All 12 parameters |
| Mostly anaerobic | Simple Monod (fix O2 terms) | mumax, Ks, Yxs, Yps |

### Step 3: Set Fixed Parameters Correctly

**Critical fixed parameters:**

```python
# Time points (MUST match your experiment)
t_batch_inicial_fin = X.X  # When did batch phase END?
t_alim_inicio = X.X        # When did feeding START?
t_alim_fin = X.X           # When did feeding END?

# Operating conditions (MUST match your setup)
V0 = X.X                   # Initial volume [L]
kLa = X.X                  # Oxygen transfer coefficient [1/h]
Cs = X.X                   # Saturation oxygen concentration [mg/L]
Sin = X.X                  # Substrate concentration in feed [g/L]

# Feeding strategy
estrategia = "Constant"    # or "Exponential", "Linear"
F_base = X.X               # Base flow rate [L/h]
```

### Step 4: Verify Initial Conditions

**Initial conditions MUST match row 1 of your data:**

```python
# From your data file, row 1 (time = 0):
X0 = [your biomass at t=0]     # g/L
S0 = [your substrate at t=0]   # g/L  
P0 = [your product at t=0]     # g/L
O2_0 = [your oxygen at t=0]    # mg/L
V0 = [your volume at t=0]      # L
```

### Step 5: Start Simple, Add Complexity

**Recommended workflow:**

1. **First attempt: Simple Monod**
   - Uses only: mumax, Ks, Yxs, Yps, Yxo
   - Assumes aerobic growth
   - If R² > 0.8, you're done!

2. **If Simple Monod fails (R² < 0.5):**
   - Check if you have fed-batch data
   - Try "Monod with restrictions"
   - Verify feeding times are correct

3. **If you have mixed aerobic/anaerobic:**
   - Use "Fermentation" model
   - Fix aerobic parameters if data is mostly anaerobic
   - Fix anaerobic parameters if data is mostly aerobic

## Quick Diagnostic Checklist

### Before Running Optimization:

- [ ] **Data units correct?**
  - Biomass, Substrate, Product: g/L
  - Oxygen: mg/L
  - Time: hours

- [ ] **Initial conditions match data row 1?**
  - X0 = biomass[0]
  - S0 = substrate[0]
  - P0 = product[0]
  - O2_0 = oxygen[0]
  - V0 = initial volume

- [ ] **Fixed parameters match experiment?**
  - Batch end time
  - Feeding start/end times
  - Initial volume
  - kLa value
  - Feed concentration (Sin)

- [ ] **Model matches data characteristics?**
  - If O2 always > 0 → Simple Monod
  - If O2 varies → Fermentation or Switched
  - If fed-batch → Include feeding strategy

- [ ] **Parameter bounds reasonable?**
  - mumax: 0.01 - 2.0 [1/h]
  - Ks: 0.01 - 20.0 [g/L]
  - Yxs: 0.01 - 0.8 [g/g]
  - Yps: 0.1 - 0.6 [g/g]
  - Yxo: 0.1 - 2.0 [g/g]

## Example: Fixing Your Current Issue

Based on your **extremely poor oxygen fit** (R² = -36,494), here's what to do:

### Option 1: Simple Monod (Recommended)
```
1. Model: Select "Simple Monod"
2. Parameters to optimize: mumax, Ks, Yxs, Yps, Yxo
3. Fixed parameters:
   - Set batch/feeding times to match YOUR experiment
   - Set V0 to YOUR initial volume
   - Set kLa to YOUR value (or typical 50-200 for lab scale)
4. Initial conditions: Use values from row 1 of your data
```

### Option 2: If You Need Fermentation Model
```
1. Model: Select "Fermentation"
2. First optimization with FIXED aerobic parameters:
   - Fix: mumax_aerob = 0.4, Ks_aerob = 0.5, KO_aerob = 0.2
   - Optimize: Only anaerobic parameters and yields
3. Check if R² improves
4. If yes, then try optimizing all parameters
```

## Common Error Messages Explained

| Message | Meaning | Solution |
|---------|---------|----------|
| "Optimization succeeded but R² < 0" | Wrong model or settings | Follow this guide |
| "Parameters near bounds" | Hitting constraint | Adjust bounds or fix parameter |
| "Poor identifiability" | Not enough info in data | Fix some parameters |
| "Singular matrix" | Redundant parameters | Simplify model |

## Getting Help

If still having issues, provide:
1. **Your data characteristics:**
   - How many time points?
   - Oxygen profile (constant, variable, zero)?
   - Is it batch or fed-batch?

2. **Your settings:**
   - Which kinetic model selected?
   - Fixed parameter values
   - Initial conditions

3. **First 5 rows of your data file**

## References

- `Body/parameter_estimation/ajuste_parametros_lote.py` - Batch models (Simple Monod)
- `Body/parameter_estimation/ajuste_parametros_lote_alim.py` - Fed-batch models
- `Body/parameter_estimation/ajuste_parametros_ferm.py` - Fermentation models
- `Utils/kinetics.py` - All kinetic model equations
- `README_IMPROVEMENTS.md` - Feature documentation
- `Data/Example_robust_estimation.xlsx` - Working example

---

**Remember**: The optimization algorithm is working correctly. The issue is model/settings don't match your experimental data. Start simple and verify settings before blaming the optimizer!
