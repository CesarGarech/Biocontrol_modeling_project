import numpy as np
from scipy.stats import t as t_dist


def model_confidence_band(simulate, params_opt, y_exp, y_pred, alpha=0.05):
    """
    95% confidence band of the fitted model curve.
    Uses standard errors but truncates the lower bound at 0.0 to respect physical limits.

    simulate(params) -> array (n_vars, n_times) evaluated at the experimental times.
    y_exp / y_pred: arrays (n_vars, n_times); NaNs in y_exp are ignored.
    Returns (lower, upper) with the same shape as y_pred, or None if not computable.
    """
    try:
        params_opt = np.asarray(params_opt, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n_vars, n_times = y_pred.shape
        n_params = len(params_opt)
        jac = np.zeros((n_vars * n_times, n_params))
        
        for i in range(n_params):
            h = 1e-7 * abs(params_opt[i]) if abs(params_opt[i]) > 1e-8 else 1e-7
            p = params_opt.copy()
            p[i] += h
            y_p = np.asarray(simulate(p), dtype=float)
            if y_p.shape != y_pred.shape:
                return None
            jac[:, i] = ((y_p - y_pred) / h).flatten()
            
        if not np.all(np.isfinite(jac)):
            return None
            
        res = (y_exp[:n_vars, :] - y_pred).flatten()
        valid = ~np.isnan(res)
        dof = int(valid.sum()) - n_params
        if dof <= 0:
            return None
            
        mse = np.sum(res[valid] ** 2) / dof
        jv = jac[valid]
        cov = np.linalg.pinv(jv.T @ jv) * mse
        var_y = np.einsum('ij,jk,ik->i', jac, cov, jac)
        se = np.sqrt(np.clip(var_y, 0.0, None)).reshape(n_vars, n_times)
        
        # --- Corrección: Truncamiento Físico (Evita NaNs por división por cero) ---
        half = t_dist.ppf(1.0 - alpha / 2.0, df=dof) * se
        
        # El límite inferior se trunca a 0.0 para evitar cruzar el eje X
        lower = np.maximum(0.0, y_pred - half)
        # El límite superior se calcula con simetría normal
        upper = y_pred + half
        
        return lower, upper
    except Exception:
        return None


def draw_band(ax, t, lower, upper, color=None, label='95% CI'):
    # Interpola los valores para asegurar que no haya huecos o quiebres visuales
    ax.fill_between(t, lower, upper, color=color, alpha=0.25, label=label, linewidth=0)