import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import PlotHelpers as phelp
import DataSelection as ds
from math import isclose
from scipy.optimize import minimize
from PlotHelpers import R


def _virial_axis_label_kwargs():
    """Match Basic_data ``mixture_isotherm_per_T_*`` (IsothermFittingPlot) / CC HOA plots."""
    return {'fontsize': phelp.AXIS_LABEL_FONTSIZE, 'fontweight': 'medium'}


# Set by Main.py after Virial degree search; prepended to virial_coefficients.txt when present.
DEGREE_SEARCH_REPORT_TEXT = None


def format_virial_fit_coefficient_report(framework, molecule, deg_a,deg_b, coeffs_a, coeffs_a_stderr, coeffs_b, coeffs_b_stderr, N, r2, SSE, P_mean, P_pred):
    """Human-readable Virial ln(P) fit summary (same content as verbose prints)."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Virial Coefficients (framework={framework}, molecule={molecule})")
    lines.append("=" * 60)
    lines.append(f"Polynomial degrees used for this fit: deg_a={deg_a}, deg_b={deg_b}")
    lines.append("Fitted model: ln(P) = ln(q*) + (1/T)*Σ a_j (q*)^j + Σ b_j (q*)^j")
    lines.append("Optimization method: Non-linear least squares in log-pressure space")
    lines.append("")
    lines.append(f"Parameters a_i (deg_a={deg_a}, {len(coeffs_a)} coefficients):")
    for i, (a, a_err) in enumerate(zip(coeffs_a, coeffs_a_stderr)):
        lines.append(f"  a_{i:<3} {a:15.6e} ± {a_err:15.6e}")
    lines.append("")
    lines.append(f"Parameters b_j (deg_b={deg_b}, {len(coeffs_b)} coefficients):")
    for j, (b, b_err) in enumerate(zip(coeffs_b, coeffs_b_stderr)):
        lines.append(f"  b_{j:<3} {b:15.6e} ± {b_err:15.6e}")
    lines.append("")
    lines.append("Goodness of Fit:")
    lines.append(f"  R²   = 1 - SSE/SS_tot  = {r2:.6f}")
    lines.append(f"  (N = {N} data points, SSE = {SSE:.6e} Pa²)")
    lines.append(f"  Mean P_exp = {P_mean:.2e} Pa, Mean P_pred = {np.mean(P_pred):.2e} Pa")
    lines.append("=" * 60)
    return "\n".join(lines)


# Help functions
def _eval_poly(coeffs, x):
    """Evaluate polynomial sum_{i} coeffs[i]*x^i using Horner's method (np.polyval).
    coeffs is ordered [c0, c1, c2, ...] (lowest degree first), so we reverse for polyval."""
    return np.polyval(np.asarray(coeffs, dtype=float)[::-1], np.asarray(x, dtype=float))

def _analytical_jacobian(q, T, deg_a, deg_b):
    """Analytical Jacobian of virial_model_lnP w.r.t. all parameters.

    Returns matrix J of shape (len(q), deg_a+1 + deg_b+1) where:
      J[:, i]       = d(lnP)/d(a_i) = q^i / T   for i in 0..deg_a
      J[:, deg_a+1+j] = d(lnP)/d(b_j) = q^j       for j in 0..deg_b
    """
    q = np.asarray(q, dtype=float)
    T = np.asarray(T, dtype=float)
    cols_a = np.vstack([(q ** i) / T for i in range(deg_a + 1)]).T
    cols_b = np.vstack([q ** j for j in range(deg_b + 1)]).T
    return np.hstack([cols_a, cols_b])

def _outlier_mask(pts, deg_a, deg_b, residual_threshold=20.0, min_sigma_floor=1e-3):
    """Return a boolean keep-mask for pts using hat-matrix standardized residuals.

    Fits the Virial model once via BFGS, computes approximate hat-matrix diagonal
    from the analytical Jacobian, and flags points with |std_resid| > threshold.
    Falls back to all-True if optimization fails.
    """
    n_all = np.array([float(p['loading'])    for p in pts], dtype=float)
    P_all = np.array([float(p['pressure'])   for p in pts], dtype=float)
    T_all = np.array([float(p['temperature']) for p in pts], dtype=float)
    y_all = np.log(P_all)

    n_pts = len(n_all)
    n_params = (deg_a + 1) + (deg_b + 1)
    if n_pts < n_params:
        return np.ones(n_pts, dtype=bool)

    try:
        p0 = np.zeros(n_params)
        try:
            V = _analytical_jacobian(n_all, T_all, deg_a, deg_b)
            y_lnPn = np.log(P_all / n_all)
            p0, *_ = np.linalg.lstsq(V, y_lnPn, rcond=None)
        except Exception:
            pass

        res = minimize(
            lambda params: virial_sse_objective(params, n_all, T_all, y_all, deg_a, deg_b),
            p0, method='BFGS', options={'maxiter': 10000}
        )
        if not (hasattr(res, 'x') and np.all(np.isfinite(res.x))):
            return np.ones(n_pts, dtype=bool)

        n_a = deg_a + 1
        lnP_pred = virial_model_lnP(n_all, T_all, res.x[:n_a], res.x[n_a:])
        resid = y_all - lnP_pred
        SSE = np.sum(resid ** 2)
        sigma = max(np.sqrt(SSE / max(1, n_pts - n_params)), float(min_sigma_floor))

        jac = _analytical_jacobian(n_all, T_all, deg_a, deg_b)
        try:
            JtJ_inv = np.linalg.pinv(jac.T @ jac)
        except Exception:
            JtJ_inv = np.eye(n_params)
        H_diag = np.clip(np.einsum('ij,jk,ik->i', jac, JtJ_inv, jac), 0.0, 0.999999)

        denom = sigma * np.sqrt(np.maximum(1.0 - H_diag, 1e-12))
        std_resid = resid / denom
        return np.isfinite(std_resid) & (np.abs(std_resid) <= float(residual_threshold))
    except Exception:
        return np.ones(n_pts, dtype=bool)

def virial_model_lnP(q, T, coeffs_a, coeffs_b, c_ln=1.0):
    """
    Compute ln(P) from Virial model per paper equation:
        ln(P) = c_ln * ln(q*) + (1/T) * Σ_{i=0}^{deg_a} a_i (q*)^i + Σ_{j=0}^{deg_b} b_j (q*)^j
    
    Parameters:
    - q: loading (q*) values (scalar or array)
    - T: temperature values (scalar or array, same shape as q)
    - coeffs_a: array of a_i coefficients
    - coeffs_b: array of b_j coefficients
    - c_ln: coefficient for ln(q*) term (default 1.0)
    
    Returns:
    - ln(P) values (same shape as q)
    """
    q = np.asarray(q, dtype=float)
    T = np.asarray(T, dtype=float)
    coeffs_a = np.asarray(coeffs_a, dtype=float)
    coeffs_b = np.asarray(coeffs_b, dtype=float)
    
    # Handle broadcasting
    if q.ndim == 0:
        q = np.array([q])
    if T.ndim == 0:
        T = np.array([T])
    if len(q) == 1 and len(T) > 1:
        q = np.broadcast_to(q, T.shape)
    elif len(T) == 1 and len(q) > 1:
        T = np.broadcast_to(T, q.shape)

    sum_a = _eval_poly(coeffs_a, q) / T
    sum_b = _eval_poly(coeffs_b, q)

    q_safe = np.where(q > 0, q, np.nan)
    lnP = float(c_ln) * np.log(q_safe) + sum_a + sum_b
    return lnP

def virial_sse_objective(params, q_arr, T_arr, lnP_obs, deg_a, deg_b, c_ln=1.0):
    """
    Objective function for non-linear optimization: minimizes SSE in log-pressure space.
    
    SSE = Σ (ln(P_i,exp) - ln(P_i,pred))^2
    
    Parameters:
    - params: flattened array [a_0, a_1, ..., a_{deg_a}, b_0, b_1, ..., b_{deg_b}]
    - q_arr: observed loading values
    - T_arr: observed temperature values
    - lnP_obs: observed ln(P) values
    - deg_a, deg_b: polynomial degrees
    - c_ln: coefficient for ln(q*) term (default 1.0)
    
    Returns:
    - SSE value (sum of squared errors in log-pressure space)
    """
    n_a = deg_a + 1
    coeffs_a = params[:n_a]
    coeffs_b = params[n_a:]
    
    lnP_pred = virial_model_lnP(q_arr, T_arr, coeffs_a, coeffs_b, c_ln)
    
    # Check for invalid predictions
    if not np.all(np.isfinite(lnP_pred)):
        # Return a large penalty value if prediction is invalid
        return 1e20
    
    # Compute SSE in log-pressure space
    residuals = lnP_obs - lnP_pred
    sse = np.sum(residuals ** 2)
    
    # Check for invalid SSE
    if not np.isfinite(sse) or sse < 0:
        return 1e20
    
    return sse

def find_optimal_virial_degrees(RASPA_data, framework, molecule, max_deg=6, min_points=3, temperatures=None, verbose=True, p_min=None):
    """
    Find the optimal polynomial degrees (deg_a, deg_b) that give the highest R² value
    for the Virial model fit.
    
    Tests all combinations of deg_a and deg_b from 0 to max_deg (inclusive) and returns
    the combination with the highest R². If multiple combinations have the same R²,
    prefers lower values (lower deg_a first, then lower deg_b).
    
    Parameters:
    - RASPA_data: iterable of dict rows with keys 'pressure' (in Pa), 'loading' (n) and 'temperature'
    - framework, molecule: selection keys used to filter data
    - max_deg: maximum degree to test for both deg_a and deg_b (default: 6)
    - min_points: minimum number of valid data points required for fitting
    - temperatures: optional list of temperatures to restrict the fit (tolerant matching)
    - verbose: if True, print progress and results
    - p_min: minimum pressure threshold (in Pa). Points with pressure < p_min are excluded before fitting.
    
    Returns:
    - dict with keys:
        - 'deg_a': optimal deg_a value
        - 'deg_b': optimal deg_b value
        - 'r2': best R² value achieved
        - 'coeffs_a': optimal a_j coefficients
        - 'coeffs_b': optimal b_j coefficients
        - 'all_results': list of all tested combinations with their R² values
    """
    # Handle lists by taking the first element (for compatibility with multi-select)
    if isinstance(framework, (list, tuple, set, np.ndarray)):
        framework = list(framework)[0] if framework else None
    if isinstance(molecule, (list, tuple, set, np.ndarray)):
        molecule = list(molecule)[0] if molecule else None
    
    if framework is None or molecule is None:
        raise ValueError("framework and molecule must be provided (not None or empty)")
    
    pts = phelp.filter_raspa_data(
        RASPA_data, frameworks=[framework], molecules=[molecule], temperatures=temperatures,
        only_pure_adsorption=True,
    )
    if not pts:
        raise ValueError(f"No data for {framework},{molecule}")
    
    # Filter by P_min first (before loading range logic)
    if p_min is not None:
        p_min_val = float(p_min)
        pts = [p for p in pts if p.get('pressure') is not None and float(p['pressure']) >= p_min_val]
    
    # Filter valid points
    valid = [p for p in pts if p.get('pressure') and p.get('loading') and p.get('temperature')
             and float(p['pressure']) > 0 and float(p['loading']) > 0]
    if len(valid) < min_points:
        raise ValueError(f"Insufficient valid data points: {len(valid)} < {min_points}")
    
    n_arr = np.array([float(p['loading']) for p in valid], float)
    P_arr = np.array([float(p['pressure']) for p in valid], float)
    T_arr = np.array([float(p['temperature']) for p in valid], float)
    # Fit ln(P) = ln(q*) + (1/T)*Σ a_j q*^j + Σ b_j q*^j (non-linear optimization in log-pressure space)
    lnP_obs = np.log(P_arr)  # ln(P) where P is in Pa
    
    best_r2 = -np.inf
    best_deg_a = None
    best_deg_b = None
    best_coeffs_a = None
    best_coeffs_b = None
    best_r2_by_temp = {}
    
    # Track best R² per temperature across all combinations
    best_r2_per_temp = {}  # {T: best_r2_value}
    best_combo_per_temp = {}  # {T: (deg_a, deg_b)}
    
    all_results = []
    
    if verbose:
        print(f"Testing {max_deg + 1} x {max_deg + 1} = {(max_deg + 1)**2} combinations of (deg_a, deg_b)...")
    
    for deg_a in range(max_deg + 1):
        for deg_b in range(max_deg + 1):
            try:
                n_pts = len(n_arr)
                n_params = (deg_a + 1) + (deg_b + 1)
                
                # Check if we have enough points
                if n_pts < n_params:
                    if verbose:
                        print(f"  ({deg_a}, {deg_b}): insufficient points ({n_pts} < {n_params})")
                    all_results.append({
                        'deg_a': deg_a,
                        'deg_b': deg_b,
                        'r2': np.nan,
                        'error': 'insufficient_points'
                    })
                    continue
                
                # Non-linear optimization: minimize SSE in log-pressure space
                # SSE = Σ (ln(P_i,exp) - ln(P_i,pred))^2
                # Initial guess: use linear least squares on ln(P/n) as starting point
                y_lnPn = np.log(P_arr / n_arr)
                Va_init = np.vstack([(n_arr ** j) / T_arr for j in range(deg_a + 1)]).T
                Vb_init = np.vstack([n_arr ** j for j in range(deg_b + 1)]).T
                V_init = np.hstack([Va_init, Vb_init])
                try:
                    coeffs_init, _, _, _ = np.linalg.lstsq(V_init, y_lnPn, rcond=None)
                    p0 = coeffs_init
                except Exception:
                    # Fallback: zero initialization
                    p0 = np.zeros(n_params)
                
                # Objective function wrapper
                def objective(params):
                    return virial_sse_objective(params, n_arr, T_arr, lnP_obs, deg_a, deg_b, c_ln=1.0)
                
                # Perform non-linear optimization with more robust settings
                result = minimize(objective, p0, method='BFGS',
                                options={'maxiter': 10000, 'gtol': 1e-6})
                
                # Check if optimization succeeded or at least produced reasonable results
                if not result.success:
                    # Check if we got a reasonable result despite the warning
                    if hasattr(result, 'x') and result.x is not None and len(result.x) == n_params:
                        if np.all(np.isfinite(result.x)):
                            # Use the result anyway if it's finite (precision loss warnings are often non-fatal)
                            if verbose:
                                print(f"  ({deg_a}, {deg_b}): optimization warning - {result.message} (using result anyway)")
                        else:
                            if verbose:
                                print(f"  ({deg_a}, {deg_b}): optimization failed - {result.message} (non-finite result)")
                            all_results.append({
                                'deg_a': deg_a,
                                'deg_b': deg_b,
                                'r2': np.nan,
                                'error': f'optimization_failed: {result.message}'
                            })
                            continue
                    else:
                        if verbose:
                            print(f"  ({deg_a}, {deg_b}): optimization failed - {result.message}")
                        all_results.append({
                            'deg_a': deg_a,
                            'deg_b': deg_b,
                            'r2': np.nan,
                            'error': f'optimization_failed: {result.message}'
                        })
                        continue
                
                # Extract coefficients
                n_a = deg_a + 1
                coeffs_a = result.x[:n_a]
                coeffs_b = result.x[n_a:]
                
                # Calculate predicted ln(P) values
                lnP_pred = virial_model_lnP(n_arr, T_arr, coeffs_a, coeffs_b, c_ln=1.0)
                
                # Convert to pressure scale for R² calculation
                P_pred = np.exp(lnP_pred)
                P_exp = P_arr
                
                # Calculate SSE on pressure scale
                SSE = np.sum((P_exp - P_pred) ** 2)
                
                # Calculate R² = 1 - (SSE / sum((P_i,exp - mean(P_exp))²))
                P_mean = np.mean(P_exp)
                SS_tot = np.sum((P_exp - P_mean) ** 2)
                if SS_tot > 0:
                    r2_global = 1.0 - (SSE / SS_tot)
                else:
                    r2_global = np.nan
                
                # Round values for output
                r2_global_rounded = round(r2_global, 4) if np.isfinite(r2_global) else np.nan
                
                # Compute per-temperature R² values
                r2_by_temp = {}
                temps_unique = sorted(set(T_arr))
                for T_val in temps_unique:
                    mask = np.abs(T_arr - T_val) < 0.1  # tolerance for temperature matching
                    if np.sum(mask) >= 2:  # need at least 2 points for R²
                        lnP_obs_temp = lnP_obs[mask]
                        lnP_pred_temp = lnP_pred[mask]
                        ss_res_temp = np.sum((lnP_obs_temp - lnP_pred_temp) ** 2)
                        ss_tot_temp = np.sum((lnP_obs_temp - np.mean(lnP_obs_temp)) ** 2)
                        r2_temp = 1.0 - (ss_res_temp / ss_tot_temp) if ss_tot_temp > 0 else np.nan
                        r2_temp_rounded = round(r2_temp, 4) if np.isfinite(r2_temp) else np.nan
                        r2_by_temp[T_val] = r2_temp_rounded
                        
                        # Track best R² per temperature across all combinations
                        if np.isfinite(r2_temp):
                            if T_val not in best_r2_per_temp or r2_temp > best_r2_per_temp[T_val]:
                                best_r2_per_temp[T_val] = r2_temp
                                best_combo_per_temp[T_val] = (deg_a, deg_b)
                
                all_results.append({
                    'deg_a': deg_a,
                    'deg_b': deg_b,
                    'r2': r2_global_rounded,
                    'r2_by_temp': r2_by_temp
                })
                
                # Check if this is the best
                # Prefer highest R², but when R² values are equal (rounded to 3 decimals), prefer lowest deg_a
                if np.isfinite(r2_global):
                    r2_rounded = r2_global_rounded
                    best_r2_rounded = round(best_r2, 4) if np.isfinite(best_r2) else -np.inf
                    
                    # If this R² is higher, select it
                    # If R² is equal (within rounding tolerance) and deg_a is lower, select it
                    is_better = False
                    if r2_rounded > best_r2_rounded:
                        is_better = True
                    elif np.isclose(r2_rounded, best_r2_rounded, atol=0.0005) and best_deg_a is not None:
                        # R² values are equal (rounded), prefer lower deg_a
                        if deg_a < best_deg_a:
                            is_better = True
                    
                    if is_better:
                        best_r2 = r2_global
                        best_deg_a = deg_a
                        best_deg_b = deg_b
                        best_coeffs_a = coeffs_a
                        best_coeffs_b = coeffs_b
                        best_r2_by_temp = r2_by_temp
                
                if verbose:
                    temp_r2_str = ", ".join([f"T={int(T)}K: R²={r2_val:.4f}" for T, r2_val in sorted(r2_by_temp.items())])
                    print(f"  ({deg_a}, {deg_b}): Global R² = {r2_global_rounded:.4f} | {temp_r2_str}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ({deg_a}, {deg_b}): Error - {e}")
                all_results.append({
                    'deg_a': deg_a,
                    'deg_b': deg_b,
                    'r2': np.nan,
                    'error': str(e)
                })
                continue
    
    if best_deg_a is None or best_deg_b is None:
        raise ValueError("Could not find valid fit for any combination of degrees")
    
    if verbose:
        best_r2_rounded = round(best_r2, 4) if np.isfinite(best_r2) else np.nan
        print(f"\nOptimal combination: deg_a={best_deg_a}, deg_b={best_deg_b}, Global R²={best_r2_rounded:.4f}")
        if best_r2_by_temp:
            print("Per-temperature R² values (for globally best combination):")
            for T, r2 in sorted(best_r2_by_temp.items()):
                print(f"  T={int(T)}K: R²={r2:.4f}")
        
        # Show best R² per temperature across all combinations
        if best_r2_per_temp:
            print("\nBest R² per temperature (across all combinations):")
            for T in sorted(best_r2_per_temp.keys()):
                best_r2_val = round(best_r2_per_temp[T], 4)
                best_deg_a_temp, best_deg_b_temp = best_combo_per_temp[T]
                print(f"  T={int(T)}K: R²={best_r2_val:.4f} (deg_a={best_deg_a_temp}, deg_b={best_deg_b_temp})")
    
    return {
        'deg_a': best_deg_a,
        'deg_b': best_deg_b,
        'r2': round(best_r2, 4) if np.isfinite(best_r2) else np.nan,
        'r2_by_temp': best_r2_by_temp,
        'coeffs_a': best_coeffs_a,
        'coeffs_b': best_coeffs_b,
        'all_results': all_results
    }

def compute_lnP_per_temperature_separate(
    RASPA_data,
    framework,
    molecule,
    deg_a=2,
    deg_b=2,
    min_points=3,
    temperatures=None,
    verbose=True,
    p_min=None,
    coefficient_report_path=None,
    report_preamble=None,
):
    """
    Fit the Virial equation using non-linear optimization in log-pressure space:
        ln(P) = ln(q*) + (1/T)*Σ_{j=0}^{deg_a} a_j (q*)^j + Σ_{j=0}^{deg_b} b_j (q*)^j
    
    Minimizes SSE in log-pressure space: SSE = Σ (ln(P_i,exp) - ln(P_i,pred))^2
    
    This function builds a global non-linear fit across temperatures and
    returns the Virial coefficients `a_j` and `b_j` used elsewhere (e.g. Qst).
    
    Parameters:
    - verbose: if True, print coefficients and fit statistics (default: True)
    - p_min: minimum pressure threshold (in Pa). Points with pressure < p_min are excluded before fitting.
    
    NOTE: In the paper, m and n refer to the NUMBER of coefficients (a_i and b_j respectively).
    In this code, deg_a and deg_b refer to the MAXIMUM DEGREE of the polynomial.
    Mapping: if paper uses m=3 coefficients, use deg_a=2 (gives 3 coefficients: a_0, a_1, a_2).
             if paper uses n=3 coefficients, use deg_b=2 (gives 3 coefficients: b_0, b_1, b_2).
    
    The a_j coefficients are temperature-dependent virial parameters fitted globally
    across all temperatures. The temperature dependence is built into the model via
    the (1/T) factor, making a_j effectively represent temperature-dependent parameters.
    This allows the classical Virial expression for Qst per paper Eq. (31):
        Q_st(n) = -R * Σ_{i=0}^{m-1} a_i (q*)^i

    Parameters
    - RASPA_data: iterable of dict rows with keys 'pressure' (in Pa), 'loading' (n) and 'temperature'
    - framework, molecule: selection keys used to filter data
    - deg_a, deg_b: maximum polynomial degrees for the (1/T)*a(n) and b(n) expansions
                    (number of coefficients = deg_a + 1 and deg_b + 1 respectively)
    - min_points: minimum number of valid data points required
    - temperatures: optional list of temperatures to restrict the fit (tolerant matching)

    Returns a dict with keys 'coeffs_a', 'coeffs_b', 'coeffs_a_stderr', 'coeffs_b_stderr', 'c_ln', 'residuals', 'rank', 's', 'r2', 'SSE', 'N'
        - 'coeffs_a': array of a_i coefficients (deg_a+1 values)
        - 'coeffs_b': array of b_j coefficients (deg_b+1 values)
        - 'coeffs_a_stderr': standard errors for a_i coefficients
        - 'coeffs_b_stderr': standard errors for b_j coefficients
        - 'c_ln': coefficient for ln(q*) term (fixed at 1.0 when fitting ln(P/n) directly)
        - 'residuals', 'rank', 's': from least squares fit
        - 'r2': R² on pressure scale
        - 'SSE': Sum of Squared Errors on pressure scale
        - 'N': Number of data points
    """
    pts = phelp.filter_raspa_data(
        RASPA_data, frameworks=[framework], molecules=[molecule], temperatures=temperatures,
        only_pure_adsorption=True,
    )
    if not pts:
        raise ValueError(f"No data for {framework},{molecule}")

    # Filter by P_min first (before loading range logic)
    if p_min is not None:
        p_min_val = float(p_min)
        pts = [p for p in pts if p.get('pressure') is not None and float(p['pressure']) >= p_min_val]

    # keep only rows with positive pressure and positive loading (n)
    valid = [p for p in pts if p.get('pressure') and p.get('loading') and p.get('temperature')
             and float(p['pressure']) > 0 and float(p['loading']) > 0]
    if len(valid) < min_points:
        raise ValueError("Insufficient valid data points")

    # use n_arr to emphasize that this is loading (n or q*)
    n_arr = np.array([float(p['loading']) for p in valid], float)
    P_arr = np.array([float(p['pressure']) for p in valid], float)
    T_arr = np.array([float(p['temperature']) for p in valid], float)

    # Validate data
    if len(n_arr) < min_points:
        raise ValueError(f"Insufficient valid data points: {len(n_arr)} < {min_points}")
    if np.any(n_arr <= 0) or np.any(P_arr <= 0) or np.any(T_arr <= 0):
        raise ValueError("Data contains non-positive values for loading, pressure, or temperature")
    
    # Fit ln(P) = ln(q*) + (1/T)*Σ_{j=0}^{deg_a} a_j (q*)^j + Σ_{j=0}^{deg_b} b_j (q*)^j
    # Using non-linear optimization in log-pressure space per paper
    lnP_obs = np.log(P_arr)  # observed ln(P) where P is in Pa
    c_ln = 1.0  # coefficient for ln(q*) term
    
    # Check for invalid values
    if not np.all(np.isfinite(lnP_obs)) or not np.all(np.isfinite(n_arr)) or not np.all(np.isfinite(T_arr)):
        raise ValueError("Data contains non-finite values (NaN or Inf)")
    
    N = len(lnP_obs)  # number of data points
    n_params = (deg_a + 1) + (deg_b + 1)
    
    if N < n_params:
        raise ValueError(f"Insufficient data points ({N}) for number of parameters ({n_params})")
    
    # Initial guess: use linear least squares on ln(P/n) as starting point
    y_lnPn = np.log(P_arr / n_arr)
    Va_init = np.vstack([(n_arr ** j) / T_arr for j in range(deg_a + 1)]).T
    Vb_init = np.vstack([n_arr ** j for j in range(deg_b + 1)]).T
    V_init = np.hstack([Va_init, Vb_init])
    try:
        coeffs_init, _, rank, s = np.linalg.lstsq(V_init, y_lnPn, rcond=None)
        p0 = coeffs_init
    except Exception:
        # Fallback: zero initialization
        p0 = np.zeros(n_params)
        rank = 0
        s = None
    
    # Objective function wrapper
    def objective(params):
        return virial_sse_objective(params, n_arr, T_arr, lnP_obs, deg_a, deg_b, c_ln=c_ln)
    
    # Perform non-linear optimization with robust settings
    result = minimize(
        objective,
        p0,
        method='BFGS',
        options={
            'maxiter': 10000,
            'gtol': 1e-6,  # gradient tolerance (BFGS supports gtol)
        }
    )
    
    # Check if optimization succeeded or at least produced reasonable results
    if not result.success:
        # Check if we got a reasonable result despite the warning
        if hasattr(result, 'x') and result.x is not None and len(result.x) == n_params:
            # Try to use the result anyway if it's finite
            if np.all(np.isfinite(result.x)):
                # Only print warning if SSE is unreasonably large (e.g., > 1e3)
                # Small SSE values (< 1e3) indicate the fit is good despite convergence message
                if result.fun > 1e3:
                    print(f"Warning: Non-linear optimization did not fully converge: {result.message}")
                    print(f"  Using result anyway (SSE = {result.fun:.6e})")
                # Otherwise, silently use the result (SSE is reasonable)
            else:
                raise ValueError(f"Non-linear optimization failed: {result.message} (result contains non-finite values)")
        else:
            raise ValueError(f"Non-linear optimization failed: {result.message} (no valid result)")
    
    # Extract coefficients
    n_a = deg_a + 1
    coeffs_a = result.x[:n_a]  # a_i coefficients
    coeffs_b = result.x[n_a:]  # b_j coefficients
    
    # Calculate predicted ln(P) values
    lnP_pred = virial_model_lnP(n_arr, T_arr, coeffs_a, coeffs_b, c_ln=c_ln)
    residuals_lnP = lnP_obs - lnP_pred
    SSE_lnP = np.sum(residuals_lnP ** 2)  # SSE in log-pressure space

    # Calculate standard errors (uncertainties) for coefficients
    # For non-linear optimization, use the inverse Hessian (approximate covariance matrix)
    # Covariance matrix ≈ sigma^2 * H^(-1), where H is the Hessian of SSE
    # Residual variance (on ln(P) scale)
    M = n_params
    if N > M:
        sigma2 = SSE_lnP / (N - M)
    else:
        sigma2 = np.nan
    
    # Covariance from analytical Jacobian: J[:, i] = d(lnP)/d(param_i)
    try:
        jac = _analytical_jacobian(n_arr, T_arr, deg_a, deg_b)
        try:
            JtJ_inv = np.linalg.pinv(jac.T @ jac)
        except Exception:
            JtJ_inv = np.eye(M)
        if np.isfinite(sigma2) and sigma2 > 0:
            cov_diag = np.maximum(np.diag(sigma2 * JtJ_inv), 0)
            coeffs_stderr = np.sqrt(cov_diag)
            coeffs_a_stderr = coeffs_stderr[:deg_a + 1]
            coeffs_b_stderr = coeffs_stderr[deg_a + 1:]
        else:
            coeffs_a_stderr = np.full(len(coeffs_a), np.nan)
            coeffs_b_stderr = np.full(len(coeffs_b), np.nan)
    except Exception:
        coeffs_a_stderr = np.full(len(coeffs_a), np.nan)
        coeffs_b_stderr = np.full(len(coeffs_b), np.nan)
    
    c_ln_stderr = np.nan  # No error for fixed c_ln = 1.0

    # Calculate goodness of fit metrics (R²) on pressure scale
    P_pred = np.exp(lnP_pred)
    P_exp = P_arr
    
    # Calculate SSE on pressure scale
    SSE = np.sum((P_exp - P_pred) ** 2)
    
    # Calculate R² = 1 - (SSE / sum((P_i,exp - mean(P_exp))²))
    P_mean = np.mean(P_exp)
    SS_tot = np.sum((P_exp - P_mean) ** 2)
    if SS_tot > 0:
        r2 = 1.0 - (SSE / SS_tot)
    else:
        r2 = np.nan

    report_text = format_virial_fit_coefficient_report(
        framework,
        molecule,
        deg_a,
        deg_b,
        coeffs_a,
        coeffs_a_stderr,
        coeffs_b,
        coeffs_b_stderr,
        N,
        r2,
        SSE,
        P_mean,
        P_pred,
    )
    pre = report_preamble if report_preamble is not None else (DEGREE_SEARCH_REPORT_TEXT or "")
    if pre.strip():
        report_text = pre.rstrip() + "\n\n" + report_text
    if coefficient_report_path:
        out = Path(coefficient_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_text, encoding="utf-8")
    elif verbose:
        print("\n" + report_text + "\n")

    return {
        'coeffs_a': coeffs_a,
        'coeffs_b': coeffs_b,
        'coeffs_a_stderr': coeffs_a_stderr,  # Standard errors for a_i coefficients
        'coeffs_b_stderr': coeffs_b_stderr,  # Standard errors for b_j coefficients
        'c_ln': c_ln,  # coefficient for ln(q*) term (fixed at 1.0)
        'c_ln_stderr': c_ln_stderr,  # Standard error for c_ln (NaN since it's fixed)
        'residuals': residuals_lnP,  # Residuals in log-pressure space
        'rank': rank,  # Rank from initial guess (linear LSQ)
        's': s,  # Singular values from initial guess (linear LSQ)
        'r2': r2,      # R² on pressure scale
        'SSE': SSE,    # Sum of Squared Errors on pressure scale
        'SSE_lnP': SSE_lnP,  # Sum of Squared Errors in log-pressure space
        'N': N         # Number of data points
    }

def compute_lnP_from_coeffs(coeffs_a, coeffs_b, q_grid, T, c_ln=1.0):
    """
    Given Virial coefficients (a_i and b_j) fitted per paper Eq. (30), compute both 
    ln(P) and ln(P/n) where P is in Pa.

    The Virial model per paper Eq. (30) is:
        ln(P) = ln(q*) + (1/T) * Σ_{i=0}^{m-1} a_i (q*)^i + Σ_{j=0}^{n-1} b_j (q*)^j
    
    This can be rearranged as:
        ln(P/n) = (1/T) * Σ a_i n^i + Σ b_j n^j
        ln(P) = c_ln * ln(n) + (1/T) * Σ a_i n^i + Σ b_j n^j
    
    where c_ln is the coefficient for ln(q*) term (typically 1.0 per paper's equation).

    Parameters:
    - coeffs_a, coeffs_b: arrays of a_i and b_j coefficients (from fitting)
    - q_grid: array of n (loading, q*) values where the model is evaluated
    - T: temperature (scalar) used for the (1/T) terms
    - c_ln: coefficient for ln(q*) term (default 1.0 to match paper's equation)

    Returns a tuple `(lnP, lnP_over_q)` arrays of the same shape as `q_grid`.
        - lnP: ln(P) where P is in Pa
        - lnP_over_q: ln(P/n) where P is in Pa
    """
    coeffs_a = np.asarray(coeffs_a, dtype=float)
    coeffs_b = np.asarray(coeffs_b, dtype=float)
    qg = np.asarray(q_grid, dtype=float)

    lnP_over_q = _eval_poly(coeffs_a, qg) / float(T) + _eval_poly(coeffs_b, qg)

    # ln(P) = c_ln * ln(q*) + (1/T) * Σ a_i q*^i + Σ b_j q*^j
    # Protect against non-positive q values.
    q_safe = np.where(qg > 0, qg, np.nan)
    lnq = np.log(q_safe)
    lnP = float(c_ln) * lnq + lnP_over_q

    return lnP, lnP_over_q

def compute_Qst_from_coef_slopes(
    RASPA_data,
    framework,
    molecule,
    deg_a=2,
    deg_b=2,
    min_points=3,
    n_min=None,
    n_max=None,
    n_points=200,
    R=R,
    temperatures=None,
    coeffs_a_override=None,
    coeffs_b_override=None,
    p_min=None,
    verbose=True,
    coefficient_report_path=None,
    report_preamble=None,
    eval_loadings=None,
):
    """
    Compute Qst(n) directly from global Virial fit using the classical Virial expression
    per paper Eq. (31):
        Q_st(q*) = -R * Σ_{i=0}^{m-1} a_i (q*)^i
    
    This is derived from the temperature-dependent Virial model per paper Eq. (30):
        ln P = ln q* + (1/T) * Σ_{i=0}^{m-1} a_i (q*)^i + Σ_{j=0}^{n-1} b_j (q*)^j
    
    where the fit includes T in the regression and the a_i coefficients represent
    temperature-dependent virial parameters fitted globally across temperatures.
    
    The a_i coefficients come from the global fit where temperature is included
    in the regression, making them effectively temperature-dependent parameters.
    Taking the derivative with respect to (1/T) at constant loading gives the
    isosteric heat of adsorption as shown above.
    
    NOTE: Zero-coverage enthalpy Q^0_st = -R * a_0 per paper Eq. (32).
    
    Parameters:
    - p_min: minimum pressure threshold (in Pa). Points with pressure < p_min are excluded before fitting.
            This filtering is applied BEFORE the loading range logic.
    - coefficient_report_path: optional path to write the Virial coefficient report (UTF-8).
    - report_preamble: optional text prepended to the report file; if None, uses DEGREE_SEARCH_REPORT_TEXT.
    - eval_loadings: optional 1D array of loadings; when set, Qst is evaluated on these
      values (sorted, unique) instead of ``linspace`` from the intersection range.  Use
      together with CC ``compute_isosteric_heat(..., loadings=...)`` so both curves
      share identical abscissas.
    """
    # Input rows arrive pre-selected from DataSelection (p-bounded, common loading grid).
    # No p_min filtering or loading-range selection needed here.
    filtered = phelp.filter_raspa_data(
        RASPA_data, frameworks=[framework], molecules=[molecule], temperatures=temperatures,
        only_pure_adsorption=True,
    )
    if not filtered:
        raise ValueError(f"No data for {framework},{molecule} to compute n-range")

    # Same loading span as Clausius–Clapeyron (compute_isosteric_heat): per-temperature
    # intersection via DataSelection, not global min/max over all rows (which is wider).
    try:
        n_min_val, n_max_val = ds._intersection_loading_range(filtered)
    except Exception:
        n_min_val, n_max_val = None, None
    if n_min_val is None or n_max_val is None or n_max_val <= n_min_val:
        all_n = np.array(
            [float(p['loading']) for p in filtered if p.get('loading') and float(p['loading']) > 0],
            dtype=float,
        )
        if all_n.size == 0:
            raise ValueError(f"No positive loadings for {framework},{molecule}")
        n_min_val = float(np.min(all_n))
        n_max_val = float(np.max(all_n))

    # User-specified overrides still respected.
    if n_min is not None:
        n_min_val = float(n_min)
    if n_max is not None:
        n_max_val = float(n_max)

    if n_max_val < n_min_val:
        n_max_val = n_min_val

    filtered_for_fit = filtered
    
    # Check if override coefficients are provided (skip fitting if so)
    if coeffs_a_override is not None and coeffs_b_override is not None:
        # Use provided coefficients directly (skip fitting)
        coeffs_a = np.asarray(coeffs_a_override, dtype=float)
        coeffs_b = np.asarray(coeffs_b_override, dtype=float)
    else:
        # Fit coefficients ONCE using only the filtered data points (within n_min to n_max range)
        # This is the ONLY place coefficients are calculated
        results = compute_lnP_per_temperature_separate(
            filtered_for_fit,
            framework,
            molecule,
            deg_a=deg_a,
            deg_b=deg_b,
            min_points=min_points,
            temperatures=temperatures,
            verbose=verbose and coefficient_report_path is None,
            p_min=None,
            coefficient_report_path=coefficient_report_path,
            report_preamble=report_preamble,
        )

        # Reuse the coefficients from the fit (do NOT recalculate)
        coeffs_a = np.asarray(results['coeffs_a'], float)
        coeffs_b = np.asarray(results['coeffs_b'], float)

    if eval_loadings is not None:
        n_grid = np.asarray(eval_loadings, dtype=float).ravel()
        n_grid = n_grid[np.isfinite(n_grid) & (n_grid > 0)]
        if n_grid.size < 2:
            raise ValueError("eval_loadings must contain at least two finite positive values")
        n_grid = np.sort(np.unique(n_grid))
        n_min_val = float(n_grid[0])
        n_max_val = float(n_grid[-1])
    else:
        n_grid = np.linspace(n_min_val, n_max_val, int(max(2, n_points)))

    # Qst(n) = -R * Σ a_j n^j  (Virial expression, paper Eq. 31)
    Q_grid = -float(R) * _eval_poly(coeffs_a, n_grid)  # J/mol

    return {
        'coeffs_a': coeffs_a,
        'coeffs_b': coeffs_b,
        'n_grid': n_grid,
        'Qst': Q_grid,        # J/mol
        'Qst_kJmol': Q_grid / 1000.0,  # kJ/mol for plotting
        'fitted_points': filtered_for_fit,  # Data points used for fitting (for scatter plotting)
        'n_min': n_min_val,  # Min loading of intersection range
        'n_max': n_max_val   # Max loading of intersection range
    }

def plot_Qst(
    RASPA_data=None,
    framework=None,
    molecule=None,
    temperatures=None,
    standard_colors=None,
    deg_a=2,
    deg_b=2,
    min_points=3,
    results=None,
    n_points=200,
    R=R,
    save_fw_list=None,
    save_mol_list=None,
    save_temp_list=None,
    residual_threshold=20.0,
    min_sigma_floor=1e-3,
    verbose=False,
    coeffs_a_override=None,
    coeffs_b_override=None,
    p_min=None,
    show_all_loadings=False,
    degrees_per_combo=None,
    method_linestyles=None,
    save_data=False,
):
    """Plot Virial Qst(n) over the intersection loading range (outlier mask + ln(P) fit).

    ``results`` is accepted for API compatibility with Main but is not used
    (this plot always refits using intersection + :func:`_outlier_mask`).

    When ``method_linestyles`` is set (e.g. from Main), uses HoA colours and
    :func:`PlotHelpers.build_hoa_proxy_legend`; otherwise uses the classic
    legend label with intersection point count.
    """
    if isinstance(framework, (list, tuple, set, np.ndarray)):
        fw_list_iter = list(framework)
    elif framework is None and save_fw_list is not None:
        fw_list_iter = list(save_fw_list)
    else:
        fw_list_iter = [framework] if framework is not None else []

    if isinstance(molecule, (list, tuple, set, np.ndarray)):
        mol_list_iter = list(molecule)
    elif molecule is None and save_mol_list is not None:
        mol_list_iter = list(save_mol_list)
    else:
        mol_list_iter = [molecule] if molecule is not None else []

    if not fw_list_iter or not mol_list_iter:
        raise ValueError("plot_Qst: no framework or molecule provided to plot")

    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    combo_count = 0
    global_max_loading = 0.0
    global_min_qst = np.inf
    global_max_qst = -np.inf
    export_rows = []

    for fw in fw_list_iter:
        for mol in mol_list_iter:
            try:
                # Per-combo degree override
                if degrees_per_combo and (fw, mol) in degrees_per_combo:
                    deg_a_use, deg_b_use = degrees_per_combo[(fw, mol)]
                else:
                    deg_a_use, deg_b_use = deg_a, deg_b

                # Input rows are pre-selected by DataSelection; just filter by fw/mol/T.
                pts_all = phelp.filter_raspa_data(
                    RASPA_data, frameworks=[fw], molecules=[mol], temperatures=temperatures,
                    only_pure_adsorption=True,
                )
                pts_all = [p for p in pts_all
                           if p.get('pressure') and p.get('loading') and p.get('temperature')
                           and float(p['pressure']) > 0 and float(p['loading']) > 0]
                if not pts_all:
                    if verbose:
                        print(f"[plot_Qst] No valid data for {fw},{mol}; skipping")
                    continue

                # Match CC / compute_Qst_from_coef_slopes: intersection over temperatures.
                try:
                    n_min_val, n_max_val = ds._intersection_loading_range(pts_all)
                except Exception:
                    n_min_val, n_max_val = None, None
                if n_min_val is None or n_max_val is None or n_max_val <= n_min_val:
                    all_n = np.array(
                        [float(p['loading']) for p in pts_all if float(p['loading']) > 0],
                        dtype=float,
                    )
                    if all_n.size == 0:
                        if verbose:
                            print(f"[plot_Qst] No positive loadings for {fw},{mol}; skipping")
                        continue
                    n_min_val = float(np.min(all_n))
                    n_max_val = float(np.max(all_n))

                pts_in_range = pts_all

                # --- outlier removal (single BFGS, analytical Jacobian) ---
                if coeffs_a_override is None or coeffs_b_override is None:
                    keep = _outlier_mask(pts_in_range, deg_a_use, deg_b_use,
                                         residual_threshold=residual_threshold,
                                         min_sigma_floor=min_sigma_floor)
                    pts_for_fit = [p for p, k in zip(pts_in_range, keep) if k]
                    if len(pts_for_fit) < min_points:
                        if verbose:
                            print(f"[plot_Qst] Outlier removal left {len(pts_for_fit)} pts for {fw},{mol}; using range-filtered data.")
                        pts_for_fit = pts_in_range
                else:
                    pts_for_fit = pts_in_range

                # --- coefficients ---
                if coeffs_a_override is not None and coeffs_b_override is not None:
                    coeffs_a = np.asarray(coeffs_a_override, dtype=float)
                    if verbose:
                        print(f"[plot_Qst] Using override coefficients for {fw},{mol}")
                else:
                    fit = compute_lnP_per_temperature_separate(
                        pts_for_fit, fw, mol,
                        deg_a=deg_a_use, deg_b=deg_b_use, min_points=min_points,
                        temperatures=temperatures, verbose=False, p_min=None
                    )
                    coeffs_a = np.asarray(fit['coeffs_a'], float)

                # --- Qst curve (Horner via _eval_poly) ---
                n_grid = np.linspace(n_min_val, n_max_val, int(max(2, n_points)))
                Q_plot = -float(R) * _eval_poly(coeffs_a, n_grid) / 1000.0  # kJ/mol

                if method_linestyles is not None:
                    color = phelp.get_color_for_molecule(mol) or 'C0'
                    # Match PlotHelpers.build_hoa_proxy_legend: single method → structure linestyles
                    _hoa_ls_mode = phelp.choose_hoa_proxy_linestyle_mode(['virial'])
                    ls = phelp.get_hoa_linestyle(
                        fw, 'virial', _hoa_ls_mode, method_linestyles=method_linestyles
                    )
                    ax.plot(
                        n_grid,
                        Q_plot,
                        lw=phelp.LINEWIDTH,
                        alpha=phelp.ALPHA,
                        color=color,
                        linestyle=ls,
                        label="_nolegend_",
                    )
                elif standard_colors and 'virial' in standard_colors:
                    color = standard_colors['virial']
                    clean_fw = phelp.clean_material_name(fw)
                    clean_mol = phelp.get_molecule_display_name(mol)
                    ax.plot(
                        n_grid,
                        Q_plot,
                        lw=phelp.LINEWIDTH,
                        alpha=phelp.ALPHA,
                        color=color,
                        label=f"{clean_fw}, {clean_mol}, intersection range ({len(pts_for_fit)} pts)",
                    )
                else:
                    try:
                        color = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0'])[combo_count % 10]
                    except Exception:
                        color = 'C0'
                    clean_fw = phelp.clean_material_name(fw)
                    clean_mol = phelp.get_molecule_display_name(mol)
                    ax.plot(
                        n_grid,
                        Q_plot,
                        lw=phelp.LINEWIDTH,
                        alpha=phelp.ALPHA,
                        color=color,
                        label=f"{clean_fw}, {clean_mol}, intersection range ({len(pts_for_fit)} pts)",
                    )

                mask_valid = np.isfinite(Q_plot)
                if mask_valid.any():
                    global_max_loading = max(global_max_loading, np.max(n_grid))
                    global_min_qst = min(global_min_qst, np.min(Q_plot[mask_valid]))
                    global_max_qst = max(global_max_qst, np.max(Q_plot[mask_valid]))

                marker = phelp.get_marker_for_material(fw)
                n_fit = np.array([float(p['loading']) for p in pts_for_fit], dtype=float)
                in_range = (n_fit >= n_min_val) & (n_fit <= n_max_val)
                if in_range.any():
                    if method_linestyles is not None:
                        sc_color = phelp.get_color_for_molecule(mol) or color
                    elif standard_colors:
                        sc_color = standard_colors.get('virial', color)
                    else:
                        sc_color = color
                    ax.scatter(
                        n_fit[in_range],
                        np.interp(n_fit[in_range], n_grid, Q_plot),
                        color=sc_color,
                        s=phelp.AXIS_S_SIZE,
                        alpha=phelp.ALPHA,
                        marker=marker,
                        zorder=6,
                    )

                if save_data:
                    for n_val, q_val in zip(n_grid, Q_plot):
                        if np.isfinite(n_val) and np.isfinite(q_val):
                            export_rows.append({
                                'framework': fw,
                                'molecule': mol,
                                'loading': float(n_val),
                                'Qst_kJmol': float(q_val),
                            })

                combo_count += 1
            except Exception as e:
                import traceback
                print(f"[plot_Qst] Failed to compute/plot Virial for {fw},{mol}: {e}")
                if verbose:
                    traceback.print_exc()
                continue

    if combo_count == 0:
        raise ValueError("plot_Qst: no valid framework/molecule combinations were plotted")

    ax.set_xlabel("Loading [mol/kg]", **_virial_axis_label_kwargs())
    ax.set_ylabel("Qst [kJ/mol]", **_virial_axis_label_kwargs())
    if method_linestyles is None:
        ax.set_title("Virial")
    if np.isfinite(global_min_qst) and np.isfinite(global_max_qst):
        margin = max(abs(global_max_qst - global_min_qst) * 0.05, 0.5)
        ax.set_ylim(global_min_qst - margin, global_max_qst + margin)
    if global_max_loading > 0:
        ax.set_xlim(left=0, right=global_max_loading * 1.05)
    ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
    if method_linestyles is not None:
        phelp.build_hoa_proxy_legend(
            ax,
            molecules_present=mol_list_iter,
            frameworks_present=fw_list_iter,
            methods_present=['virial'],
            method_linestyles=method_linestyles,
            fontsize=phelp.AXIS_LEGEND_SIZE,
            loc='best',
        )
    else:
        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
    phelp.apply_unified_axes_layout(fig, ax)

    out_path = None
    try:
        fw_list = save_fw_list if save_fw_list is not None else ([framework] if framework is not None else [])
        mol_list = save_mol_list if save_mol_list is not None else ([molecule] if molecule is not None else [])
        temp_list = save_temp_list if save_temp_list is not None else (list(temperatures) if temperatures is not None else [])
        save_prefix = 'virial_' if method_linestyles is not None else 'virial_equation_control_intersection'
        out_path = phelp._save_plot(
            save_prefix, 'plot_controls', fw_list, mol_list, temp_list, fig=fig, out_dir=None
        )
    except Exception:
        pass

    if export_rows and save_data and out_path is not None:
        try:
            base = Path(out_path)
            saved_dir = base.parent / "saved"
            saved_dir.mkdir(parents=True, exist_ok=True)
            stem = base.stem
            if stem.startswith('virial_equation_control_intersection'):
                stem = stem.replace('virial_equation_control_intersection', 'virial_equation', 1)
            data_path = saved_dir / (stem + '_data.txt')
            with data_path.open('w', encoding='utf-8') as f:
                f.write("framework\tmolecule\tloading_mol_per_kg\tQst_kJmol\n")
                for r in export_rows:
                    f.write(
                        f"{r['framework']}\t{r['molecule']}\t"
                        f"{r['loading']}\t{r['Qst_kJmol']}\n"
                    )
        except Exception as e:
            print(f"Warning: failed to write Virial Qst data file: {e}")

    plt.show()
    plt.close(fig)
    return fig, ax


def plot_mixture_heat_hoa_pure_virial(
        mixture_data, RASPA_data_pure,
        selected_frameworks, mixture_name, selected_temperatures,
        p_min, p_max,
        deg_a=2, deg_b=2, degrees_per_combo=None,
        min_points=3, n_loadings=50, smoothing_sigma=None,
        combo_colors=None, out_dir=None, save_data=False):
    """
    Mixture HOA via weighted pure Virial Qst (entry point in Virial).

    Writes ``mixture_hoa_pure_virial_heat_*`` and ``pure_hoa_virial_heat_*`` under
    Heat_of_adsorption, using :func:`compute_Qst_from_coef_slopes` per component.
    Shared layout, mixture weighting, and file naming reuse
    ``ClausiusClapeyron._plot_hoa_mix`` (lazy import to avoid circular imports).
    """
    import ClausiusClapeyron as _cc

    if smoothing_sigma is None:
        smoothing_sigma = 1.5
    if degrees_per_combo is None:
        degrees_per_combo = {}

    if not mixture_data:
        print("plot_mixture_heat_hoa_pure_virial: no mixture data, skipping.")
        return
    if not RASPA_data_pure:
        print("plot_mixture_heat_hoa_pure_virial: no pure RASPA data available; skipping.")
        return

    components = sorted({d['molecule'] for d in mixture_data})
    temps = [float(t) for t in selected_temperatures]
    p_min_b = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b = float(p_max) if p_max is not None else 1e8
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for fw in selected_frameworks:
        qst_interp, pure_curves = {}, {}
        for comp in components:
            da, db = degrees_per_combo.get((fw, comp), (deg_a, deg_b))
            try:
                n_grid = _cc._hoa_pure_loading_grid_n(
                    RASPA_data_pure, fw, comp, temps, n_loadings)
                if n_grid is None:
                    continue
                res = compute_Qst_from_coef_slopes(
                    RASPA_data=RASPA_data_pure,
                    framework=fw, molecule=comp,
                    deg_a=da, deg_b=db,
                    min_points=min_points,
                    n_points=n_loadings,
                    R=R,
                    temperatures=temps,
                    p_min=p_min,
                    verbose=False,
                    eval_loadings=n_grid,
                )
            except Exception as e:
                print(f"plot_mixture_heat_hoa_pure_virial: Virial failed for {fw}/{comp}: {e}")
                continue

            interp, n_arr, qst_arr = _cc._build_qst_interp(res['n_grid'], res['Qst_kJmol'])
            if interp is None:
                continue
            qst_interp[comp] = interp
            pure_curves[comp] = (
                np.asarray(res['n_grid'], dtype=float),
                np.asarray(res['Qst_kJmol'], dtype=float),
            )

        if not qst_interp:
            print(f"plot_mixture_heat_hoa_pure_virial: no Virial Qst for any component in {fw}.")
            continue

        _cc._plot_hoa_mix(
            fw, components, qst_interp, pure_curves,
            mixture_data, temps, p_min_b, p_max_b,
            n_loadings, smoothing_sigma, combo_colors,
            mixture_name, default_cycle,
            method_label='pure Virial',
            mix_prefix='mixture_hoa_pure_virial_heat',
            pure_prefix='pure_hoa_virial_heat',
            selected_temperatures=selected_temperatures,
            out_dir=out_dir,
            save_data=save_data,
        )

    plt.show()
    plt.close('all')
