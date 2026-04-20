import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import PlotHelpers as phelp
import Initialize as init
import DataSelection as ds
from math import isclose
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from matplotlib.ticker import MultipleLocator, NullLocator
from PlotHelpers import R


def _cc_axis_label_kwargs():
    """Match Basic_data ``mixture_isotherm_per_T_*`` (IsothermFittingPlot)."""
    return {'fontsize': phelp.AXIS_LABEL_FONTSIZE, 'fontweight': 'medium'}


def plot_pure_hoa_from_file(filepath, selected_frameworks, selected_molecules,
                            selected_temperatures, colors, base_temp,
                            method_linestyles=None):
    """
    Plot pure heat of adsorption (Qst vs loading) from a file and
    return a qst_cache ``{(fw, mol): (loads_arr, qst_arr)}`` suitable for
    storage-density calculations with method='hoa_file'.
    """
    try:
        hoa_rows = init.load_hoa_data(str(filepath))
    except Exception as e:
        print(f"Warning: Failed to load HOA data from {filepath}: {e}")
        return {}

    curves = init.build_hoa_curves(
        hoa_rows,
        frameworks=selected_frameworks,
        molecules=selected_molecules,
        min_points=2,
    )

    if not curves:
        return {}

    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    any_plotted = False
    vary_T = len(selected_temperatures or []) > 1
    vary_mol = len(selected_molecules or []) > 1
    vary_fw = len(selected_frameworks or []) > 1
    molecules_present = []
    frameworks_present = []
    for (fw, mol), (loads, qst) in curves.items():
        # Single-method HoA plot: color by molecule, linestyle by structure.
        style = phelp.resolve_series_style(
            fw, mol, base_temp,
            vary_fw=vary_fw, vary_mol=vary_mol, vary_T=vary_T,
            plot_kind="hoa",
            combo_colors=None,
            method="hoa_file",
            method_linestyles=method_linestyles,
        )
        color = style.get("color") or phelp.get_color_for_molecule(mol)
        linestyle = phelp.get_hoa_linestyle(fw, "hoa_file", "structure", method_linestyles=method_linestyles)
        marker = style.get("marker") or 'o'
        label = "_nolegend_"
        ax.plot(
            loads, qst, marker=marker, linestyle=linestyle, color=color, label=label,
            lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, markersize=phelp.MARKER_SIZE,
        )
        molecules_present.append(mol)
        frameworks_present.append(fw)
        any_plotted = True

    if any_plotted:
        ax.set_xlabel("Loading [mol/kg]", **_cc_axis_label_kwargs())
        ax.set_ylabel("Qst [kJ/mol]", **_cc_axis_label_kwargs())
        # ax.set_title("Heat of Adsorption from file")
        phelp.build_hoa_proxy_legend(
            ax,
            molecules_present=molecules_present,
            frameworks_present=frameworks_present,
            methods_present=["hoa_file"],
            method_linestyles=method_linestyles,
            fontsize=phelp.AXIS_LEGEND_SIZE,
            loc='best',
        )
        phelp.apply_unified_axes_layout(fig, ax)
        phelp._save_plot(
            "hoa_file", "plot_virial",
            selected_frameworks, selected_molecules, selected_temperatures,
            fig=fig,
        )
        plt.close(fig)

    return curves


def _rows_to_p_mat(rows, framework, molecule, temperatures):
    """Pivot pre-built DataSelection rows into (loadings, temps_with_data, P_mat, lnP_mat).

    Reads values directly from the common loading grid — no PCHIP re-interpolation.
    All temperatures share the same loading values because DataSelection already
    placed every isotherm on the same linspace grid.
    """
    temps = [float(t) for t in temperatures]
    pts = phelp.filter_raspa_data(
        rows, frameworks=[framework], molecules=[molecule], temperatures=temps,
        only_pure_adsorption=True,
    )
    if not pts:
        raise ValueError(f"No pre-built rows found for {framework}, {molecule}")

    data_by_temp = {}
    for pt in pts:
        T = round(float(pt['temperature']))
        data_by_temp.setdefault(T, []).append(
            (float(pt['loading']), float(pt['pressure']))
        )

    temps_with_data = sorted(
        [float(t) for t in temps if round(float(t)) in data_by_temp]
    )
    if len(temps_with_data) < 2:
        raise ValueError(
            f"Need at least two temperatures with data for {framework}, {molecule}"
        )

    first_T = round(float(temps_with_data[0]))
    loadings = np.array(
        sorted({q for q, _ in data_by_temp[first_T]}), dtype=float
    )
    nL = loadings.size
    nT = len(temps_with_data)

    P_mat = np.full((nT, nL), np.nan)
    lnP_mat = np.full((nT, nL), np.nan)

    for i, T in enumerate(temps_with_data):
        T_rounded = round(float(T))
        if T_rounded not in data_by_temp:
            continue
        lq_dict = {q: p for q, p in data_by_temp[T_rounded]}
        for j, q in enumerate(loadings):
            p = lq_dict.get(q, np.nan)
            if np.isfinite(p) and p > 0:
                P_mat[i, j] = p
                with np.errstate(divide='ignore', invalid='ignore'):
                    lnP_mat[i, j] = np.log(p)

    return loadings, np.array(temps_with_data, dtype=float), P_mat, lnP_mat


def compute_isosteric_heat(framework, molecule, temperatures, selected_fit_types, fittings,
                           RASPA_data=None, loadings=None, n_loadings=60, p_grid=None, p_min=None, p_max=None,
                           r2_min=0.95, min_temps=3, smooth=False, smooth_width=3, smoothing_sigma=1.5,
                           use_direct_interpolation=False):
    """
    Compute isosteric heat (Clausius-Clapeyron) using supplied isotherm fits or direct interpolation.
    
    Parameters:
    - use_direct_interpolation: If True, uses RASPA_data directly for interpolation instead of fits.
                                If False (default), uses fitted isotherms (original approach).

    Returns Qst (kJ/mol) and uncertainties propagated from the standard error of slope.

    Notes:
    - Expects temperatures in K.
    - When use_direct_interpolation=True, RASPA_data must be provided.
    
    Loading Range Determination:
    - The loading range is determined from the pressure bounds (p_min, p_max) from Input.py
    - For each temperature, find the loading that corresponds to p_min and p_max
    - The valid loading range is where at least min_temps temperatures have coverage
    - This same loading range is used for BOTH interpolation and formula approaches
    """
    
    temps = [float(t) for t in temperatures]
    min_temps_req = int(min_temps)
    
    # ===== PRESSURE BOUNDS =====
    p_min_bound = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_bound = float(p_max) if p_max is not None else 1e8
    
    # ===== DIRECT INTERPOLATION APPROACH =====
    # Reads the pre-built DataSelection grid directly — no re-interpolation.
    if use_direct_interpolation:
        if RASPA_data is None:
            raise ValueError("RASPA_data must be provided when use_direct_interpolation=True")
        loadings, t_array, P_mat, lnP_mat = _rows_to_p_mat(
            RASPA_data, framework, molecule, temps
        )
        temps_with_data = list(t_array)
        nL = loadings.size

        valid_counts = np.sum(np.isfinite(lnP_mat), axis=0)
        
        # Prepare outputs
        slopes = np.full(nL, np.nan)
        intercepts = np.full(nL, np.nan)
        Qst_kJmol = np.full(nL, np.nan)
        r2_arr = np.full(nL, np.nan)
        slope_stderr = np.full(nL, np.nan)
        Qst_kJmol_stderr = np.full(nL, np.nan)
        
        invT = 1.0 / t_array
        min_temps_req = int(min_temps)
        
        # Fit ln(P) vs 1/T at each loading
        for j in range(nL):
            y = lnP_mat[:, j]
            valid = np.isfinite(y) & np.isfinite(invT)
            n_valid = np.sum(valid)
            if n_valid < min_temps_req:
                continue
            
            x = invT[valid]
            yy = y[valid]
            
            # Linear fit ln(p) = a*(1/T) + b
            a, b = np.polyfit(x, yy, 1)
            y_fit = a * x + b
            ss_res = np.sum((yy - y_fit) ** 2)
            ss_tot = np.sum((yy - np.mean(yy)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            r2_arr[j] = r2
            
            # Standard error of slope a
            dof = int(n_valid) - 2
            if dof > 0:
                s2 = ss_res / dof
                Sxx = np.sum((x - np.mean(x)) ** 2)
                a_stderr = np.sqrt(s2 / Sxx) if (Sxx > 0 and s2 >= 0) else np.nan
            else:
                a_stderr = np.nan
            
            # Accept fit only when R^2 meets threshold
            if np.isfinite(r2) and (r2_min is None or r2 >= float(r2_min)):
                slopes[j] = a
                intercepts[j] = b
                Qst_kJmol[j] = -R * a / 1000.0
                slope_stderr[j] = a_stderr
                Qst_kJmol_stderr[j] = (R / 1000.0) * a_stderr if np.isfinite(a_stderr) else np.nan
        
        # Apply smoothing if requested - only at transitions where valid_counts changes
        if smooth:
            def _gaussian_smooth_1d_selective(arr, sigma, transition_mask):
                """
                Apply Gaussian smoothing only at transition points where valid_counts changes.
                transition_mask: boolean array indicating which points are at transitions
                """
                a = np.array(arr, dtype=float)
                n = a.size
                result = a.copy()  # Start with original values
                
                # Only smooth at transition points
                if not np.any(transition_mask):
                    return result
                
                # For each transition point, apply local smoothing
                x = np.arange(n)
                nans = np.isnan(a)
                
                # Interpolate NaNs for smoothing calculation
                if np.any(nans):
                    good = ~nans
                    if np.sum(good) >= 2:
                        a_interp = a.copy()
                        a_interp[nans] = np.interp(x[nans], x[good], a[good])
                    else:
                        return result
                else:
                    a_interp = a
                
                if sigma is None or sigma <= 0:
                    return result
                
                radius = int(max(1, int(3 * float(sigma))))
                
                # Apply smoothing only at transition points
                for idx in np.where(transition_mask)[0]:
                    # Define local window around transition point
                    start_idx = max(0, idx - radius)
                    end_idx = min(n, idx + radius + 1)
                    window_size = end_idx - start_idx
                    
                    if window_size < 2:
                        continue
                    
                    # Create local kernel for this window
                    center_offset = idx - start_idx
                    kx = np.arange(-center_offset, window_size - center_offset)
                    kernel = np.exp(-0.5 * (kx / float(sigma)) ** 2)
                    kernel = kernel / np.sum(kernel)
                    
                    # Extract local window
                    window = a_interp[start_idx:end_idx]
                    
                    # Apply convolution
                    if window_size > 1:
                        smoothed_val = np.sum(window * kernel)
                        result[idx] = smoothed_val
                
                # Restore NaNs where original was NaN
                result[nans] = np.nan
                return result
            
            try:
                sigma = float(smoothing_sigma)
            except Exception:
                sigma = 1.5
            
            slopes_arr = np.array(slopes, dtype=float)
            finite_mask = np.isfinite(slopes_arr)
            
            # Identify transition points where valid_counts changes
            try:
                vc = np.asarray(valid_counts, dtype=float)
                # Exclude points with insufficient temperatures
                sufficient_temps_mask = vc >= min_temps_req
                
                # Find transitions: points where valid_counts changes
                transition_mask = np.zeros(len(vc), dtype=bool)
                if len(vc) > 1:
                    # Check for changes in valid_counts
                    vc_diff = np.diff(vc)
                    # Mark points at transitions (both the point before and after the change)
                    transition_indices = np.where(vc_diff != 0)[0]
                    for trans_idx in transition_indices:
                        if trans_idx < len(transition_mask):
                            transition_mask[trans_idx] = True
                        if trans_idx + 1 < len(transition_mask):
                            transition_mask[trans_idx + 1] = True
                
                # Only smooth at transitions AND where we have sufficient temperatures
                smooth_mask = transition_mask & sufficient_temps_mask & finite_mask
            except Exception:
                # Fallback: no smoothing if we can't determine transitions
                smooth_mask = np.zeros(len(slopes_arr), dtype=bool)
            
            if np.sum(finite_mask) >= 2:
                if np.any(smooth_mask):
                    # There are transition points - apply selective smoothing
                    slopes_sm = _gaussian_smooth_1d_selective(slopes_arr, sigma, smooth_mask)
                    slopes_sm[~finite_mask] = np.nan
                    # Also exclude points with insufficient temperatures
                    try:
                        vc = np.asarray(valid_counts, dtype=float)
                        slopes_sm[vc < min_temps_req] = np.nan
                    except Exception:
                        pass
                else:
                    # No transition points - smoothed data is same as original (smoothing not needed)
                    slopes_sm = slopes_arr.copy()
                    slopes_sm[~finite_mask] = np.nan
                    # Also exclude points with insufficient temperatures
                    try:
                        vc = np.asarray(valid_counts, dtype=float)
                        slopes_sm[vc < min_temps_req] = np.nan
                    except Exception:
                        pass
                
                Qst_kJmol_smoothed = -R * slopes_sm / 1000.0
                slope_stderr_arr = np.array(slope_stderr, dtype=float)
                if np.any(np.isfinite(slope_stderr_arr)):
                    if np.any(smooth_mask):
                        # There are transition points - apply selective smoothing
                        slope_stderr_sm = _gaussian_smooth_1d_selective(slope_stderr_arr, sigma, smooth_mask)
                    else:
                        # No transition points - smoothed stderr is same as original
                        slope_stderr_sm = slope_stderr_arr.copy()
                    slope_stderr_sm[~np.isfinite(slope_stderr_arr)] = np.nan
                    # Exclude points with insufficient temperatures
                    try:
                        vc = np.asarray(valid_counts, dtype=float)
                        slope_stderr_sm[vc < min_temps_req] = np.nan
                    except Exception:
                        pass
                    Qst_kJmol_stderr_smoothed = (R / 1000.0) * slope_stderr_sm
                else:
                    Qst_kJmol_stderr_smoothed = np.full_like(Qst_kJmol_smoothed, np.nan)
            else:
                Qst_kJmol_smoothed = None
                Qst_kJmol_stderr_smoothed = None
        
        return {
            'loading': loadings,
            'Qst_kJmol': Qst_kJmol,
            'Qst_kJmol_stderr': Qst_kJmol_stderr,
            'Qst_kJmol_smoothed': Qst_kJmol_smoothed if 'Qst_kJmol_smoothed' in locals() else None,
            'Qst_kJmol_stderr_smoothed': Qst_kJmol_stderr_smoothed if 'Qst_kJmol_stderr_smoothed' in locals() else None,
            'slope': slopes,
            'slope_stderr': slope_stderr,
            'intercept': intercepts,
            'pressures': P_mat,
            'lnP': lnP_mat,
            'temperatures': t_array,
            'p_grid': None,  # Not applicable for direct interpolation
            'r2': r2_arr,
            'valid_counts': valid_counts
        }
    
    # ===== FIT-BASED APPROACH =====

    # Determine pressure grid
    if p_grid is None:
        p_grid = np.logspace(np.log10(p_min_bound), np.log10(p_max_bound), int(n_loadings) * 5)
    try:
        p_grid = np.asarray(
            list(p_grid) if not isinstance(p_grid, (np.ndarray, list, tuple)) else p_grid,
            dtype=float,
        )
    except Exception as exc:
        raise ValueError(f"invalid p_grid supplied: {exc}")
    p_grid = p_grid.ravel().astype(float)
    p_grid = p_grid[(p_grid >= p_min_bound) & (p_grid <= p_max_bound) & (p_grid > 0)]
    if p_grid.size == 0:
        raise ValueError("pressure grid is empty after applying p_min/p_max")

    # build fit cache (integer-tolerant)
    fit_cache = {}
    for fit in fittings:
        if fit['framework'] == framework and fit['molecule'].strip().lower() == molecule.strip().lower() and fit['fit_type'] in selected_fit_types:
            fit_cache.setdefault(round(float(fit['temperature'])), []).append(fit)

    # select one fit per requested temperature
    fit_rows = []
    for t in temps:
        rows = fit_cache.get(round(float(t)), [])
        if not rows:
            continue
        chosen = next((r for r in rows if r['fit_type'] == 'Langmuir_Freundlich'), rows[0])
        fit_rows.append((float(chosen['temperature']), chosen['fit_type'], np.array(chosen['params'], dtype=float)))

    if len(fit_rows) < 2:
        raise ValueError(f"Need at least two temperatures with fits for {framework},{molecule}; found {len(fit_rows)}")

    # evaluate q(p) for each selected fit on the p_grid
    q_t_list = []
    t_list = []
    for (T, ft_type, params) in fit_rows:
        q_vals = phelp.evaluate_fit(p_grid, params, ft_type)
        q_vals = np.array(q_vals, dtype=float)
        q_t_list.append(q_vals)
        t_list.append(float(T))

    q_t_array = np.vstack(q_t_list)   # shape (n_temps, n_p)
    t_array = np.array(t_list, dtype=float)

    # Determine loading range from pre-built RASPA_data rows if available,
    # otherwise fall back to the intersection of fit evaluations.
    if loadings is None:
        unified_loading_range = None
        if RASPA_data is not None:
            pts_for_range = phelp.filter_raspa_data(
                RASPA_data, frameworks=[framework], molecules=[molecule], temperatures=temps,
                only_pure_adsorption=True,
            )
            if pts_for_range:
                try:
                    n_lo, n_hi = ds._intersection_loading_range(pts_for_range)
                    unified_loading_range = (n_lo, n_hi)
                except Exception:
                    pass

        if unified_loading_range is not None:
            overall_min, overall_max = unified_loading_range
        else:
            loading_ranges = []
            for q_vals in q_t_list:
                finite_q = q_vals[np.isfinite(q_vals)]
                if finite_q.size > 0:
                    loading_ranges.append((np.min(finite_q), np.max(finite_q)))
            if loading_ranges:
                overall_min = max(r[0] for r in loading_ranges)
                overall_max = min(r[1] for r in loading_ranges)
            else:
                overall_min = 0.0
                overall_max = 0.0

        overall_min = max(overall_min, 1e-8)
        if overall_max <= overall_min:
            raise ValueError(f"Unable to determine a valid loading range for {framework},{molecule}")
        loadings = np.linspace(overall_min, overall_max, int(n_loadings))
    else:
        loadings = np.array(loadings, dtype=float)

    loadings = np.ravel(loadings).astype(float)
    if loadings.size > 1:
        loadings = np.sort(loadings)

    nL = loadings.size
    nT = len(t_array)

    # build matrices for pressures and ln pressures per (T, loading)
    P_mat = np.full((nT, nL), np.nan)
    lnP_mat = np.full((nT, nL), np.nan)

    # For formula-based approach: use direct formula inversion to find P at each loading
    # This numerically solves q = f(P) for P, using the analytical fit formula
    for i, (T, ft_type, params) in enumerate(fit_rows):
        # Use inverse_fit to find P for each target loading
        # This directly inverts the fit formula without interpolation
        p_values = phelp.inverse_fit(loadings, params, ft_type, 
                                      p_min=p_min_bound * 0.01, p_max=p_max_bound * 100)
        p_values = np.asarray(p_values, dtype=float)
        
        # Apply pressure bounds filtering: reject pressures outside valid range
        valid_pressure = (p_values >= p_min_bound) & (p_values <= p_max_bound) & np.isfinite(p_values)
        p_values_filtered = np.where(valid_pressure, p_values, np.nan)
        
        P_mat[i, :] = p_values_filtered
        with np.errstate(divide='ignore', invalid='ignore'):
            lnP_mat[i, :] = np.log(p_values_filtered)

    valid_counts = np.sum(np.isfinite(lnP_mat), axis=0)

    # prepare outputs
    slopes = np.full(nL, np.nan)
    intercepts = np.full(nL, np.nan)
    Qst_kJmol = np.full(nL, np.nan)
    r2_arr = np.full(nL, np.nan)
    slope_stderr = np.full(nL, np.nan)
    Qst_kJmol_stderr = np.full(nL, np.nan)

    invT = 1.0 / t_array

    # determine minimum number of temperatures required for regression:
    # use the user-specified `min_temps` directly (caller controls the threshold)
    min_temps_req = int(min_temps)

    for j in range(nL):
        y = lnP_mat[:, j]
        valid = np.isfinite(y) & np.isfinite(invT)
        n_valid = np.sum(valid)
        if n_valid < min_temps_req:
            continue

        x = invT[valid]
        yy = y[valid]

        # linear fit ln(p) = a*(1/T) + b
        a, b = np.polyfit(x, yy, 1)
        y_fit = a * x + b
        ss_res = np.sum((yy - y_fit) ** 2)
        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_arr[j] = r2

        # standard error of slope a
        dof = int(n_valid) - 2
        if dof > 0:
            s2 = ss_res / dof
            Sxx = np.sum((x - np.mean(x)) ** 2)
            a_stderr = np.sqrt(s2 / Sxx) if (Sxx > 0 and s2 >= 0) else np.nan
        else:
            a_stderr = np.nan

        # accept fit only when R^2 meets threshold (or r2_min is None to disable)
        if np.isfinite(r2) and (r2_min is None or r2 >= float(r2_min)):
            slopes[j] = a
            intercepts[j] = b
            Qst_kJmol[j] = -R * a / 1000.0
            slope_stderr[j] = a_stderr
            Qst_kJmol_stderr[j] = (R / 1000.0) * a_stderr if np.isfinite(a_stderr) else np.nan
        else:
            # leave NaN if rejected
            continue

    # Optionally smooth by applying a Gaussian filter to the regression
    # slopes `a`, then converting back to Qst=-R*a/1000. This preserves the
    # physical relation and avoids smoothing across unsupported points.
    # Smoothing only occurs at transitions where valid_counts changes.
    if smooth:
        def _gaussian_smooth_1d_selective(arr, sigma, transition_mask):
            """
            Apply Gaussian smoothing only at transition points where valid_counts changes.
            transition_mask: boolean array indicating which points are at transitions
            """
            a = np.array(arr, dtype=float)
            n = a.size
            result = a.copy()  # Start with original values
            
            # Only smooth at transition points
            if not np.any(transition_mask):
                return result
            
            # For each transition point, apply local smoothing
            x = np.arange(n)
            nans = np.isnan(a)
            
            # Interpolate NaNs for smoothing calculation
            if np.any(nans):
                good = ~nans
                if np.sum(good) >= 2:
                    a_interp = a.copy()
                    a_interp[nans] = np.interp(x[nans], x[good], a[good])
                else:
                    return result
            else:
                a_interp = a
            
            if sigma is None or sigma <= 0:
                return result
            
            radius = int(max(1, int(3 * float(sigma))))
            
            # Apply smoothing only at transition points
            for idx in np.where(transition_mask)[0]:
                # Define local window around transition point
                start_idx = max(0, idx - radius)
                end_idx = min(n, idx + radius + 1)
                window_size = end_idx - start_idx
                
                if window_size < 2:
                    continue
                
                # Create local kernel for this window
                center_offset = idx - start_idx
                kx = np.arange(-center_offset, window_size - center_offset)
                kernel = np.exp(-0.5 * (kx / float(sigma)) ** 2)
                kernel = kernel / np.sum(kernel)
                
                # Extract local window
                window = a_interp[start_idx:end_idx]
                
                # Apply convolution
                if window_size > 1:
                    smoothed_val = np.sum(window * kernel)
                    result[idx] = smoothed_val
            
            # Restore NaNs where original was NaN
            result[nans] = np.nan
            return result

        try:
            sigma = float(smoothing_sigma)
        except Exception:
            sigma = 1.5

        slopes_arr = np.array(slopes, dtype=float)
        finite_mask = np.isfinite(slopes_arr)
        
        # Identify transition points where valid_counts changes
        try:
            vc = np.asarray(valid_counts, dtype=float)
            # Exclude points with insufficient temperatures
            sufficient_temps_mask = vc >= min_temps_req
            
            # Find transitions: points where valid_counts changes
            transition_mask = np.zeros(len(vc), dtype=bool)
            if len(vc) > 1:
                # Check for changes in valid_counts
                vc_diff = np.diff(vc)
                # Mark points at transitions (both the point before and after the change)
                transition_indices = np.where(vc_diff != 0)[0]
                for trans_idx in transition_indices:
                    if trans_idx < len(transition_mask):
                        transition_mask[trans_idx] = True
                    if trans_idx + 1 < len(transition_mask):
                        transition_mask[trans_idx + 1] = True
            
            # Only smooth at transitions AND where we have sufficient temperatures
            smooth_mask = transition_mask & sufficient_temps_mask & finite_mask
        except Exception:
            # Fallback: no smoothing if we can't determine transitions
            smooth_mask = np.zeros(len(slopes_arr), dtype=bool)
        
        if np.sum(finite_mask) >= 2:
            if np.any(smooth_mask):
                # There are transition points - apply selective smoothing
                slopes_sm = _gaussian_smooth_1d_selective(slopes_arr, sigma, smooth_mask)
            else:
                # No transition points - smoothed data is same as original (smoothing not needed)
                slopes_sm = slopes_arr.copy()
            # restore NaNs where original slopes were NaN
            slopes_sm[~finite_mask] = np.nan
            # mask by valid_counts threshold
            try:
                vc = np.asarray(valid_counts, dtype=float)
                slopes_sm[vc < min_temps_req] = np.nan
            except Exception:
                pass

            Qst_kJmol_smoothed = -R * slopes_sm / 1000.0
            # smooth slope stderr if available
            slope_stderr_arr = np.array(slope_stderr, dtype=float)
            if np.any(np.isfinite(slope_stderr_arr)):
                if np.any(smooth_mask):
                    # There are transition points - apply selective smoothing
                    slope_stderr_sm = _gaussian_smooth_1d_selective(slope_stderr_arr, sigma, smooth_mask)
                else:
                    # No transition points - smoothed stderr is same as original
                    slope_stderr_sm = slope_stderr_arr.copy()
                slope_stderr_sm[~np.isfinite(slope_stderr_arr)] = np.nan
                # Exclude points with insufficient temperatures
                try:
                    vc = np.asarray(valid_counts, dtype=float)
                    slope_stderr_sm[vc < min_temps_req] = np.nan
                except Exception:
                    pass
                Qst_kJmol_stderr_smoothed = (R / 1000.0) * slope_stderr_sm
            else:
                Qst_kJmol_stderr_smoothed = np.full_like(Qst_kJmol_smoothed, np.nan)
        else:
            Qst_kJmol_smoothed = None
            Qst_kJmol_stderr_smoothed = None

    return {
        'loading': loadings,
        'Qst_kJmol': Qst_kJmol,
        'Qst_kJmol_stderr': Qst_kJmol_stderr,
        'Qst_kJmol_smoothed': Qst_kJmol_smoothed if 'Qst_kJmol_smoothed' in locals() else None,
        'Qst_kJmol_stderr_smoothed': Qst_kJmol_stderr_smoothed if 'Qst_kJmol_stderr_smoothed' in locals() else None,
        'slope': slopes,
        'slope_stderr': slope_stderr,
        'intercept': intercepts,
        'pressures': P_mat,
        'lnP': lnP_mat,
        'temperatures': t_array,
        'p_grid': p_grid,
        'r2': r2_arr,
        'valid_counts': valid_counts
    }

def plot_clausius_clapeyron(selected_frameworks, selected_molecules, temperatures, selected_fit_types, 
                fittings, RASPA_data=None, x_fit=None, loadings=None, n_loadings=60, p_min=None, p_max=None, 
                show_RASPA=False, r2_min=0.95, show_calc_points=True, plot_smoothed=True, out_dir=None,
                use_direct_interpolation=False, smooth=False, smoothing_sigma=1.5, 
                show_original=True, plot_suffix='', num_of_isotherm=None, method_linestyles=None, combo_colors=None,
                hoa_scatter_data=None, save_data=False):
    if x_fit is None:
        x_fit = np.logspace(np.log10(0.5), 7, 300)

    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    any_plotted = False
    # Points that define the final CC curve (smoothed or not), for optional export
    export_rows = []
    global_max_loading = 0.0  # Track maximum loading across all lines
    global_min_qst = np.inf  # Track minimum Qst across all lines
    global_max_qst = -np.inf  # Track maximum Qst across all lines

    for fw in selected_frameworks:
        for mol in selected_molecules:
            # When not using direct interpolation, find all unique (fit_type, num_params) combinations
            # and plot each separately to avoid mixing
            if not use_direct_interpolation:
                # Determine which parameter counts to accept based on num_of_isotherm
                if num_of_isotherm is None:
                    allowed_param_counts = None  # Allow any
                elif isinstance(num_of_isotherm, list):
                    allowed_param_counts = [n * 3 for n in num_of_isotherm]
                else:
                    allowed_param_counts = [num_of_isotherm * 3]
                
                # Find all unique (fit_type, num_params) combinations available in the fittings
                # that match the user's selection
                fit_combinations = set()
                for fit in fittings:
                    if fit['framework'] == fw and fit['molecule'].strip().lower() == mol.strip().lower() and fit['fit_type'] in selected_fit_types:
                        num_params = len(fit['params'])
                        # Only include if matches the requested num_of_isotherm
                        if allowed_param_counts is None or num_params in allowed_param_counts:
                            fit_combinations.add((fit['fit_type'], num_params))
                
                # Sort for consistent ordering
                fit_combinations = sorted(list(fit_combinations))
                
                if not fit_combinations:
                    print(f"Clausius-Clapeyron: no fits found for {fw},{mol} matching the requested fit_types and num_of_isotherm")
                    continue
            else:
                # Direct interpolation - just one pass
                fit_combinations = [(None, None)]
            
            for fit_type, num_params in fit_combinations:
                try:
                    # Adjust min_temps to match available temperatures (need at least 2, but not more than available)
                    min_temps_adj = max(2, min(3, len(temperatures)))
                    # If plot_smoothed is True, we need smoothed data even if smooth parameter is False
                    smooth_for_compute = smooth or plot_smoothed
                    data = compute_isosteric_heat(fw, mol, temperatures, selected_fit_types, fittings,
                                                  RASPA_data=RASPA_data, loadings=loadings, n_loadings=n_loadings,
                                                  p_grid=x_fit, p_min=p_min, p_max=p_max, r2_min=r2_min, smooth=smooth_for_compute,
                                                  use_direct_interpolation=use_direct_interpolation, min_temps=min_temps_adj)
                except Exception as e:
                    if fit_type is not None:
                        print(f"Clausius-Clapeyron: skipping {fw},{mol} with {fit_type} ({num_params} params): {e}")
                    else:
                        print(f"Clausius-Clapeyron: skipping {fw},{mol}: {e}")
                    continue

                loading = data.get('loading')
                Qst = data.get('Qst_kJmol')
                Qst_err = data.get('Qst_kJmol_stderr')
                slopes_arr = data.get('slope')
                r2_arr = data.get('r2')
                valid_counts = data.get('valid_counts')
                if loading is None or Qst is None:
                    if fit_type is not None:
                        print(f"Clausius-Clapeyron: no data for {fw},{mol} with {fit_type} ({num_params} params) - skipping")
                    else:
                        print(f"Clausius-Clapeyron: no data for {fw},{mol} - skipping")
                    continue

                mask = np.isfinite(loading) & np.isfinite(Qst)
                if not np.any(mask):
                    if fit_type is not None:
                        print(f"Clausius-Clapeyron: no finite Qst for {fw},{mol} with {fit_type} ({num_params} params) - skipping")
                    else:
                        print(f"Clausius-Clapeyron: no finite Qst for {fw},{mol} - skipping")
                    continue
                
                # Track maximum loading and Qst range from this line
                if len(loading[mask]) > 0:
                    global_max_loading = max(global_max_loading, np.max(loading[mask]))
                    global_min_qst = min(global_min_qst, np.min(Qst[mask]))
                    global_max_qst = max(global_max_qst, np.max(Qst[mask]))

                # plot original Qst curve (if show_original is True)
                clean_fw = phelp.clean_material_name(fw)
                clean_mol = phelp.get_molecule_display_name(mol)

                # Vary flags for legend compression
                vary_fw = len(selected_frameworks or []) > 1
                vary_mol = len(selected_molecules or []) > 1
                vary_T = len(temperatures or []) > 1

                # Build label core (fw/mol/T as needed)
                core_label = phelp.build_series_label(
                    fw, mol, None,
                    vary_fw=vary_fw, vary_mol=vary_mol, vary_T=False,
                )

                # Append fit-type / site info when relevant
                if fit_type is not None and num_params is not None:
                    fit_type_label = fit_type.replace('_', ' ')
                    num_sites = num_params // 3
                    base_label = core_label
                else:
                    base_label = core_label
                
                # Decide which curve defines the exported / visual Qst values:
                # if smoothed data is available and requested, use that; otherwise use raw Qst.
                Qst_sm = data.get('Qst_kJmol_smoothed')
                use_smoothed = plot_smoothed and Qst_sm is not None
                # Keep HoA color mapping consistent across original and smoothed curves.
                cc_color = phelp.get_color_for_molecule(mol)

                if show_original:
                    # HoA rule: color encodes adsorbate (molecule), linestyle encodes method
                    cc_ls = phelp.get_hoa_linestyle(
                        fw, 'clausius_clapeyron', 'structure', method_linestyles=method_linestyles
                    )
                    marker = phelp.get_marker_for_molecule(mol) if len(selected_molecules or []) > 1 else 'o'
                    orig_line, = ax.plot(
                        loading[mask], Qst[mask],
                        marker=marker, linestyle=cc_ls, label="_nolegend_",
                        color=cc_color, lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, markersize=phelp.MARKER_SIZE
                    )
                    
                    # Track maximum loading and Qst range from original line
                    if len(loading[mask]) > 0:
                        global_max_loading = max(global_max_loading, np.max(loading[mask]))
                        global_min_qst = min(global_min_qst, np.min(Qst[mask]))
                        global_max_qst = max(global_max_qst, np.max(Qst[mask]))
                    
                any_plotted = True

                # Plot smoothed curve if requested
                if plot_smoothed and Qst_sm is not None:
                    try:
                        Qst_sm = np.asarray(Qst_sm, dtype=float)
                        mask_sm = np.isfinite(loading) & np.isfinite(Qst_sm)
                        if np.any(mask_sm):
                            # Label depends on whether we're showing original line or not
                            if show_original:
                                sm_label = "_nolegend_"
                                sm_linestyle = '--'
                            else:
                                sm_label = "_nolegend_"
                                sm_linestyle = phelp.get_hoa_linestyle(
                                    fw, 'clausius_clapeyron', 'structure', method_linestyles=method_linestyles
                                )
                            sm_line, = ax.plot(
                                loading[mask_sm], Qst_sm[mask_sm],
                                marker='', linestyle=sm_linestyle, lw=phelp.LINEWIDTH,
                                alpha=phelp.ALPHA, label=sm_label,
                                color=cc_color
                            )
                            # Track maximum loading and Qst range from smoothed line
                            if len(loading[mask_sm]) > 0:
                                global_max_loading = max(global_max_loading, np.max(loading[mask_sm]))
                                global_min_qst = min(global_min_qst, np.min(Qst_sm[mask_sm]))
                                global_max_qst = max(global_max_qst, np.max(Qst_sm[mask_sm]))
                            any_plotted = True

                    except Exception:
                        pass

                # Collect export points for the curve that is actually used:
                # smoothed if available and requested, otherwise raw Qst.
                export_Qst = np.asarray(Qst, dtype=float)
                export_loading = np.asarray(loading, dtype=float)
                mask_export = np.isfinite(export_loading) & np.isfinite(export_Qst)
                for L, q_val in zip(export_loading[mask_export], export_Qst[mask_export]):
                    export_rows.append({
                    'framework': fw,
                    'molecule': mol,
                    'loading': float(L),
                    'Qst_kJmol': float(q_val),
                })

                if show_calc_points:
                    marker = phelp.get_marker_for_material(fw)
                    cc_color = phelp.get_color_for_molecule(mol)
                    hoa_scattered = False

                    if hoa_scatter_data:
                        try:
                            hoa_rows = [
                                r for r in hoa_scatter_data
                                if str(r.get('framework', '')).strip() == str(fw).strip()
                                and str(r.get('molecule', '')).strip().lower() == str(mol).strip().lower()
                            ]
                            if hoa_rows:
                                loads_hoa = np.asarray([float(r.get('loading')) for r in hoa_rows], dtype=float)
                                qst_hoa = np.asarray([float(r.get('qst_kjmol')) for r in hoa_rows], dtype=float)
                                mask_hoa = np.isfinite(loads_hoa) & np.isfinite(qst_hoa) & (loads_hoa > 0)
                                if np.any(mask_hoa):
                                    ax.scatter(
                                        loads_hoa[mask_hoa], qst_hoa[mask_hoa],
                                        marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                                        color=cc_color, zorder=6,
                                    )
                                    hoa_scattered = True
                        except Exception:
                            hoa_scattered = False

                    if not hoa_scattered:
                        # Get the pressure and loading matrices
                        P_mat = data.get('pressures')
                        lnP_mat = data.get('lnP')
                        temps_array = data.get('temperatures')
                        loads_all = np.array(data.get('loading', []), dtype=float)
                        qst_all = np.array(data.get('Qst_kJmol', []), dtype=float)

                        if P_mat is not None and lnP_mat is not None and temps_array is not None:
                            # Extract all valid (loading, Qst) points that were used in the calculation.
                            for j, load_val in enumerate(loads_all):
                                if j >= len(qst_all) or not np.isfinite(qst_all[j]):
                                    continue
                                if j < P_mat.shape[1]:
                                    pressures_at_loading = P_mat[:, j]
                                    valid_temp_mask = np.isfinite(pressures_at_loading) & (pressures_at_loading > 0)
                                    if np.any(valid_temp_mask):
                                        n_valid = int(np.sum(valid_temp_mask))
                                        ax.scatter(
                                            [load_val] * n_valid, [qst_all[j]] * n_valid,
                                            marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                                            color=cc_color, zorder=5,
                                        )

                        # Also show points based on valid_counts (original behavior)
                        if valid_counts is not None:
                            vc = np.array(valid_counts, dtype=float)
                            mask_pts = np.isfinite(loads_all) & np.isfinite(qst_all) & (vc > 0)
                            if np.any(mask_pts):
                                sizes = np.clip(3.0 * vc[mask_pts], 5, 60)
                                ax.scatter(
                                    loads_all[mask_pts], qst_all[mask_pts],
                                    marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                                    color=cc_color, zorder=6,
                                )

                        if show_RASPA and RASPA_data is not None:
                            pts = phelp.filter_raspa_data(
                                RASPA_data, frameworks=[fw], molecules=[mol],
                                only_pure_adsorption=True,
                            )
                            pts = [p for p in pts if p.get('enthalpy') is not None and p.get('loading') is not None]
                            if pts:
                                loads = np.array([p['loading'] for p in pts], dtype=float)
                                enths = np.array([p['enthalpy'] for p in pts], dtype=float)
                                ax.scatter(loads, enths, marker='.', color='k', s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA, label=f"RASPA H: {fw},{mol}")

        if not any_plotted:
            print("Clausius-Clapeyron: nothing plotted (no valid fits)")
            return

        ax.set_xlabel('Loading [mol/kg]', **_cc_axis_label_kwargs())
        ax.set_ylabel('Heat of adsorption [kJ/mol]', **_cc_axis_label_kwargs())

        # X-axis: start at 0, round max up to next multiple of 2, tick every 2 units.
        _x_max = math.ceil(max(global_max_loading, 0.1) / 2) * 2
        ax.set_xlim(left=0, right=_x_max)
        ax.xaxis.set_major_locator(MultipleLocator(2))

        # Y-axis: 15 % margin below data min, 5 % above data max.
        _y_range = (global_max_qst - global_min_qst) if (np.isfinite(global_min_qst) and np.isfinite(global_max_qst)) else 1.0
        ax.set_ylim(
            bottom=global_min_qst - 0.15 * _y_range,
            top=global_max_qst + 0.05 * _y_range,
        )

        ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)

        # ax.set_title('Clausius-Clapeyron')
        phelp.build_hoa_proxy_legend(
            ax,
            molecules_present=selected_molecules,
            frameworks_present=selected_frameworks,
            methods_present=['clausius_clapeyron'],
            method_linestyles=method_linestyles,
            fontsize=phelp.AXIS_LEGEND_SIZE,
            loc='best',
        )
    phelp.apply_unified_axes_layout(fig, ax)
    # Use plot_suffix to distinguish between different versions of the plot
    plot_name = 'clausius_clapeyron' + plot_suffix
    out_path = phelp._save_plot(plot_name, 'plot_clausius_clapeyron', selected_frameworks, selected_molecules, temperatures, fig=fig, out_dir=out_dir)
    # Save the points defining the final CC curve (smoothed or raw) independently of show_original.
    if export_rows and save_data and out_path is not None:
        try:
            base = Path(out_path)
            saved_dir = base.parent / "saved"
            saved_dir.mkdir(parents=True, exist_ok=True)
            data_path = saved_dir / (base.stem + '_data.txt')
            with data_path.open('w', encoding='utf-8') as f:
                f.write("framework\tmolecule\tloading_mol_per_kg\tQst_kJmol\n")
                for r in export_rows:
                    f.write(
                        f"{r['framework']}\t{r['molecule']}\t"
                        f"{r['loading']}\t{r['Qst_kJmol']}\n"
                    )
        except Exception as e:
            print(f"Warning: failed to write Clausius-Clapeyron data file next to {out_path}: {e}")
    plt.show()
    plt.close(fig)

def compute_mixture_isosteric_heat_cc(mixture_data, components, temperatures, framework, mixture_name,
                                      p_min, p_max, n_loadings=40, min_temps=2,
                                      trim_components_overlap=True,
                                      trim_components_ordering_cutoff=True,
                                      trim_past_tmin_crossover=True,
                                      trim_past_tmin_crossover_total=False,
                                      trim_mixture_total_ordering_cutoff=False):
    """
    Mixture isosteric heat from Clausius-Clapeyron for mixed-gas adsorption:
    Q_st,i^mix = R*T^2 * (d ln(P*y_i)/dT)_{n_i} = -R * d(ln(P*y_i))/d(1/T),
    where i is the component, y_i = q_i / q_total is the adsorbed-phase mole fraction (Hamid et al.).
    For the total mixture heat, y_i=1: fit ln(P) vs 1/T at constant total loading.
    Returns dict: {comp: {'loading': array, 'Qst_kJmol': array}, 'all': {'loading': array, 'Qst_kJmol': array}}.

    When ``trim_past_tmin_crossover`` is True (default), per-component crossover trimming runs: for each
    component ``c`` alone, walk increasing P on ``c``'s lowest-T isotherm and drop pressures above the first
    P where ``q(c,T_lo,P) < q(c,T,P)`` for any warmer ``T``. Only ``data_by_temp[T][c]`` is filtered — other
    components keep their own pressure ranges.

    **Mixture total:** ``data_by_temp[T]['all']`` (q_tot vs P) is filled from the saved
       ``mixture_isotherm_log.txt`` when that file exists and contains mixture-total rows
       (``molecule`` = mixture name, or legacy ``__MIXTURE_TOTAL__``) for every temperature (see
       ``IsothermFittingPlot.plot_mixture_isotherms`` with ``save_data``). Otherwise it falls back to the
       same sum-at-each-pressure convention as the ``mixture_isotherm_total_*`` figure.

       **Optional trims on ``'all'`` only:** ``trim_past_tmin_crossover_total`` (default False) and
       ``trim_mixture_total_ordering_cutoff`` (default False) apply the same physical-ordering cuts as for
       smooth single-component isotherms. Sum-at-P mixture totals often violate strict
       ``q(T_cold,P) >= q(T_warm,P)`` at identical ``P`` because species grids do not align, which would
       otherwise truncate ``'all'`` to a tiny pressure window and break the mixture-total CC curve.

    Total ``q_tot(P)`` rows are read from ``Output/<run>/Basic_Data/saved/mixture_isotherm_log.txt``
    when present (same file written by ``plot_mixture_isotherms`` with ``save_data``).

    Partial-molar CC still interpolates ``q_tot`` from these ``'all'`` curves; if totals are shorter than a
    component isotherm in P, some high-loading points can miss ``q_tot`` (NaN / skipped) — expected.
    """
    p_min_b = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b = float(p_max) if p_max is not None else 1e8
    temps = [float(t) for t in temperatures]
    result = {}

    def _safe_interp(x_query, x_data, y_data):
        """Interpolate y at x_query; sorts x, deduplicates, returns NaN out-of-bounds."""
        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        order = np.argsort(x_arr)
        x_arr, y_arr = x_arr[order], y_arr[order]
        _, u = np.unique(x_arr, return_index=True)
        x_arr, y_arr = x_arr[u], y_arr[u]
        if len(x_arr) < 2 or x_query < x_arr[0] or x_query > x_arr[-1]:
            return np.nan
        return float(np.interp(x_query, x_arr, y_arr))

    # Build per-component data; total loading derived by summing components at shared pressures.
    # Data field is 'mixture_pure' (from Initialize.load_RASPA_data).
    data_by_temp = {}
    for T in temps:
        data_by_temp[T] = {c: [] for c in components}
    for d in mixture_data:
        t_fw  = d.get('framework')
        t_mix = d.get('mixture_pure', d.get('mixture'))
        if t_fw != framework or t_mix != mixture_name:
            continue
        T = float(d['temperature'])
        if T not in data_by_temp:
            continue
        P = float(d.get('pressure', np.nan))
        q = float(d.get('loading', np.nan))
        if not (np.isfinite(P) and np.isfinite(q) and P > 0 and q >= 0 and p_min_b <= P <= p_max_b):
            continue
        mol = d.get('molecule')
        if mol in components:
            data_by_temp[T][mol].append((P, q))

    def _pq_sorted_unique(pts):
        if len(pts) < 2:
            return None, None
        Pv = np.array([x[0] for x in pts], dtype=float)
        qv = np.array([x[1] for x in pts], dtype=float)
        order = np.argsort(Pv)
        Pv, qv = Pv[order], qv[order]
        _, u = np.unique(Pv, return_index=True)
        return Pv[u], qv[u]

    repo_root = init.get_pipeline_run_root()

    def _resolve_mixture_total_log_path():
        return phelp.mixture_isotherm_saved_log_path(
            repo_root, [framework], [mixture_name], temperatures
        )

    def _populate_mixture_total_rows():
        """Set ``data_by_temp[T]['all']`` from saved isotherm log or sum-at-P fallback."""
        path = _resolve_mixture_total_log_path()
        loaded = {}
        if path.is_file():
            loaded = phelp.load_mixture_total_from_isotherm_log(
                path, framework, p_min_b, p_max_b, mixture_name=mixture_name
            )
        if phelp.mixture_total_log_covers_temperatures(loaded, temps, tol=1.0, min_points=2):
            for T in temps:
                key = None
                for kk in loaded:
                    if abs(float(kk) - float(T)) <= 1.0:
                        key = kk
                        break
                if key is not None:
                    data_by_temp[T]['all'] = list(loaded[key])
            return
        if path.is_file():
            print(
                f"Warning: mixture CC '{framework}': '{path.name}' has no usable "
                f"mixture-total rows (molecule '{mixture_name}' or legacy tag) for all temperatures; "
                "using sum-at-P totals.",
                flush=True,
            )
        for T in temps:
            data_by_temp[T]['all'] = phelp.mixture_total_pq_tuples(
                mixture_data, framework, T, components,
                p_min=p_min_b, p_max=p_max_b, require_positive_pressure=True,
            )

    # Drop pressures past where T_min loading falls below a warmer T at the same P (per component).
    if trim_past_tmin_crossover and len(temps) >= 2:
        temps_asc = sorted(temps)
        T_lo = temps_asc[0]
        for c in components:
            pts_lo = data_by_temp.get(T_lo, {}).get(c, [])
            P_ref, _ = _pq_sorted_unique(pts_lo)
            if P_ref is None or len(P_ref) < 2:
                continue
            pq_T = {}
            for T in temps_asc:
                Pv, qv = _pq_sorted_unique(data_by_temp[T].get(c, []))
                if Pv is not None and len(Pv) >= 2:
                    pq_T[T] = (Pv, qv)
            if T_lo not in pq_T:
                continue
            p_cut = np.inf
            for k, P in enumerate(P_ref):
                q_lo = _safe_interp(float(P), pq_T[T_lo][0], pq_T[T_lo][1])
                if not np.isfinite(q_lo):
                    continue
                crossed = False
                for T in temps_asc[1:]:
                    if T not in pq_T:
                        continue
                    q_T = _safe_interp(float(P), pq_T[T][0], pq_T[T][1])
                    if np.isfinite(q_T) and q_lo < q_T:
                        crossed = True
                        break
                if crossed:
                    p_cut = float(P_ref[k - 1]) if k > 0 else np.inf
                    break
            if np.isfinite(p_cut):
                for T in temps:
                    data_by_temp[T][c] = [
                        (P, q) for P, q in data_by_temp[T][c]
                        if float(P) <= p_cut
                    ]

    _populate_mixture_total_rows()

    # Mixture-total only: pts_all for partial-molar CC comes from these rows; component (P,q) unchanged.
    # Optional: same T_lo vs warmer crossover on ``'all'`` (usually leave off for sum-at-P / log totals).
    if trim_past_tmin_crossover and trim_past_tmin_crossover_total and len(temps) >= 2:
        temps_asc = sorted(temps)
        T_lo = temps_asc[0]
        pts_all_lo = data_by_temp.get(T_lo, {}).get('all', [])
        P_ref_all, _ = _pq_sorted_unique(pts_all_lo)
        if P_ref_all is not None and len(P_ref_all) >= 2:
            pq_tot = {}
            for T in temps_asc:
                Pv, qv = _pq_sorted_unique(data_by_temp.get(T, {}).get('all', []))
                if Pv is not None and len(Pv) >= 2:
                    pq_tot[T] = (Pv, qv)
            if T_lo in pq_tot:
                p_cut_all = np.inf
                for k, P in enumerate(P_ref_all):
                    q_lo = _safe_interp(float(P), pq_tot[T_lo][0], pq_tot[T_lo][1])
                    if not np.isfinite(q_lo):
                        continue
                    crossed = False
                    for T in temps_asc[1:]:
                        if T not in pq_tot:
                            continue
                        q_T = _safe_interp(float(P), pq_tot[T][0], pq_tot[T][1])
                        if np.isfinite(q_T) and q_lo < q_T:
                            crossed = True
                            break
                    if crossed:
                        p_cut_all = float(P_ref_all[k - 1]) if k > 0 else np.inf
                        break
                if np.isfinite(p_cut_all):
                    for T in temps:
                        data_by_temp[T]['all'] = [
                            (P, q) for P, q in data_by_temp[T].get('all', [])
                            if float(P) <= p_cut_all
                        ]

    # Per-component Q_st,i^mix at constant n_i: fit ln(P*y_i) vs 1/T
    for comp in components:
        by_t = {}
        for T in temps:
            pts = data_by_temp[T].get(comp, [])
            pts_all = data_by_temp[T].get('all', [])
            if len(pts) < 3 or len(pts_all) < 3:
                continue
            P_c = np.array([x[0] for x in pts])
            q_c = np.array([x[1] for x in pts])
            order = np.argsort(P_c)
            P_c = P_c[order]
            q_c = q_c[order]
            P_a = np.array([x[0] for x in pts_all])
            q_a = np.array([x[1] for x in pts_all])
            order_a = np.argsort(P_a)
            P_a = P_a[order_a]
            q_a = q_a[order_a]
            by_t[T] = (P_c, q_c, P_a, q_a)
        if len(by_t) < min_temps:
            continue
        # Determine the component loading range.
        # - With overlap trimming enabled: use overlap intersection (min..max over overlap).
        # - With overlap trimming disabled: use union range across all temperatures available.
        if trim_components_overlap:
            q_mins = [np.min(by_t[T][1]) for T in temps if T in by_t]
            q_maxs = [np.max(by_t[T][1]) for T in temps if T in by_t]
            if not q_mins or not q_maxs or min(q_maxs) <= max(q_mins):
                continue
            q_lower = max(q_mins)
            q_upper = min(q_maxs)
        else:
            q_vals_min = [np.min(by_t[T][1]) for T in by_t]
            q_vals_max = [np.max(by_t[T][1]) for T in by_t]
            if not q_vals_min or not q_vals_max:
                continue
            q_lower = float(min(q_vals_min))
            # No overlap trimming for relaxed component curves:
            # use the union range across temperatures.
            q_upper = float(max(q_vals_max))
            if q_upper <= q_lower:
                continue

        # Optional ordering cutoff trimming (CC-invalid multi-valued region).
        cutoff_P = np.inf
        if trim_components_ordering_cutoff:
            sorted_T_in = sorted(T for T in temps if T in by_t)
            if len(sorted_T_in) >= 2:
                P_ref = by_t[sorted_T_in[0]][0]
                for k, P in enumerate(P_ref):
                    q_prev = None
                    ordering_ok = True
                    for T in sorted_T_in:
                        q_T = _safe_interp(P, by_t[T][0], by_t[T][1])
                        if not np.isfinite(q_T):
                            continue
                        if q_prev is not None and q_T > q_prev:
                            ordering_ok = False
                            break
                        q_prev = q_T
                    if not ordering_ok:
                        cutoff_P = P_ref[k - 1] if k > 0 else P_ref[0]
                        q_upper = _safe_interp(
                            cutoff_P,
                            by_t[sorted_T_in[0]][0],
                            by_t[sorted_T_in[0]][1],
                        )
                        break

            if np.isfinite(cutoff_P):
                for T in sorted_T_in:
                    P_c_t, q_c_t, P_a_t, q_a_t = by_t[T]
                    mask_c = P_c_t <= cutoff_P
                    mask_a = P_a_t <= cutoff_P
                    # Keep only if we still have at least 2 points for
                    # interpolation stability.
                    if np.sum(mask_c) >= 2 and np.sum(mask_a) >= 2:
                        by_t[T] = (P_c_t[mask_c], q_c_t[mask_c], P_a_t[mask_a], q_a_t[mask_a])

        # After optional ordering-cutoff slicing, recompute the loading bounds
        # based on the actually trimmed `by_t` ranges. This avoids generating
        # loadings that fall into CC-invalid regions.
        if trim_components_ordering_cutoff:
            if trim_components_overlap:
                q_mins = [np.min(by_t[T][1]) for T in temps if T in by_t and len(by_t[T][1]) > 0]
                q_maxs = [np.max(by_t[T][1]) for T in temps if T in by_t and len(by_t[T][1]) > 0]
                if not q_mins or not q_maxs or min(q_maxs) <= max(q_mins):
                    continue
                q_lower = max(q_mins)
                q_upper = min(q_maxs)
            else:
                q_vals_min = [np.min(by_t[T][1]) for T in by_t if len(by_t[T][1]) > 0]
                q_vals_max = [np.max(by_t[T][1]) for T in by_t if len(by_t[T][1]) > 0]
                if not q_vals_min or not q_vals_max:
                    continue
                q_lower = float(min(q_vals_min))
                # For relaxed component curves we still want the lower bound
                # to be as wide as possible, but the upper bound must be
                # capped so that all temperatures keep coverage up to the
                # CC ordering cutoff (otherwise we see unstable "kinks"
                # near the component's high-loading end).
                q_upper = float(min(q_vals_max))

        # If overlap trimming is enabled, keep the original small safety margin clipping.
        if trim_components_overlap:
            q_upper = float(np.clip(q_upper, q_lower * 1.001, q_upper))
        if not np.isfinite(q_upper) or q_upper <= q_lower:
            continue

        loadings = np.linspace(q_lower, q_upper, n_loadings)
        Qst_arr = np.full(len(loadings), np.nan)
        for j, q_fix in enumerate(loadings):
            ln_Pxi = []
            T_used = []
            for T in temps:
                if T not in by_t:
                    continue
                P_c, q_c, P_a, q_a = by_t[T]
                P_at_q = _safe_interp(q_fix, q_c, P_c)
                if not np.isfinite(P_at_q) or P_at_q <= 0:
                    continue
                q_tot = _safe_interp(P_at_q, P_a, q_a)
                if not np.isfinite(q_tot) or q_tot <= 0:
                    continue
                y_i = q_fix / q_tot
                if y_i <= 0 or y_i > 1.01:
                    continue
                ln_Pxi.append(np.log(P_at_q * y_i))
                T_used.append(T)
            if len(ln_Pxi) < min_temps:
                continue
            invT = 1.0 / np.array(T_used)
            a, _ = np.polyfit(invT, np.array(ln_Pxi), 1)
            qst_val = -R * a / 1000.0
            # Guard against CC-invalid / non-physical values.
            if not np.isfinite(qst_val) or qst_val <= 0:
                continue
            Qst_arr[j] = qst_val
        result[comp] = {'loading': loadings, 'Qst_kJmol': Qst_arr}

    # Mixture total (y_i=1): fit ln(P) vs 1/T at constant total loading
    by_t_all = {}
    for T in temps:
        pts = data_by_temp[T].get('all', [])
        if len(pts) < 3:
            continue
        P_a = np.array([x[0] for x in pts])
        q_a = np.array([x[1] for x in pts])
        order = np.argsort(P_a)
        by_t_all[T] = (P_a[order], q_a[order])
    if len(by_t_all) >= min_temps:
        q_mins = [np.min(by_t_all[T][1]) for T in temps if T in by_t_all]
        q_maxs = [np.max(by_t_all[T][1]) for T in temps if T in by_t_all]
        if q_mins and q_maxs and min(q_maxs) > max(q_mins):
            sorted_T_all = sorted(T for T in temps if T in by_t_all)
            q_upper_all = min(q_maxs)
            cutoff_P_all = np.inf
            if trim_mixture_total_ordering_cutoff and len(sorted_T_all) >= 2:
                P_ref_all = by_t_all[sorted_T_all[0]][0]
                for k, P in enumerate(P_ref_all):
                    q_prev_all = None
                    ok = True
                    for T in sorted_T_all:
                        q_T = _safe_interp(P, by_t_all[T][0], by_t_all[T][1])
                        if not np.isfinite(q_T):
                            continue
                        if q_prev_all is not None and q_T > q_prev_all:
                            ok = False
                            break
                        q_prev_all = q_T
                    if not ok:
                        cutoff_P_all = P_ref_all[k - 1] if k > 0 else P_ref_all[0]
                        q_upper_all = _safe_interp(cutoff_P_all, by_t_all[sorted_T_all[0]][0],
                                                   by_t_all[sorted_T_all[0]][1])
                        break
            if np.isfinite(cutoff_P_all):
                for T in sorted_T_all:
                    P_a_t, q_a_t = by_t_all[T]
                    mask = P_a_t <= cutoff_P_all
                    if np.sum(mask) >= 2:
                        by_t_all[T] = (P_a_t[mask], q_a_t[mask])
            q_upper_all = float(np.clip(q_upper_all, max(q_mins) * 1.001, min(q_maxs)))
            if q_upper_all > max(q_mins):
                loadings_all = np.linspace(max(q_mins), q_upper_all, n_loadings)
                Qst_all = np.full(len(loadings_all), np.nan)
                for j, q_fix in enumerate(loadings_all):
                    ln_P = []
                    T_used = []
                    for T in temps:
                        if T not in by_t_all:
                            continue
                        P_a, q_a = by_t_all[T]
                        P_val = _safe_interp(q_fix, q_a, P_a)
                        if np.isfinite(P_val) and P_val > 0:
                            ln_P.append(np.log(P_val))
                            T_used.append(T)
                    if len(ln_P) < min_temps:
                        continue
                    invT = 1.0 / np.array(T_used)
                    a, _ = np.polyfit(invT, np.array(ln_P), 1)
                    qst_all_val = -R * a / 1000.0
                    # Same guard as per-component curves: omit non-finite / non-positive Qst.
                    # Otherwise totals can sit entirely below y=0 and disappear when plots clip
                    # at bottom=0 while the legend entry remains.
                    if np.isfinite(qst_all_val) and qst_all_val > 0:
                        Qst_all[j] = qst_all_val
                result['all'] = {'loading': loadings_all, 'Qst_kJmol': Qst_all}
    return result

def plot_mixture_heat_cc(mixture_data, selected_frameworks, selected_molecules,
                         selected_temperatures, p_min, p_max, n_loadings=40,
                         min_temps=3, smoothing_sigma=1.5,
                         component_colors=None, out_dir=None, save_data=False):
    """
    Compute and plot mixture isosteric heat Q_st^mix (Clausius-Clapeyron) vs loading.
    Internally calls compute_mixture_isosteric_heat_cc for each framework.
    One figure per framework: one line per component plus the total mixture.
    Saved under the Heat_of_adsorption subfolder.
    """
    if not mixture_data:
        print("plot_mixture_heat_cc: no mixture data provided, skipping.")
        return

    components  = sorted({d['molecule'] for d in mixture_data})
    mixture_name = selected_molecules[0] if selected_molecules else 'mixture'

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if component_colors is None:
        component_colors = {}
    comp_colors = {c: component_colors.get(c, default_colors[i % len(default_colors)])
                   for i, c in enumerate(components)}

    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    export_rows = []
    any_plotted = False
    q_cc_plot_min = np.inf
    q_cc_plot_max = -np.inf
    fw_first = selected_frameworks[0] if selected_frameworks else None

    for fw in selected_frameworks:
        try:
            result = compute_mixture_isosteric_heat_cc(
                mixture_data=mixture_data,
                components=components,
                temperatures=selected_temperatures,
                framework=fw,
                mixture_name=mixture_name,
                p_min=p_min,
                p_max=p_max,
                n_loadings=n_loadings,
                min_temps=min_temps,
                # Defaults match pure-style CC: intersection loading window across
                # temperatures + ordering trim (same as StorageDensity / function default).
            )
        except Exception as e:
            print(f"plot_mixture_heat_cc: computation failed for {fw}: {e}")
            continue

        if not result:
            print(f"plot_mixture_heat_cc: no results for {fw}, skipping.")
            continue

        mols = [c for c in components if c in result] + (['all'] if 'all' in result else [])
        for comp in mols:
            d = result[comp]
            loadings = np.asarray(d['loading'], dtype=float)
            Qst      = np.asarray(d['Qst_kJmol'], dtype=float)
            mask = np.isfinite(loadings) & np.isfinite(Qst)
            if not np.any(mask):
                continue
            loadings, Qst = loadings[mask], Qst[mask]
            order = np.argsort(loadings)
            loadings, Qst = loadings[order], Qst[order]
            if len(Qst) >= 5:
                Qst = gaussian_filter1d(Qst, sigma=smoothing_sigma)

            q_cc_plot_min = min(q_cc_plot_min, float(np.nanmin(Qst)))
            q_cc_plot_max = max(q_cc_plot_max, float(np.nanmax(Qst)))

            if comp == 'all':
                color, ls = 'black', '--'
                label = f'{mixture_name}' if fw == fw_first else "_nolegend_"
                mol_label = mixture_name  # Use mixture name for overall contribution
            else:
                color = comp_colors[comp]
                ls = phelp.get_linestyle_for_structure(fw)
                label = phelp.get_molecule_display_name(comp) if fw == fw_first else "_nolegend_"
                mol_label = comp
            ax.plot(loadings, Qst, ls, color=color, lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label=label)
            any_plotted = True

            # Collect export rows for this curve: one per (loading, Qst)
            for L_val, q_val in zip(loadings, Qst):
                if not np.isfinite(L_val) or not np.isfinite(q_val):
                    continue
                export_rows.append({
                    'framework': fw,
                    'molecule': mol_label,
                    'loading': float(L_val),
                    'Qst_kJmol': float(q_val),
                })

    if any_plotted:
        ax.set_xlabel('Loading [mol/kg]', **_cc_axis_label_kwargs())
        ax.set_ylabel(r'$Q_{st}^{mix}$ [kJ/mol]', **_cc_axis_label_kwargs())
        ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
        phelp.set_axis_limits_nice(ax)
        x_right_cc = ax.get_xlim()[1]
        ax.set_xlim(left=0.0, right=x_right_cc)
        _hoa_mix_yaxis_pre_legend(ax, q_cc_plot_min, q_cc_plot_max)
        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
        phelp.apply_unified_axes_layout(fig, ax)
        _hoa_mix_yaxis_post_layout(ax, q_cc_plot_min, q_cc_plot_max)

        out_path = phelp._save_plot(
            'mixture_cc_heat',
            'Heat_of_adsorption',
            selected_frameworks, selected_molecules, selected_temperatures,
            fig=fig, out_dir=out_dir,
        )

        # Save a separate data file for this CC mixture plot (overall + per-component),
        # independent of the HOA pure helpers.
        if export_rows and save_data and out_path is not None:
            try:
                base = Path(out_path)
                saved_dir = base.parent / "saved"
                saved_dir.mkdir(parents=True, exist_ok=True)
                data_path = saved_dir / (base.stem + '_data.txt')
                with data_path.open('w', encoding='utf-8') as f:
                    f.write("framework\tmolecule\tloading_mol_per_kg\tQst_kJmol\n")
                    for r in export_rows:
                        f.write(
                            f"{r['framework']}\t{r['molecule']}\t"
                            f"{r['loading']}\t{r['Qst_kJmol']}\n"
                        )
            except Exception as e:
                print(f"Warning: failed to write mixture CC data file next to {out_path}: {e}")
    plt.close(fig)

    plt.show()
    plt.close('all')


def _build_qst_interp(n_arr_raw, qst_arr_raw):
    """Validate, sort, deduplicate and build a PchipInterpolator from raw arrays.
    Returns (interp, n_arr, qst_arr) or (None, None, None) if data are insufficient."""
    n_arr   = np.asarray(n_arr_raw,   dtype=float)
    qst_arr = np.asarray(qst_arr_raw, dtype=float)
    ok = np.isfinite(n_arr) & np.isfinite(qst_arr)
    if ok.sum() < 3:
        return None, None, None
    n_arr, qst_arr = n_arr[ok], qst_arr[ok]
    idx = np.argsort(n_arr)
    n_arr, qst_arr = n_arr[idx], qst_arr[idx]
    _, u = np.unique(n_arr, return_index=True)
    n_arr, qst_arr = n_arr[u], qst_arr[u]
    if len(n_arr) < 3:
        return None, None, None
    return PchipInterpolator(n_arr, qst_arr, extrapolate=False), n_arr, qst_arr


def _safe_join_mix(lst):
    """Join labels into a filesystem-safe segment for mixture HOA outputs."""
    return "-".join([str(x).replace(" ", "_") for x in lst]) if lst else "all"


def _save_mix_hoa_rows_to_run_folder(rows, fw, mixture_name, selected_temperatures,
                                     prefix, base_dir=None):
    """Save mixture HOA rows to Output/<run>/Heat_of_Adsorption/saved.

    Columns: framework, molecule, temperature_K, loading_mol_per_kg, Qst_kJmol
    Molecule is the mixture name (entire mixture contribution).
    """
    if not rows:
        return
    try:
        from pathlib import Path as _Path

        base_dir = base_dir or init.get_pipeline_run_root()
        plots_root = base_dir / 'Output'

        fw_part = _safe_join_mix([fw])
        mix_part = _safe_join_mix([mixture_name])
        temp_part = _safe_join_mix(selected_temperatures)

        run_folder_name = f"{fw_part}_{mix_part}_{temp_part}"
        saved_dir = plots_root / run_folder_name / 'Heat_of_Adsorption' / 'saved'
        saved_dir.mkdir(parents=True, exist_ok=True)

        data_path = saved_dir / f"{prefix}.txt"
        with data_path.open('w', encoding='utf-8') as f:
            f.write("framework\tmolecule\ttemperature_K\tloading_mol_per_kg\tQst_kJmol\n")
            for r in rows:
                f.write(
                    f"{r['framework']}\t{r['molecule']}\t{r['temperature']}\t"
                    f"{r['loading']}\t{r['Qst_kJmol']}\n"
                )
    except Exception as e:
        print(f"Warning: failed to write mixture HOA data file: {e}")


# Mixture / pure HOA figures from ``_plot_hoa_mix``: y-span (kJ/mol) and tick step.
_HOA_MIX_Y_SPAN_START_KJMOL = 16.0  # first try: centred window of this height
_HOA_MIX_Y_SPAN_STEP_KJMOL = 4.0   # expand 16 -> 20 -> 24 -> ... until data fits
_HOA_MIX_Y_TICK_STEP = 4.0


def _hoa_mix_ylim_kjmol(q_min, q_max):
    """Return ``(y_bottom, y_top)`` in kJ/mol for mixture-style Q_st plots.

    Uses a **centred** window on the data midpoint, with limits snapped to multiples of
    ``_HOA_MIX_Y_TICK_STEP`` (4 kJ/mol) so major ticks stay on whole fours.

    Starts with span ``_HOA_MIX_Y_SPAN_START_KJMOL`` (16). If any part of ``[q_min,
    q_max]`` lies outside the snapped window, increases the span by
    ``_HOA_MIX_Y_SPAN_STEP_KJMOL`` (4) and repeats until the full range is visible
    (16, 20, 24, 28, ...). Caps at a large span to avoid an infinite loop.
    """
    q_min = float(q_min)
    q_max = float(q_max)
    if not (np.isfinite(q_min) and np.isfinite(q_max)):
        return None, None
    if q_max < q_min:
        q_min, q_max = q_max, q_min
    step = float(_HOA_MIX_Y_TICK_STEP)
    span0 = float(_HOA_MIX_Y_SPAN_START_KJMOL)
    dspan = float(_HOA_MIX_Y_SPAN_STEP_KJMOL)
    mid = 0.5 * (q_min + q_max)
    eps = 1e-9
    target = span0
    # Upper bound for search: data span plus slack for floor-snapping the lower limit.
    max_target = max(span0, (q_max - q_min) + 16.0 * dspan, 512.0)

    while target <= max_target:
        y_lo_raw = mid - 0.5 * target
        y_bottom = step * np.floor(y_lo_raw / step)
        y_top = y_bottom + target
        if y_bottom <= q_min + eps and y_top >= q_max - eps:
            return y_bottom, y_top
        target += dspan

    # Fallback: widest attempt (should be rare)
    y_lo_raw = mid - 0.5 * target
    y_bottom = step * np.floor(y_lo_raw / step)
    y_top = y_bottom + target
    return y_bottom, y_top


def _apply_hoa_mix_y_tick_step(ax):
    """Major y-ticks every ``_HOA_MIX_Y_TICK_STEP`` kJ/mol; no minor y-ticks."""
    ax.yaxis.set_major_locator(MultipleLocator(_HOA_MIX_Y_TICK_STEP))
    ax.yaxis.set_minor_locator(NullLocator())


def _hoa_mix_yaxis_pre_legend(ax, q_min, q_max):
    """If Q range is not tracked, set y from 0 to ``set_axis_limits_nice`` top (before legend)."""
    if not (np.isfinite(q_min) and np.isfinite(q_max)):
        y_top = ax.get_ylim()[1]
        ax.set_ylim(bottom=0.0, top=y_top)


def _hoa_mix_yaxis_post_layout(ax, q_min, q_max):
    """Centred expandable y-window (``_hoa_mix_ylim_kjmol``) + tick step; call after legend + layout."""
    if np.isfinite(q_min) and np.isfinite(q_max):
        y_lo, y_hi = _hoa_mix_ylim_kjmol(q_min, q_max)
        if y_lo is not None and y_hi is not None:
            ax.set_ylim(bottom=y_lo, top=y_hi)
            _apply_hoa_mix_y_tick_step(ax)


def _plot_hoa_mix(fw, components, qst_interp, pure_curves,
                  mixture_data, temps, p_min_b, p_max_b,
                  n_loadings, smoothing_sigma, combo_colors,
                  mixture_name, default_cycle, method_label,
                  mix_prefix, pure_prefix,
                  selected_temperatures, out_dir, save_data=False):
    """
    Shared computation + plotting for both HOA-pure variants (CC and Virial).

    Given per-component pure Qst interpolators, computes:
        Qst_mix(n_tot) = sum_i  y_i * Qst_i^pure(n_i)
    where y_i = n_i / n_tot (adsorbed-phase mole fraction from mixture RASPA data).

    Produces two figures:
      - mixture HOA vs loading (one curve per temperature)
      - pure component HOA curves used in the mixing
    Both saved under Heat_of_adsorption/ with the mixture folder pattern.
    """
    # ---------- 1. Collect per-component mixture RASPA loadings ----------
    comp_data = {c: {T: [] for T in temps} for c in components}
    for d in mixture_data:
        if d.get('framework') != fw:
            continue
        mol = d.get('molecule')
        if mol not in components:
            continue
        T = float(d['temperature'])
        if T not in set(temps):
            continue
        P = float(d.get('pressure', np.nan))
        q = float(d.get('loading',  np.nan))
        if (np.isfinite(P) and np.isfinite(q)
                and P > 0 and q >= 0 and p_min_b <= P <= p_max_b):
            comp_data[mol][T].append((P, q))

    # ---------- 2. Compute Qst_mix per temperature ----------------------
    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    has_any = False
    export_rows = []
    q_mix_plot_min = np.inf
    q_mix_plot_max = -np.inf

    for i_t, T in enumerate(temps):
        by_pressure = {}
        for comp in components:
            for P, q in comp_data[comp][T]:
                by_pressure.setdefault(P, {})[comp] = q
        shared = [(P, cd) for P, cd in by_pressure.items()
                  if len(cd) == len(components)]
        if len(shared) < 3:
            continue

        shared.sort(key=lambda x: x[0])
        n_tot_arr = np.array([sum(s[1].values()) for s in shared], dtype=float)
        order = np.argsort(n_tot_arr)
        n_tot_sorted  = n_tot_arr[order]
        shared_sorted = [shared[i] for i in order]

        _, uniq_idx = np.unique(n_tot_sorted, return_index=True)
        n_tot_sorted  = n_tot_sorted[uniq_idx]
        shared_sorted = [shared_sorted[i] for i in uniq_idx]
        if len(n_tot_sorted) < 3 or n_tot_sorted.min() >= n_tot_sorted.max():
            continue

        load_grid = np.linspace(n_tot_sorted.min(), n_tot_sorted.max(), n_loadings)
        Qst_mix   = np.full_like(load_grid, np.nan)

        for j, n_tot in enumerate(load_grid):
            if not np.isfinite(n_tot) or n_tot <= 0:
                continue
            n_i_vals, skip = [], False
            for comp in components:
                n_tot_c, n_comp_c = [], []
                for P, comp_dict in shared_sorted:
                    nt = sum(comp_dict.values())
                    nc = comp_dict.get(comp, np.nan)
                    if np.isfinite(nt) and np.isfinite(nc):
                        n_tot_c.append(float(nt))
                        n_comp_c.append(float(nc))
                if len(n_tot_c) < 2:
                    skip = True; break
                nt_arr = np.array(n_tot_c); nc_arr = np.array(n_comp_c)
                idx_c  = np.argsort(nt_arr)
                nt_arr, nc_arr = nt_arr[idx_c], nc_arr[idx_c]
                if n_tot < nt_arr[0] or n_tot > nt_arr[-1]:
                    skip = True; break
                n_i = float(np.interp(n_tot, nt_arr, nc_arr))
                if not np.isfinite(n_i) or n_i < 0:
                    skip = True; break
                n_i_vals.append((comp, n_i))

            if skip or not n_i_vals:
                continue
            n_sum = sum(n_i for _, n_i in n_i_vals)
            if n_sum <= 0:
                continue

            # Strict rule: only accept a mixture point if every component has
            # finite interpolated Qst at its computed n_i.
            n_i_by_comp = {comp: n_i for comp, n_i in n_i_vals}
            if any(comp not in n_i_by_comp for comp in components):
                continue
            if any(comp not in qst_interp for comp in components):
                continue

            qst_j = 0.0
            all_finite = True
            for comp in components:
                qv = float(qst_interp[comp](n_i_by_comp[comp]))
                if not np.isfinite(qv):
                    all_finite = False
                    break
                qst_j += (n_i_by_comp[comp] / n_sum) * qv

            if all_finite and np.isfinite(qst_j):
                Qst_mix[j] = qst_j

        valid = np.isfinite(Qst_mix)
        if not np.any(valid):
            continue
        lf, qf = load_grid[valid], Qst_mix[valid]
        lf, qf = lf[np.argsort(lf)], qf[np.argsort(lf)]
        if len(qf) >= 5:
            qf = gaussian_filter1d(qf, sigma=smoothing_sigma)

        # Mixture HOA: vary color by temperature when `combo_colors` is provided
        # (it is built in Main via get_combo_colors and respects TEMPERATURE settings).
        # Fallback to mixture-name color when no combo mapping exists.
        color = None
        if combo_colors is not None:
            try:
                color = combo_colors.get((fw, mixture_name, float(T)))
            except Exception:
                color = None
            if color is None:
                color = combo_colors.get((fw, mixture_name, T))
        color = color or phelp.get_color_for_molecule(mixture_name) or default_cycle[i_t % len(default_cycle)]
        ls = '-' if method_label is None else ('--' if str(method_label).strip().lower().startswith('virial') else '-')
        q_mix_plot_min = min(q_mix_plot_min, float(np.nanmin(qf)))
        q_mix_plot_max = max(q_mix_plot_max, float(np.nanmax(qf)))
        ax.plot(lf, qf, ls, color=color, lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label=f'{int(T)}K')
        has_any = True

        # Collect export rows for optional data saving: one per (loading, Qst_mix), with
        # the mixture name as the "molecule".
        for L_val, q_val in zip(lf, qf):
            if not np.isfinite(L_val) or not np.isfinite(q_val):
                continue
            export_rows.append({
                'framework': fw,
                'molecule': mixture_name,
                'temperature': float(T),
                'loading': float(L_val),
                'Qst_kJmol': float(q_val),
            })

    if not has_any:
        plt.close(fig)
    else:
        fw_display = phelp.clean_material_name(fw)
        ax.set_xlabel('Loading [mol/kg]', **_cc_axis_label_kwargs())
        ax.set_ylabel(r'$Q_{st}^{mix}$ [kJ/mol]', **_cc_axis_label_kwargs())
        # ax.set_title(f'Mixture HOA ({method_label}) – {mixture_name}, {fw_display}')
        ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
        phelp.set_axis_limits_nice(ax)
        x_right = ax.get_xlim()[1]
        ax.set_xlim(left=0.0, right=x_right)
        _hoa_mix_yaxis_pre_legend(ax, q_mix_plot_min, q_mix_plot_max)
        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
        phelp.apply_unified_axes_layout(fig, ax)
        _hoa_mix_yaxis_post_layout(ax, q_mix_plot_min, q_mix_plot_max)
        phelp._save_plot(mix_prefix, 'Heat_of_adsorption',
                         [fw], [mixture_name], selected_temperatures,
                         fig=fig, out_dir=out_dir)
        # Save mixture HOA data once per (fw, mixture) when requested. Use the mixture
        # name as the "molecule" in the output file to represent the total contribution.
        if export_rows and save_data and out_dir is None:
            _save_mix_hoa_rows_to_run_folder(
                export_rows, fw, mixture_name, selected_temperatures,
                prefix=mix_prefix,
            )
        plt.close(fig)

    # ---------- 3. Pure-component HOA figure ----------------------------
    if pure_curves:
        fig_p, ax_p = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        q_pure_plot_min = np.inf
        q_pure_plot_max = -np.inf
        for i_c, (comp, (n_arr, q_arr)) in enumerate(pure_curves.items()):
            if n_arr.size < 2:
                continue
            q_ok = np.asarray(q_arr, dtype=float)
            q_ok = q_ok[np.isfinite(q_ok)]
            if q_ok.size:
                q_pure_plot_min = min(q_pure_plot_min, float(np.min(q_ok)))
                q_pure_plot_max = max(q_pure_plot_max, float(np.max(q_ok)))
            ax_p.plot(n_arr, q_arr, '-',
                      color=default_cycle[i_c % len(default_cycle)],
                      lw=phelp.LINEWIDTH, alpha=phelp.ALPHA,
                      label=phelp.get_molecule_display_name(comp))
        fw_display = phelp.clean_material_name(fw)
        ax_p.set_xlabel('Loading [mol/kg]', **_cc_axis_label_kwargs())
        ax_p.set_ylabel(r'$Q_{st}^{pure}$ [kJ/mol]', **_cc_axis_label_kwargs())
        # ax_p.set_title(f'Pure-component HOA ({method_label}) – {fw_display}')
        ax_p.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
        phelp.set_axis_limits_nice(ax_p)
        x_right_p = ax_p.get_xlim()[1]
        ax_p.set_xlim(left=0.0, right=x_right_p)
        _hoa_mix_yaxis_pre_legend(ax_p, q_pure_plot_min, q_pure_plot_max)
        ax_p.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
        phelp.apply_unified_axes_layout(fig_p, ax_p)
        _hoa_mix_yaxis_post_layout(ax_p, q_pure_plot_min, q_pure_plot_max)
        phelp._save_plot(pure_prefix, 'Heat_of_adsorption',
                         [fw], [mixture_name], selected_temperatures,
                         fig=fig_p, out_dir=out_dir)
        plt.close(fig_p)


def _hoa_pure_loading_grid_n(RASPA_data_pure, fw, comp, temps, n_loadings):
    """One loading vector per component for mixture pure HOA (CC + Virial).

    Same rule as ``compute_isosteric_heat`` with ``loadings=None``: intersection
    over temperatures, then ``linspace`` with ``n_loadings`` points.
    """
    if not RASPA_data_pure:
        return None
    pts = phelp.filter_raspa_data(
        RASPA_data_pure,
        frameworks=[fw],
        molecules=[comp],
        temperatures=list(temps),
        only_pure_adsorption=True,
    )
    if not pts:
        return None
    try:
        n_lo, n_hi = ds._intersection_loading_range(pts)
    except Exception:
        return None
    n_lo = max(float(n_lo), 1e-8)
    n_hi = float(n_hi)
    if not np.isfinite(n_lo) or not np.isfinite(n_hi) or n_hi <= n_lo:
        return None
    return np.linspace(n_lo, n_hi, int(max(2, n_loadings)))


def plot_mixture_heat_hoa_pure_cc(
        mixture_data, fits_pure, RASPA_data_pure,
        selected_frameworks, mixture_name, selected_temperatures,
        selected_fit_types, p_min, p_max,
        n_loadings=50, min_temps=3, smoothing_sigma=None,
        combo_colors=None, out_dir=None, save_data=False, **_ignored):
    """
    Mixture HOA via weighted pure CC Qst:
        Qst_mix(n_tot) = sum_i  y_i * Qst_i^CC(n_i)
    Pure Qst computed with Clausius-Clapeyron (compute_isosteric_heat).

    Always uses fitting values (use_direct_interpolation=False).  The loading
    range for each component is derived from RASPA_data_pure via
    ``_intersection_loading_range`` — the same logic as the pure-component CC
    HOA plot.  Pass a DataSelection-built dataset (one built with the
    individual gas-component names) as RASPA_data_pure so both CC and Virial
    share the identical loading span.
    """
    if min_temps    is None: min_temps    = 2
    if smoothing_sigma is None: smoothing_sigma = 1.5

    if not mixture_data:
        print("plot_mixture_heat_hoa_pure_cc: no mixture data, skipping.")
        return
    if not fits_pure:
        print("plot_mixture_heat_hoa_pure_cc: no pure-component fits available; skipping.")
        return

    components    = sorted({d['molecule'] for d in mixture_data})
    temps         = [float(t) for t in selected_temperatures]
    p_min_b       = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b       = float(p_max)            if p_max is not None else 1e8
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for fw in selected_frameworks:
        qst_interp, pure_curves = {}, {}
        for comp in components:
            try:
                n_grid = _hoa_pure_loading_grid_n(
                    RASPA_data_pure, fw, comp, temps, n_loadings)
                if n_grid is None:
                    continue
                res = compute_isosteric_heat(
                    framework=fw, molecule=comp,
                    temperatures=temps,
                    selected_fit_types=selected_fit_types,
                    fittings=fits_pure,
                    RASPA_data=RASPA_data_pure,
                    loadings=n_grid,
                    n_loadings=n_loadings,
                    p_min=p_min, p_max=p_max,
                    min_temps=min_temps,
                    smooth=True, smoothing_sigma=smoothing_sigma,
                    use_direct_interpolation=False,
                )
            except Exception as e:
                print(f"plot_mixture_heat_hoa_pure_cc: CC failed for {fw}/{comp}: {e}")
                continue
            qst_raw = (res.get('Qst_kJmol_smoothed')
                       if res.get('Qst_kJmol_smoothed') is not None
                       else res.get('Qst_kJmol'))
            if qst_raw is None:
                continue
            interp, n_arr, qst_arr = _build_qst_interp(res['loading'], qst_raw)
            if interp is None:
                continue
            qst_interp[comp] = interp
            # Pure figure: keep full abscissa (NaN Qst where CC has no valid slope)
            # so x-limits match Virial, which uses the same ``n_grid``.
            pure_curves[comp] = (
                np.asarray(res['loading'], dtype=float),
                np.asarray(qst_raw, dtype=float),
            )

        if not qst_interp:
            print(f"plot_mixture_heat_hoa_pure_cc: no CC Qst for any component in {fw}.")
            continue

        _plot_hoa_mix(
            fw, components, qst_interp, pure_curves,
            mixture_data, temps, p_min_b, p_max_b,
            n_loadings, smoothing_sigma, combo_colors,
            mixture_name, default_cycle,
            method_label='pure CC',
            mix_prefix='mixture_hoa_pure_cc_heat',
            pure_prefix='pure_hoa_cc_heat',
            selected_temperatures=selected_temperatures,
            out_dir=out_dir,
            save_data=save_data,
        )

    plt.show()
    plt.close('all')

def plot_mixture_heat_hoa_pure_file(
        mixture_data, hoa_pure_curves,
        selected_frameworks, mixture_name, selected_temperatures,
        p_min, p_max,
        n_loadings=50, smoothing_sigma=None,
        combo_colors=None, out_dir=None, save_data=False):
    """
    Mixture HOA via weighted pure-file Qst:
        Qst_mix(n_tot) = sum_i  y_i * Qst_i^file(n_i)
    Pure Qst_i^file(n_i) comes directly from HoA-file curves.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    if smoothing_sigma is None:
        smoothing_sigma = 1.5

    if not mixture_data:
        print("plot_mixture_heat_hoa_pure_file: no mixture data, skipping.")
        return
    if not hoa_pure_curves:
        print("plot_mixture_heat_hoa_pure_file: no pure HoA-file curves; skipping.")
        return

    components    = sorted({d['molecule'] for d in mixture_data})
    temps         = [float(t) for t in selected_temperatures]
    p_min_b       = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b       = float(p_max)            if p_max is not None else 1e8
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Build pure interpolators per framework/component from hoa_pure_curves
    def _build_interp_for_fw(fw):
        qst_interp, pure_curves = {}, {}
        for comp in components:
            key = (fw, comp)
            if key not in hoa_pure_curves:
                continue
            loads_arr, qst_arr = hoa_pure_curves[key]
            n_arr = np.asarray(loads_arr, dtype=float)
            q_arr = np.asarray(qst_arr, dtype=float)
            ok = np.isfinite(n_arr) & np.isfinite(q_arr)
            if ok.sum() < 3:
                continue
            n_arr, q_arr = n_arr[ok], q_arr[ok]
            idx = np.argsort(n_arr)
            n_arr, q_arr = n_arr[idx], q_arr[idx]
            _, u = np.unique(n_arr, return_index=True)
            n_arr, q_arr = n_arr[u], q_arr[u]
            if len(n_arr) < 3:
                continue
            try:
                interp = PchipInterpolator(n_arr, q_arr, extrapolate=False)
            except Exception:
                continue
            qst_interp[comp]  = interp
            pure_curves[comp] = (n_arr, q_arr)
        return qst_interp, pure_curves

    for fw in selected_frameworks:
        qst_interp, pure_curves = _build_interp_for_fw(fw)
        if not qst_interp:
            print(f"plot_mixture_heat_hoa_pure_file: no HoA-file Qst for any component in {fw}.")
            continue

        _plot_hoa_mix(
            fw, components, qst_interp, pure_curves,
            mixture_data, temps, p_min_b, p_max_b,
            n_loadings, smoothing_sigma, combo_colors,
            mixture_name, default_cycle,
            method_label='HoA file',
            mix_prefix='mixture_hoa_file_heat',
            pure_prefix='pure_hoa_file_heat',
            selected_temperatures=selected_temperatures,
            out_dir=out_dir,
            save_data=save_data,
        )

    plt.show()
    plt.close('all')


def plot_mixture_heat_hoa_pure_virial(*args, **kwargs):
    """Backward-compatible alias; implementation is :func:`Virial.plot_mixture_heat_hoa_pure_virial`."""
    import Virial as virial
    return virial.plot_mixture_heat_hoa_pure_virial(*args, **kwargs)
