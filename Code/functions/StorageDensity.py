"""StorageDensity.py — unified thermochemical storage-density plotting.

The three core plot variants each accept a *method* argument:
    'cc'     – isosteric heat from Clausius-Clapeyron (compute_isosteric_heat)
    'virial' – isosteric heat from the Virial two-term model (compute_Qst_from_coef_slopes)

All plot variants support both method='cc' and method='virial'.

Mixture mode
------------
Pass *qst_cache* ``{(fw, mol): (loads_arr, qst_arr)}`` to any plot function.
``_get_qst`` returns the cached entry immediately instead of computing Qst.
For loading at (T, P) use synthetic 'interp' fit objects created by
``make_mixture_fittings``; the 'interp' fit type in ``Initialize.formula_fitting``
performs linear interpolation from the stored (P_arr, q_arr) pair.
"""

from math import isclose
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import PlotHelpers as phelp
import Virial as virial
from ClausiusClapeyron import compute_isosteric_heat

# 3D storage-density axis labels: \mathrm{...} for upright text (default math is italic).
SD3D_LABEL_SD = r'$\mathrm{SD\ [kJ/kg]}$'
SD3D_LABEL_T_DES = r'$\mathrm{T_{Des}\ [K]}$'
SD3D_LABEL_T_ADS = r'$\mathrm{T_{Ads}\ [K]}$'
SD3D_LABEL_P_DES = r'$\mathrm{P_{Des}\ [Pa]}$'


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def integrate_enthalpy(L_low, L_high, loads_arr, enths_arr):
    """
    Integrate enthalpy (kJ/mol) over loading (mol/kg) between L_low and L_high.
    Expects loads_arr and enths_arr as numpy arrays (sorted by loading). Returns kJ/mol integrated over loading.
    Stops integration when data is not available (no extrapolation).
    """
    if loads_arr is None or enths_arr is None or L_high <= L_low:
        return np.nan

    # Clamp integration range to available data range
    load_min = float(np.min(loads_arr))
    load_max = float(np.max(loads_arr))
    L_low_clamped = max(L_low, load_min)
    L_high_clamped = min(L_high, load_max)

    # If clamped range is invalid or completely outside data range, return NaN
    if L_high_clamped <= L_low_clamped:
        return np.nan
    # If the entire requested range is outside the data range, return NaN
    if L_high < load_min or L_low > load_max:
        return np.nan

    sample_L = np.linspace(L_low_clamped, L_high_clamped, max(8, min(200, 50)))
    if loads_arr.size == 1:
        H_sample = np.full_like(sample_L, enths_arr[0])
    else:
        # Use np.nan for left/right to avoid extrapolation beyond data range
        H_sample = np.interp(sample_L, loads_arr, enths_arr, left=np.nan, right=np.nan)
        # If any values are NaN (outside data range), return NaN
        if np.any(~np.isfinite(H_sample)):
            return np.nan
    # numpy>=2.0 removed np.trapz; use trapezoid
    return float(np.trapezoid(H_sample, sample_L))


def _as_list(value):
    """Normalize scalars/tuples/lists to a plain list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    return [value]


def _safe_join_labels(items):
    vals = _as_list(items)
    return "-".join([str(x).replace(" ", "_") for x in vals]) if vals else "all"


def _storage_density_3d_filename_method_token(method):
    """Short token for 3D PNG stems; long *method* strings hit ``_save_plot`` path-length fallback."""
    m = str(method).replace('-', '_').strip()
    if m == 'hoa_pure_virial':
        return 'hpv'
    if m == 'hoa_pure_cc':
        return 'hpc'
    return m


def _sd_filename_ps_teq(method, *, dim_3d=False):
    """(i) Pressure swing, T_ads = T_des; fixed P_ads; stem Option A."""
    t = _storage_density_3d_filename_method_token(method)
    return f"sd_PS_Teq_{t}_3d" if dim_3d else f"sd_PS_Teq_{t}"


def _sd_filename_ts_peq(method, *, dim_3d=False):
    """(ii) Temperature swing, P_ads = P_des; stem Option A."""
    t = _storage_density_3d_filename_method_token(method)
    return f"sd_TS_Peq_{t}_3d" if dim_3d else f"sd_TS_Peq_{t}"


def _sd_filename_pts_fixedpads(method, *, dim_3d=False):
    """(iii) Combined P–T swing, fixed P_ads; stem Option A."""
    t = _storage_density_3d_filename_method_token(method)
    return f"sd_PTS_fixedPads_{t}_3d" if dim_3d else f"sd_PTS_fixedPads_{t}"


def _iter_fw_mol_pairs(selected_frameworks, selected_molecules):
    for fw in _as_list(selected_frameworks):
        for mol in _as_list(selected_molecules):
            yield fw, mol


def _get_storage_density_out_dir(
    selected_frameworks,
    selected_molecules,
    selected_temperatures,
    *,
    dimension,
    fw=None,
    mol=None,
):
    """
    Build explicit output directory for storage density plots:
    - 2D: <run>/Storage_Density/2D/<mol>
    - 3D: <run>/Storage_Density/3D/<fw>_<mol>
    """
    base_dir = Path(__file__).resolve().parents[2]
    run_folder = (
        f"{_safe_join_labels(selected_frameworks)}_"
        f"{_safe_join_labels(selected_molecules)}_"
        f"{_safe_join_labels(selected_temperatures)}"
    )
    root = base_dir / "Output" / run_folder / "Storage_Density"

    dim = str(dimension).strip().upper()
    if dim == "2D":
        # For 2D plots we want one folder per molecule, while frameworks are
        # shown in the same figure (i.e. "structures should be in the figure").
        mol_one = mol if mol is not None else (_as_list(selected_molecules)[0] if _as_list(selected_molecules) else "all")
        out_dir = root / "2D" / f"{str(mol_one).replace(' ', '_')}"
    elif dim == "3D":
        fw_one = fw if fw is not None else (_as_list(selected_frameworks)[0] if _as_list(selected_frameworks) else "all")
        mol_one = mol if mol is not None else (_as_list(selected_molecules)[0] if _as_list(selected_molecules) else "all")
        out_dir = root / "3D" / f"{str(fw_one).replace(' ', '_')}_{str(mol_one).replace(' ', '_')}"
    else:
        out_dir = root / "2D"

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _get_qst(method, fw, mol, temps, selected_fit_types, fittings, RASPA_data, x_fit, *,
             smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5, min_temps=None,
             deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
             coeffs_a_override=None, coeffs_b_override=None, degrees_per_combo=None,
             qst_cache=None):
    """Return ``(loads [mol/kg], qst [kJ/mol])`` numpy arrays, or ``(None, None)`` on failure.

    If *qst_cache* ``{(fw, mol): (loads, qst)}`` contains an entry for this
    ``(fw, mol)`` the cached arrays are returned immediately, bypassing all
    computation.  This lets pre-computed mixture Qst be injected while
    loading lookups continue to use synthetic ``'interp'`` fit objects.
    """
    if qst_cache is not None and (fw, mol) in qst_cache:
        loads_c, qst_c = qst_cache[(fw, mol)]
        return (np.asarray(loads_c, dtype=float) if loads_c is not None else None,
                np.asarray(qst_c,   dtype=float) if qst_c   is not None else None)

    if method == 'cc':
        kw = dict(smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                  smoothing_sigma=smoothing_sigma)
        if min_temps is not None:
            kw['min_temps'] = min_temps
        data = compute_isosteric_heat(
            fw, mol, temps, selected_fit_types, fittings,
            RASPA_data=RASPA_data, p_grid=x_fit, **kw)
        loads = data.get('loading')
        qst_key = ('Qst_kJmol_smoothed'
                   if smooth and data.get('Qst_kJmol_smoothed') is not None
                   else 'Qst_kJmol')
        qst = data.get(qst_key)
    else:  # virial
        _da = (degrees_per_combo[(fw, mol)][0]
               if degrees_per_combo and (fw, mol) in degrees_per_combo else deg_a)
        _db = (degrees_per_combo[(fw, mol)][1]
               if degrees_per_combo and (fw, mol) in degrees_per_combo else deg_b)
        data = virial.compute_Qst_from_coef_slopes(
            RASPA_data, fw, mol, deg_a=_da, deg_b=_db, min_points=min_points,
            n_points=200, temperatures=temps,
            coeffs_a_override=coeffs_a_override, coeffs_b_override=coeffs_b_override,
            p_min=virial_p_min, verbose=False)
        loads = data.get('n_grid')
        qst_raw = data.get('Qst')
        qst = np.asarray(qst_raw, dtype=float) / 1000.0 if qst_raw is not None else None

    return (np.asarray(loads, dtype=float) if loads is not None else None,
            np.asarray(qst, dtype=float) if qst is not None else None)


def _finite_qst(loads, qst):
    """Filter arrays to finite values. Return ``(lf, qf, q_min, q_max)`` or ``(None,)*4``."""
    if loads is None or qst is None or len(loads) == 0 or len(qst) == 0:
        return None, None, None, None
    mask = np.isfinite(loads) & np.isfinite(qst)
    if not np.any(mask):
        return None, None, None, None
    lf, qf = loads[mask], qst[mask]
    return lf, qf, float(np.min(lf)), float(np.max(lf))


def _max_P_for_loading_in_qst_window(ads_params, ads_ft, q_min, q_max, *, n_scan=400):
    """Largest pressure P (Pa) with ``q_min <= q(P) <= q_max`` for the adsorption isotherm at *T_ads*.

    ``q_min`` / ``q_max`` are the loadings where Qst is defined (from :func:`_finite_qst`).
    Uses a log-spaced scan so ``interp`` tabulated fits and analytic isotherms are covered;
    may return NaN if no pressure in the search interval satisfies the band.
    """
    if not (np.isfinite(q_min) and np.isfinite(q_max) and q_max > q_min):
        return float('nan')
    q_lo = float(q_min)
    q_hi = float(q_max)

    ft_norm = str(ads_ft).replace('fitting_', '').strip()
    if ft_norm.lower() == 'interp' and ads_params is not None and len(ads_params) >= 2:
        P_arr = np.asarray(ads_params[0], dtype=float)
        P_arr = P_arr[np.isfinite(P_arr) & (P_arr > 0)]
        if P_arr.size:
            p_a, p_b = float(np.min(P_arr)), float(np.max(P_arr))
        else:
            p_a, p_b = 1e-9, 1e8
    else:
        p_a, p_b = 1e-9, 1e8

    p_a = max(p_a, 1e-30)
    p_b = max(p_b, p_a * 1.0001)
    P_grid = np.logspace(np.log10(p_a), np.log10(p_b), int(n_scan))
    try:
        q_eval = phelp.evaluate_fit(P_grid, ads_params, ads_ft)
    except Exception:
        return float('nan')
    q_eval = np.asarray(q_eval, dtype=float)
    if q_eval.shape != P_grid.shape:
        if q_eval.size == P_grid.size:
            q_eval = q_eval.reshape(P_grid.shape)
        else:
            return float('nan')
    valid = np.isfinite(q_eval) & (q_eval >= q_lo) & (q_eval <= q_hi)
    if np.any(valid):
        return float(np.max(P_grid[valid]))

    for q_star in (q_hi, q_lo):
        try:
            P_star = float(phelp.inverse_fit(q_star, ads_params, ads_ft))
        except Exception:
            continue
        if not (np.isfinite(P_star) and P_star > 0):
            continue
        try:
            qc = float(phelp.evaluate_fit(P_star, ads_params, ads_ft))
        except Exception:
            continue
        if np.isfinite(qc) and q_lo <= qc <= q_hi:
            return P_star
    return float('nan')


def _format_p_ads_qst_window_hint(ads_params, ads_ft, q_min, q_max):
    p_s = _max_P_for_loading_in_qst_window(ads_params, ads_ft, q_min, q_max)
    if np.isfinite(p_s) and p_s > 0:
        return (
            f" Largest P_ads at this T_ads with loading inside the Qst window "
            f"[{q_min:.4g}, {q_max:.4g}] mol/kg is about {p_s:.4g} Pa."
        )
    return ""


# Same P_ads / T_ads / Qst-window issue is hit from many plot branches (2D/3D, each T_des, CC/virial).
_SEEN_L_ADS_QST_PADS_WARNING_KEYS: set = set()


def reset_l_ads_qst_p_ads_warning_cache() -> None:
    """Clear :data:`_SEEN_L_ADS_QST_PADS_WARNING_KEYS` (optional; for tests or repeated in-process runs)."""
    _SEEN_L_ADS_QST_PADS_WARNING_KEYS.clear()


def _warn_l_ads_outside_qst_once(fw, mol, t_ads_iso, p_ads, q_min, q_max, l_ads, ads_params, ads_ft):
    """Emit at most one console line per (fw, mol, T_ads isotherm, P_ads, Qst loading window)."""
    key = (
        str(fw),
        str(mol),
        int(round(float(t_ads_iso))),
        round(float(p_ads), 12),
        round(float(q_min), 8),
        round(float(q_max), 8),
    )
    if key in _SEEN_L_ADS_QST_PADS_WARNING_KEYS:
        return
    _SEEN_L_ADS_QST_PADS_WARNING_KEYS.add(key)
    _hint = _format_p_ads_qst_window_hint(ads_params, ads_ft, q_min, q_max)
    L_s = f"{l_ads:.4f}" if np.isfinite(l_ads) else "nan"
    t_i = int(round(float(t_ads_iso)))
    print(
        f"Warning: L_ads={L_s} outside Qst range [{q_min:.4f}, {q_max:.4f}] mol/kg "
        f"for {fw}, {mol} at T_ads={t_i} K with P_ads={float(p_ads):g} Pa "
        f"(storage-density segments that use this adsorption state are skipped).{_hint}"
    )


def _sd_2d_axis_label_weight(ax):
    """Match Basic_data isotherm axis titles (``fontweight='medium'``) after ``format_storage_plot``."""
    ax.xaxis.label.set_fontweight('medium')
    ax.yaxis.label.set_fontweight('medium')


def _sd3d_apply_tick_params(ax, log_pressure_axis=None):
    """Match 2D ``apply_unified_axes_layout``: ``TICK_SIZE_X`` on log-pressure axis only."""

    def _tick_sz(axis_name):
        if axis_name == 'x':
            return phelp.TICK_SIZE_X if log_pressure_axis == 'x' else phelp.TICK_SIZE
        if axis_name == 'y':
            return phelp.TICK_SIZE_X if log_pressure_axis == 'y' else phelp.TICK_SIZE
        return phelp.TICK_SIZE

    for axis_name in ('x', 'y', 'z'):
        ax.tick_params(
            axis=axis_name,
            which='major',
            labelsize=_tick_sz(axis_name),
            width=phelp.AXIS_TICK_WIDTH,
            length=phelp.AXIS_TICK_LENGTH_MAJOR,
            pad=phelp.AXIS_TICK_PAD,
        )
        ax.tick_params(axis=axis_name, which='minor', length=0, width=0)


def _sd3d_bbox_extra_artists(ax, cb):
    """Axis titles only: mplot3d tight bbox often omits them, but every tick label in *bbox_extra_artists*
    can blow memory on Windows (``bad allocation``) when unioning many bboxes at high DPI.
    """
    artists = []
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        lab = axis.label
        if lab is not None and lab.get_text().strip():
            artists.append(lab)
    ylab = cb.ax.yaxis.label
    if ylab is not None and ylab.get_text().strip():
        artists.append(ylab)
    return artists


def _sd3d_apply_axis_labels_and_colorbar(fig, ax, surf, xlabel, ylabel, zlabel, log_pressure_axis=None):
    """2D-matched label and tick sizes; colorbar in a separate ``cax`` so the 3D axes are not shrunk.

    *zlabel* is drawn only on the colorbar (left side of the strip, toward the 3D axes), not on the 3D z-axis.
    Returns the :class:`matplotlib.colorbar.Colorbar` for tight-save bbox helpers.

    Matplotlib's ``fig.colorbar(..., ax=ax)`` steals width from *ax*, which kept the 3D
    plot narrow regardless of ``subplots_adjust``; we reserve a thin strip with
    ``fig.add_axes`` + ``colorbar(..., cax=…)`` instead.
    """
    if hasattr(ax, 'dist'):
        ax.dist = phelp.resolve_storage_density_3d_axes_dist()
    _sd3d_apply_tick_params(ax, log_pressure_axis=log_pressure_axis)
    fs = phelp.AXIS_LABEL_FONTSIZE
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    # SD is shown only on the colorbar; no z-axis title on the 3D axes.
    ax.set_zlabel('', fontsize=fs)
    ax.xaxis.label.set_fontweight('medium')
    ax.yaxis.label.set_fontweight('medium')
    _lp = phelp.resolve_storage_density_3d_axis_labelpads()
    ax.xaxis.labelpad = _lp['x']
    ax.yaxis.labelpad = _lp['y']

    margins = phelp.resolve_storage_density_3d_subplot_margins()
    cb_kw = phelp.resolve_storage_density_3d_colorbar_kwargs()
    fig_right = 1.0
    gap = cb_kw['gap_frac']
    cb_w = cb_kw['width_frac']
    left, bottom, top = margins['left'], margins['bottom'], margins['top']
    desired_ax_right = margins['right']
    # Reserve colorbar strip + gap on the right; place 3D axes in [left, ax_right], then cax hugs the box.
    ax_right = min(desired_ax_right, fig_right - gap - cb_w)
    h = top - bottom
    w = max(0.06, ax_right - left)
    fig.subplots_adjust(left=left, bottom=bottom, top=top, right=fig_right)
    scale = phelp.resolve_storage_density_3d_box_scale()
    w_s = w * scale
    h_s = h * scale
    halign = phelp.resolve_storage_density_3d_box_halign()
    valign = phelp.resolve_storage_density_3d_box_valign()
    cx = left + (w - w_s) * halign
    cy = bottom + (h - h_s) * valign
    ax.set_position([cx, cy, w_s, h_s])

    sh = cb_kw['shrink']
    cb_h = max(0.05, sh * h_s)
    cb_b = cy + 0.5 * (h_s - cb_h)
    cax_l = cx + w_s + gap
    cb_w_eff = min(cb_w, max(fig_right - cax_l, 0.012))
    cax = fig.add_axes([cax_l, cb_b, cb_w_eff, cb_h])
    cb = fig.colorbar(surf, cax=cax)
    lp_cb = float(cb_kw['cbar_label_labelpad'])
    y_cb = float(cb_kw['cbar_label_y'])
    # Put the SD label on the left side of the vertical colorbar (toward the 3D axes).
    try:
        cb.set_label(zlabel, fontsize=fs, fontweight='medium', loc='left', labelpad=lp_cb)
    except (TypeError, ValueError):
        try:
            cb.set_label(zlabel, fontsize=fs, fontweight='medium', labelpad=lp_cb)
        except (TypeError, ValueError):
            cb.set_label(zlabel, fontsize=fs, fontweight='medium')
        cb.ax.yaxis.set_label_position('left')
    _cb_lab = cb.ax.yaxis.get_label()
    if _cb_lab is not None:
        _cb_lab.set_y(y_cb)
    cb.ax.tick_params(
        labelsize=phelp.TICK_SIZE,
        width=phelp.AXIS_TICK_WIDTH,
        length=phelp.AXIS_TICK_LENGTH_MAJOR,
        pad=phelp.AXIS_TICK_PAD,
    )
    if cb_kw['ticks']:
        cb.set_ticks(cb_kw['ticks'])
    elif cb_kw.get('nticks', 0):
        from matplotlib import ticker as mticker
        cb.ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=int(cb_kw['nticks']), min_n_ticks=2)
        )
    return cb


def _render_surface(T_vals, P_vals, SD_vals, fig_title, xlabel, ylabel,
                    swap_xy=False, selected_frameworks=None, selected_molecules=None,
                    save_name=None, folder_temps=None,
                    combo_fw=None, combo_mol=None):
    """Build an interpolated T×P surface and render it as a 3-D plot.

    When *swap_xy* is True the surface axes are (log10 P, T, SD) instead of (T, log10 P, SD).
    """
    T_vals = np.asarray(T_vals, dtype=float)
    P_vals = np.asarray(P_vals, dtype=float)
    SD_vals = np.asarray(SD_vals, dtype=float)

    T_u = np.sort(np.unique(T_vals))
    P_u = np.sort(np.unique(P_vals))

    SD_base = np.full((len(T_u), len(P_u)), np.nan, dtype=float)
    t_idx = {t: i for i, t in enumerate(T_u)}
    p_idx = {p: j for j, p in enumerate(P_u)}
    for T, P, SD in zip(T_vals, P_vals, SD_vals):
        i, j = t_idx.get(T), p_idx.get(P)
        if i is not None and j is not None:
            SD_base[i, j] = SD

    num_T = max(2 * len(T_u) - 1, 50)
    T_fine = np.linspace(T_u.min(), T_u.max(), num_T)
    T_grid, P_grid = np.meshgrid(T_fine, P_u, indexing='ij')
    SD_grid = np.full_like(T_grid, np.nan, dtype=float)
    for j in range(len(P_u)):
        col = SD_base[:, j]
        mask = np.isfinite(col)
        if np.count_nonzero(mask) >= 2:
            SD_grid[:, j] = np.interp(T_fine, T_u[mask], col[mask])

    P_grid_log = np.log10(P_grid)
    tick_exp = np.arange(np.floor(np.nanmin(np.log10(P_u))),
                         np.ceil(np.nanmax(np.log10(P_u))) + 1)

    cmap = phelp.resolve_storage_density_3d_colormap()
    vmin, vmax = phelp.resolve_storage_density_3d_boundries()
    surf_kw = dict(cmap=cmap, edgecolor='none', antialiased=True)
    if vmin is not None:
        surf_kw['vmin'] = vmin
    if vmax is not None:
        surf_kw['vmax'] = vmax

    fig = plt.figure(figsize=phelp.resolve_storage_density_3d_figsize())
    ax = fig.add_subplot(111, projection='3d')
    if swap_xy:
        surf = ax.plot_surface(P_grid_log, T_grid, SD_grid, **surf_kw)
        ax.set_xticks(tick_exp)
        ax.set_xticklabels([r"$\mathrm{10^{%d}}$" % int(e) for e in tick_exp])
        log_axis = 'x'
    else:
        surf = ax.plot_surface(T_grid, P_grid_log, SD_grid, **surf_kw)
        ax.set_yticks(tick_exp)
        ax.set_yticklabels([r"$\mathrm{10^{%d}}$" % int(e) for e in tick_exp])
        log_axis = 'y'

    cb = _sd3d_apply_axis_labels_and_colorbar(
        fig, ax, surf, xlabel, ylabel, SD3D_LABEL_SD, log_pressure_axis=log_axis)

    if save_name:
        fw_one = combo_fw if combo_fw is not None else (_as_list(selected_frameworks)[0] if _as_list(selected_frameworks) else None)
        mol_one = combo_mol if combo_mol is not None else (_as_list(selected_molecules)[0] if _as_list(selected_molecules) else None)
        fw_safe = str(fw_one).replace(" ", "_") if fw_one is not None else "all"
        mol_safe = str(mol_one).replace(" ", "_") if mol_one is not None else "all"
        out_dir = _get_storage_density_out_dir(
            selected_frameworks, selected_molecules, folder_temps,
            dimension="3D",
            fw=fw_one,
            mol=mol_one,
        )
        # Short filename_suffix avoids Windows path-length fallback to ``prefix_<hash>.png``,
        # which shows up as an opaque "title" in many image viewers (run folder already encodes temps).
        phelp._save_plot(save_name, 'plot_storage_density',
                         selected_frameworks, selected_molecules, folder_temps, fig=fig, out_dir=out_dir,
                         fw_label_override=fw_safe if fw_one is not None else None,
                         mol_label_override=mol_safe if mol_one is not None else None,
                         filename_suffix=f"{fw_safe}__{mol_safe}",
                         tight_bbox=True,
                         bbox_extra_artists=_sd3d_bbox_extra_artists(ax, cb))
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-D storage-density plots  (method = 'cc' | 'virial')
# ---------------------------------------------------------------------------

def _save_sd_rows(out_path, rows, method_label):
    """Helper to save 2D storage-density data next to a saved PNG.

    Columns: framework, molecule, T_ads_K, T_des_K, P_ads_Pa, P_des_Pa, SD
    """
    if not rows or out_path is None:
        return
    try:
        from pathlib import Path
        base = Path(out_path)
        saved_dir = base.parent / "saved"
        saved_dir.mkdir(parents=True, exist_ok=True)
        data_path = saved_dir / (base.stem + '_data.txt')

        # Build new lines for this run
        new_lines = []
        for r in rows:
            new_lines.append(
                f"{r['framework']}\t{r['molecule']}\t"
                f"{r['T_ads']}\t{r['T_des']}\t"
                f"{r['P_ads']}\t{r['P_des']}\t"
                f"{r['SD']}\n"
            )

        # If file exists, merge with existing unique lines to avoid duplicates
        header = "framework\tmolecule\tT_ads_K\tT_des_K\tP_ads_Pa\tP_des_Pa\tSD\n"
        existing_lines = []
        if data_path.exists():
            with data_path.open('r', encoding='utf-8') as f:
                lines = f.readlines()
            if lines:
                # First line is header; keep as-is, dedupe only data lines
                header = lines[0]
                existing_lines = lines[1:]

        # Deduplicate by full line content
        all_lines_set = set(existing_lines)
        for line in new_lines:
            if line not in all_lines_set:
                all_lines_set.add(line)
                existing_lines.append(line)

        # Rewrite file with header + merged unique lines
        with data_path.open('w', encoding='utf-8') as f:
            f.write(header)
            f.writelines(existing_lines)
    except Exception as e:
        print(f"Warning: failed to write storage density ({method_label}) data file next to {out_path}: {e}")


def plot_storage_density(method, selected_frameworks, selected_molecules, selected_temperatures,
                         selected_fit_types, fittings, P_ads, x_fit, combo_colors, RASPA_data,
                         p_min=None, p_max=None, P_des_max=None,
                         smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5,
                         num_of_isotherm=None,
                         deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                         coeffs_a_override=None, coeffs_b_override=None,
                         degrees_per_combo=None, folder_temperatures=None,
                         qst_cache=None, save_data=False, scale='log'):
    """Storage density vs desorption pressure for each ``(fw, mol, T_ads)`` combination.

    *method* is ``'cc'`` (Clausius-Clapeyron) or ``'virial'``.

    Pass *qst_cache* ``{(fw, mol): (loads, qst)}`` and synthetic ``'interp'``
    *fittings* to use mixture data without any other code changes.
    """
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules,
        selected_temperatures, selected_fit_types, num_of_isotherm=num_of_isotherm)

    global_rows = phelp.filter_raspa_data(
        RASPA_data, frameworks=selected_frameworks,
        molecules=selected_molecules, temperatures=selected_temperatures)
    global_pts = [float(d['pressure']) for d in global_rows if d.get('pressure') is not None]
    global_p_min = float(min(global_pts)) if global_pts else None
    
    fw_first = selected_frameworks[0] if selected_frameworks else None
    fw_first = selected_frameworks[0] if selected_frameworks else None
    for mol_current in selected_molecules:
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        any_plotted = False
        global_max_sd = 0.0
        export_rows = []

        for fw_current in selected_frameworks:
            for ads_temp in selected_temperatures:
                ads_temp = float(ads_temp)
                for ft in selected_fit_types:
                    key = (fw_current, mol_current, ads_temp, ft)
                    if key not in fit_cache:
                        continue
                    params, ft_type = fit_cache[key]
                    L_ads = float(phelp.evaluate_fit(P_ads, params, ft_type))
                    try:
                        loads, qst = _get_qst(
                            method, fw_current, mol_current, selected_temperatures, selected_fit_types, fittings,
                            RASPA_data, x_fit,
                            smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                            smoothing_sigma=smoothing_sigma,
                            deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                            virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                            coeffs_a_override=coeffs_a_override,
                            coeffs_b_override=coeffs_b_override,
                            degrees_per_combo=degrees_per_combo,
                            qst_cache=qst_cache)
                    except Exception as e:
                        print(f"Warning: Qst failed for {fw_current}, {mol_current}, {int(ads_temp)}K: {e}")
                        loads, qst = None, None
                    lf, qf, q_min, q_max = _finite_qst(loads, qst)
                    if lf is None or not np.isfinite(L_ads) or L_ads < q_min or L_ads > q_max:
                        if lf is not None and q_min is not None and q_max is not None:
                            _warn_l_ads_outside_qst_once(
                                fw_current, mol_current, ads_temp, P_ads,
                                q_min, q_max, L_ads, params, ft_type,
                            )
                        continue
                    p_des_list, thermo_list = [], []
                    for p_des in x_fit:
                        L_des = float(phelp.evaluate_fit(p_des, params, ft_type))
                        val = integrate_enthalpy(min(L_des, L_ads), max(L_des, L_ads), lf, qf)
                        if np.isfinite(val):
                            p_des_list.append(p_des)
                            thermo_list.append(val)
                            # Keep x-axis behavior unchanged, but compute y-axis
                            # range from the displayed x-window only.
                            in_window = True
                            if p_min is not None and float(p_des) < float(p_min):
                                in_window = False
                            if p_max is not None and float(p_des) > float(p_max):
                                in_window = False
                            if P_des_max is not None and float(p_des) > float(P_des_max):
                                in_window = False
                            if in_window:
                                global_max_sd = max(global_max_sd, val)

                    if p_des_list:
                        style = phelp.resolve_series_style(
                            fw_current, mol_current, ads_temp,
                            vary_fw=True, vary_mol=False, vary_T=len(selected_temperatures or []) > 1,
                            plot_kind="storage_density",
                            combo_colors=combo_colors,
                        )
                        color = style.get("color") or combo_colors.get((fw_current, mol_current, ads_temp))
                        # Legend should not repeat structures; we only label for the first structure.
                        if fw_current == fw_first:
                            label = phelp.build_series_label(
                                fw_current, mol_current, ads_temp,
                                vary_fw=False, vary_mol=False,
                                vary_T=len(selected_temperatures or []) > 1,
                            )
                        else:
                            label = "_nolegend_"

                        ax.plot(
                            p_des_list, thermo_list,
                            label=label,
                            color=color,
                            lw=phelp.LINEWIDTH,
                            alpha=phelp.ALPHA,
                            linestyle=style.get("linestyle", "-"),
                        )
                        any_plotted = True
                        for P_des, sd_val in zip(p_des_list, thermo_list):
                            export_rows.append({
                                'framework': fw_current,
                                'molecule': mol_current,
                                'T_ads': float(ads_temp),
                                'T_des': float(ads_temp),
                                'P_ads': float(P_ads),
                                'P_des': float(P_des),
                                'SD': float(sd_val),
                                'method': method,
                            })

        if not any_plotted:
            plt.close(fig)
            continue

        subtitle = f"{method.upper()} (P_ads={P_ads} Pa)"
        phelp.format_storage_plot(ax, "Storage Density", p_min, p_max, P_des_max,
                                  global_p_min, subtitle=subtitle, scale=scale)
        ax.set_ylim(bottom=0, top=global_max_sd * 1.05 if global_max_sd > 0 else None)
        ax.legend(loc='best', fontsize=phelp.AXIS_LEGEND_SIZE)
        phelp.apply_unified_axes_layout(fig, ax)
        _sd_2d_axis_label_weight(ax)

        folder_temps = folder_temperatures if folder_temperatures is not None else selected_temperatures
        out_dir_2d = _get_storage_density_out_dir(
            selected_frameworks, selected_molecules, folder_temps,
            dimension="2D", fw=None, mol=mol_current
        )
        _stem_ps = _sd_filename_ps_teq(method)
        out_path = phelp._save_plot(
            _stem_ps, 'plot_storage_density',
            selected_frameworks, selected_molecules, folder_temps,
            fig=fig, out_dir=out_dir_2d,
            fw_label_override=None,
            mol_label_override=str(mol_current).replace(" ", "_"),
        )
        if save_data:
            _save_sd_rows(out_path, export_rows, _stem_ps)
        plt.show()
        plt.close(fig)


def plot_storage_density_fixed_ads(method, selected_frameworks, selected_molecules, T_ads, P_ads,
                                   desorption_temperatures, selected_fit_types, fittings,
                                   x_fit, combo_colors, RASPA_data,
                                   p_min=None, p_max=None, P_des_max=None,
                                   smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5,
                                   num_of_isotherm=None,
                                   deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                                   coeffs_a_override=None, coeffs_b_override=None,
                                   degrees_per_combo=None, folder_temperatures=None,
                                   qst_cache=None, save_data=False, scale='log'):
    """Storage density with a fixed *T_ads* / *P_ads*, varying *T_des*.

    *method* is ``'cc'`` or ``'virial'``.
    """
    all_temps = [T_ads] + list(desorption_temperatures)
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules,
        all_temps, selected_fit_types, num_of_isotherm=num_of_isotherm)

    # For legend clarity in fixed-adsorption SD plots, the user only wants temperatures in the legend.
    vary_T = len(desorption_temperatures or []) > 1
    vary_mol = False
    vary_fw_style = True  # encode structure by linestyle inside the same (molecule) figure

    fw_first = selected_frameworks[0] if selected_frameworks else None

    for mol_current in selected_molecules:
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        any_plotted = False
        global_max_sd = 0.0
        export_rows = []

        for fw_current in selected_frameworks:
            ads_key = next(
                (k for k in fit_cache
                 if k[0] == fw_current and k[1] == mol_current
                 and isclose(k[2], float(T_ads), abs_tol=1)
                 and k[3] in selected_fit_types), None)
            if not ads_key:
                continue
            ads_params, ads_ft = fit_cache[ads_key]
            L_ads = float(phelp.evaluate_fit(P_ads, ads_params, ads_ft))

            for t_des in desorption_temperatures:
                des_key = next(
                    (k for k in fit_cache
                     if k[0] == fw_current and k[1] == mol_current
                     and isclose(k[2], float(t_des), abs_tol=1)
                     and k[3] in selected_fit_types), None)
                if not des_key:
                    continue
                des_params, des_ft = fit_cache[des_key]
                q_des_all = phelp.evaluate_fit(x_fit, des_params, des_ft)

                try:
                    loads, qst = _get_qst(
                        method, fw_current, mol_current, all_temps, selected_fit_types, fittings,
                        RASPA_data, x_fit,
                        smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                        smoothing_sigma=smoothing_sigma,
                        min_temps=max(2, len(all_temps)),
                        deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                        virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                        coeffs_a_override=coeffs_a_override,
                        coeffs_b_override=coeffs_b_override,
                        degrees_per_combo=degrees_per_combo,
                        qst_cache=qst_cache)
                except Exception as e:
                    print(f"Warning: Qst failed for {fw_current}, {mol_current}: {e}")
                    loads, qst = None, None

                lf, qf, q_min, q_max = _finite_qst(loads, qst)
                if lf is None:
                    print(f"Warning: No valid Qst for {fw_current}, {mol_current}, Tdes={int(float(t_des))}K — skipping")
                    continue
                if not np.isfinite(L_ads) or L_ads < q_min or L_ads > q_max:
                    _warn_l_ads_outside_qst_once(
                        fw_current, mol_current, T_ads, P_ads,
                        q_min, q_max, L_ads, ads_params, ads_ft,
                    )
                    continue

                sd_list, valid_p = [], []
                for idx, q_des in enumerate(q_des_all):
                    if not np.isfinite(q_des):
                        continue
                    val = integrate_enthalpy(min(q_des, L_ads), max(q_des, L_ads), lf, qf)
                    if np.isfinite(val):
                        sd_list.append(val)
                        valid_p.append(x_fit[idx])
                        global_max_sd = max(global_max_sd, val)

                if sd_list:
                    style = phelp.resolve_series_style(
                        fw_current, mol_current, t_des,
                        vary_fw=vary_fw_style, vary_mol=vary_mol, vary_T=vary_T,
                        plot_kind="storage_density",
                        combo_colors=combo_colors,
                    )
                    color = style.get("color") or combo_colors.get((fw_current, mol_current, t_des),
                                                                      combo_colors.get((fw_current, mol_current, T_ads)))
                    # Legend label: only temperature (e.g. '283K'); don't repeat for multiple structures.
                    if fw_current == fw_first:
                        label = phelp.build_series_label(
                            fw_current, mol_current, t_des,
                            vary_fw=False, vary_mol=False, vary_T=True,
                        )
                    else:
                        label = "_nolegend_"

                    ax.plot(
                        valid_p, sd_list,
                        label=label,
                        color=color,
                        lw=phelp.LINEWIDTH,
                        alpha=phelp.ALPHA,
                        linestyle=style.get("linestyle", "-"),
                    )
                    any_plotted = True
                    for P_des, sd_val in zip(valid_p, sd_list):
                        export_rows.append({
                            'framework': fw_current,
                            'molecule': mol_current,
                            'T_ads': float(T_ads),
                            'T_des': float(t_des),
                            'P_ads': float(P_ads),
                            'P_des': float(P_des),
                            'SD': float(sd_val),
                            'method': method,
                        })

        if not any_plotted:
            plt.close(fig)
            continue

        subtitle = f"{method.upper()} (T_ads={T_ads}K, P_ads={P_ads} Pa)"
        phelp.format_storage_plot(ax, "Storage Density", p_min, p_max, P_des_max, subtitle=subtitle, scale=scale)
        ax.set_ylim(bottom=0, top=global_max_sd * 1.05 if global_max_sd > 0 else None)
        ax.legend(loc='best', fontsize=phelp.AXIS_LEGEND_SIZE)
        phelp.apply_unified_axes_layout(fig, ax)
        _sd_2d_axis_label_weight(ax)

        folder_temps = folder_temperatures if folder_temperatures is not None else desorption_temperatures
        out_dir_2d = _get_storage_density_out_dir(
            selected_frameworks, selected_molecules, folder_temps,
            dimension="2D", fw=None, mol=mol_current
        )
        _stem_pts = _sd_filename_pts_fixedpads(method)
        out_path = phelp._save_plot(
            _stem_pts, 'plot_storage_density',
            selected_frameworks, selected_molecules, folder_temps,
            fig=fig, out_dir=out_dir_2d,
            fw_label_override=None,
            mol_label_override=str(mol_current).replace(" ", "_"),
        )
        if save_data:
            _save_sd_rows(out_path, export_rows, _stem_pts)
        plt.show()
        plt.close(fig)


def plot_storage_density_temperature_series(method, selected_frameworks, selected_molecules,
                                            T_ads, P_ads, desorption_temperatures, selected_fit_types,
                                            fittings, x_fit, combo_colors, RASPA_data,
                                            p_min=None, p_max=None, P_des_max=None,
                                            smooth=False, use_direct_interpolation=False,
                                            smoothing_sigma=1.5, num_of_isotherm=None,
                                            deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                                            coeffs_a_override=None, coeffs_b_override=None,
                                            degrees_per_combo=None, folder_temperatures=None,
                                            qst_cache=None, save_data=False, scale='log'):
    """Storage density temperature series: fixed *T_ads*, multiple *T_des* on one plot.

    *method* is ``'cc'`` or ``'virial'``.
    """
    fit_cache = {}
    for fit in fittings:
        ft = fit.get('fit_type', None)
        if ft is None:
            # Some fit rows may miss `fit_type`; if the caller requested a
            # single fit type, we can safely assume it.
            if isinstance(selected_fit_types, (list, tuple)) and len(selected_fit_types) == 1:
                ft = selected_fit_types[0]
            else:
                continue
        framework = fit.get('framework')
        molecule = fit.get('molecule')
        temperature = fit.get('temperature')
        params = fit.get('params', None)
        if framework is None or molecule is None or temperature is None or params is None:
            continue
        key = (framework, molecule, temperature, ft)
        fit_cache.setdefault(key, (params, ft))

    all_temps = [T_ads] + list(desorption_temperatures)
    p_arr = np.asarray(x_fit, dtype=float)

    fw_first = selected_frameworks[0] if selected_frameworks else None
    for mol_current in selected_molecules:
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        any_plotted = False
        global_max_sd = 0.0
        export_rows = []

        for fw_current in selected_frameworks:
            base_color = combo_colors.get((fw_current, mol_current, T_ads))
            ads_key = next(
                (k for k in fit_cache
                 if k[0] == fw_current and k[1] == mol_current
                 and abs(k[2] - float(T_ads)) < 1
                 and k[3] in selected_fit_types), None)
            if not ads_key:
                continue
            ads_params, ads_ft = fit_cache[ads_key]
            q_ads_all = phelp.evaluate_fit(p_arr, ads_params, ads_ft)
            for t_des in desorption_temperatures:
                des_key = next(
                    (k for k in fit_cache
                     if k[0] == fw_current and k[1] == mol_current
                     and abs(k[2] - float(t_des)) < 1
                     and k[3] in selected_fit_types), None)
                if not des_key:
                    continue
                des_params, des_ft = fit_cache[des_key]
                q_des_all = phelp.evaluate_fit(p_arr, des_params, des_ft)
                try:
                    loads, qst = _get_qst(
                        method, fw_current, mol_current, all_temps, selected_fit_types, fittings,
                        RASPA_data, x_fit,
                        smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                        smoothing_sigma=smoothing_sigma,
                        min_temps=max(2, len(all_temps)),
                        deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                        virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                        coeffs_a_override=coeffs_a_override,
                        coeffs_b_override=coeffs_b_override,
                        degrees_per_combo=degrees_per_combo,
                        qst_cache=qst_cache)
                except Exception as e:
                    print(f"Warning: Qst failed for {fw_current}, {mol_current}: {e}")
                    loads, qst = None, None
                lf, qf, _, _ = _finite_qst(loads, qst)
                if lf is None:
                    continue
                sd_list, valid_p = [], []
                for idx in range(len(p_arr)):
                    q_a = float(q_ads_all[idx]); q_d = float(q_des_all[idx])
                    if not np.isfinite(q_a) or not np.isfinite(q_d):
                        continue
                    val = integrate_enthalpy(min(q_d, q_a), max(q_d, q_a), lf, qf)
                    if np.isfinite(val):
                        sd_list.append(val)
                        valid_p.append(float(p_arr[idx]))
                        global_max_sd = max(global_max_sd, val)

                if sd_list:
                    style = phelp.resolve_series_style(
                        fw_current, mol_current, t_des,
                        vary_fw=True, vary_mol=False, vary_T=True,
                        plot_kind="storage_density",
                        combo_colors=combo_colors,
                    )
                    color = style.get("color") or combo_colors.get((fw_current, mol_current, t_des), base_color)
                    # Legend should only show temperature; framework distinction
                    # remains visible via linestyle.
                    if fw_current == fw_first:
                        label = f"{int(float(t_des))}K"
                    else:
                        label = "_nolegend_"
                    ax.plot(
                        valid_p, sd_list,
                        label=label,
                        color=color,
                        lw=phelp.LINEWIDTH,
                        alpha=phelp.ALPHA,
                        linestyle=style.get("linestyle", "-"),
                    )
                    any_plotted = True
                    for P_des, sd_val in zip(valid_p, sd_list):
                        export_rows.append({
                            'framework': fw_current,
                            'molecule': mol_current,
                            'T_ads': float(T_ads),
                            'T_des': float(t_des),
                            'P_ads': float(P_ads),
                            'P_des': float(P_des),
                            'SD': float(sd_val),
                            'method': method
                        })

        if not any_plotted:
            plt.close(fig)
            continue

        subtitle = f"{method.upper()} (T_ads={T_ads}K)"
        phelp.format_storage_plot(ax, "Storage Density", p_min, p_max, P_des_max, None, subtitle=subtitle, scale=scale)
        ax.set_ylim(bottom=0, top=global_max_sd * 1.05 if global_max_sd > 0 else None)
        ax.legend(loc='best', fontsize=phelp.AXIS_LEGEND_SIZE)
        phelp.apply_unified_axes_layout(fig, ax)
        _sd_2d_axis_label_weight(ax)

        folder_temps = folder_temperatures if folder_temperatures is not None else desorption_temperatures
        out_dir_2d = _get_storage_density_out_dir(
            selected_frameworks, selected_molecules, folder_temps,
            dimension="2D", fw=None, mol=mol_current
        )
        _stem_ts = _sd_filename_ts_peq(method)
        out_path = phelp._save_plot(
            _stem_ts, 'plot_storage_density',
            selected_frameworks, selected_molecules, folder_temps,
            fig=fig, out_dir=out_dir_2d,
            fw_label_override=None,
            mol_label_override=str(mol_current).replace(" ", "_"),
        )
        if save_data:
            _save_sd_rows(out_path, export_rows, _stem_ts)
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3-D surface plots  (CC only)
# ---------------------------------------------------------------------------

def plot_storage_density_3d(method, selected_frameworks, selected_molecules, selected_temperatures,
                            selected_fit_types, fittings, P_ads, x_fit, combo_colors, RASPA_data,
                            p_min=None, p_max=None, P_des_max=None,
                            smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5,
                            num_of_isotherm=None,
                            deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                            coeffs_a_override=None, coeffs_b_override=None,
                            degrees_per_combo=None, folder_temperatures=None,
                            qst_cache=None):
    """3-D surface: adsorption/desorption temperature × desorption pressure × storage density.

    *method* is ``'cc'`` or ``'virial'``.  Axes: X = temperature [K], Y = log10(P_des), Z = SD.
    """
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules, selected_temperatures,
        selected_fit_types, num_of_isotherm=num_of_isotherm)

    p_list = phelp.filter_pressures(x_fit, P_des_max) if P_des_max is not None else x_fit
    if len(p_list) == 0:
        p_list = x_fit

    any_plotted = False
    for fw in selected_frameworks:
        for mol in selected_molecules:
            T_vals, P_vals, SD_vals = [], [], []
            for ads_temp in selected_temperatures:
                ads_temp = float(ads_temp)
                for ft in selected_fit_types:
                    key = (fw, mol, ads_temp, ft)
                    if key not in fit_cache:
                        continue
                    params, ft_type = fit_cache[key]
                    L_ads = float(phelp.evaluate_fit(P_ads, params, ft_type))

                    try:
                        loads, qst = _get_qst(
                            method, fw, mol, selected_temperatures, selected_fit_types, fittings,
                            RASPA_data, x_fit,
                            smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                            smoothing_sigma=smoothing_sigma,
                            deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                            virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                            coeffs_a_override=coeffs_a_override,
                            coeffs_b_override=coeffs_b_override,
                            degrees_per_combo=degrees_per_combo,
                            qst_cache=qst_cache)
                    except Exception:
                        loads, qst = None, None

                    lf, qf, q_min, q_max = _finite_qst(loads, qst)
                    if lf is None or not np.isfinite(L_ads) or L_ads < q_min or L_ads > q_max:
                        if lf is not None and q_min is not None and q_max is not None:
                            _warn_l_ads_outside_qst_once(
                                fw, mol, ads_temp, P_ads, q_min, q_max, L_ads, params, ft_type,
                            )
                        continue

                    for p_des in p_list:
                        L_des = float(phelp.evaluate_fit(p_des, params, ft_type))
                        val = integrate_enthalpy(min(L_des, L_ads), max(L_des, L_ads), lf, qf)
                        if np.isfinite(val):
                            T_vals.append(ads_temp)
                            P_vals.append(p_des)
                            SD_vals.append(val)

            if not T_vals:
                continue
            label = 'Clausius\u2013Clapeyron' if method == 'cc' else method.capitalize()
            title = (f"Storage Density from {label}\n"
                     r"$P_{{ads}}$ = {:.2g} Pa, $T_{{ads}}$ = $T_{{des}}$".format(P_ads))
            folder_temps = folder_temperatures if folder_temperatures is not None else selected_temperatures
            _render_surface(T_vals, P_vals, SD_vals, title,
                            xlabel=SD3D_LABEL_T_DES,
                            ylabel=SD3D_LABEL_P_DES,
                            swap_xy=False,
                            selected_frameworks=selected_frameworks,
                            selected_molecules=selected_molecules,
                            save_name=_sd_filename_ps_teq(method, dim_3d=True),
                            folder_temps=folder_temps,
                            combo_fw=fw,
                            combo_mol=mol)
            any_plotted = True
    if not any_plotted:
        print(f"Warning: No valid data for 3D storage density plot (method={method}).")


def plot_storage_density_fixed_ads_3d(method, selected_frameworks, selected_molecules, T_ads, P_ads,
                                      desorption_temperatures, selected_fit_types, fittings,
                                      x_fit, combo_colors, RASPA_data,
                                      p_min=None, p_max=None, P_des_max=None,
                                      smooth=False, use_direct_interpolation=False,
                                      smoothing_sigma=1.5, num_of_isotherm=None,
                                      deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                                      coeffs_a_override=None, coeffs_b_override=None,
                                      degrees_per_combo=None, folder_temperatures=None,
                                      qst_cache=None):
    """3-D surface: desorption temperature × desorption pressure × storage density (fixed T_ads, P_ads).

    *method* is ``'cc'`` or ``'virial'``.  Axes: X = log10(P_des), Y = T_des [K], Z = SD.
    """
    all_temps = [T_ads] + list(desorption_temperatures)
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules, all_temps,
        selected_fit_types, num_of_isotherm=num_of_isotherm)

    p_list = phelp.filter_pressures(x_fit, P_des_max) if P_des_max is not None else x_fit
    if len(p_list) == 0:
        p_list = x_fit

    any_plotted = False
    for fw in selected_frameworks:
        for mol in selected_molecules:
            T_vals, P_vals, SD_vals = [], [], []
            ads_key = next(
                (k for k in fit_cache
                 if k[0] == fw and k[1] == mol
                 and isclose(k[2], float(T_ads), abs_tol=1)
                 and k[3] in selected_fit_types), None)
            if not ads_key:
                continue
            ads_params, ads_ft = fit_cache[ads_key]
            L_ads = float(phelp.evaluate_fit(P_ads, ads_params, ads_ft))

            for t_des in desorption_temperatures:
                des_key = next(
                    (k for k in fit_cache
                     if k[0] == fw and k[1] == mol
                     and isclose(k[2], float(t_des), abs_tol=1)
                     and k[3] in selected_fit_types), None)
                if not des_key:
                    continue
                des_params, des_ft = fit_cache[des_key]
                q_des_all = phelp.evaluate_fit(p_list, des_params, des_ft)

                try:
                    loads, qst = _get_qst(
                        method, fw, mol, all_temps, selected_fit_types, fittings,
                        RASPA_data, x_fit,
                        smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                        smoothing_sigma=smoothing_sigma,
                        min_temps=max(2, len(all_temps)),
                        deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                        virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                        coeffs_a_override=coeffs_a_override,
                        coeffs_b_override=coeffs_b_override,
                        degrees_per_combo=degrees_per_combo,
                        qst_cache=qst_cache)
                except Exception:
                    loads, qst = None, None

                lf, qf, q_min, q_max = _finite_qst(loads, qst)
                if lf is None or not np.isfinite(L_ads) or L_ads < q_min or L_ads > q_max:
                    if lf is not None and q_min is not None and q_max is not None:
                        _warn_l_ads_outside_qst_once(
                            fw, mol, T_ads, P_ads, q_min, q_max, L_ads, ads_params, ads_ft,
                        )
                    continue

                for idx, q_des in enumerate(q_des_all):
                    if not np.isfinite(q_des):
                        continue
                    p_val = float(p_list[idx])
                    if P_des_max is not None and p_val > P_des_max:
                        continue
                    val = integrate_enthalpy(min(float(q_des), L_ads), max(float(q_des), L_ads), lf, qf)
                    if np.isfinite(val):
                        T_vals.append(float(t_des))
                        P_vals.append(p_val)
                        SD_vals.append(val)
            if not T_vals:
                continue
            label = 'Clausius\u2013Clapeyron' if method == 'cc' else method.capitalize()
            title = (f"Storage Density from {label} (fixed $T_{{ads}}$, $P_{{ads}}$)\n"
                     f"$T_{{ads}}$ = {int(T_ads)} K, $P_{{ads}}$ = {P_ads:.2g} Pa")
            folder_temps = folder_temperatures if folder_temperatures is not None else desorption_temperatures
            _render_surface(T_vals, P_vals, SD_vals, title,
                            xlabel=SD3D_LABEL_P_DES,
                            ylabel=SD3D_LABEL_T_DES,
                            swap_xy=True,
                            selected_frameworks=selected_frameworks,
                            selected_molecules=selected_molecules,
                            save_name=_sd_filename_pts_fixedpads(method, dim_3d=True),
                            folder_temps=folder_temps,
                            combo_fw=fw,
                            combo_mol=mol)
            any_plotted = True
    if not any_plotted:
        print(f"Warning: No valid data for 3D fixed-ads storage density plot (method={method}).")


def plot_storage_density_temperature_series_3d(method, selected_frameworks, selected_molecules,
                                               T_ads, desorption_temperatures, selected_fit_types,
                                               fittings, x_fit, combo_colors, RASPA_data,
                                               p_min=None, p_max=None, P_des_max=None,
                                               smooth=False, use_direct_interpolation=False,
                                               smoothing_sigma=1.5, num_of_isotherm=None,
                                               deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                                               coeffs_a_override=None, coeffs_b_override=None,
                                               degrees_per_combo=None, folder_temperatures=None,
                                               qst_cache=None):
    """3-D surface: desorption temperature × pressure × storage density (temperature series).

    *method* is ``'cc'`` or ``'virial'``.  Axes: X = T_des [K], Y = log10(P), Z = SD.

    ``fit_cache`` and ``p_arr`` match ``plot_storage_density_temperature_series`` (2D)
    so mixture / synthetic fits and pressure samples stay aligned between plots.
    """
    fit_cache = {}
    for fit in fittings:
        ft = fit.get('fit_type', None)
        if ft is None:
            if isinstance(selected_fit_types, (list, tuple)) and len(selected_fit_types) == 1:
                ft = selected_fit_types[0]
            else:
                continue
        framework = fit.get('framework')
        molecule = fit.get('molecule')
        temperature = fit.get('temperature')
        params = fit.get('params', None)
        if framework is None or molecule is None or temperature is None or params is None:
            continue
        key = (framework, molecule, temperature, ft)
        fit_cache.setdefault(key, (params, ft))

    all_temps = [T_ads] + list(desorption_temperatures)
    p_arr = np.asarray(x_fit, dtype=float)

    any_plotted = False
    for fw in selected_frameworks:
        for mol in selected_molecules:
            T_vals, P_vals, SD_vals = [], [], []
            ads_key = next(
                (k for k in fit_cache
                 if k[0] == fw and k[1] == mol
                 and abs(k[2] - float(T_ads)) < 1
                 and k[3] in selected_fit_types), None)
            if not ads_key:
                continue
            ads_params, ads_ft = fit_cache[ads_key]
            q_ads_all = phelp.evaluate_fit(p_arr, ads_params, ads_ft)

            for t_des in desorption_temperatures:
                des_key = next(
                    (k for k in fit_cache
                     if k[0] == fw and k[1] == mol
                     and abs(k[2] - float(t_des)) < 1
                     and k[3] in selected_fit_types), None)
                if not des_key:
                    continue
                des_params, des_ft = fit_cache[des_key]
                q_des_all = phelp.evaluate_fit(p_arr, des_params, des_ft)

                try:
                    loads, qst = _get_qst(
                        method, fw, mol, all_temps, selected_fit_types, fittings,
                        RASPA_data, x_fit,
                        smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                        smoothing_sigma=smoothing_sigma,
                        min_temps=max(2, len(all_temps)),
                        deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                        virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                        coeffs_a_override=coeffs_a_override,
                        coeffs_b_override=coeffs_b_override,
                        degrees_per_combo=degrees_per_combo,
                        qst_cache=qst_cache)
                except Exception:
                    loads, qst = None, None

                lf, qf, _, _ = _finite_qst(loads, qst)
                if lf is None:
                    continue

                for idx in range(len(p_arr)):
                    q_a = float(q_ads_all[idx])
                    q_d = float(q_des_all[idx])
                    if not np.isfinite(q_a) or not np.isfinite(q_d):
                        continue
                    val = integrate_enthalpy(min(q_d, q_a), max(q_d, q_a), lf, qf)
                    if np.isfinite(val):
                        T_vals.append(float(t_des))
                        P_vals.append(float(p_arr[idx]))
                        SD_vals.append(val)
            if not T_vals:
                continue
            label = 'Clausius\u2013Clapeyron' if method == 'cc' else method.capitalize()
            title = (f"Storage Density from {label} (temperature series)\n"
                     r"$T_{{ads}}$ = {} K".format(int(T_ads)))
            folder_temps = folder_temperatures if folder_temperatures is not None else desorption_temperatures
            _render_surface(T_vals, P_vals, SD_vals, title,
                            xlabel=SD3D_LABEL_T_DES,
                            ylabel=SD3D_LABEL_P_DES,
                            swap_xy=False,
                            selected_frameworks=selected_frameworks,
                            selected_molecules=selected_molecules,
                            save_name=_sd_filename_ts_peq(method, dim_3d=True),
                            folder_temps=folder_temps,
                            combo_fw=fw,
                            combo_mol=mol)
            any_plotted = True
    if not any_plotted:
        print(f"Warning: No valid data for 3D temperature-series storage density plot (method={method}).")


# ---------------------------------------------------------------------------
# Special 3-D plot: T_ads × T_des surface
# ---------------------------------------------------------------------------

def plot_storage_density_3d_Tads_Tdes(method, selected_frameworks, selected_molecules,
                                      adsorption_temperatures, desorption_temperatures,
                                      selected_fit_types, fittings,
                                      P_ads_const, P_des_const, x_fit, combo_colors, RASPA_data,
                                      p_min=None, p_max=None, P_des_max=None,
                                      smooth=False, use_direct_interpolation=False,
                                      smoothing_sigma=1.5, num_of_isotherm=None,
                                      deg_a=2, deg_b=2, min_points=3, virial_p_min=None,
                                      coeffs_a_override=None, coeffs_b_override=None,
                                      degrees_per_combo=None, folder_temperatures=None,
                                      qst_cache=None, save_data=False):
    """3-D surface: T_ads × T_des × storage density at fixed P_ads and P_des.

    *method* is ``'cc'`` or ``'virial'``.
    """
    all_temps = sorted({float(t) for t in list(adsorption_temperatures) + list(desorption_temperatures)})
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules, all_temps,
        selected_fit_types, num_of_isotherm=num_of_isotherm)

    any_plotted = False
    for fw in selected_frameworks:
        for mol in selected_molecules:
            T_ads_vals, T_des_vals, SD_vals = [], [], []
            try:
                loads, qst = _get_qst(
                    method, fw, mol, all_temps, selected_fit_types, fittings,
                    RASPA_data, x_fit,
                    smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                    smoothing_sigma=smoothing_sigma,
                    min_temps=max(2, len(all_temps)),
                    deg_a=deg_a, deg_b=deg_b, min_points=min_points,
                    virial_p_min=virial_p_min if virial_p_min is not None else p_min,
                    coeffs_a_override=coeffs_a_override,
                    coeffs_b_override=coeffs_b_override,
                    degrees_per_combo=degrees_per_combo,
                    qst_cache=qst_cache)
            except Exception:
                loads, qst = None, None

            lf, qf, _, _ = _finite_qst(loads, qst)
            if lf is None:
                continue

            for T_ads_val in adsorption_temperatures:
                T_ads_val = float(T_ads_val)
                ads_key = next(
                    (k for k in fit_cache
                     if k[0] == fw and k[1] == mol
                     and isclose(k[2], T_ads_val, abs_tol=1)
                     and k[3] in selected_fit_types), None)
                if not ads_key:
                    continue
                ads_params, ads_ft = fit_cache[ads_key]

                for T_des_val in desorption_temperatures:
                    T_des_val = float(T_des_val)
                    des_key = next(
                        (k for k in fit_cache
                         if k[0] == fw and k[1] == mol
                         and isclose(k[2], T_des_val, abs_tol=1)
                         and k[3] in selected_fit_types), None)
                    if not des_key:
                        continue
                    des_params, des_ft = fit_cache[des_key]

                    L_ads = float(phelp.evaluate_fit(P_ads_const, ads_params, ads_ft))
                    L_des = float(phelp.evaluate_fit(P_des_const, des_params, des_ft))
                    if not np.isfinite(L_ads) or not np.isfinite(L_des):
                        continue
                    val = integrate_enthalpy(min(L_des, L_ads), max(L_des, L_ads), lf, qf)
                    if np.isfinite(val):
                        T_ads_vals.append(T_ads_val)
                        T_des_vals.append(T_des_val)
                        SD_vals.append(val)

            if not SD_vals:
                continue

            T_ads_vals = np.asarray(T_ads_vals, dtype=float)
            T_des_vals = np.asarray(T_des_vals, dtype=float)
            SD_vals = np.asarray(SD_vals, dtype=float)

            T_ads_u = np.sort(np.unique(T_ads_vals))
            T_des_u = np.sort(np.unique(T_des_vals))
            SD_base = np.full((len(T_ads_u), len(T_des_u)), np.nan, dtype=float)
            ai = {t: i for i, t in enumerate(T_ads_u)}
            di = {t: j for j, t in enumerate(T_des_u)}
            for Ta, Td, SD in zip(T_ads_vals, T_des_vals, SD_vals):
                i, j = ai.get(Ta), di.get(Td)
                if i is not None and j is not None:
                    SD_base[i, j] = SD

            nTa = max(2 * len(T_ads_u) - 1, 50)
            nTd = max(2 * len(T_des_u) - 1, 50)
            T_ads_fine = np.linspace(T_ads_u.min(), T_ads_u.max(), nTa)
            T_des_fine = np.linspace(T_des_u.min(), T_des_u.max(), nTd)

            SD_mid = np.full((nTa, len(T_des_u)), np.nan, dtype=float)
            for j in range(len(T_des_u)):
                col = SD_base[:, j]
                mask = np.isfinite(col)
                if np.count_nonzero(mask) >= 2:
                    SD_mid[:, j] = np.interp(T_ads_fine, T_ads_u[mask], col[mask])

            SD_grid = np.full((nTa, nTd), np.nan, dtype=float)
            for i in range(nTa):
                row = SD_mid[i, :]
                mask = np.isfinite(row)
                if np.count_nonzero(mask) >= 2:
                    SD_grid[i, :] = np.interp(T_des_fine, T_des_u[mask], row[mask])

            T_ads_grid, T_des_grid = np.meshgrid(T_ads_fine, T_des_fine, indexing='ij')

            cmap = phelp.resolve_storage_density_3d_colormap()
            vmin, vmax = phelp.resolve_storage_density_3d_boundries()
            surf_kw = dict(cmap=cmap, edgecolor='none', antialiased=True)
            if vmin is not None:
                surf_kw['vmin'] = vmin
            if vmax is not None:
                surf_kw['vmax'] = vmax
            fig = plt.figure(figsize=phelp.resolve_storage_density_3d_figsize())
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(T_ads_grid, T_des_grid, SD_grid, **surf_kw)
            cb = _sd3d_apply_axis_labels_and_colorbar(
                fig, ax, surf, SD3D_LABEL_T_ADS, SD3D_LABEL_T_DES, SD3D_LABEL_SD)
            folder_temps = folder_temperatures if folder_temperatures is not None else desorption_temperatures
            out_dir_3d = _get_storage_density_out_dir(
                selected_frameworks, selected_molecules, folder_temps,
                dimension="3D",
                fw=fw, mol=mol,
            )
            fw_safe = str(fw).replace(" ", "_")
            mol_safe = str(mol).replace(" ", "_")
            _stem_tadt = f"sd_Tads_Tdes_{_storage_density_3d_filename_method_token(method)}_3d"
            out_path = phelp._save_plot(
                _stem_tadt, 'plot_storage_density',
                selected_frameworks, selected_molecules, folder_temps, fig=fig, out_dir=out_dir_3d,
                fw_label_override=fw_safe,
                mol_label_override=mol_safe,
                filename_suffix=f"{fw_safe}__{mol_safe}",
                tight_bbox=True,
                bbox_extra_artists=_sd3d_bbox_extra_artists(ax, cb),
            )

            if save_data:
                export_rows = []
                for Ta, Td, SD in zip(T_ads_vals, T_des_vals, SD_vals):
                    export_rows.append({
                        'framework': fw,
                        'molecule': mol,
                        'T_ads': float(Ta),
                        'T_des': float(Td),
                        'P_ads': float(P_ads_const),
                        'P_des': float(P_des_const),
                        'SD': float(SD),
                    })
                _save_sd_rows(out_path, export_rows, _stem_tadt)
            plt.show()
            plt.close(fig)
            any_plotted = True
    if not any_plotted:
        print("Warning: No valid data for 3D T_ads/T_des storage density plot.")


# ---------------------------------------------------------------------------
# Mixture helpers — build synthetic fittings and Qst cache
# ---------------------------------------------------------------------------

def _build_mixture_loading_table(mixture_data, frameworks, mixture_name, temperatures, tol=1.0):
    """Pre-build ``{(fw, T): (P_arr, q_total_arr)}`` for fast loading lookups.

    Sums component loadings at each shared pressure point.
    """
    from collections import defaultdict

    table = {}
    for fw in frameworks:
        for T in temperatures:
            T = float(T)
            rows = [d for d in mixture_data
                    if d.get('framework') == fw
                    and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()
                    and abs(float(d['temperature']) - T) < tol]
            if not rows:
                continue

            mol_pd: dict = defaultdict(lambda: defaultdict(list))
            for d in rows:
                mol_pd[d.get('molecule', '')][float(d['pressure'])].append(float(d['loading']))

            if not mol_pd:
                continue

            mol_means = {mol: {P: float(np.mean(qs)) for P, qs in pd.items()}
                         for mol, pd in mol_pd.items()}

            common_P = sorted(set.intersection(*[set(md.keys()) for md in mol_means.values()]))
            if len(common_P) < 2:
                continue

            P_arr = np.asarray(common_P, dtype=float)
            q_tot = sum(np.array([md[P] for P in common_P], dtype=float)
                        for md in mol_means.values())
            table[(fw, T)] = (P_arr, q_tot)
    return table


def make_mixture_fittings(mixture_data, frameworks, mixture_name, temperatures):
    """Create synthetic ``'interp'`` fit objects from RASPA total-loading data.

    Each returned dict looks like a normal fitting entry with
    ``fit_type='interp'`` and ``params=(P_arr, q_arr)``.  These work with
    ``phelp.evaluate_fit`` / ``Initialize.formula_fitting`` once the
    ``'interp'`` branch is present in ``formula_fitting``.
    """
    table = _build_mixture_loading_table(mixture_data, frameworks, mixture_name, temperatures)
    fittings = []
    for (fw, T), (P_arr, q_arr) in table.items():
        fittings.append({
            'framework':   fw,
            'molecule':    mixture_name,
            'temperature': T,
            'fit_type':    'interp',
            'params':      (P_arr, q_arr),
        })
    return fittings


def make_mixture_component_fittings(mixture_data, frameworks, mixture_name, components, temperatures, tol=1.0):
    """Create synthetic ``'interp'`` fit objects for each component from mixture RASPA data.

    Returns fitting dicts with molecule=<component>, fit_type='interp', params=(P_arr, q_arr).
    This is used to look up component loadings at (T, P) for SD decomposition.
    """
    fittings = []
    temps = [float(t) for t in temperatures] if temperatures else []
    for fw in frameworks:
        for T in temps:
            rows_T = [d for d in mixture_data
                      if d.get('framework') == fw
                      and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()
                      and abs(float(d.get('temperature', np.nan)) - float(T)) < float(tol)]
            if not rows_T:
                continue
            for comp in components:
                pts = []
                for d in rows_T:
                    if str(d.get('molecule', '')).strip().lower() != str(comp).strip().lower():
                        continue
                    P = d.get('pressure')
                    q = d.get('loading')
                    if P is None or q is None:
                        continue
                    try:
                        P = float(P)
                        q = float(q)
                    except Exception:
                        continue
                    if np.isfinite(P) and np.isfinite(q) and P > 0 and q >= 0:
                        pts.append((P, q))
                if len(pts) < 2:
                    continue
                pts.sort(key=lambda x: x[0])
                P_arr = np.asarray([p for p, _ in pts], dtype=float)
                q_arr = np.asarray([q for _, q in pts], dtype=float)
                fittings.append({
                    'framework': fw,
                    'molecule': comp,
                    'temperature': float(T),
                    'fit_type': 'interp',
                    'params': (P_arr, q_arr),
                })
    return fittings


def make_pure_interp_fittings_from_raspa(RASPA_data, frameworks, molecules, temperatures):
    """Synthetic ``interp`` fitting dicts from RASPA-like rows (``DATA_SOURCE=points``).

    Each entry has ``fit_type='interp'`` and ``params=(P_arr, q_arr)`` so
    :func:`PlotHelpers.build_fit_cache` and :func:`Initialize.formula_fitting` can
    evaluate loadings at ``P_ads`` / ``P_des``. Use with ``selected_fit_types=['interp']``
    and ``num_of_isotherm=None`` in storage-density calls so parameter-count filters do
    not drop these rows.

    Rows are filtered with ``only_pure_adsorption=True`` when ``mixture_pure`` is present
    so mixture-component isotherms are not mixed into pure-component SD.
    """
    if not RASPA_data or not frameworks or not molecules or not temperatures:
        return []
    temps_f = [float(t) for t in temperatures]
    out = []
    for fw in frameworks:
        for mol in molecules:
            for T in temps_f:
                pts = phelp.filter_raspa_data(
                    RASPA_data,
                    frameworks=[fw],
                    molecules=[mol],
                    temperatures=[T],
                    only_pure_adsorption=True,
                )
                pairs = []
                for d in pts:
                    P, q = d.get('pressure'), d.get('loading')
                    if P is None or q is None:
                        continue
                    try:
                        P = float(P)
                        q = float(q)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(P) and np.isfinite(q) and P > 0 and q >= 0:
                        pairs.append((P, q))
                if len(pairs) < 2:
                    continue
                pairs.sort(key=lambda x: x[0])
                P_arr = np.asarray([p for p, _ in pairs], dtype=float)
                q_arr = np.asarray([q for _, q in pairs], dtype=float)
                out.append({
                    'framework': fw,
                    'molecule': mol,
                    'temperature': float(T),
                    'fit_type': 'interp',
                    'params': (P_arr, q_arr),
                })
    return out


def make_mixture_qst_cache(mixture_data, frameworks, mixture_name, temperatures,
                            p_min, p_max, n_loadings=40, min_temps=2, smoothing_sigma=1.5):
    """Build ``{(fw, mixture_name): (loads, qst)}`` from mixture CC for use as *qst_cache*."""
    from scipy.ndimage import gaussian_filter1d
    from ClausiusClapeyron import compute_mixture_isosteric_heat_cc

    cache = {}
    for fw in frameworks:
        components = sorted({d['molecule'] for d in mixture_data
                              if d.get('framework') == fw
                              and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()})
        if not components:
            continue

        result = compute_mixture_isosteric_heat_cc(
            mixture_data=mixture_data,
            components=components,
            temperatures=temperatures,
            framework=fw,
            mixture_name=mixture_name,
            p_min=p_min,
            p_max=p_max,
            n_loadings=n_loadings,
            min_temps=min_temps,
        )

        if 'all' not in result:
            continue

        entry = result['all']
        loads = np.asarray(entry['loading'],   dtype=float)
        qst   = np.asarray(entry['Qst_kJmol'], dtype=float)
        mask  = np.isfinite(loads) & np.isfinite(qst)
        if not np.any(mask):
            continue
        loads, qst = loads[mask], qst[mask]
        order = np.argsort(loads)
        loads, qst = loads[order], qst[order]
        if len(qst) >= 3:
            qst = gaussian_filter1d(qst, sigma=smoothing_sigma)
        cache[(fw, mixture_name)] = (loads, qst)
    return cache


def make_mixture_qst_cache_hoa_pure_cc(mixture_data, frameworks, mixture_name,
                                       pure_temperatures, mix_temperature,
                                       fits_pure, RASPA_data_pure, selected_fit_types,
                                       p_min, p_max, n_loadings=40, min_temps=2,
                                       smoothing_sigma=1.5,
                                       use_direct_interpolation=False):
    """
    Build ``{(fw, mixture_name): (loads, qst)}`` for mixture HOA using the
    weighted-pure CC approach (hoa_pure_cc) at a *single* mixture temperature.

    - ``pure_temperatures``: temperatures used to compute pure-component Qst_i^0
      via Clausius-Clapeyron (same as for the pure HOA plots).
    - ``mix_temperature``: the mixture temperature (e.g. T_ads) at which the
      adsorbed-phase mole fractions y_i are taken from the mixture RASPA data.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d
    from ClausiusClapeyron import compute_isosteric_heat

    cache = {}
    temps_pure = [float(t) for t in pure_temperatures] if pure_temperatures else []
    T_mix = float(mix_temperature)
    p_min_b = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b = float(p_max) if p_max is not None else 1e8

    if not temps_pure:
        return cache

    for fw in frameworks:
        # Components present for this framework / mixture
        components = sorted({d['molecule'] for d in mixture_data
                             if d.get('framework') == fw
                             and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()})
        if not components:
            continue

        # 1) Pure-component CC Qst interpolators comp -> callable(n)
        qst_interp = {}
        for comp in components:
            try:
                res = compute_isosteric_heat(
                    framework=fw, molecule=comp,
                    temperatures=temps_pure,
                    selected_fit_types=selected_fit_types,
                    fittings=fits_pure,
                    RASPA_data=RASPA_data_pure,
                    n_loadings=n_loadings,
                    p_min=p_min, p_max=p_max,
                    min_temps=min_temps,
                    smooth=True, smoothing_sigma=smoothing_sigma,
                    use_direct_interpolation=use_direct_interpolation,
                )
            except Exception:
                continue

            qst_arr = (res.get('Qst_kJmol_smoothed')
                       if res.get('Qst_kJmol_smoothed') is not None
                       else res.get('Qst_kJmol'))
            if qst_arr is None:
                continue
            n_arr = np.asarray(res.get('loading'), dtype=float)
            qst_arr = np.asarray(qst_arr, dtype=float)
            ok = np.isfinite(n_arr) & np.isfinite(qst_arr)
            if ok.sum() < 3:
                continue
            n_arr, qst_arr = n_arr[ok], qst_arr[ok]
            idx = np.argsort(n_arr)
            n_arr, qst_arr = n_arr[idx], qst_arr[idx]
            _, u = np.unique(n_arr, return_index=True)
            n_arr, qst_arr = n_arr[u], qst_arr[u]
            if len(n_arr) < 3:
                continue
            try:
                qst_interp[comp] = PchipInterpolator(n_arr, qst_arr, extrapolate=False)
            except Exception:
                continue

        if not qst_interp:
            continue

        # 2) Mixture per-component RASPA data at the chosen mixture temperature
        comp_data = {c: [] for c in components}
        for d in mixture_data:
            if d.get('framework') != fw:
                continue
            mol = d.get('molecule')
            if mol not in components:
                continue
            T = float(d.get('temperature', T_mix))
            if not np.isclose(T, T_mix):
                continue
            P = float(d.get('pressure', np.nan))
            q = float(d.get('loading', np.nan))
            if (np.isfinite(P) and np.isfinite(q)
                    and P > 0 and q >= 0 and p_min_b <= P <= p_max_b):
                comp_data[mol].append((P, q))

        # 3) Build shared states and compute Qst_mix vs total loading
        by_pressure = {}
        for comp in components:
            for P, q in comp_data[comp]:
                by_pressure.setdefault(P, {})[comp] = q
        shared = [(P, cd) for P, cd in by_pressure.items()
                  if len(cd) == len(components)]
        if len(shared) < 3:
            continue

        shared.sort(key=lambda x: x[0])
        n_tot_arr = np.array([sum(s[1].values()) for s in shared], dtype=float)
        order = np.argsort(n_tot_arr)
        n_tot_sorted = n_tot_arr[order]
        shared_sorted = [shared[i] for i in order]
        _, uniq_idx = np.unique(n_tot_sorted, return_index=True)
        n_tot_sorted = n_tot_sorted[uniq_idx]
        shared_sorted = [shared_sorted[i] for i in uniq_idx]
        if len(n_tot_sorted) < 3 or n_tot_sorted.min() >= n_tot_sorted.max():
            continue

        load_grid = np.linspace(n_tot_sorted.min(), n_tot_sorted.max(), int(n_loadings))
        Qst_mix = np.full_like(load_grid, np.nan, dtype=float)

        for j, n_tot in enumerate(load_grid):
            if not np.isfinite(n_tot) or n_tot <= 0:
                continue
            n_i_vals = []
            skip = False
            for comp in components:
                n_tot_list, n_comp_list = [], []
                for _, comp_dict in shared_sorted:
                    nt = sum(comp_dict.values())
                    nc = comp_dict.get(comp, np.nan)
                    if np.isfinite(nt) and np.isfinite(nc):
                        n_tot_list.append(float(nt))
                        n_comp_list.append(float(nc))
                if len(n_tot_list) < 2:
                    skip = True
                    break
                nt_arr = np.array(n_tot_list, dtype=float)
                nc_arr = np.array(n_comp_list, dtype=float)
                idx_c = np.argsort(nt_arr)
                nt_arr, nc_arr = nt_arr[idx_c], nc_arr[idx_c]
                if n_tot < nt_arr[0] or n_tot > nt_arr[-1]:
                    skip = True
                    break
                n_i = float(np.interp(n_tot, nt_arr, nc_arr))
                if not np.isfinite(n_i) or n_i < 0:
                    skip = True
                    break
                n_i_vals.append((comp, n_i))

            if skip or not n_i_vals:
                continue
            n_sum = sum(n_i for _, n_i in n_i_vals)
            if n_sum <= 0:
                continue

            qst_j = 0.0
            any_v = False
            for comp, n_i in n_i_vals:
                if comp not in qst_interp:
                    continue
                qv = float(qst_interp[comp](n_i))
                if np.isfinite(qv):
                    qst_j += (n_i / n_sum) * qv
                    any_v = True
            if any_v and np.isfinite(qst_j):
                Qst_mix[j] = qst_j

        mask = np.isfinite(Qst_mix)
        if not np.any(mask):
            continue

        loads = load_grid[mask]
        qst_vals = Qst_mix[mask]
        order = np.argsort(loads)
        loads, qst_vals = loads[order], qst_vals[order]
        if len(qst_vals) >= 3:
            qst_vals = gaussian_filter1d(qst_vals, sigma=smoothing_sigma)

        cache[(fw, mixture_name)] = (loads, qst_vals)

    return cache


def make_mixture_qst_cache_hoa_pure_file(mixture_data, frameworks, mixture_name,
                                         mix_temperature, hoa_pure_curves,
                                         p_min, p_max, n_loadings=40,
                                         smoothing_sigma=1.5):
    """
    Build ``{(fw, mixture_name): (loads, qst)}`` for mixture HOA using the
    weighted-pure approach with Qst_i^0(n_i) taken directly from HoA-file
    curves (no CC/Virial recomputation).
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    cache = {}
    T_mix = float(mix_temperature)
    p_min_b = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b = float(p_max) if p_max is not None else 1e8

    if not hoa_pure_curves:
        return cache

    for fw in frameworks:
        # Components present for this framework / mixture
        components = sorted({d['molecule'] for d in mixture_data
                             if d.get('framework') == fw
                             and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()})
        if not components:
            continue

        # 1) Pure-component Qst interpolators comp -> callable(n), from HoA file curves
        qst_interp = {}
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
            # Remove duplicate loadings
            _, u = np.unique(n_arr, return_index=True)
            n_arr, q_arr = n_arr[u], q_arr[u]
            if len(n_arr) < 3:
                continue
            try:
                qst_interp[comp] = PchipInterpolator(n_arr, q_arr, extrapolate=False)
            except Exception:
                continue

        if not qst_interp:
            continue

        # 2) Mixture per-component RASPA data at the chosen mixture temperature
        comp_data = {c: [] for c in components}
        for d in mixture_data:
            if d.get('framework') != fw:
                continue
            mol = d.get('molecule')
            if mol not in components:
                continue
            T = float(d.get('temperature', T_mix))
            if not np.isclose(T, T_mix):
                continue
            P = float(d.get('pressure', np.nan))
            q = float(d.get('loading', np.nan))
            if (np.isfinite(P) and np.isfinite(q)
                    and P > 0 and q >= 0 and p_min_b <= P <= p_max_b):
                comp_data[mol].append((P, q))

        # 3) Build shared states and compute Qst_mix vs total loading
        by_pressure = {}
        for comp in components:
            for P, q in comp_data[comp]:
                by_pressure.setdefault(P, {})[comp] = q
        shared = [(P, cd) for P, cd in by_pressure.items()
                  if len(cd) == len(components)]
        if len(shared) < 3:
            continue

        shared.sort(key=lambda x: x[0])
        n_tot_arr = np.array([sum(s[1].values()) for s in shared], dtype=float)
        order = np.argsort(n_tot_arr)
        n_tot_sorted = n_tot_arr[order]
        shared_sorted = [shared[i] for i in order]
        _, uniq_idx = np.unique(n_tot_sorted, return_index=True)
        n_tot_sorted = n_tot_sorted[uniq_idx]
        shared_sorted = [shared_sorted[i] for i in uniq_idx]
        if len(n_tot_sorted) < 3 or n_tot_sorted.min() >= n_tot_sorted.max():
            continue

        load_grid = np.linspace(n_tot_sorted.min(), n_tot_sorted.max(), int(n_loadings))
        Qst_mix = np.full_like(load_grid, np.nan, dtype=float)

        for j, n_tot in enumerate(load_grid):
            if not np.isfinite(n_tot) or n_tot <= 0:
                continue
            n_i_vals = []
            skip = False
            for comp in components:
                n_tot_list, n_comp_list = [], []
                for _, comp_dict in shared_sorted:
                    nt = sum(comp_dict.values())
                    nc = comp_dict.get(comp, np.nan)
                    if np.isfinite(nt) and np.isfinite(nc):
                        n_tot_list.append(float(nt))
                        n_comp_list.append(float(nc))
                if len(n_tot_list) < 2:
                    skip = True
                    break
                nt_arr = np.array(n_tot_list, dtype=float)
                nc_arr = np.array(n_comp_list, dtype=float)
                idx_c = np.argsort(nt_arr)
                nt_arr, nc_arr = nt_arr[idx_c], nc_arr[idx_c]
                if n_tot < nt_arr[0] or n_tot > nt_arr[-1]:
                    skip = True
                    break
                n_i = float(np.interp(n_tot, nt_arr, nc_arr))
                if not np.isfinite(n_i) or n_i < 0:
                    skip = True
                    break
                n_i_vals.append((comp, n_i))

            if skip or not n_i_vals:
                continue
            n_sum = sum(n_i for _, n_i in n_i_vals)
            if n_sum <= 0:
                continue

            qst_j = 0.0
            any_v = False
            for comp, n_i in n_i_vals:
                if comp not in qst_interp:
                    continue
                qv = float(qst_interp[comp](n_i))
                if np.isfinite(qv):
                    qst_j += (n_i / n_sum) * qv
                    any_v = True
            if any_v and np.isfinite(qst_j):
                Qst_mix[j] = qst_j

        mask = np.isfinite(Qst_mix)
        if not np.any(mask):
            continue

        loads = load_grid[mask]
        qst_vals = Qst_mix[mask]
        order = np.argsort(loads)
        loads, qst_vals = loads[order], qst_vals[order]
        if len(qst_vals) >= 3:
            qst_vals = gaussian_filter1d(qst_vals, sigma=smoothing_sigma)

        cache[(fw, mixture_name)] = (loads, qst_vals)

    return cache

def make_mixture_qst_cache_hoa_pure_virial(mixture_data, frameworks, mixture_name,
                                           temperatures, RASPA_data_pure,
                                           deg_a=2, deg_b=2, degrees_per_combo=None,
                                           min_points=3, n_loadings=40,
                                           p_min=None, p_max=None,
                                           smoothing_sigma=1.5):
    """
    Build ``{(fw, mixture_name): (loads, qst)}`` for mixture HOA using the
    weighted-pure Virial approach (hoa_pure_virial) at a single mixture
    temperature set by the mixture RASPA data (same logic as the HOA plot).

    - Pure Qst_i^0(n_i) are obtained from Virial.compute_Qst_from_coef_slopes.
    - Adsorbed-phase mole fractions y_i come from mixture RASPA data at the
      temperatures present in that data set.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import PchipInterpolator
    import Virial as virial

    cache = {}
    temps = [float(t) for t in temperatures] if temperatures else []
    if not temps or not RASPA_data_pure:
        return cache

    p_min_b = max(float(p_min), 1e-8) if p_min is not None else 1e-8
    p_max_b = float(p_max) if p_max is not None else 1e8

    for fw in frameworks:
        components = sorted({d['molecule'] for d in mixture_data
                             if d.get('framework') == fw
                             and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()})
        if not components:
            continue

        # 1) Pure-component Virial Qst interpolators comp -> callable(n)
        qst_interp = {}
        for comp in components:
            da = (degrees_per_combo[(fw, comp)][0]
                  if degrees_per_combo and (fw, comp) in degrees_per_combo else deg_a)
            db = (degrees_per_combo[(fw, comp)][1]
                  if degrees_per_combo and (fw, comp) in degrees_per_combo else deg_b)
            try:
                res = virial.compute_Qst_from_coef_slopes(
                    RASPA_data=RASPA_data_pure,
                    framework=fw, molecule=comp,
                    deg_a=da, deg_b=db,
                    min_points=min_points,
                    n_points=n_loadings,
                    R=virial.R,
                    temperatures=temps,
                    p_min=p_min,
                    verbose=False,
                )
            except Exception:
                continue

            n_arr = np.asarray(res.get('n_grid'), dtype=float)
            qst_arr = np.asarray(res.get('Qst_kJmol'), dtype=float)
            ok = np.isfinite(n_arr) & np.isfinite(qst_arr)
            if ok.sum() < 3:
                continue
            n_arr, qst_arr = n_arr[ok], qst_arr[ok]
            idx = np.argsort(n_arr)
            n_arr, qst_arr = n_arr[idx], qst_arr[idx]
            _, u = np.unique(n_arr, return_index=True)
            n_arr, qst_arr = n_arr[u], qst_arr[u]
            if len(n_arr) < 3:
                continue
            try:
                qst_interp[comp] = PchipInterpolator(n_arr, qst_arr, extrapolate=False)
            except Exception:
                continue

        if not qst_interp:
            continue

        # 2) Mixture per-component RASPA data at all mixture temperatures
        comp_data = {c: {T: [] for T in temps} for c in components}
        for d in mixture_data:
            if d.get('framework') != fw:
                continue
            mol = d.get('molecule')
            if mol not in components:
                continue
            T = float(d.get('temperature', temps[0]))
            if T not in comp_data[mol]:
                continue
            P = float(d.get('pressure', np.nan))
            q = float(d.get('loading', np.nan))
            if (np.isfinite(P) and np.isfinite(q)
                    and P > 0 and q >= 0 and p_min_b <= P <= p_max_b):
                comp_data[mol][T].append((P, q))

        # 3) Aggregate over mixture temperatures to build Qst_mix vs total loading
        by_pressure = {}
        for T in temps:
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
        n_tot_sorted = n_tot_arr[order]
        shared_sorted = [shared[i] for i in order]
        _, uniq_idx = np.unique(n_tot_sorted, return_index=True)
        n_tot_sorted = n_tot_sorted[uniq_idx]
        shared_sorted = [shared_sorted[i] for i in uniq_idx]
        if len(n_tot_sorted) < 3 or n_tot_sorted.min() >= n_tot_sorted.max():
            continue

        load_grid = np.linspace(n_tot_sorted.min(), n_tot_sorted.max(), int(n_loadings))
        Qst_mix = np.full_like(load_grid, np.nan, dtype=float)

        for j, n_tot in enumerate(load_grid):
            if not np.isfinite(n_tot) or n_tot <= 0:
                continue
            n_i_vals = []
            skip = False
            for comp in components:
                n_tot_list, n_comp_list = [], []
                for _, comp_dict in shared_sorted:
                    nt = sum(comp_dict.values())
                    nc = comp_dict.get(comp, np.nan)
                    if np.isfinite(nt) and np.isfinite(nc):
                        n_tot_list.append(float(nt))
                        n_comp_list.append(float(nc))
                if len(n_tot_list) < 2:
                    skip = True
                    break
                nt_arr = np.array(n_tot_list, dtype=float)
                nc_arr = np.array(n_comp_list, dtype=float)
                idx_c = np.argsort(nt_arr)
                nt_arr, nc_arr = nt_arr[idx_c], nc_arr[idx_c]
                if n_tot < nt_arr[0] or n_tot > nt_arr[-1]:
                    skip = True
                    break
                n_i = float(np.interp(n_tot, nt_arr, nc_arr))
                if not np.isfinite(n_i) or n_i < 0:
                    skip = True
                    break
                n_i_vals.append((comp, n_i))

            if skip or not n_i_vals:
                continue
            n_sum = sum(n_i for _, n_i in n_i_vals)
            if n_sum <= 0:
                continue

            qst_j = 0.0
            any_v = False
            for comp, n_i in n_i_vals:
                if comp not in qst_interp:
                    continue
                qv = float(qst_interp[comp](n_i))
                if np.isfinite(qv):
                    qst_j += (n_i / n_sum) * qv
                    any_v = True
            if any_v and np.isfinite(qst_j):
                Qst_mix[j] = qst_j

        mask = np.isfinite(Qst_mix)
        if not np.any(mask):
            continue

        loads = load_grid[mask]
        qst_vals = Qst_mix[mask]
        order = np.argsort(loads)
        loads, qst_vals = loads[order], qst_vals[order]
        if len(qst_vals) >= 3:
            qst_vals = gaussian_filter1d(qst_vals, sigma=smoothing_sigma)

        cache[(fw, mixture_name)] = (loads, qst_vals)

    return cache

## mixture
def _integrate_qst(L_low, L_high, loads_arr, qst_arr):
    """
    Integrate Qst(kJ/mol) over loading (mol/kg) between L_low and L_high.
    Similar to integrate_enthalpy but for mixture HOA Qst.
    """
    if loads_arr is None or qst_arr is None or L_high <= L_low:
        return np.nan

    load_min = float(np.min(loads_arr))
    load_max = float(np.max(loads_arr))
    L_low_clamped = max(L_low, load_min)
    L_high_clamped = min(L_high, load_max)

    if L_high_clamped <= L_low_clamped:
        return np.nan
    if L_high < load_min or L_low > load_max:
        return np.nan

    sample_L = np.linspace(L_low_clamped, L_high_clamped, max(8, min(200, 50)))
    if loads_arr.size == 1:
        Q_sample = np.full_like(sample_L, qst_arr[0])
    else:
        Q_sample = np.interp(sample_L, loads_arr, qst_arr, left=np.nan, right=np.nan)
        if np.any(~np.isfinite(Q_sample)):
            return np.nan
    # kJ/mol * mol/kg → kJ/kg
    # numpy>=2.0 removed np.trapz; use trapezoid
    return float(np.trapezoid(Q_sample, sample_L))

def plot_storage_density_mix_components_cc(
    selected_frameworks, mixture_name, qst_temperatures, T_ads, T_des_list,
    P_ads, x_fit, mixture_data, selected_fit_types,
    fits_pure, RASPA_data_pure,
    p_min=None, p_max=None, n_loadings=40, min_temps=2,
    smoothing_sigma=1.5, combo_colors=None,
    out_dir=None, save_data=False, scale='both'):
    """
    Mixture storage density vs desorption pressure, per component + total,
    using CC mixture HOA (compute_mixture_isosteric_heat_cc).

    One figure per T_des in T_des_list.
    """
    from ClausiusClapeyron import compute_mixture_isosteric_heat_cc

    if not mixture_data:
        print("plot_storage_density_mix_components_cc: no mixture data, skipping.")
        return

    # Temperatures for CC (Qst vs loading) – typically the isotherm temperatures
    cc_temps = [float(t) for t in (qst_temperatures or [])]

    # Temperatures for component loading fits (interp) – need T_ads and all T_des
    fit_temps = sorted({float(T_ads)} | {float(t) for t in T_des_list})

    fw_first = selected_frameworks[0] if selected_frameworks else None
    # Put mixture-component SD plots in a dedicated per-mixture folder.
    mix_out_dir = None
    if out_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        run_folder = f"{_safe_join_labels(selected_frameworks)}_{_safe_join_labels([mixture_name])}_{_safe_join_labels(list(qst_temperatures) if qst_temperatures else fit_temps)}"
        mix_out_dir = base_dir / "Output" / run_folder / "Storage_Density_components" / str(mixture_name).replace(" ", "_")
        mix_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        mix_out_dir = Path(out_dir)
        mix_out_dir.mkdir(parents=True, exist_ok=True)

    # One plot per T_des, with all frameworks overlaid
    for T_des in T_des_list:
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        any_plotted = False
        export_rows = []

        for fw in selected_frameworks:
                # Components present for this framework / mixture
                components = sorted({d['molecule'] for d in mixture_data
                                    if d.get('framework') == fw
                                    and str(d.get('mixture_pure', '')).lower() == str(mixture_name).lower()})
                if not components:
                    continue

                # Build component loading fits: (fw, component, T) -> loading(P)
                comp_fits = make_mixture_component_fittings(
                    mixture_data, [fw], mixture_name, components, fit_temps
                )
                comp_fit_cache = phelp.build_fit_cache(
                    comp_fits, [fw], components, fit_temps, ['interp'], num_of_isotherm=None
                )

                cc_temps_use = cc_temps if cc_temps else fit_temps

                try:
                    mix_cc = compute_mixture_isosteric_heat_cc(
                        mixture_data=mixture_data,
                        components=components,
                        temperatures=cc_temps_use,
                        framework=fw,
                        mixture_name=mixture_name,
                        p_min=p_min, p_max=p_max,
                        n_loadings=n_loadings,
                        min_temps=min_temps,
                    )
                except Exception as e:
                    print(f"plot_storage_density_mix_components_cc: CC mixture HOA failed for {fw}: {e}")
                    continue

                # Component Qst vs loading (each component is defined on its own loading axis)
                comp_curves = {}
                for comp in components:
                    if comp not in mix_cc:
                        continue
                    loads_c = np.asarray(mix_cc[comp]['loading'], dtype=float)
                    qst_c = np.asarray(mix_cc[comp]['Qst_kJmol'], dtype=float)
                    ok = np.isfinite(loads_c) & np.isfinite(qst_c)
                    loads_c, qst_c = loads_c[ok], qst_c[ok]
                    if loads_c.size < 2:
                        continue
                    order = np.argsort(loads_c)
                    loads_c, qst_c = loads_c[order], qst_c[order]
                    # Remove duplicate loadings to keep np.interp well-defined
                    loads_c_u, idx_u = np.unique(loads_c, return_index=True)
                    qst_c_u = qst_c[idx_u]
                    if loads_c_u.size < 2:
                        continue
                    comp_curves[comp] = (loads_c_u, qst_c_u)

                # Adsorption loading per component at (T_ads, P_ads)
                L_ads_comp = {}
                for comp in components:
                    key_ads = (fw, comp, float(T_ads), 'interp')
                    if key_ads not in comp_fit_cache:
                        L_ads_comp[comp] = np.nan
                    else:
                        ads_params_c, ads_ft_c = comp_fit_cache[key_ads]
                        L_ads_comp[comp] = float(phelp.evaluate_fit(P_ads, ads_params_c, ads_ft_c))

                p_arr = np.asarray(x_fit, dtype=float)
            # Apply P_max if provided
                if p_max is not None:
                    p_arr = p_arr[p_arr <= float(p_max)]
                if p_min is not None:
                    p_arr = p_arr[p_arr >= float(p_min)]
                if p_arr.size == 0:
                    continue

                # Desorption loading arrays per component at this T_des
                L_des_comp_arr = {}
                for comp in components:
                    key_des = (fw, comp, float(T_des), 'interp')
                    if key_des not in comp_fit_cache:
                        L_des_comp_arr[comp] = None
                    else:
                        des_params_c, des_ft_c = comp_fit_cache[key_des]
                        arr = phelp.evaluate_fit(p_arr, des_params_c, des_ft_c)
                        L_des_comp_arr[comp] = np.asarray(arr, dtype=float)

                sd_comp = {c: [] for c in components}
                p_valid = []
                sd_sum = []

                for idx, P_des in enumerate(p_arr):
                    comp_vals = {}
                    ok_all = True
                    for comp in components:
                        loads_c, qst_c = comp_curves.get(comp, (None, None))
                        if loads_c is None or qst_c is None:
                            ok_all = False
                            break
                        L_ads_c = float(L_ads_comp.get(comp, np.nan))
                        L_des_arr_c = L_des_comp_arr.get(comp)
                        if L_des_arr_c is None or idx >= len(L_des_arr_c):
                            ok_all = False
                            break
                        L_des_c = float(L_des_arr_c[idx])
                        if not np.isfinite(L_ads_c) or not np.isfinite(L_des_c):
                            ok_all = False
                            break
                        L1, L2 = min(L_ads_c, L_des_c), max(L_ads_c, L_des_c)
                        sd_c_val = _integrate_qst(L1, L2, loads_c, qst_c)
                        if not np.isfinite(sd_c_val):
                            ok_all = False
                            break
                        comp_vals[comp] = float(sd_c_val)
                    if not ok_all:
                        continue

                    p_valid.append(float(P_des))
                    s = 0.0
                    for comp in components:
                        sd_comp[comp].append(comp_vals[comp])
                        s += comp_vals[comp]
                    sd_sum.append(float(s))

                if not p_valid:
                    continue

                # Plot per component
                for comp in components:
                    if not sd_comp[comp]:
                        continue
                    color = phelp.get_color_for_molecule(comp) or None
                    label = phelp.get_molecule_display_name(comp) if fw == fw_first else "_nolegend_"
                    ls_fw = phelp.get_linestyle_for_structure(fw)
                    ax.plot(
                        p_valid, sd_comp[comp],
                        lw=phelp.LINEWIDTH, alpha=phelp.ALPHA,
                        label=label, color=color, linestyle=ls_fw,
                    )
                    any_plotted = True

                # Plot total mixture – use mixture color; no framework names in legend
                mix_color = phelp.get_color_for_molecule(mixture_name) or 'black'
                total_label = phelp.get_molecule_display_name(mixture_name) if fw == fw_first else "_nolegend_"
                ls_fw = phelp.get_linestyle_for_structure(fw)
                ax.plot(
                    p_valid, sd_sum,
                    lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, linestyle=ls_fw, color=mix_color,
                    label=total_label,
                )
                any_plotted = True

                if save_data:
                    for P_des, sd_val in zip(p_valid, sd_sum):
                        export_rows.append({
                            'framework': fw,
                            'molecule': mixture_name,
                            'T_ads': float(T_ads),
                            'T_des': float(T_des),
                            'P_ads': float(P_ads),
                            'P_des': float(P_des),
                            'SD': float(sd_val),
                        })
                    for comp in components:
                        for P_des, sd_val in zip(p_valid, sd_comp[comp]):
                            export_rows.append({
                                'framework': fw,
                                'molecule': comp,
                                'T_ads': float(T_ads),
                                'T_des': float(T_des),
                                'P_ads': float(P_ads),
                                'P_des': float(P_des),
                                'SD': float(sd_val),
                            })

        if not any_plotted:
            plt.close(fig)
            continue

        subtitle = f"{mixture_name}, T_ads={int(T_ads)}K, T_des={int(T_des)}K, P_ads={P_ads:.2g} Pa"
        phelp.format_storage_plot(
            ax, "Storage Density (components)",
            p_min=p_min, p_max=p_max, P_des_max=None,
            global_p_min=None, subtitle=subtitle,
            scale=scale,
        )
        ax.legend(loc='best', fontsize=phelp.AXIS_LEGEND_SIZE)
        phelp.apply_unified_axes_layout(fig, ax)
        _sd_2d_axis_label_weight(ax)

        mix_safe = str(mixture_name).replace(" ", "_")
        out_path = phelp._save_plot(
            f"storage_density_components_cc_Tdes{int(T_des)}K",
            "plot_storage_density_components",
            selected_frameworks, [mixture_name], list(qst_temperatures) if qst_temperatures else fit_temps,
            fig=fig, out_dir=str(mix_out_dir),
            filename_suffix=mix_safe,
        )

        if save_data and out_path is not None:
            _save_sd_rows(out_path, export_rows, "storage_density_components_cc")

        plt.show()
        plt.close(fig)