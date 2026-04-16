"""
DataSelection.py
================
Single preprocessing module that prepares the shared dataset used by both
Virial and ClausiusClapeyron, independent of the data_source setting.

Responsibilities
----------------
- Pressure bounds  [p_min, p_max]
- PCHIP interpolation onto a common loading grid
- Minimum temperature-coverage constraint (min_temps)

Outlier removal is intentionally NOT done here because Virial and
ClausiusClapeyron apply different outlier criteria.

Public API
----------
build_dataset(input_rows, selected_frameworks, selected_molecules,
              selected_temperatures, n_loadings, p_min, p_max, min_temps=3)
    -> list[dict]

save_dataset(rows, filepath)
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import PlotHelpers as phelp


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deduplicate_loading_pressure(q_arr, p_arr):
    """Remove duplicate loading values by averaging pressure at each duplicate."""
    order = np.argsort(q_arr)
    q_sorted = q_arr[order]
    p_sorted = p_arr[order]
    q_unique, _, inv = np.unique(q_sorted, return_index=True, return_inverse=True)
    p_unique = np.array(
        [np.mean(p_sorted[inv == i]) for i in range(len(q_unique))], dtype=float
    )
    return q_unique, p_unique


def _intersection_loading_range(pts):
    """Return (n_min, n_max) as the intersection of per-temperature loading ranges.

    n_min = max over temperatures of each temperature's minimum loading.
    n_max = min over temperatures of each temperature's maximum loading.
    Falls back to overall [min, max] when per-temperature grouping is impossible.
    """
    by_temp = {}
    for p in pts:
        try:
            t = float(p['temperature'])
            n = float(p['loading'])
        except (KeyError, TypeError, ValueError):
            continue
        if n > 0:
            by_temp.setdefault(t, []).append(n)

    if by_temp:
        mins = [min(ns) for ns in by_temp.values()]
        maxs = [max(ns) for ns in by_temp.values()]
        return float(max(mins)), float(min(maxs))

    all_n = np.array(
        [float(p['loading']) for p in pts
         if p.get('loading') is not None and float(p['loading']) > 0],
        dtype=float,
    )
    if all_n.size == 0:
        raise ValueError("No positive loading values found")
    return float(np.min(all_n)), float(np.max(all_n))


def _unified_loading_range(input_rows, framework, molecule, temperatures,
                            p_min_bound, p_max_bound, min_temps_req):
    """Loading range where P is in [p_min, p_max] for >= min_temps temperatures.

    Works directly from the supplied rows — no fittings fallback needed because
    both data-source modes (raw RASPA and synthetic from synthesize_points_from_fittings)
    provide sufficiently dense rows.

    Returns (q_min, q_max) or None when no valid range exists.
    """
    temps = [float(t) for t in temperatures]
    pts_all = phelp.filter_raspa_data(
        input_rows, frameworks=[framework], molecules=[molecule], temperatures=temps,
        only_pure_adsorption=True,
    )
    if not pts_all:
        return None

    data_by_temp = {}
    for pt in pts_all:
        if pt.get('pressure') is None or pt.get('loading') is None:
            continue
        T = round(float(pt['temperature']))
        data_by_temp.setdefault(T, []).append({
            'pressure': float(pt['pressure']),
            'loading': float(pt['loading']),
        })

    loading_ranges_by_temp = {}
    for T_rounded, data_points in data_by_temp.items():
        if len(data_points) < 3:
            continue
        q_arr = np.array([d['loading'] for d in data_points])
        p_arr = np.array([d['pressure'] for d in data_points])

        mask = np.isfinite(q_arr) & np.isfinite(p_arr) & (p_arr > 0) & (q_arr >= 0)
        if np.sum(mask) < 3:
            continue

        q_arr, p_arr = q_arr[mask], p_arr[mask]
        q_sorted, p_sorted = _deduplicate_loading_pressure(q_arr, p_arr)
        if len(q_sorted) < 3:
            continue

        try:
            interp_q_to_p = PchipInterpolator(q_sorted, p_sorted, extrapolate=False)
            q_test = np.linspace(q_sorted.min(), q_sorted.max(), 500)
            p_test = interp_q_to_p(q_test)
            valid_mask = (
                (p_test >= p_min_bound) & (p_test <= p_max_bound) & np.isfinite(p_test)
            )
            if np.any(valid_mask):
                q_valid = q_test[valid_mask]
                loading_ranges_by_temp[T_rounded] = (q_valid.min(), q_valid.max())
        except Exception:
            continue

    if not loading_ranges_by_temp:
        return None

    all_q_mins = [r[0] for r in loading_ranges_by_temp.values()]
    all_q_maxs = [r[1] for r in loading_ranges_by_temp.values()]
    q_grid_fine = np.linspace(min(all_q_mins), max(all_q_maxs), 1000)

    coverage_count = np.zeros(len(q_grid_fine))
    for q_min, q_max in loading_ranges_by_temp.values():
        coverage_count += ((q_grid_fine >= q_min) & (q_grid_fine <= q_max)).astype(int)

    sufficient = coverage_count >= min_temps_req
    if not np.any(sufficient):
        return None

    indices = np.where(sufficient)[0]
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    segments = np.split(indices, breaks)
    largest = max(segments, key=len)

    q_range_min = q_grid_fine[largest[0]]
    q_range_max = q_grid_fine[largest[-1]]
    return (max(q_range_min, 1e-8), q_range_max)


def _build_interpolation_matrices(input_rows, framework, molecule, temperatures,
                                   unified_range, n_loadings, p_min_bound, p_max_bound):
    """PCHIP(loading -> pressure) per temperature on a shared loading grid.

    Returns (loadings, temps_with_data, P_mat, lnP_mat).
    """
    temps = [float(t) for t in temperatures]
    pts_all = phelp.filter_raspa_data(
        input_rows, frameworks=[framework], molecules=[molecule], temperatures=temps,
        only_pure_adsorption=True,
    )
    if not pts_all:
        raise ValueError(
            f"No data found for {framework}, {molecule} at temperatures {temperatures}"
        )

    data_by_temp = {}
    for pt in pts_all:
        if pt.get('pressure') is None or pt.get('loading') is None:
            continue
        T = round(float(pt['temperature']))
        data_by_temp.setdefault(T, []).append({
            'pressure': float(pt['pressure']),
            'loading': float(pt['loading']),
        })

    temps_with_data = [
        float(t) for t in temps
        if round(float(t)) in data_by_temp and len(data_by_temp[round(float(t))]) >= 3
    ]
    if len(temps_with_data) < 2:
        raise ValueError(
            f"Need at least two temperatures with sufficient data for "
            f"{framework}, {molecule}; found {len(temps_with_data)}"
        )

    nT = len(temps_with_data)

    if unified_range is not None:
        overall_min, overall_max = unified_range
    else:
        all_loadings = [
            d['loading']
            for T_r in (round(float(t)) for t in temps_with_data)
            for d in data_by_temp.get(T_r, [])
        ]
        if not all_loadings:
            raise ValueError(f"No loading data found for {framework}, {molecule}")
        overall_min = max(float(min(all_loadings)), 1e-8)
        overall_max = float(max(all_loadings))

    if overall_max <= overall_min:
        raise ValueError(
            f"Unable to determine a valid loading range for {framework}, {molecule}"
        )

    loadings = np.linspace(overall_min, overall_max, int(n_loadings))
    nL = loadings.size

    P_mat = np.full((nT, nL), np.nan)
    lnP_mat = np.full((nT, nL), np.nan)

    for i, T in enumerate(temps_with_data):
        T_rounded = round(float(T))
        if T_rounded not in data_by_temp:
            continue

        data_temp = data_by_temp[T_rounded]
        loadings_obs = np.array([d['loading'] for d in data_temp], dtype=float)
        pressures_obs = np.array([d['pressure'] for d in data_temp], dtype=float)

        mask = (
            np.isfinite(loadings_obs)
            & np.isfinite(pressures_obs)
            & (pressures_obs > 0)
        )
        if np.sum(mask) < 3:
            continue

        loadings_obs, pressures_obs = loadings_obs[mask], pressures_obs[mask]
        q_sorted, p_sorted = _deduplicate_loading_pressure(loadings_obs, pressures_obs)
        if len(q_sorted) < 3:
            continue

        try:
            interpolator = PchipInterpolator(q_sorted, p_sorted, extrapolate=False)
            p_interp = interpolator(loadings)
        except Exception:
            p_interp = np.interp(loadings, q_sorted, p_sorted, left=np.nan, right=np.nan)

        valid_pressure = (
            (p_interp >= p_min_bound) & (p_interp <= p_max_bound) & np.isfinite(p_interp)
        )
        p_interp_filtered = np.where(valid_pressure, p_interp, np.nan)

        P_mat[i, :] = p_interp_filtered
        with np.errstate(divide='ignore', invalid='ignore'):
            lnP_mat[i, :] = np.log(p_interp_filtered)

    return loadings, temps_with_data, P_mat, lnP_mat


def _flatten_to_rows(framework, molecule, temps_with_data, loadings, P_mat):
    """Convert P_mat back to a flat list of row dicts."""
    out = []
    nL = loadings.size
    for i, T in enumerate(temps_with_data):
        for j in range(nL):
            p_ij = P_mat[i, j]
            if np.isfinite(p_ij) and p_ij > 0 and loadings[j] > 0:
                out.append({
                    'framework': framework,
                    'molecule': molecule,
                    'temperature': float(T),
                    'pressure': float(p_ij),
                    'loading': float(loadings[j]),
                })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(input_rows, selected_frameworks, selected_molecules,
                  selected_temperatures, n_loadings, p_min, p_max, min_temps=3):
    """Build the shared dataset for Virial and ClausiusClapeyron.

    Parameters
    ----------
    input_rows : list[dict]
        Raw RASPA data **or** synthetic points from
        ``IsothermFittingPlot.synthesize_points_from_fittings``.
        Both modes are handled identically.
    selected_frameworks, selected_molecules, selected_temperatures :
        Selections to process.
    n_loadings : int
        Number of points on the common loading grid per (fw, mol).
    p_min, p_max : float
        Pressure bounds [Pa].
    min_temps : int
        Minimum number of temperatures that must have pressure coverage
        in [p_min, p_max] for a loading value to be included.

    Returns
    -------
    list[dict]
        Rows ``{framework, molecule, temperature, pressure, loading}`` on a
        common loading grid, already p-bounded and multi-temperature filtered.
    """
    if (not input_rows or not selected_frameworks
            or not selected_molecules or not selected_temperatures):
        return []

    p_min_bound = max(float(p_min), 1e-8)
    p_max_bound = float(p_max)
    min_temps_req = int(min_temps)
    n_loadings = int(max(2, n_loadings))
    temps = [float(t) for t in selected_temperatures]

    out = []
    for framework in selected_frameworks:
        for molecule in selected_molecules:
            try:
                unified = _unified_loading_range(
                    input_rows, framework, molecule, temps,
                    p_min_bound, p_max_bound, min_temps_req,
                )
                loadings, temps_with_data, P_mat, _ = _build_interpolation_matrices(
                    input_rows, framework, molecule, temps,
                    unified, n_loadings, p_min_bound, p_max_bound,
                )
            except Exception:
                continue
            out.extend(_flatten_to_rows(framework, molecule, temps_with_data, loadings, P_mat))
    return out


def save_dataset(rows, filepath):
    """Write dataset rows to a RASPA-compatible text file.

    The file can be re-read by ``Initialize.load_RASPA_data``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            "# DataSelection output: framework, molecule, mixture, "
            "T[K], P[Pa], loading [mol/kg]\n"
        )
        for r in rows:
            f.write(
                f"{r['framework']}\t{r['molecule']}\tpure\t"
                f"{r['temperature']}\t{r['pressure']}\t{r['loading']}\n"
            )
