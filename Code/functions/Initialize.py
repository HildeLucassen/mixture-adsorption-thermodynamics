import numpy as np
import os
import itertools
from pathlib import Path


def get_pipeline_run_root() -> Path:
    """Root directory for ``config.in``, ``design.in``, and ``Output/``.

    When ``run.py`` sets ``PIPELINE_REPO_ROOT`` (e.g. an example folder), use that.
    Otherwise use the repository root (parent of the ``Code/`` package), i.e. two
    levels above this file under ``Code/functions/``.
    """
    env = os.environ.get("PIPELINE_REPO_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    return Path(__file__).resolve().parents[2]

#Data loading
def load_fitting_data(filepath, pressure_unit='kPa'):
    """
    Load fitting data from file.

    Supported layout (tab- or space-separated):
    - Columns: Framework, Temperature, Molecule, Mixture/Pure, FittingType, FinalParameters...
    - pressure_unit: 'kPa' or 'Pa'; if 'kPa', K parameters are converted to 1/Pa.
    """
    fittings = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Framework, Temperature, Molecule, Mixture/Pure, FittingType, then at least 3 params
            if len(parts) < 8:
                continue
            fw = parts[0]
            temp = parts[1]
            mol = parts[2]
            fit_type = parts[3]
            try:
                params = [float(x) for x in parts[4:]]
            except (ValueError, TypeError):
                continue

            if pressure_unit == 'kPa':
                for i in range(1, len(params), 3):
                    params[i] = params[i] / 1000.0

            fittings.append({
                "framework": fw,
                "temperature": float(temp),
                "molecule": mol,
                "fit_type": fit_type,
                "params": params,
            })
    return fittings

def load_RASPA_data(filepath, pure_only=True):
    """
    Load data points from file.

    Supported layouts (whitespace-separated):
    - Compact (6 cols): framework, molecule, mixture_pure, T[K], P[Pa], loading.
    - RASPA wide export (>=11 cols): T, P, loading, ... optional columns ...,
      then molecule, framework, mixture (``-`` or ``pure`` for pure component).
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Structure":
                continue

            if len(parts) == 6:
                try:
                    framework = parts[0]
                    molecule = parts[1]
                    mixture_pure = parts[2]
                    temperature = float(parts[3])
                    pressure = float(parts[4])
                    loading = float(parts[5])
                except (ValueError, IndexError):
                    continue
                if pure_only and str(mixture_pure).strip().lower() != "pure":
                    continue
                data.append({
                    "framework": framework,
                    "molecule": molecule,
                    "mixture_pure": mixture_pure,
                    "temperature": temperature,
                    "pressure": pressure,
                    "loading": loading,
                })
                continue

            # RASPA simulation output: first three fields are T, P, loading; last three are
            # molecule, framework, mixture token (often "-" for pure).
            if len(parts) >= 11:
                try:
                    temperature = float(parts[0])
                    pressure = float(parts[1])
                    loading = float(parts[2])
                    molecule = parts[-3]
                    framework = parts[-2]
                    mixture_raw = parts[-1]
                except (ValueError, IndexError):
                    continue
                mixture_pure = mixture_raw
                if mixture_raw in ("-", "–", "—"):
                    mixture_pure = "pure"
                if pure_only and str(mixture_pure).strip().lower() != "pure":
                    continue
                data.append({
                    "framework": framework,
                    "molecule": molecule,
                    "mixture_pure": mixture_pure,
                    "temperature": temperature,
                    "pressure": pressure,
                    "loading": loading,
                })

    return data


def load_hoa_data(filepath):
    """
    Load pure heat-of-adsorption data from file.
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and parts[0].lower() == "structure":
                continue
            if len(parts) < 4:
                continue
            try:
                framework = parts[0]
                molecule = parts[1]
                # File columns: 3rd = loading (mol/kg), 4th = Qst (kJ/mol)
                loading = float(parts[2])
                qst_kjmol = float(parts[3])
            except (ValueError, IndexError):
                continue
            data.append({
                "framework": framework,
                "molecule": molecule,
                "loading": loading,
                "qst_kjmol": qst_kjmol,
            })
    return data


def build_hoa_curves(hoa_rows, frameworks=None, molecules=None, *, min_points=2):
    """
    Build HoA curves {(fw, mol): (loads_arr, qst_arr)} from rows produced by load_hoa_data.

    - Sorts by loading
    - De-duplicates identical loading values
    - Filters to finite values
    """
    fw_filter = set(frameworks) if frameworks else None
    mol_filter = set(molecules) if molecules else None

    curves = {}
    buckets = {}
    for r in hoa_rows or []:
        fw = str(r.get("framework", "")).strip()
        mol = str(r.get("molecule", "")).strip()
        if fw_filter is not None and fw not in fw_filter:
            continue
        if mol_filter is not None and mol not in mol_filter:
            continue
        try:
            L = float(r.get("loading", np.nan))
            Q = float(r.get("qst_kjmol", np.nan))
        except Exception:
            continue
        if not (np.isfinite(L) and np.isfinite(Q)):
            continue
        buckets.setdefault((fw, mol), []).append((L, Q))

    for key, pts in buckets.items():
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])
        loads = np.asarray([p[0] for p in pts], dtype=float)
        qst = np.asarray([p[1] for p in pts], dtype=float)
        u_loads, idx = np.unique(loads, return_index=True)
        qst = qst[idx]
        if u_loads.size < int(min_points):
            continue
        curves[key] = (u_loads, qst)

    return curves


# --- Data selection helpers ---    
def get_pressure_bounds(data):
    pressures = [d["pressure"] for d in data if d.get("pressure") is not None]
    return (min(pressures), max(pressures)) if pressures else (None, None)

# -- plot initializers --
def _lookup_temperature_color(temperature_color_mapping, temp):
    """Match design.in keys (often int K) to runtime temps (int/float/str)."""
    if not temperature_color_mapping:
        return None
    try:
        tf = float(temp)
    except (TypeError, ValueError):
        return None
    candidates = [temp, tf]
    try:
        ri = int(round(tf))
        if abs(tf - ri) < 1e-6:
            candidates.append(ri)
    except (TypeError, ValueError, OverflowError):
        pass
    for k in candidates:
        if k in temperature_color_mapping:
            return temperature_color_mapping[k]
    for k in (str(int(round(tf))), str(tf)):
        if k in temperature_color_mapping:
            return temperature_color_mapping[k]
    return None


def get_combo_colors(selected_frameworks, selected_molecules, selected_temperatures,
                     basic_colors=None, temperature_color_mapping=None,
                     molecule_color_mapping=None, structure_color_mapping=None):
    """
    Generate color mapping for (framework, molecule, temperature) combinations.
    
    Parameters:
    - selected_frameworks: List of framework names
    - selected_molecules: List of molecule names
    - selected_temperatures: List of temperatures
    - basic_colors: List of fallback colors if temperature mapping is not provided
    - temperature_color_mapping: Dictionary mapping temperature values to colors
                              If provided, colors are assigned based on temperature first,
                              then falls back to basic_colors for unmapped temperatures
    
    Returns:
    - Dictionary mapping (framework, molecule, temperature) tuples to color strings
    """
    if basic_colors is None:
        basic_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    if temperature_color_mapping is None:
        temperature_color_mapping = {}
    if molecule_color_mapping is None:
        molecule_color_mapping = {}
    if structure_color_mapping is None:
        structure_color_mapping = {}

    vary_T = len(selected_temperatures or []) > 1
    vary_mol = len(selected_molecules or []) > 1
    vary_fw = len(selected_frameworks or []) > 1
    
    combo_list = list(itertools.product(selected_frameworks, selected_molecules, selected_temperatures))
    
    # Track automatic color assignments within the chosen encoding
    auto_map = {}
    auto_idx = 0
    
    result = {}
    for combo in combo_list:
        fw, mol, temp = combo
        # Encoding precedence:
        # - If T varies: color encodes temperature.
        # - Else if molecule varies: color encodes adsorbate (molecule).
        # - Else if framework varies: color encodes adsorbent (framework).
        # - Else: fall back to temperature key (single series anyway).

        if vary_T:
            key = float(temp)
            mapped = _lookup_temperature_color(temperature_color_mapping, temp)
            if mapped is not None:
                result[combo] = mapped
            elif key in auto_map:
                result[combo] = auto_map[key]
            else:
                color = basic_colors[auto_idx % len(basic_colors)]
                auto_map[key] = color
                result[combo] = color
                auto_idx += 1
            continue

        if vary_mol:
            key = str(mol)
            if key in molecule_color_mapping:
                result[combo] = molecule_color_mapping[key]
            elif key in auto_map:
                result[combo] = auto_map[key]
            else:
                color = basic_colors[auto_idx % len(basic_colors)]
                auto_map[key] = color
                result[combo] = color
                auto_idx += 1
            continue

        if vary_fw:
            key = str(fw)
            if key in structure_color_mapping:
                result[combo] = structure_color_mapping[key]
            elif key in auto_map:
                result[combo] = auto_map[key]
            else:
                color = basic_colors[auto_idx % len(basic_colors)]
                auto_map[key] = color
                result[combo] = color
                auto_idx += 1
            continue

        # Single-series fallback
        result[combo] = _lookup_temperature_color(temperature_color_mapping, temp) or basic_colors[0]
    
    return result

# -- isotherm fitting calculations
def formula_fitting(x, p, ft_type):
    """
    Unified dispatcher function for all fitting formulas.
    
    Parameters:
    - x: pressure (scalar or array)
    - p: parameter list (format depends on fit_type)
    - ft_type: string indicating the fitting type ('Langmuir_Freundlich', 'Sips', 'fitting_Sips', 'Toth', 'fitting_toth')
    
    Returns:
    - Calculated loading values based on the specified fit_type
    
    This function centralizes the logic for selecting the appropriate fitting formula,
    making it easier to add new fit types in the future.
    """
    # Normalize fit_type to handle both with and without "fitting_" prefix
    ft_normalized = ft_type.replace("fitting_", "").strip()
    
    if ft_normalized == "Langmuir_Freundlich" or ft_normalized == "Langmuir-Freundlich":
        return formula_Langmuir_Freundlich(x, p)
    elif ft_normalized == "Sips":
        return formula_fitting_Sips(x, p)
    elif ft_normalized == "Toth" or ft_normalized == "toth":
        return formula_fitting_toth(x, p)
    elif ft_normalized == "interp":
        # Linear interpolation from pre-built (P_arr, q_arr) pair stored in p.
        # Returns NaN outside the data range so SD functions skip those points.
        P_arr = np.asarray(p[0], dtype=float)
        q_arr = np.asarray(p[1], dtype=float)
        x_arr = np.asarray(x, dtype=float)
        result = np.interp(x_arr, P_arr, q_arr, left=np.nan, right=np.nan)
        return result if x_arr.ndim > 0 else float(result)
    else:
        # Default fallback to Langmuir-Freundlich
        return formula_Langmuir_Freundlich(x, p)

def formula_Langmuir_Freundlich(x, p):
    """
    Generalized multi-site Langmuir-Freundlich (LF) isotherm model.

    Parameters:
    - x: pressure (scalar or array)
    - p: parameter list where parameters are grouped per site in triples
         [q_s0, K0, n0, q_s1, K1, n1, ...] where each site contributes
         q_s_i * K_i * x^n_i / (1.0 + K_i * x^n_i)

    Formula: q(p) = q_s * K * p^n / (1 + K * p^n)
    Where:
    - q_s: saturation capacity
    - K: affinity constant
    - n: heterogeneity parameter

    Supports 1, 2, or 3 (or more) sites as long as len(p) is a multiple of 3.
    """
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        p = list(p)
        n_params = len(p)
        if n_params % 3 != 0:
            # fallback: try to use at least the first 3 parameters
            n_sites = max(1, n_params // 3)
        else:
            n_sites = n_params // 3
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for i in range(n_sites):
            A = float(p[3*i + 0])
            B = float(p[3*i + 1])
            C = float(p[3*i + 2])
            num = A * B * np.power(x, C)
            den = 1.0 + B * np.power(x, C)
            term = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            result = result + term
        # Replace inf/nan with np.nan for safety
        result = np.nan_to_num(result, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return result

def formula_fitting_Sips(x, p):
    """
    Generalized multi-site Sips model.

    Parameters:
    - x: pressure (scalar or array)
    - p: parameter list where parameters are grouped per site in triples
         [q0, K0, m0, q1, K1, m1, ...] and each site contributes
         q_i * (K_i * x)^(1/m_i) / (1 + (K_i * x)^(1/m_i)).

    Supports any number of sites where len(p) is a multiple of 3 (1,2,3...).
    """
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        p = list(p)
        n_params = len(p)
        if n_params % 3 != 0:
            n_sites = max(1, n_params // 3)
        else:
            n_sites = n_params // 3
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for i in range(n_sites):
            q_i = float(p[3*i + 0])
            K_i = float(p[3*i + 1])
            m_i = float(p[3*i + 2])
            # compute (K_i * x)^(1/m_i) safely
            base = np.power(K_i * x, 1.0 / m_i)
            den = 1.0 + base
            term = np.divide(q_i * base, den, out=np.zeros_like(base), where=den!=0)
            result = result + term
        result = np.nan_to_num(result, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return result

def formula_fitting_toth(x, p):
    """
    Generalized multi-site Tóth isotherm model.

    Parameters:
    - x: pressure (scalar or array)
    - p: parameter list where parameters are grouped per site in triples
         [q_s0, b0, t0, q_s1, b1, t1, ...] where each site contributes
         q_s_i * b_i * x / (1 + (b_i * x)^t_i)^(1/t_i)

    Formula: q(p) = q_s * b * p / (1 + (b * p)^t)^(1/t)
    Where:
    - q_s: saturation capacity
    - b: affinity constant (related to adsorption strength)
    - t: heterogeneity parameter

    Supports any number of sites where len(p) is a multiple of 3 (1,2,3...).
    """
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        p = list(p)
        n_params = len(p)
        if n_params % 3 != 0:
            n_sites = max(1, n_params // 3)
        else:
            n_sites = n_params // 3
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for i in range(n_sites):
            q_s_i = float(p[3*i + 0])
            b_i = float(p[3*i + 1])
            t_i = float(p[3*i + 2])
            # Compute b_i * x
            bx = b_i * x
            # Compute (b_i * x)^t_i
            bx_power_t = np.power(bx, t_i)
            # Compute denominator: (1 + (b_i * x)^t_i)^(1/t_i)
            den_base = 1.0 + bx_power_t
            den = np.power(den_base, 1.0 / t_i)
            # Compute numerator: q_s_i * b_i * x
            num = q_s_i * bx
            # Compute term: q_s_i * b_i * x / (1 + (b_i * x)^t_i)^(1/t_i)
            term = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            result = result + term
        result = np.nan_to_num(result, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return result



