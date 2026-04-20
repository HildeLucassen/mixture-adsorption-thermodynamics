import ast
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def _parse_config_in(path):
    """Return {KEY: value_or_list_of_values} from a config.in-style file.

    Duplicate keys (e.g. multiple VIRIAL_DEGREES_COMBO lines) are collected
    into a list rather than overwriting each other.
    """
    raw = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key = parts[0].upper()
                val = parts[1].strip()
                if key in raw:
                    if not isinstance(raw[key], list):
                        raw[key] = [raw[key]]
                    raw[key].append(val)
                else:
                    raw[key] = val
    return raw

def _parse_val(s):
    """Convert a raw string to a Python value (number, list, tuple, bool, or str)."""
    s = s.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    low = s.lower()
    if low in ('yes', 'true'):
        return True
    if low in ('no', 'false'):
        return False
    if low == 'none':
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if ',' in s:
        return [x.strip() for x in s.split(',') if x.strip()]
    parts = s.split()
    return parts if len(parts) > 1 else s

def _listify(x):
    return x if isinstance(x, list) else ([x] if x is not None else [])


def _config_yes_no(value, default=True):
    """Normalize yes/no style config values to bool (e.g. SHOW_PLOTS, OUT_DIR, SHOW_POINTS)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return _config_yes_no(value[0], default) if value else default
    if isinstance(value, (int, float)):
        return int(value) != 0
    s = str(value).strip().lower()
    if s in ('yes', 'true', '1', 'y'):
        return True
    if s in ('no', 'false', '0', 'n'):
        return False
    return default


def _parse_virial_degrees_combo(raw):
    """Parse VIRIAL_DEGREES_COMBO into {(framework, molecule): (deg_a, deg_b)}.

    Each line is: adsorbent adsorbate deg_a deg_b (whitespace-separated).
    Multiple lines are supported (duplicate key in config.in).
    """
    if not raw:
        return {}
    lines = []
    if isinstance(raw, str):
        lines = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                lines.append(item.strip())
    out = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        fw, mol = parts[0], parts[1]
        try:
            da, db = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        out[(fw, mol)] = (da, db)
    return out


def _as_virial_deg_pair(v):
    """Normalize VIRIAL_DEGREES to (int, int), or None if missing."""
    if v is None:
        return None
    t = tuple(v)
    if len(t) != 2:
        raise ValueError(f"VIRIAL_DEGREES must be a pair (deg_a, deg_b), got {v!r}")
    return (int(t[0]), int(t[1]))


def _has_explicit_virial_degrees_in_config(read_cfg, combo_map):
    """True if the user supplied VIRIAL_DEGREES_COMBO lines and/or a valid VIRIAL_DEGREES pair."""
    if combo_map:
        return True
    try:
        return _as_virial_deg_pair(read_cfg.get('VIRIAL_DEGREES')) is not None
    except (ValueError, TypeError):
        return False


def _resolve_virial_degrees_for_config(suggestion_yes, read_cfg, selection_fw, selection_mol, combo_map):
    """Return global (deg_a, deg_b) stored as config['virial_degrees'] (may be None).

    When ``SUGGESTION_VIRIAL`` is omitted from ``config.in``, it is inferred: if any
    ``VIRIAL_DEGREES_COMBO`` line parses or ``VIRIAL_DEGREES`` is a valid pair, behaviour
    matches ``SUGGESTION_VIRIAL no``; otherwise it matches ``yes`` (auto-search).

    SUGGESTION_VIRIAL=no: use VIRIAL_DEGREES unless at least one VIRIAL_DEGREES_COMBO
    line parses; then COMBO sets the global fallback (first ADSORBENT/ADSORBATE match,
    else first combo line). No implicit (4, 3) — one of the two must be set.

    SUGGESTION_VIRIAL=yes: VIRIAL_DEGREES / COMBO are optional; return optional explicit
    VIRIAL_DEGREES or None (Main uses an internal numeric fallback only when needed).
    """
    if suggestion_yes:
        return _as_virial_deg_pair(read_cfg.get('VIRIAL_DEGREES'))
    if combo_map:
        fw0 = selection_fw[0] if selection_fw else None
        mol0 = selection_mol[0] if selection_mol else None
        if fw0 is not None and mol0 is not None and (fw0, mol0) in combo_map:
            return combo_map[(fw0, mol0)]
        return next(iter(combo_map.values()))
    out = _as_virial_deg_pair(read_cfg.get('VIRIAL_DEGREES'))
    if out is None:
        raise ValueError(
            "config.in: when SUGGESTION_VIRIAL=no, set VIRIAL_DEGREES or at least one "
            "valid VIRIAL_DEGREES_COMBO line (framework molecule deg_a deg_b)."
        )
    return out


def _runtime_repo_root() -> Path:
    """Same semantics as ``Initialize.get_pipeline_run_root`` (cannot import that here)."""
    env = os.environ.get("PIPELINE_REPO_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    return Path(__file__).resolve().parents[1]


_repo_root = _runtime_repo_root()
_cfg_raw = _parse_config_in(str(_repo_root / 'config.in'))
# VIRIAL_DEGREES_COMBO must stay full-line strings (not token-split by _parse_val).
read_input_file = {}
for _k, _v in _cfg_raw.items():
    if _k == 'VIRIAL_DEGREES_COMBO':
        read_input_file[_k] = _v
    elif isinstance(_v, list):
        read_input_file[_k] = [_parse_val(_x) for _x in _v]
    else:
        read_input_file[_k] = _parse_val(_v)


def _load_design_in(path: Path) -> dict:
    """Load design variables from design.in (safe Python literals only)."""
    if not path.exists():
        return {}
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return {}

    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name):
            continue
        name = t.id
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        out[name] = value
    return out

selection = {
    'fw':              _listify(read_input_file.get('ADSORBENT')),
    'mol':             _listify(read_input_file.get('ADSORBATE')),
    'temp':            read_input_file.get('TEMPERATURE'),
    'fit_types':       _listify(read_input_file.get('FIT_TYPE')),
    'num_of_isotherm': _listify(read_input_file.get('NUM_ISOTHERM_SITES')),
    'pressure_unit': str(read_input_file.get('PRESSURE_UNIT', 'Pa')),
}

# Normalise HOA mode strings (strip whitespace; duplicate config keys can become lists).
def _hoa_flag_token(v):
    if v is None:
        return ''
    if isinstance(v, list):
        v = v[0] if v else ''
    return str(v).strip().lower()

_heat_hoa = _hoa_flag_token(read_input_file.get('HEAT_OF_ADSORPTION'))
_mix_hoa = _hoa_flag_token(read_input_file.get('HEAT_OF_ADSORPTION_MIX'))

plot_flags = {
    'Virial':                  _heat_hoa in ('virial', 'both'),
    'Clausius_Clapeyron':      _heat_hoa in ('cc', 'both'),
    'HOA_From_File':           _heat_hoa in ('data_file', 'both'),
    'Storage_Density_Method':  _hoa_flag_token(read_input_file.get('STORAGE_DENSITY')),
    'Storage_Density_Dim':     str(read_input_file.get('STORAGE_DENSITY_DIMENSION', '')).strip().lower(),
    'Mixture_CC':              _mix_hoa in ('cc', 'both'),
    'Mixture_HOA_Pure_CC':     _mix_hoa in ('hoa_pure_cc',    'both'),
    'Mixture_HOA_Pure_Virial': _mix_hoa in ('hoa_pure_virial', 'both'),
    'Mixture_HOA_Pure_File':   _mix_hoa in ('hoa_file', 'data_file', 'both'),
}

# ---------------------------------------------------------------------------
# Data file path (used by Main to load fitting + RASPA from same directory)
# ---------------------------------------------------------------------------

_data_file_fitting_str = str(read_input_file.get('DATA_FILE_FITTING', 'none')).strip()
_data_file_points_str  = str(read_input_file.get('DATA_FILE_POINTS',  'none')).strip()
_data_source_str       = str(read_input_file.get('DATA_SOURCE', 'fitting')).lower()
# Primary file drives load_files_dir (kept for any legacy usage)
_data_file_str = (_data_file_fitting_str if _data_source_str == 'fitting'
                  else _data_file_points_str)
load_files_dir = os.path.dirname(_data_file_str) or '.'

_isotherm_raw = read_input_file.get('ISOTHERM_TYPE')
_isotherm_st = '' if _isotherm_raw is None else str(_isotherm_raw).strip().lower()
_isotherm_type_cfg = _isotherm_st if _isotherm_st in ('pure', 'mixture') else 'auto'

_virial_fitting_degrees = _parse_virial_degrees_combo(read_input_file.get('VIRIAL_DEGREES_COMBO'))
# Missing SUGGESTION_VIRIAL: use explicit COMBO / VIRIAL_DEGREES when given (→ suggestion off),
# otherwise default to auto-search (same as ``SUGGESTION_VIRIAL yes``).
if 'SUGGESTION_VIRIAL' in read_input_file:
    _suggestion_virial = _config_yes_no(read_input_file.get('SUGGESTION_VIRIAL'), default=False)
else:
    _suggestion_virial = not _has_explicit_virial_degrees_in_config(read_input_file, _virial_fitting_degrees)
_virial_degrees_resolved = _resolve_virial_degrees_for_config(
    _suggestion_virial,
    read_input_file,
    selection['fw'],
    selection['mol'],
    _virial_fitting_degrees,
)

# ---------------------------------------------------------------------------
# Storage-density defaults when keys are omitted (P_MIN / P_MAX = isotherm window)
# ---------------------------------------------------------------------------
_p_min_iso = read_input_file.get('P_MIN')
_p_max_iso = read_input_file.get('P_MAX')
_t_des_for_config = read_input_file.get('T_DES', selection['temp'])
_t_des_floats: list[float] = []
for _t in _listify(_t_des_for_config):
    try:
        _t_des_floats.append(float(_t))
    except (TypeError, ValueError):
        continue
_t_ads_raw = read_input_file.get('T_ADS')
if _t_ads_raw is not None:
    _t_ads_resolved = float(_t_ads_raw)
elif _t_des_floats:
    _t_ads_resolved = min(_t_des_floats)
else:
    raise ValueError(
        "config.in: set T_ADS, or set TEMPERATURE and/or T_DES with at least one temperature "
        "so T_ADS can default to the lowest desorption temperature."
    )

config = {
    # Pressure range
    'P_MIN':     float(read_input_file.get('P_MIN')),
    'P_MAX':     float(read_input_file.get('P_MAX')),
    'P_MIN_SD':  float(read_input_file.get('P_DES_MIN', _p_min_iso)),
    'P_des_max': float(read_input_file.get('P_DES_MAX', _p_max_iso)),
    # Fixed pressures for T_ads–T_des storage-density 3D plot
    'P_ads_TT':  float(read_input_file.get('P_ADS_TT', _p_max_iso)),
    'P_des_TT':  float(read_input_file.get('P_DES_TT', _p_min_iso)),

    # Storage density
    'T_ads':     float(_t_ads_resolved),
    'P_ads':     float(read_input_file.get('P_ADS', _p_max_iso)),
    'T_des':     _t_des_for_config,

    # Fitting
    'suggestion_virial': _suggestion_virial,
    'virial_degrees': _virial_degrees_resolved,
    'virial_fitting_degrees': _virial_fitting_degrees,
    'n_loadings':    int(read_input_file.get('N_LOADINGS', 50)),

    # Data
    'data_file_fitting': _data_file_fitting_str,
    'data_file_points':  _data_file_points_str,
    'data_file_hoa':     str(read_input_file.get('DATA_FILE_HOA', 'none')).strip(),
    'data_source':       _data_source_str,
    # 'auto' when ISOTHERM_TYPE omitted: Main infers from RASPA + ADSORBATE (see Main.py).
    'isotherm_type':     _isotherm_type_cfg,

    # Output (omit SHOW_POINTS / OUT_DIR → off, same as ``no``)
    'out_dir':     _config_yes_no(read_input_file.get('OUT_DIR'), False),
    # Pressure scale for isotherm plots: 'log', 'linear', or 'both'
    'pressure_scale': str(read_input_file.get('PRESSURE_SCALE', 'both')).strip().lower(),
    'show_points': _config_yes_no(read_input_file.get('SHOW_POINTS'), False),
    'show_plots':  _config_yes_no(read_input_file.get('SHOW_PLOTS'), False),

    # HoA file-curve legend label only (DATA_FILE_HOA); CC and Virial are fixed in code
    'hoa_legend_file': str(read_input_file.get('HOA_LEGEND_FILE') or 'Data').strip(),
}

_DESIGN_DEFAULTS = {
    # When these mappings are empty, PlotHelpers will assign styles/markers
    # automatically from its internal palettes, purely based on ordering of
    # structures / fit types / molecules.
    'structure_linestyle_mapping': plt.get_cmap("Dark2").colors,
    'fit_type_linestyle_mapping': {},
    'molecule_marker_mapping': {},
    'molecule_linestyle_mapping': {},
    'molecule_color_mapping': plt.get_cmap("Set2").colors,
    'structure_color_mapping': {},

    # Optional palettes to control the automatic assignment order when the
    # corresponding mapping is empty.
    'structure_linestyle_palette': ['-', '--', '-.', ':'],
    'fit_type_linestyle_palette': ['-', '--', '-.', ':'],
    'molecule_marker_palette': ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd'],
    'molecule_linestyle_palette': ['-', '--', '-.', ':'],

    'HoA_method_linestyles': {
        'clausius_clapeyron': '-',
        'virial':             '--',
        'hoa_file':           ':',
    },
    # When temperature_color_mapping is empty, Main will fall back to
    # temperature_palette for ordering-based colors.
    'temperature_color_mapping': {},
    'temperature_palette': plt.get_cmap("Set1").colors,
    # Storage-density 3D colormap / clim (layout: PlotHelpers.STORAGE_DENSITY_3D_DEFAULTS)
    'storage_density_3d_colormap': 'viridis',
    'storage_density_3d_boundries': None,
}

_design = {**_DESIGN_DEFAULTS, **_load_design_in(_repo_root / 'design.in')}

_functions_dir = Path(__file__).resolve().parent / 'functions'
if str(_functions_dir) not in sys.path:
    sys.path.insert(0, str(_functions_dir))

import PlotHelpers as _ph_sd3d
_SD3D = _ph_sd3d.STORAGE_DENSITY_3D_DEFAULTS


def _sd3d(key):
    return _design.get(key, _SD3D[key])

# Plot styling — defined in design.in (with defaults above)
structure_linestyle_mapping = _design['structure_linestyle_mapping']
fit_type_linestyle_mapping = _design['fit_type_linestyle_mapping']
molecule_marker_mapping = _design['molecule_marker_mapping']
molecule_linestyle_mapping = _design.get('molecule_linestyle_mapping', {})
molecule_color_mapping = _design.get('molecule_color_mapping')
structure_color_mapping = _design.get('structure_color_mapping', {})
structure_linestyle_palette = _design.get('structure_linestyle_palette', ['-', '--', '-.', ':'])
fit_type_linestyle_palette = _design.get('fit_type_linestyle_palette', ['-', '--', '-.', ':'])
molecule_marker_palette = _design.get('molecule_marker_palette', ['o', 's', '^'])
molecule_linestyle_palette = _design.get('molecule_linestyle_palette', ['-', '--', '-.', ':'])
hoa_method_linestyles = _design['HoA_method_linestyles']
temperature_color_mapping = _design['temperature_color_mapping']
temperature_palette = _design.get('temperature_palette')
storage_density_3d_figsize = _design.get('storage_density_3d_figsize')
storage_density_3d_figwidth_pct_extra = float(_sd3d('storage_density_3d_figwidth_pct_extra'))
storage_density_3d_colormap = _design.get('storage_density_3d_colormap', 'viridis')
storage_density_3d_boundries = _design.get('storage_density_3d_boundries')
if storage_density_3d_boundries is None and 'storage_density_3d_clim' in _design:
    storage_density_3d_boundries = _design.get('storage_density_3d_clim')
storage_density_3d_subplot_left = float(_sd3d('storage_density_3d_subplot_left'))
storage_density_3d_subplot_bottom = float(_sd3d('storage_density_3d_subplot_bottom'))
storage_density_3d_subplot_top = float(_sd3d('storage_density_3d_subplot_top'))
storage_density_3d_subplot_right = float(_sd3d('storage_density_3d_subplot_right'))
storage_density_3d_axes_dist = float(_sd3d('storage_density_3d_axes_dist'))
storage_density_3d_box_scale = float(_sd3d('storage_density_3d_box_scale'))
storage_density_3d_box_halign = float(_sd3d('storage_density_3d_box_halign'))
storage_density_3d_box_valign = float(_sd3d('storage_density_3d_box_valign'))
storage_density_3d_labelpad_x = float(_sd3d('storage_density_3d_labelpad_x'))
storage_density_3d_labelpad_y = float(_sd3d('storage_density_3d_labelpad_y'))
storage_density_3d_labelpad_z = float(_sd3d('storage_density_3d_labelpad_z'))
storage_density_3d_colorbar_gap = float(_sd3d('storage_density_3d_colorbar_gap'))
storage_density_3d_cbar_label_labelpad = float(_sd3d('storage_density_3d_cbar_label_labelpad'))
storage_density_3d_cbar_label_y = float(_sd3d('storage_density_3d_cbar_label_y'))
storage_density_3d_colorbar_width = float(_sd3d('storage_density_3d_colorbar_width'))
storage_density_3d_colorbar_shrink = float(_sd3d('storage_density_3d_colorbar_shrink'))
storage_density_3d_colorbar_ticks = _sd3d('storage_density_3d_colorbar_ticks')
storage_density_3d_colorbar_nticks = int(_sd3d('storage_density_3d_colorbar_nticks'))
storage_density_3d_save_bbox_pad = float(_sd3d('storage_density_3d_save_bbox_pad'))
storage_density_3d_save_dpi = int(_sd3d('storage_density_3d_save_dpi'))
