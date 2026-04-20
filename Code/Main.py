import warnings
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # for now to no have the plots pop up on the screen
from matplotlib import pyplot as _plt

# Suppress non-critical Agg backend user warnings that are expected in headless runs
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Tight layout not applied.*",
    category=UserWarning,
)


R2_MIN              = 0.95   # minimum R² to accept a Clausius-Clapeyron fit
R2_VIRIAL_THRESHOLD = 0.99   # minimum R² target for virial degree suggestion
VIRIAL_MAX_DEG      = 6      # highest degree tested in the virial degree search
VIRIAL_MIN_POINTS   = 3      # minimum data points required per virial fit
CC_SMOOTHING_SIGMA  = 1.5    # Gaussian smoothing sigma for Clausius-Clapeyron
CC_MIN_TEMPS        = 3      # minimum number of temperatures required for a CC fit


visualisation_dir = Path(__file__).resolve().parent
_env_root = os.environ.get("PIPELINE_REPO_ROOT", "").strip()
if _env_root:
    repo_root = Path(_env_root).resolve()
else:
    # Allow ``python Main.py`` from an example directory (same folder as config.in)
    # without going through example ``run.py`` — match run.py behaviour via cwd.
    _cwd_cfg = Path.cwd().resolve() / "config.in"
    if _cwd_cfg.is_file():
        repo_root = Path.cwd().resolve()
        os.environ["PIPELINE_REPO_ROOT"] = str(repo_root)
    else:
        repo_root = visualisation_dir.parent
functions_dir = visualisation_dir / 'functions'

# Prevent the built-in 'code' module from shadowing the local one
if 'code' in sys.modules and hasattr(sys.modules['code'], '__file__') and 'code.py' in str(sys.modules['code'].__file__):
    if '/usr/local/lib' in str(sys.modules['code'].__file__) or '/usr/lib' in str(sys.modules['code'].__file__):
        del sys.modules['code']

if str(functions_dir) not in sys.path:
    sys.path.insert(0, str(functions_dir))
if str(visualisation_dir) not in sys.path:
    sys.path.insert(0, str(visualisation_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ---------------------------------------------------------------------------
# Imports from local code directory
# ---------------------------------------------------------------------------
import Initialize as init           # type: ignore
import IsothermFittingPlot          # type: ignore
import MolFraction                  # type: ignore
import PlotHelpers as phelp         # type: ignore
import ClausiusClapeyron as cc      # type: ignore
import Control as control           # type: ignore
import Virial as virial             # type: ignore
import StorageDensity as sd         # type: ignore
import DataSelection as ds          # type: ignore
import Input                        # type: ignore


config    = Input.config
selection = Input.selection

# ---------------------------------------------------------------------------
# Resolve the two data file paths
# ---------------------------------------------------------------------------
def _resolve(s):
    """Return an absolute Path, or None when the value is 'none'/empty."""
    if not s or str(s).strip().lower() == 'none':
        return None
    p = Path(str(s).strip())
    return p if p.is_absolute() else repo_root / p

_fitting_file = _resolve(config['data_file_fitting'])
_points_file  = _resolve(config['data_file_points'])
_hoa_points_file = _resolve(config.get('data_file_hoa', 'none'))


def _fitting_has_rows_for_molecule(fits_list, mol, frameworks, temperatures, tol=1.0):
    """True if the fitting table contains at least one row for ``mol`` (optional fw / T filter)."""
    if not fits_list or not mol:
        return False
    mol_l = str(mol).strip().lower()
    fw_set = {str(x) for x in (frameworks or [])}
    temp_list = [float(t) for t in (temperatures or [])]
    for row in fits_list:
        if str(row.get('molecule', '')).strip().lower() != mol_l:
            continue
        if fw_set and str(row.get('framework', '')).strip() not in fw_set:
            continue
        try:
            t_r = float(row.get('temperature'))
        except (TypeError, ValueError):
            continue
        if not temp_list:
            return True
        for tt in temp_list:
            if abs(t_r - float(tt)) <= tol:
                return True
    return False


# Fitting table from disk (used for isotherm curves when DATA_SOURCE=fitting, and
# always available for mixture pure-HOA CC / component synthesis even in points mode).
_fits_from_disk = (
    init.load_fitting_data(str(_fitting_file), pressure_unit=selection['pressure_unit'])
    if _fitting_file
    else []
)

# Primary data load — controlled by DATA_SOURCE
if config['data_source'] == 'fitting':
    fits = _fits_from_disk
    # When using fitting-derived isotherms, still optionally show RASPA
    # points (scatter) if the user enabled it.
    if config.get('show_points', True) and _points_file:
        data_points = init.load_RASPA_data(str(_points_file), pure_only=True)
    else:
        data_points = []
else:  # 'points'
    fits = []
    data_points = init.load_RASPA_data(str(_points_file), pure_only=True) if _points_file else []

# Full RASPA table (pure + mixture rows) for autodetection and mixture filtering.
_raspa_full = []
if _points_file:
    try:
        _raspa_full = init.load_RASPA_data(str(_points_file), pure_only=False)
    except Exception as e:
        print(f"Warning: could not load RASPA points file for mixture/autodetect: {e}")

# Pure vs mixture: default is ``auto`` (omit ISOTHERM_TYPE in config.in) — decided from RASPA
# using ADSORBATE vs ``mixture_pure`` on mixture rows. Optional explicit ISOTHERM_TYPE pure|mixture.
_iso_cfg = str(config.get('isotherm_type') or 'auto').strip().lower()
if _iso_cfg not in ('pure', 'mixture', 'auto'):
    print(f"Warning: ISOTHERM_TYPE {_iso_cfg!r} is not pure, mixture, or auto; using auto.")
    _iso_cfg = 'auto'

_mixture_use_points_fallback = False
_mols_sel = selection['mol']
if len(_mols_sel) > 1:
    if _iso_cfg == 'mixture':
        print("Warning: ISOTHERM_TYPE is mixture but ADSORBATE lists several species; using pure.")
    config['isotherm_type'] = 'pure'
elif len(_mols_sel) == 1:
    _ads = str(_mols_sel[0]).strip()
    if _raspa_full:
        _has_blend = any(
            str(d.get('mixture_pure', '')).strip().lower() == _ads.lower()
            and str(d.get('mixture_pure', '')).strip().lower() != 'pure'
            for d in _raspa_full
        )
        if _iso_cfg == 'pure':
            config['isotherm_type'] = 'pure'
        elif _iso_cfg == 'mixture':
            if _has_blend:
                config['isotherm_type'] = 'mixture'
            else:
                print(
                    f"Warning: ISOTHERM_TYPE is mixture but RASPA has no mixture rows with "
                    f"mixture_pure={_ads!r} (non-pure); using pure."
                )
                config['isotherm_type'] = 'pure'
        else:
            if _has_blend:
                print(f"Autodetection mixture components: {_ads}")
                config['isotherm_type'] = 'mixture'
            else:
                print(f"Autodetection pure component: {_ads}")
                config['isotherm_type'] = 'pure'
    else:
        if _iso_cfg == 'mixture':
            print("Warning: ISOTHERM_TYPE is mixture but RASPA points are missing or empty; using pure.")
            config['isotherm_type'] = 'pure'
        elif _iso_cfg == 'pure':
            config['isotherm_type'] = 'pure'
        else:
            print(f"Autodetection pure component: {_ads}")
            config['isotherm_type'] = 'pure'
else:
    config['isotherm_type'] = 'pure'

# Mixture data is always RASPA data points regardless of DATA_SOURCE.
# In the RASPA file col-3 holds the mixture name (e.g. R407F) for mixture rows,
# and 'pure' for pure rows — so we filter by the selected adsorbate name.
mixture_data = []
if config['isotherm_type'] == 'mixture' and selection['mol']:
    _mixture_name = selection['mol'][0]
    try:
        mixture_data = [d for d in _raspa_full
                        if str(d.get('mixture_pure', '')).lower() == _mixture_name.lower()]
        if not mixture_data:
            print(f"No mixture data found for '{_mixture_name}' in {_points_file}. "
                  f"Ensure DATA_FILE_POINTS points to the RASPA simulation results.")
    except Exception as e:
        print(f"Mixture data loading failed: {e}")

# Auto-select temperatures if not set in config.in
if selection['temp'] is None:
    if config['isotherm_type'] == 'mixture' and mixture_data:
        # For mixture: derive temperatures from the mixture rows themselves
        selection['temp'] = sorted({d['temperature'] for d in mixture_data
                                     if d.get('framework') in selection['fw']})
    else:
        source = fits if fits else data_points
        selection['temp'] = sorted({e['temperature'] for e in source
                                     if e.get('framework') in selection['fw']
                                     and e.get('molecule') in selection['mol']})
    print(f"Auto-selected temperatures: {selection['temp']}")

# Mixture + fitting source: if the fitting table has no row for the adsorbate name (blend),
# do not synthesize calculation grids from fits — use RASPA points instead.
if (
    config.get('isotherm_type') == 'mixture'
    and str(config.get('data_source', '')).lower() == 'fitting'
    and selection.get('mol')
):
    _mol_chk = selection['mol'][0]
    if not _fitting_has_rows_for_molecule(
        _fits_from_disk, _mol_chk, selection.get('fw'), selection.get('temp')
    ):
        _mixture_use_points_fallback = True
        if _points_file and not data_points:
            try:
                data_points = init.load_RASPA_data(str(_points_file), pure_only=True)
            except Exception as _e_pf:
                print(f"Warning: RASPA points load for mixture fallback failed: {_e_pf}")

# ---------------------------------------------------------------------------
# Plot styling — push Input.py mappings into PlotHelpers
# ---------------------------------------------------------------------------
phelp.set_marker_map(getattr(Input, 'marker_mapping', {}))
phelp.set_structure_linestyle_map(getattr(Input, 'structure_linestyle_mapping', {}))
phelp.set_structure_linestyle_palette(getattr(Input, 'structure_linestyle_palette', None))
phelp.set_structure_color_map(getattr(Input, 'structure_color_mapping', {}))
phelp.set_fit_type_linestyle_map(getattr(Input, 'fit_type_linestyle_mapping', {}))
phelp.set_fit_type_linestyle_palette(getattr(Input, 'fit_type_linestyle_palette', None))
phelp.set_molecule_marker_map(getattr(Input, 'molecule_marker_mapping', {}))
phelp.set_molecule_marker_palette(getattr(Input, 'molecule_marker_palette', None))
phelp.set_molecule_linestyle_map(getattr(Input, 'molecule_linestyle_mapping', {}))
phelp.set_molecule_linestyle_palette(getattr(Input, 'molecule_linestyle_palette', None))
phelp.set_molecule_color_map(getattr(Input, 'molecule_color_mapping', {}))
phelp.set_molecule_display_name_override(getattr(Input, 'molecule_display_name_override', {}))

x      = np.logspace(np.log10(config['P_MIN']), np.log10(config['P_MAX']), 300)

# Build colors over the union of all temperatures that can appear in any plot
# (TEMPERATURE, T_DES, and T_ADS may each add distinct values)
_t_des_list = config['T_des'] if isinstance(config['T_des'], list) else [config['T_des']]
_all_plot_temps = sorted(
    {float(t) for t in (selection['temp'] or [])} |
    {float(t) for t in _t_des_list} |
    {float(config['T_ads'])}
)

colors = init.get_combo_colors(
    selection['fw'],
    selection['mol'],
    _all_plot_temps,
    getattr(Input, 'temperature_palette', ['orange', 'red', 'green']),
    temperature_color_mapping=getattr(Input, 'temperature_color_mapping', None),
    molecule_color_mapping=getattr(Input, 'molecule_color_mapping', None),
    structure_color_mapping=getattr(Input, 'structure_color_mapping', None),
)
hoa_method_linestyles = getattr(Input, 'hoa_method_linestyles', {'clausius_clapeyron': '-', 'virial': '--'})

# ---------------------------------------------------------------------------
# Settings summary (saved to Output/<run>/settings_summary.txt)
# ---------------------------------------------------------------------------
def _fmt_list(vals):
    if vals is None:
        return "[]"
    return "[" + ", ".join(str(v) for v in vals) + "]"

def _subset_map(mapping, keys):
    m = mapping or {}
    if not isinstance(m, dict):
        return {}
    return {k: m[k] for k in keys if k in m}


def _as_dict(mapping):
    return mapping if isinstance(mapping, dict) else {}


def _norm_fit_type(ft):
    if ft is None:
        return ''
    return str(ft).replace('-', '_').strip().lower()


def _fit_type_mapping_lookup(ft, fm):
    """Return value from fit_type_linestyle_mapping if any key matches ``ft`` (hyphen/underscore tolerant)."""
    if not fm:
        return None
    ft_n = _norm_fit_type(ft)
    if ft in fm:
        return fm[ft]
    for k, v in fm.items():
        if _norm_fit_type(k) == ft_n:
            return v
    return None


def _build_effective_style_summary_lines(
    sel_fw,
    sel_mol,
    sel_fit,
    active_hoa_methods,
    hoa_ls_dict,
    *,
    combo_colors=None,
    plot_temperatures=None,
):
    """Each line states the resolved value used in plots (no source tags)."""
    lines = [
        "Effective plot styles (this run — each line is the value actually used):",
    ]
    ls_map = _as_dict(getattr(Input, 'structure_linestyle_mapping', {}))
    if not sel_fw:
        lines.append("    (n/a — no frameworks in selection)")
    else:
        for fw in sel_fw:
            if fw in ls_map:
                lines.append(f"    {fw}: {ls_map[fw]!r}")
            else:
                _ls = phelp.get_linestyle_for_structure(fw)
                lines.append(f"    {fw}: {_ls!r}")

    lines.append("  Structure colours (when used as structure colour):")
    sc_map = _as_dict(getattr(Input, 'structure_color_mapping', {}))
    if not sel_fw:
        lines.append("    (n/a — no frameworks in selection)")
    else:
        for fw in sel_fw:
            if fw in sc_map:
                lines.append(f"    {fw}: {sc_map[fw]!r}")
            else:
                _c = phelp.get_color_for_structure(fw)
                lines.append(f"    {fw}: {_c!r}")

    lines.append("  Molecule colours & markers:")
    mc_map = _as_dict(getattr(Input, 'molecule_color_mapping', {}))
    mm_map = _as_dict(getattr(Input, 'molecule_marker_mapping', {}))
    if not sel_mol:
        lines.append("    (n/a — no molecules in selection)")
    else:
        for mol in sel_mol:
            if mol in mc_map:
                c_note = f"colour={mc_map[mol]!r}"
            else:
                _cc = phelp.get_color_for_molecule(mol)
                c_note = f"colour={_cc!r}"
            if mol in mm_map:
                m_note = f"marker={mm_map[mol]!r}"
            else:
                _mk = phelp.get_marker_for_molecule(mol)
                m_note = f"marker={_mk!r}"
            lines.append(f"    {mol}: {c_note}; {m_note}")

    mat_map = _as_dict(getattr(Input, 'marker_mapping', {}))
    lines.append("  Framework markers (material / RASPA scatter via marker_mapping):")
    if not sel_fw:
        lines.append("    (n/a — no frameworks in selection)")
    else:
        for fw in sel_fw:
            if fw in mat_map:
                lines.append(f"    {fw}: {mat_map[fw]!r}")
            else:
                _mm = phelp.get_marker_for_material(fw)
                lines.append(f"    {fw}: {_mm!r}")

    if sel_fit:
        lines.append("  Fit-type line styles:")
        fm = _as_dict(getattr(Input, 'fit_type_linestyle_mapping', {}))
        for ft in sel_fit:
            hit = _fit_type_mapping_lookup(ft, fm)
            if hit is not None:
                lines.append(f"    {ft}: {hit!r}")
            else:
                _ftls = phelp.get_linestyle_for_fit_type(ft)
                lines.append(f"    {ft}: {_ftls!r}")

    mls_map = _as_dict(getattr(Input, 'molecule_linestyle_mapping', {}))
    if any(m in mls_map for m in (sel_mol or [])):
        lines.append("  Molecule line styles (where set):")
        for mol in sel_mol or []:
            if mol in mls_map:
                lines.append(f"    {mol}: {mls_map[mol]!r}")

    if active_hoa_methods:
        lines.append("  HoA method line styles (active flags):")
        for method in active_hoa_methods:
            ls = (hoa_ls_dict or {}).get(method, '-')
            lines.append(f"    {method}: {ls!r}")

    _temps_list = plot_temperatures
    if _temps_list is not None and not isinstance(_temps_list, (list, tuple)):
        _temps_list = [_temps_list]
    if combo_colors is not None and _temps_list and sel_fw and sel_mol:
        lines.append("  Temperature–combo colours (first framework & first adsorbate in selection):")
        rw, rm = sel_fw[0], sel_mol[0]
        for T in _temps_list:
            col = phelp.get_combo_color_temperature(combo_colors, rw, rm, T)
            if col is None:
                col = phelp.get_combo_color_temperature(combo_colors, rw, rm, float(T))
            try:
                t_lab = f"{int(round(float(T)))}K"
            except Exception:
                t_lab = f"{T}K"
            lines.append(f"    ({rw}, {rm}) @ {t_lab}: {col!r}")

    return lines


# Only print design choices relevant for this run selection
_sel_fw = [str(x) for x in (selection.get('fw') or [])]
_sel_mol = [str(x) for x in (selection.get('mol') or [])]
_sel_fit = [str(x) for x in (selection.get('fit_types') or [])]
_sel_temp_raw = selection.get('temp')
if isinstance(_sel_temp_raw, (list, tuple, set)):
    _sel_temp = [str(x) for x in _sel_temp_raw]
elif _sel_temp_raw is None:
    _sel_temp = []
else:
    _sel_temp = [str(_sel_temp_raw)]
_active_hoa_methods = []
if Input.plot_flags.get('Clausius_Clapeyron'):
    _active_hoa_methods.append('clausius_clapeyron')
if Input.plot_flags.get('Virial'):
    _active_hoa_methods.append('virial')
if Input.plot_flags.get('HOA_From_File'):
    _active_hoa_methods.append('hoa_file')

_effective_style_lines = _build_effective_style_summary_lines(
    _sel_fw,
    _sel_mol,
    _sel_fit,
    _active_hoa_methods,
    hoa_method_linestyles,
    combo_colors=colors,
    plot_temperatures=selection.get('temp'),
)

_run_folder = (
    f"{'-'.join(str(x).replace(' ', '_') for x in (_sel_fw or ['all']))}_"
    f"{'-'.join(str(x).replace(' ', '_') for x in (_sel_mol or ['all']))}_"
    f"{'-'.join(str(x).replace(' ', '_') for x in (_sel_temp or ['all']))}"
)
_settings_out_dir = repo_root / "Output" / _run_folder
_settings_out_dir.mkdir(parents=True, exist_ok=True)
_summary_name = (
    f"Summary_{'-'.join(str(x).replace(' ', '_') for x in (_sel_fw or ['all']))}"
    f"_{'-'.join(str(x).replace(' ', '_') for x in (_sel_mol or ['all']))}"
    f"_{'-'.join(str(x).replace(' ', '_') for x in (_sel_temp or ['all']))}.txt"
)
_settings_path = _settings_out_dir / _summary_name

_design_choice_lines = []
_m = _subset_map(getattr(Input, 'structure_linestyle_mapping', {}), _sel_fw)
if _m:
    _design_choice_lines.append(f"  Structure line styles (selected): {_m}")
_m = _subset_map(getattr(Input, 'structure_color_mapping', {}), _sel_fw)
if _m:
    _design_choice_lines.append(f"  Structure colors (selected): {_m}")
_m = _subset_map(getattr(Input, 'molecule_color_mapping', {}), _sel_mol)
if _m:
    _design_choice_lines.append(f"  Molecule colors (selected): {_m}")
_m = _subset_map(getattr(Input, 'molecule_marker_mapping', {}), _sel_mol)
if _m:
    _design_choice_lines.append(f"  Molecule markers (selected): {_m}")
_m = _subset_map(getattr(Input, 'fit_type_linestyle_mapping', {}), _sel_fit)
if _m:
    _design_choice_lines.append(f"  Fit-type line styles (selected): {_m}")
_m = _subset_map(hoa_method_linestyles, _active_hoa_methods)
if _m:
    _design_choice_lines.append(f"  HoA method line styles (active): {_m}")

_settings_lines = [
    "=== Settings Summary ===",
    f"Isotherm type: {config.get('isotherm_type')}",
    f"Frameworks (structures): {_fmt_list(selection.get('fw'))}",
    f"Molecules (selected): {_fmt_list(selection.get('mol'))}",
    f"Temperatures (selected): {_fmt_list(selection.get('temp'))}",
    f"Fit types: {_fmt_list(selection.get('fit_types'))}",
    f"Data source: {config.get('data_source')}",
    *(
        ["Mixture: calculation data from RASPA (no fitting rows for this adsorbate)."]
        if _mixture_use_points_fallback
        else []
    ),
    f"Show points: {config.get('show_points')}",
    f"Show plots: {config.get('show_plots')}",
    f"Pressure range (isotherm): P_MIN={config.get('P_MIN')} Pa, P_MAX={config.get('P_MAX')} Pa",
    f"Pressure range (storage-density desorption): P_DES_MIN={config.get('P_MIN_SD')} Pa, P_DES_MAX={config.get('P_des_max')} Pa",
    f"Fixed adsorption pressure: P_ads={config.get('P_ads')} Pa",
]
if _design_choice_lines:
    _settings_lines.extend(["", "Design choices:", *_design_choice_lines])
_settings_lines.extend([*_effective_style_lines, "=== End Settings ===", ""])
try:
    _settings_path.write_text("\n".join(_settings_lines), encoding="utf-8")
except Exception as _e:
    print(f"Warning: failed to write settings summary file at {_settings_path}: {_e}")

# ---------------------------------------------------------------------------
# Build calculation data set (synthetic from fits, or raw data points)
# ---------------------------------------------------------------------------
_use_fitting_for_calc = (
    str(config.get('data_source', 'fitting')).lower() == 'fitting'
    and not _mixture_use_points_fallback
)
if _use_fitting_for_calc:
    # Log-spaced pressures for Virial/CC (not tied to PRESSURE_SCALE in config.in).
    data_points_calc = IsothermFittingPlot.synthesize_points_from_fittings(
        fits, selection['fw'], selection['mol'],
        selection['temp'], selection['fit_types'],
        n_loadings=config['n_loadings'], p_min=config['P_MIN'], p_max=config['P_MAX'],
        p_grid=None, formula_fit_types=selection['fit_types'],
        num_of_isotherm=selection.get('num_of_isotherm'),
        pressure_scale='log',
        save_data=config.get('out_dir', False))
    synth_pressures = sorted({float(d['pressure']) for d in data_points_calc
                              if d.get('pressure') is not None and float(d['pressure']) > 0})
    x_calc = np.array(synth_pressures, dtype=float) if synth_pressures else x
    print(
        f"Using Adsorption Isotherm Models. For Heat of Adsorption created "
        f"{len(data_points_calc)} points."
    )
else:
    data_points_calc = data_points
    x_calc = x
    if (config.get('isotherm_type') != 'mixture' and selection.get('fw')
            and selection.get('mol') and selection.get('temp')):
        _n_pts_calc = len(phelp.filter_raspa_data(
            data_points_calc,
            frameworks=selection['fw'],
            molecules=selection['mol'],
            temperatures=selection['temp'],
            only_pure_adsorption=True,
        ))
    else:
        _n_pts_calc = len(data_points_calc)
    print(f"Using data points for calculations: {_n_pts_calc} points")

# Build shared dataset for Virial and CC via DataSelection.
# Works for both data_source modes: 'fitting' (data_points_calc = synthetic points from
# synthesize_points_from_fittings) and 'points' (data_points_calc = raw RASPA data).
# DataSelection applies PCHIP re-gridding, p-bounds, and the min_temps coverage constraint.
data_points_virial = data_points_calc
raspa_for_cc = data_points_calc
_sd_m = str(Input.plot_flags.get('Storage_Density_Method', '') or '').lower()
_want_dataset = (
    Input.plot_flags.get('Virial')
    or Input.plot_flags.get('Clausius_Clapeyron')
    or _sd_m in ('virial', 'both', 'cc')
    or Input.plot_flags.get('Mixture_HOA_Pure_CC')
    or Input.plot_flags.get('Mixture_HOA_Pure_Virial')
)
if _want_dataset:
    try:
        _rows = ds.build_dataset(
            data_points_calc,
            selection['fw'], selection['mol'], selection['temp'],
            n_loadings=config['n_loadings'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            min_temps=CC_MIN_TEMPS,
        )
        if _rows:
            data_points_virial = _rows
            raspa_for_cc = _rows
    except Exception as _e:
        print(f"Warning: DataSelection failed ({_e}); Virial/CC use raw input data.")

# Heat-of-adsorption paths (pure or mixture) need at least three isotherm temperatures for CC/Virial.
_hoa_needs_three_temps = bool(
    Input.plot_flags.get('Virial')
    or Input.plot_flags.get('Clausius_Clapeyron')
    or Input.plot_flags.get('Mixture_CC')
    or Input.plot_flags.get('Mixture_HOA_Pure_CC')
    or Input.plot_flags.get('Mixture_HOA_Pure_Virial')
    or _sd_m in ('virial', 'both', 'cc')
)
if _hoa_needs_three_temps:
    _ts_warn = selection.get('temp')
    if isinstance(_ts_warn, (list, tuple, set)):
        _n_ts_warn = len(_ts_warn)
    elif _ts_warn is None:
        _n_ts_warn = 0
    else:
        _n_ts_warn = 1
    if _n_ts_warn < 3:
        print(
            "> Select at least 3 temperatures to calculate the heat of adsorption."
        )

# ---------------------------------------------------------------------------
# Per-gas-component DataSelection dataset for mixture pure HOA (CC + Virial).
# For BOTH plots the loading range must come from the same DataSelection grid —
# identical to what the pure-component CC/Virial plots use.  We always
# synthesise from fittings so that both paths use fitting values.
# Uses _fits_from_disk so DATA_SOURCE=points still gets per-gas rows from DATA_FILE_FITTING.
# ---------------------------------------------------------------------------
raspa_pure_mixture_components = None
if (
    config.get('isotherm_type') == 'mixture'
    and mixture_data
    and selection.get('fw')
    and _fits_from_disk
    and (Input.plot_flags.get('Mixture_HOA_Pure_CC') or Input.plot_flags.get('Mixture_HOA_Pure_Virial'))
):
    _cmp_mols = sorted({d['molecule'] for d in mixture_data})
    if _cmp_mols:
        try:
            _cmp_synth = IsothermFittingPlot.synthesize_points_from_fittings(
                fittings=_fits_from_disk,
                selected_frameworks=selection['fw'],
                selected_molecules=_cmp_mols,
                selected_temperatures=selection['temp'],
                selected_fit_types=selection['fit_types'],
                n_loadings=config['n_loadings'],
                p_min=config['P_MIN'],
                p_max=config['P_MAX'],
                p_grid=None,
                formula_fit_types=selection['fit_types'],
                num_of_isotherm=selection.get('num_of_isotherm'),
                pressure_scale='log',
                save_data=False,
                folder_molecule_label=str(selection['mol'][0]),
            )
            if _cmp_synth:
                _cmp_ds = ds.build_dataset(
                    _cmp_synth,
                    selection['fw'],
                    _cmp_mols,
                    selection['temp'],
                    n_loadings=config['n_loadings'],
                    p_min=config['P_MIN'],
                    p_max=config['P_MAX'],
                    min_temps=CC_MIN_TEMPS,
                )
                if _cmp_ds:
                    raspa_pure_mixture_components = _cmp_ds
                    print(f"Mixture components: {_cmp_mols}")
        except Exception as _e_cmp:
            print(
                f"Warning: mixture pure-component DataSelection failed ({_e_cmp}); "
                "pure HOA CC/Virial will derive loading range from synthesised points directly."
            )

# ---------------------------------------------------------------------------
# Virial polynomial degrees — resolved before the isotherm-type split so they
# are available in both mixture and pure branches.
# ---------------------------------------------------------------------------
_virial_deg_cfg = config.get('virial_degrees')
if _virial_deg_cfg is None:
    # SUGGESTION_VIRIAL=yes with VIRIAL_DEGREES omitted: internal fallback for
    # dict.get(..., fallback) only; per-(fw, mol) degrees come from search / COMBO.
    _virial_deg_a, _virial_deg_b = 2, 2
else:
    _virial_deg_a, _virial_deg_b = _virial_deg_cfg
# Per-(fw, mol) degrees: start from VIRIAL_DEGREES_COMBO in config.in; optional search overwrites below.
_virial_degrees_dict = dict(config.get('virial_fitting_degrees') or {})

# Molecules used in find_optimal_virial_degrees: for mixture runs use each **gas
# component** (R32, R125, …), matching ``degrees_per_combo[(fw, comp)]`` in
# ``plot_mixture_heat_hoa_pure_virial`` / storage-density virial — not the mixture label.
_virial_search_molecules = list(selection['mol'] or [])
if config.get('isotherm_type') == 'mixture' and mixture_data:
    _fw_sel = selection.get('fw') or []
    _from_mix = sorted(
        {d['molecule'] for d in mixture_data if d.get('framework') in _fw_sel}
    )
    if _from_mix:
        _virial_search_molecules = _from_mix

# Only search / apply Virial degrees when something actually uses the Virial model (HOA, SD-virial, or mixture pure-virial HOA).
_sd_for_virial_degrees = str(Input.plot_flags.get('Storage_Density_Method', '') or '').lower()
_need_virial_degrees = bool(
    Input.plot_flags.get('Virial')
    or _sd_for_virial_degrees in ('virial', 'both')
    or (config.get('isotherm_type') == 'mixture' and Input.plot_flags.get('Mixture_HOA_Pure_Virial'))
)

if config.get('suggestion_virial', False) and _need_virial_degrees:
    _deg_lines = [f"\n>>> Virial degree search (R² ≥ {R2_VIRIAL_THRESHOLD}, deg_a ≥ deg_b):"]
    for fw in selection['fw']:
        for mol in _virial_search_molecules:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    search = virial.find_optimal_virial_degrees(
                        RASPA_data=data_points_virial,
                        framework=fw, molecule=mol,
                        max_deg=VIRIAL_MAX_DEG,
                        min_points=VIRIAL_MIN_POINTS,
                        temperatures=selection['temp'],
                        verbose=False,
                        p_min=config['P_MIN']
                    )
                valid = [r for r in search['all_results']
                         if isinstance(r.get('r2'), float) and r['deg_a'] >= r['deg_b']]
                candidates = [r for r in valid if r['r2'] >= R2_VIRIAL_THRESHOLD]
                sug = None
                if candidates:
                    sug = min(candidates, key=lambda r: (r['deg_a'] + r['deg_b'], r['deg_a'], r['deg_b']))
                    _deg_lines.append(
                        f"    {fw} / {mol}:  deg_a={sug['deg_a']}, deg_b={sug['deg_b']}  →  R²={sug['r2']:.4f}  [applied]"
                    )
                elif valid:
                    best_r2 = max(round(r['r2'], 2) for r in valid)
                    sug = min(
                        (r for r in valid if round(r['r2'], 2) == best_r2),
                        key=lambda r: (r['deg_a'] + r['deg_b'], r['deg_a'], r['deg_b']))
                    _deg_lines.append(
                        f"    {fw} / {mol}:  no combination reached R² ≥ {R2_VIRIAL_THRESHOLD} — "
                        f"fallback: deg_a={sug['deg_a']}, deg_b={sug['deg_b']}  →  R²={sug['r2']:.4f}  [applied]"
                    )
                else:
                    _deg_lines.append(f"    {fw} / {mol}:  no valid results found")
                if sug is not None:
                    _virial_degrees_dict[(fw, mol)] = (sug['deg_a'], sug['deg_b'])
            except Exception as e:
                _deg_lines.append(f"    {fw} / {mol}:  search failed — {e}")
    _gf = (
        "    Global fallback (internal placeholder, VIRIAL_DEGREES omitted): "
        f"deg_a={_virial_deg_a}, deg_b={_virial_deg_b}"
        if _virial_deg_cfg is None
        else f"    Global fallback: deg_a={_virial_deg_a}, deg_b={_virial_deg_b}"
    )
    _deg_lines.append(_gf)
    _deg_lines.append("")
    virial.DEGREE_SEARCH_REPORT_TEXT = "\n".join(_deg_lines)
elif config.get('suggestion_virial', False):
    if _virial_deg_cfg is None:
        virial.DEGREE_SEARCH_REPORT_TEXT = (
            "Virial degree search: skipped (HEAT_OF_ADSORPTION is not virial/both; no storage-density virial; "
            "no mixture HOA–pure–virial).\n"
            "SUGGESTION_VIRIAL=yes: VIRIAL_DEGREES and VIRIAL_DEGREES_COMBO are not required for this run.\n"
        )
    else:
        virial.DEGREE_SEARCH_REPORT_TEXT = (
            "Virial degree search: skipped (HEAT_OF_ADSORPTION is not virial/both; no storage-density virial; "
            "no mixture HOA–pure–virial).\n"
            f"Optional VIRIAL_DEGREES from config.in: deg_a={_virial_deg_a}, deg_b={_virial_deg_b}.\n"
        )
else:
    if config.get('virial_fitting_degrees'):
        virial.DEGREE_SEARCH_REPORT_TEXT = (
            "Virial degree search: disabled (SUGGESTION_VIRIAL = no in config.in).\n"
            f"Using VIRIAL_DEGREES_COMBO (global fallback deg_a={_virial_deg_a}, deg_b={_virial_deg_b}; "
            "VIRIAL_DEGREES is not used when COMBO lines are present).\n"
        )
    else:
        virial.DEGREE_SEARCH_REPORT_TEXT = (
            "Virial degree search: disabled (SUGGESTION_VIRIAL = no in config.in).\n"
            f"Using VIRIAL_DEGREES: deg_a={_virial_deg_a}, deg_b={_virial_deg_b}.\n"
        )

# Isotherm plots: pure mode uses fitting curves/scatter; mixture mode plots components
if config['isotherm_type'] == 'mixture':
    IsothermFittingPlot.plot_mixture_isotherms(
        selection['fw'], selection['mol'], selection['temp'],
        mixture_data, colors,
        p_min=config['P_MIN'], p_max=config['P_MAX'],
        out_dir=None,
        pressure_scale=config.get('pressure_scale', 'both'),
        show_points=config.get('show_points', True),
        save_data=config.get('out_dir', False))
    MolFraction.plot_mol_fraction_vs_pressure(
        mixture_data, selection['fw'], selection['mol'], selection['temp'],
        colors, p_min=config['P_MIN'], p_max=config['P_MAX'],
        out_dir=None,
        scale=config.get('pressure_scale', 'both'),
        show_points=config.get('show_points', True),
        save_data=config.get('out_dir', False))

    if Input.plot_flags['Mixture_CC'] and mixture_data:
        cc.plot_mixture_heat_cc(
            mixture_data, selection['fw'], selection['mol'], selection['temp'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            min_temps=CC_MIN_TEMPS,
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            out_dir=None,
            save_data=config.get('out_dir', False),
        )

    if Input.plot_flags['Mixture_HOA_Pure_CC'] and mixture_data:
        # raspa_pure_mixture_components: DataSelection grid built from fitting
        # synthesis for each gas component — identical DataSelection logic to the
        # pure-component CC HOA plot.  Falls back to raspa_for_cc when unavailable.
        _rpmc_cc = raspa_pure_mixture_components or raspa_for_cc
        cc.plot_mixture_heat_hoa_pure_cc(
            mixture_data=mixture_data,
            fits_pure=_fits_from_disk,
            RASPA_data_pure=_rpmc_cc,
            selected_frameworks=selection['fw'],
            mixture_name=_mixture_name,
            selected_temperatures=selection['temp'],
            selected_fit_types=selection['fit_types'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            min_temps=CC_MIN_TEMPS,
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            combo_colors=colors,
            out_dir=None,
            save_data=config.get('out_dir', False),
        )

    if Input.plot_flags['Mixture_HOA_Pure_Virial'] and mixture_data:
        # Same DataSelection grid as CC: both plots share the loading range.
        _rpmc_v = raspa_pure_mixture_components or raspa_for_cc
        virial.plot_mixture_heat_hoa_pure_virial(
            mixture_data=mixture_data,
            RASPA_data_pure=_rpmc_v,
            selected_frameworks=selection['fw'],
            mixture_name=_mixture_name,
            selected_temperatures=selection['temp'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            deg_a=_virial_deg_a, deg_b=_virial_deg_b,
            degrees_per_combo=_virial_degrees_dict or {},
            min_points=VIRIAL_MIN_POINTS,
            n_loadings=config['n_loadings'],
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            combo_colors=colors,
            out_dir=None,
        )
        # Same layout as pure-mode ``Heat_of_Adsorption/virial_control/<fw>_<mol>/``:
        # ln(P) vs loading (log + linear) and virial_coefficients.txt per gas component.
        _rpmc_v_ctrl = raspa_pure_mixture_components or raspa_for_cc
        if _rpmc_v_ctrl:
            _comps_vctrl = sorted({d['molecule'] for d in mixture_data})
            _fw_vc = "-".join([str(x).replace(" ", "_") for x in selection['fw']]) if selection['fw'] else "all"
            _mol_vc = "-".join([str(x).replace(" ", "_") for x in selection['mol']]) if selection['mol'] else "all"
            _temp_vc = "-".join([str(x).replace(" ", "_") for x in selection['temp']]) if selection['temp'] else "all"
            _virial_mix_ctrl_root = (
                repo_root / "Output" / f"{_fw_vc}_{_mol_vc}_{_temp_vc}"
                / "Heat_of_Adsorption" / "virial_control"
            )
            _virial_mix_ctrl_root.mkdir(parents=True, exist_ok=True)
            for _fw_v in selection['fw']:
                for _comp_v in _comps_vctrl:
                    try:
                        _combo_v = _virial_mix_ctrl_root / (
                            f"{str(_fw_v).replace(' ', '_')}_{str(_comp_v).replace(' ', '_')}"
                        )
                        _combo_v.mkdir(parents=True, exist_ok=True)
                        _da_v, _db_v = _virial_degrees_dict.get(
                            (_fw_v, _comp_v), (_virial_deg_a, _virial_deg_b)
                        )
                        control.plot_lnP_vs_loading_from_virial(
                            RASPA_data=_rpmc_v_ctrl,
                            framework=_fw_v,
                            molecule=_comp_v,
                            temperatures=selection['temp'],
                            deg_a=_da_v,
                            deg_b=_db_v,
                            min_points=VIRIAL_MIN_POINTS,
                            n_points=config['n_loadings'],
                            selected_frameworks=[_fw_v],
                            selected_molecules=[_comp_v],
                            selected_temperatures=selection['temp'],
                            out_dir=str(_combo_v),
                            show=False,
                            p_min=config['P_MIN'],
                        )
                    except Exception as _e_mix_vc:
                        print(f"Mixture virial_control: skipping {_fw_v} {_comp_v}: {_e_mix_vc}")
else:
    IsothermFittingPlot.plot_isotherm_fitting(
        selection['fw'], selection['mol'], selection['temp'], selection['fit_types'],
        fits, data_points, colors, x,
        p_min=None, p_max=config['P_MAX'], plot_RASPA=True,
        num_of_isotherm=selection.get('num_of_isotherm'),
        pressure_scale=config.get('pressure_scale', 'both'),
        save_data=config.get('out_dir', False))


# Single virial Qst calculation and pure CC plot — only for pure component mode
virial_results = None
if config['isotherm_type'] != 'mixture':
    if Input.plot_flags['Virial']:
        try:
            _virial_coeff_report = None
            if selection['fw'] and selection['mol']:
                _vr_fw = "-".join([str(x).replace(" ", "_") for x in selection['fw']]) if selection['fw'] else "all"
                _vr_mol = "-".join([str(x).replace(" ", "_") for x in selection['mol']]) if selection['mol'] else "all"
                _vr_temp = "-".join([str(x).replace(" ", "_") for x in selection['temp']]) if selection['temp'] else "all"
                _vr_combo = f"{str(selection['fw'][0]).replace(' ', '_')}_{str(selection['mol'][0]).replace(' ', '_')}"
                _virial_coeff_report = (
                    repo_root / "Output" / f"{_vr_fw}_{_vr_mol}_{_vr_temp}"
                    / "Heat_of_Adsorption" / "virial_control" / _vr_combo / "virial_coefficients.txt"
                )
            _da_qst, _db_qst = _virial_degrees_dict.get(
                (selection['fw'][0], selection['mol'][0]),
                (_virial_deg_a, _virial_deg_b),
            )
            virial_results = virial.compute_Qst_from_coef_slopes(
                RASPA_data=data_points_virial,
                framework=selection['fw'][0],
                molecule=selection['mol'][0],
                deg_a=_da_qst,
                deg_b=_db_qst,
                min_points=VIRIAL_MIN_POINTS,
                n_points=config['n_loadings'],
                R=virial.R,
                temperatures=selection['temp'],
                p_min=config['P_MIN'],
                verbose=False,
                coefficient_report_path=str(_virial_coeff_report) if _virial_coeff_report is not None else None,
            )
        except Exception as e:
            print(f"Warning: Virial calculation failed: {e}")
            virial_results = None

    if Input.plot_flags['Clausius_Clapeyron']:
        hoa_scatter_data = (
            init.load_hoa_data(str(_hoa_points_file))
            if (config['data_source'] == 'fitting' and _hoa_points_file and config.get('show_points', True))
            else None
        )
        cc.plot_clausius_clapeyron(
            selection['fw'], selection['mol'], selection['temp'], selection['fit_types'], fits,
            RASPA_data=raspa_for_cc, x_fit=x_calc, loadings=None,
            n_loadings=config['n_loadings'], p_min=config['P_MIN'], p_max=config['P_MAX'],
            show_RASPA=config['show_points'], show_calc_points=config['show_points'], out_dir=None,
            smooth=True, smoothing_sigma=CC_SMOOTHING_SIGMA,
            use_direct_interpolation=(config['data_source'] == 'points'),
            show_original=False, plot_smoothed=True, plot_suffix='',
            num_of_isotherm=selection.get('num_of_isotherm'), method_linestyles=hoa_method_linestyles,
            hoa_scatter_data=hoa_scatter_data,
            save_data=config.get('out_dir', False))
    # Pure HOA from file: delegate plotting + cache-building to Clausius_Clapeyron
    if Input.plot_flags.get('HOA_From_File'):
        _hoa_path = _resolve(config.get('data_file_hoa', 'none'))
        if _hoa_path and _hoa_path.is_file():
            base_temp = float(config.get('T_ads', selection['temp'][0] if selection['temp'] else 298))
            curves = cc.plot_pure_hoa_from_file(
                str(_hoa_path), selection['fw'], selection['mol'], selection['temp'],
                colors, base_temp, hoa_method_linestyles
            )
            if curves:
                globals()['qst_cache_file'] = {k: v for k, v in curves.items()}

_sd_method = str(Input.plot_flags.get('Storage_Density_Method', '') or '').strip().lower()
_run_cc_sd = _sd_method in ('cc', 'both')
_run_virial_sd = _sd_method in ('virial', 'both')
_run_data_file_sd = _sd_method in ('data_file', 'both')

_has_sd_request = bool(_sd_method) and _sd_method not in ('none', 'no', '')

_sd_dim_str = str(Input.plot_flags.get('Storage_Density_Dim', '') or '').strip().lower()
_run_sd_2d = False
_run_sd_3d = False
_run_sd_mix_components = False
if _has_sd_request:
    # ``both`` / ``all`` = 2D + 3D only. ``per_component`` (or ``components``) = mixture
    # Per-component mixture SD only when those tokens are present (not implied by ``all``).
    _dim_src = _sd_dim_str or 'both'
    _dim_tokens = {t for t in _dim_src.replace(',', ' ').split() if t}
    _run_sd_2d = bool(_dim_tokens & {'2d', 'both', 'all'})
    _run_sd_3d = bool(_dim_tokens & {'3d', 'both', 'all'})
    _run_sd_mix_components = bool(_dim_tokens & {'per_component', 'components'})

_sd_methods_to_run = []
if _run_cc_sd:
    _sd_methods_to_run.append('cc')
if _run_virial_sd:
    _sd_methods_to_run.append('virial')

# virial_results already computed above when CC or Virial is enabled
virial_coeffs_a = None
virial_coeffs_b = None

if config['isotherm_type'] != 'mixture' and Input.plot_flags['Virial']:
    # Plot Virial Qst using only the intersection loading range.
    virial.plot_Qst(RASPA_data=data_points_virial, framework=None, molecule=None, temperatures=selection['temp'],
        deg_a=_virial_deg_a, deg_b=_virial_deg_b,
        min_points=VIRIAL_MIN_POINTS, results=virial_results, n_points=config['n_loadings'],
        R=virial.R, save_fw_list=selection['fw'], save_mol_list=selection['mol'], save_temp_list=selection['temp'], p_min=config['P_MIN'],
        method_linestyles=hoa_method_linestyles, show_all_loadings=False,
        degrees_per_combo=_virial_degrees_dict or None,
        save_data=config.get('out_dir', False))

_sd_shared_kw = dict(
    p_min=config['P_MIN_SD'], p_max=config['P_MAX'], P_des_max=config['P_des_max'],
    num_of_isotherm=selection.get('num_of_isotherm'),
    folder_temperatures=selection['temp'])

_sd_per_method = {
    'cc':     (raspa_for_cc, x_calc, dict(
                   smooth=True,
                   use_direct_interpolation=(config['data_source'] == 'points'),
                   smoothing_sigma=CC_SMOOTHING_SIGMA)),
    'virial': (data_points_virial,    x,    dict(
                   deg_a=_virial_deg_a, deg_b=_virial_deg_b,
                   min_points=VIRIAL_MIN_POINTS,
                   virial_p_min=config['P_MIN'],
                   coeffs_a_override=virial_coeffs_a, coeffs_b_override=virial_coeffs_b,
                   degrees_per_combo=_virial_degrees_dict or None)),
}

# Storage-density isotherms: use fitting file, or synthetic P–q ``interp`` when points-only.
_sd_fits_for_sd = fits
_sd_fit_types_for_sd = selection['fit_types']
_sd_shared_kw_sd = dict(_sd_shared_kw)
if (
    config['isotherm_type'] != 'mixture'
    and config['data_source'] == 'points'
    and not fits
    and _sd_methods_to_run
):
    _t_des_list = config['T_des'] if isinstance(config['T_des'], (list, tuple)) else [config['T_des']]
    _all_t_sd = sorted(
        {float(config['T_ads'])}
        | {float(t) for t in (selection.get('temp') or [])}
        | {float(t) for t in _t_des_list}
    )
    _raspa_sd_src = raspa_for_cc if raspa_for_cc else data_points_calc
    _sd_fits_for_sd = sd.make_pure_interp_fittings_from_raspa(
        _raspa_sd_src, selection['fw'], selection['mol'], _all_t_sd,
    )
    _sd_fit_types_for_sd = ['interp']
    _sd_shared_kw_sd['num_of_isotherm'] = None

if config['isotherm_type'] != 'mixture':
    for _m in _sd_methods_to_run:
        _raspa, _x, _extra = _sd_per_method[_m]
        _kw = {**_sd_shared_kw_sd, **_extra}

        if _run_sd_2d:
            sd.plot_storage_density(
                _m, selection['fw'], selection['mol'], config['T_des'], _sd_fit_types_for_sd,
                _sd_fits_for_sd, config['P_ads'], _x, colors, _raspa,
                save_data=config.get('out_dir', False),
                scale=config.get('pressure_scale', 'both'), **_kw)

            sd.plot_storage_density_fixed_ads(
                _m, selection['fw'], selection['mol'], config['T_ads'], config['P_ads'],
                config['T_des'], _sd_fit_types_for_sd, _sd_fits_for_sd, _x, colors, _raspa,
                save_data=config.get('out_dir', False),
                scale=config.get('pressure_scale', 'both'), **_kw)

            sd.plot_storage_density_temperature_series(
                _m, selection['fw'], selection['mol'], config['T_ads'], config['P_ads'], config['T_des'],
                _sd_fit_types_for_sd, _sd_fits_for_sd, _x, colors, _raspa,
                save_data=config.get('out_dir', False),
                scale=config.get('pressure_scale', 'both'), **_kw)

        if _run_sd_3d:
            sd.plot_storage_density_3d(
                _m, selection['fw'], selection['mol'], config['T_des'], _sd_fit_types_for_sd,
                _sd_fits_for_sd, config['P_ads'], _x, colors, _raspa, **_kw)

            sd.plot_storage_density_fixed_ads_3d(
                _m, selection['fw'], selection['mol'], config['T_ads'], config['P_ads'],
                config['T_des'], _sd_fit_types_for_sd, _sd_fits_for_sd, _x, colors, _raspa, **_kw)

            sd.plot_storage_density_temperature_series_3d(
                _m, selection['fw'], selection['mol'], config['T_ads'], config['T_des'],
                _sd_fit_types_for_sd, _sd_fits_for_sd, _x, colors, _raspa, **_kw)

            sd.plot_storage_density_3d_Tads_Tdes(
                _m, selection['fw'], selection['mol'], selection['temp'], config['T_des'],
                _sd_fit_types_for_sd, _sd_fits_for_sd,
                config['P_ads_TT'], config['P_des_TT'], _x, colors, _raspa,
                save_data=config.get('out_dir', False), **_kw)

    # Optional pure-component storage density using HoA-from-file (method='data_file').
    # This reuses the CC isotherm grid but injects Qst from an external file via qst_cache_file.
    if Input.plot_flags.get('HOA_From_File') and 'cc' in _sd_per_method and _has_sd_request:
        qst_cache_file = globals().get('qst_cache_file')
        if qst_cache_file:
            _hoa_raspa, _hoa_x, _hoa_extra = _sd_per_method['cc']
            _hoa_extra = dict(_hoa_extra)
            _hoa_extra['smooth'] = False
            _hoa_kw = {**_sd_shared_kw_sd, **_hoa_extra, 'qst_cache': qst_cache_file}

            if _run_sd_2d:
                sd.plot_storage_density(
                    'data_file', selection['fw'], selection['mol'], config['T_des'], _sd_fit_types_for_sd,
                    _sd_fits_for_sd, config['P_ads'], _hoa_x, colors, _hoa_raspa,
                    save_data=config.get('out_dir', False),
                    scale=config.get('pressure_scale', 'both'), **_hoa_kw)

                sd.plot_storage_density_fixed_ads(
                    'data_file', selection['fw'], selection['mol'], config['T_ads'], config['P_ads'],
                    config['T_des'], _sd_fit_types_for_sd, _sd_fits_for_sd, _hoa_x, colors, _hoa_raspa,
                    save_data=config.get('out_dir', False),
                    scale=config.get('pressure_scale', 'both'), **_hoa_kw)

                sd.plot_storage_density_temperature_series(
                    'data_file', selection['fw'], selection['mol'], config['T_ads'], config['P_ads'], config['T_des'],
                    _sd_fit_types_for_sd, _sd_fits_for_sd, _hoa_x, colors, _hoa_raspa,
                    save_data=config.get('out_dir', False),
                    scale=config.get('pressure_scale', 'both'), **_hoa_kw)

            if _run_sd_3d:
                sd.plot_storage_density_3d(
                    'data_file', selection['fw'], selection['mol'], config['T_des'], _sd_fit_types_for_sd,
                    _sd_fits_for_sd, config['P_ads'], _hoa_x, colors, _hoa_raspa, **_hoa_kw)

                sd.plot_storage_density_fixed_ads_3d(
                    'data_file', selection['fw'], selection['mol'], config['T_ads'], config['P_ads'],
                    config['T_des'], _sd_fit_types_for_sd, _sd_fits_for_sd, _hoa_x, colors, _hoa_raspa, **_hoa_kw)

                sd.plot_storage_density_temperature_series_3d(
                    'data_file', selection['fw'], selection['mol'], config['T_ads'], config['T_des'],
                    _sd_fit_types_for_sd, _sd_fits_for_sd, _hoa_x, colors, _hoa_raspa, **_hoa_kw)

                sd.plot_storage_density_3d_Tads_Tdes(
                    'data_file', selection['fw'], selection['mol'], selection['temp'], config['T_des'],
                    _sd_fit_types_for_sd, _sd_fits_for_sd,
                    config['P_ads_TT'], config['P_des_TT'], _hoa_x, colors, _hoa_raspa,
                    save_data=config.get('out_dir', False), **_hoa_kw)

def _run_mixture_storage_density_suite(mix_name, mix_fits, qst_cache, method_label='cc'):
    """Run the full set of mixture storage-density plots for a given Qst cache.
    method_label is used in filenames (e.g. 'cc', 'hoa_pure_cc', 'hoa_pure_virial')."""
    _sd_kw = dict(
        p_min=config['P_MIN_SD'], p_max=config['P_MAX'], P_des_max=config['P_des_max'],
        folder_temperatures=selection['temp'],
        qst_cache=qst_cache)

    if _run_sd_2d:
        sd.plot_storage_density(
            method_label, selection['fw'], [mix_name], config['T_des'], ['interp'],
            mix_fits, config['P_ads'], x, colors, [],
            save_data=config.get('out_dir', False),
            scale=config.get('pressure_scale', 'both'), **_sd_kw)

        sd.plot_storage_density_fixed_ads(
            method_label, selection['fw'], [mix_name], config['T_ads'], config['P_ads'],
            config['T_des'], ['interp'], mix_fits, x, colors, [],
            save_data=config.get('out_dir', False),
            scale=config.get('pressure_scale', 'both'), **_sd_kw)

        sd.plot_storage_density_temperature_series(
            method_label, selection['fw'], [mix_name],
            config['T_ads'],                  # T_ads
            config['P_ads'],                  # P_ads (for Tseries equal to P_des)
            config['T_des'],                  # desorption_temperatures
            ['interp'], mix_fits, x, colors, [],
            save_data=config.get('out_dir', False),
            scale=config.get('pressure_scale', 'both'), **_sd_kw)

    if _run_sd_3d:
        sd.plot_storage_density_3d(
            method_label, selection['fw'], [mix_name], config['T_des'], ['interp'],
            mix_fits, config['P_ads'], x, colors, [], **_sd_kw)

        sd.plot_storage_density_fixed_ads_3d(
            method_label, selection['fw'], [mix_name], config['T_ads'], config['P_ads'],
            config['T_des'], ['interp'], mix_fits, x, colors, [], **_sd_kw)

        sd.plot_storage_density_temperature_series_3d(
            method_label, selection['fw'], [mix_name], config['T_ads'], config['T_des'],
            ['interp'], mix_fits, x, colors, [], **_sd_kw)

        sd.plot_storage_density_3d_Tads_Tdes(
            method_label, selection['fw'], [mix_name], selection['temp'], config['T_des'],
            ['interp'], mix_fits,
            config['P_ads_TT'], config['P_des_TT'], x, colors, [],
            save_data=config.get('out_dir', False), **_sd_kw)

if (
    config['isotherm_type'] == 'mixture'
    and mixture_data
    and Input.plot_flags['Mixture_CC']
    and _has_sd_request
    and _run_cc_sd
):
    _mix_name   = selection['mol'][0]
    _all_sd_T   = sorted({float(config['T_ads'])} |
                         {float(t) for t in config['T_des']} |
                         {float(t) for t in selection['temp']})

    if _run_sd_2d or _run_sd_3d:
        # Synthetic 'interp' fittings from RASPA total-loading data
        _mix_fits   = sd.make_mixture_fittings(
            mixture_data, selection['fw'], _mix_name, _all_sd_T)
        # Pre-computed mixture Qst (total) to inject via qst_cache
        _mix_qst    = sd.make_mixture_qst_cache(
            mixture_data, selection['fw'], _mix_name, selection['temp'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            min_temps=CC_MIN_TEMPS,
            smoothing_sigma=CC_SMOOTHING_SIGMA,
        )
        _run_mixture_storage_density_suite(_mix_name, _mix_fits, _mix_qst, method_label='cc')

    if _run_sd_mix_components and config.get('data_source') == 'fitting':
        sd.plot_storage_density_mix_components_cc(
            selected_frameworks=selection['fw'],
            mixture_name=_mix_name,
            qst_temperatures=selection['temp'],
            T_ads=config['T_ads'],
            T_des_list=config['T_des'],
            P_ads=config['P_ads'],
            x_fit=x,
            mixture_data=mixture_data,
            selected_fit_types=selection['fit_types'],
            fits_pure=_fits_from_disk,
            RASPA_data_pure=raspa_for_cc,
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            min_temps=CC_MIN_TEMPS,
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            combo_colors=colors,
            out_dir=None,
            save_data=config.get('out_dir', False),
            scale=config.get('pressure_scale', 'both'),
        )

if (
    config['isotherm_type'] == 'mixture'
    and mixture_data
    and Input.plot_flags['Mixture_HOA_Pure_CC']
    and _has_sd_request
    and _run_cc_sd
):
    _mix_name_hoa   = selection['mol'][0]
    _all_sd_T_hoa   = sorted({float(config['T_ads'])} |
                             {float(t) for t in config['T_des']} |
                             {float(t) for t in selection['temp']})

    if _run_sd_2d or _run_sd_3d:
        # Synthetic 'interp' fittings from RASPA total-loading data (reuse same construction)
        _mix_fits_hoa   = sd.make_mixture_fittings(
            mixture_data, selection['fw'], _mix_name_hoa, _all_sd_T_hoa)
        # Pre-computed mixture Qst from hoa_pure_cc (weighted pure CC) at T_ads
        _mix_qst_hoa_cc = sd.make_mixture_qst_cache_hoa_pure_cc(
            mixture_data=mixture_data,
            frameworks=selection['fw'],
            mixture_name=_mix_name_hoa,
            pure_temperatures=selection['temp'],
            mix_temperature=config['T_ads'],
            fits_pure=_fits_from_disk,
            RASPA_data_pure=raspa_pure_mixture_components or raspa_for_cc,
            selected_fit_types=selection['fit_types'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            min_temps=CC_MIN_TEMPS,
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            use_direct_interpolation=False,
        )
        _run_mixture_storage_density_suite(_mix_name_hoa, _mix_fits_hoa, _mix_qst_hoa_cc, method_label='hoa_pure_cc')

if config['isotherm_type'] == 'mixture' and mixture_data and Input.plot_flags['Mixture_HOA_Pure_Virial']:
    _mix_name_hoa_v   = selection['mol'][0]
    _all_sd_T_hoa_v   = sorted({float(config['T_ads'])} |
                               {float(t) for t in config['T_des']} |
                               {float(t) for t in selection['temp']})
    # Shared DataSelection grid for Virial HOA — same as CC (raspa_pure_mixture_components).
    _rpmc_virial_sd = raspa_pure_mixture_components or raspa_for_cc

    if _has_sd_request and _run_virial_sd and (_run_sd_2d or _run_sd_3d):
        _mix_fits_hoa_v   = sd.make_mixture_fittings(
            mixture_data, selection['fw'], _mix_name_hoa_v, _all_sd_T_hoa_v)

        _mix_qst_hoa_virial = sd.make_mixture_qst_cache_hoa_pure_virial(
            mixture_data=mixture_data,
            frameworks=selection['fw'],
            mixture_name=_mix_name_hoa_v,
            temperatures=selection['temp'],
            RASPA_data_pure=_rpmc_virial_sd,
            deg_a=_virial_deg_a, deg_b=_virial_deg_b,
            degrees_per_combo=_virial_degrees_dict or {},
            min_points=VIRIAL_MIN_POINTS,
            n_loadings=config['n_loadings'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            smoothing_sigma=CC_SMOOTHING_SIGMA,
        )

        _run_mixture_storage_density_suite(_mix_name_hoa_v, _mix_fits_hoa_v, _mix_qst_hoa_virial, method_label='hoa_pure_virial')

    virial.plot_mixture_heat_hoa_pure_virial(
        mixture_data=mixture_data,
        RASPA_data_pure=_rpmc_virial_sd,
        selected_frameworks=selection['fw'],
        mixture_name=_mix_name_hoa_v,
        selected_temperatures=selection['temp'],
        p_min=config['P_MIN'], p_max=config['P_MAX'],
        deg_a=_virial_deg_a, deg_b=_virial_deg_b,
        degrees_per_combo=_virial_degrees_dict or {},
        min_points=VIRIAL_MIN_POINTS,
        n_loadings=config['n_loadings'],
        smoothing_sigma=CC_SMOOTHING_SIGMA,
        combo_colors=colors,
        out_dir=None,
        save_data=config.get('out_dir', False),
    )

if config['isotherm_type'] == 'mixture' and mixture_data and Input.plot_flags.get('Mixture_HOA_Pure_File'):
    _mix_name_hoa_file = selection['mol'][0]
    _all_sd_T_hoa_file = sorted({float(config['T_ads'])} |
                                {float(t) for t in config['T_des']} |
                                {float(t) for t in selection['temp']})
    _mix_fits_hoa_file = sd.make_mixture_fittings(
        mixture_data, selection['fw'], _mix_name_hoa_file, _all_sd_T_hoa_file)

    _hoa_path_mix = _resolve(config.get('data_file_hoa', 'none'))
    hoa_curves_pure = {}
    if _hoa_path_mix and _hoa_path_mix.is_file():
            try:
                hoa_rows_mix = init.load_hoa_data(str(_hoa_path_mix))
            except Exception as e:
                print(f"Warning: Failed to load HOA data from {_hoa_path_mix}: {e}")
                hoa_rows_mix = []

            # Components actually present in the selected mixture data
            comps_needed = sorted({d['molecule'] for d in mixture_data
                                   if d.get('framework') in selection['fw']
                                   and str(d.get('mixture_pure', '')).lower() == str(_mix_name_hoa_file).lower()})
            hoa_curves_pure = init.build_hoa_curves(hoa_rows_mix, frameworks=selection['fw'], molecules=comps_needed)

    _mix_qst_hoa_file = sd.make_mixture_qst_cache_hoa_pure_file(
        mixture_data=mixture_data,
        frameworks=selection['fw'],
        mixture_name=_mix_name_hoa_file,
        mix_temperature=config['T_ads'],
        hoa_pure_curves=hoa_curves_pure,
        p_min=config['P_MIN'], p_max=config['P_MAX'],
        n_loadings=config['n_loadings'],
        smoothing_sigma=CC_SMOOTHING_SIGMA,
    )

    if (
        _mix_qst_hoa_file
        and _has_sd_request
        and _run_data_file_sd
        and (_run_sd_2d or _run_sd_3d)
    ):
        _run_mixture_storage_density_suite(_mix_name_hoa_file, _mix_fits_hoa_file, _mix_qst_hoa_file, method_label='data_file')
    if _mix_qst_hoa_file:
        cc.plot_mixture_heat_hoa_pure_file(
            mixture_data=mixture_data,
            hoa_pure_curves=hoa_curves_pure,
            selected_frameworks=selection['fw'],
            mixture_name=_mix_name_hoa_file,
            selected_temperatures=selection['temp'],
            p_min=config['P_MIN'], p_max=config['P_MAX'],
            n_loadings=config['n_loadings'],
            smoothing_sigma=CC_SMOOTHING_SIGMA,
            combo_colors=colors,
            out_dir=None,
            save_data=config.get('out_dir', False),
        )

if config['isotherm_type'] != 'mixture' and Input.plot_flags['Virial'] and Input.plot_flags['Clausius_Clapeyron']:
    qst_cache_file = globals().get('qst_cache_file')
    control.plot_clausius_clapeyron_with_virial(
        selection['fw'], selection['mol'], selection['temp'], selection['fit_types'],
        fits, RASPA_data=data_points_virial, x_fit=x_calc,
        loadings=None, n_loadings=config['n_loadings'], p_min=config['P_MIN'], p_max=config['P_MAX'],
        r2_min=R2_MIN, out_dir=None,
        deg_a=_virial_deg_a, deg_b=_virial_deg_b, virial_plot=virial_results,
        degrees_per_combo=_virial_degrees_dict or None,
        smooth=True, use_direct_interpolation=(config['data_source'] == 'points'),
        smoothing_sigma=CC_SMOOTHING_SIGMA, method_linestyles=hoa_method_linestyles,
        show_markers=False,
        qst_cache_file=qst_cache_file)

# ln(P) vs loading + virial_coefficients.txt — only for Virial HOA (virial/both), not cc- or file-only runs.
if (config['isotherm_type'] != 'mixture'
        and Input.plot_flags['Virial']
        and selection['fw'] and selection['mol']):
    _fw_part = "-".join([str(x).replace(" ", "_") for x in selection['fw']]) if selection['fw'] else "all"
    _mol_part = "-".join([str(x).replace(" ", "_") for x in selection['mol']]) if selection['mol'] else "all"
    _temp_part = "-".join([str(x).replace(" ", "_") for x in selection['temp']]) if selection['temp'] else "all"
    _virial_control_root = repo_root / "Output" / f"{_fw_part}_{_mol_part}_{_temp_part}" / "Heat_of_Adsorption" / "virial_control"
    _virial_control_root.mkdir(parents=True, exist_ok=True)
    for _fw in selection['fw']:
        for _mol in selection['mol']:
            try:
                _combo_dir = _virial_control_root / f"{str(_fw).replace(' ', '_')}_{str(_mol).replace(' ', '_')}"
                _combo_dir.mkdir(parents=True, exist_ok=True)
                _da_ln, _db_ln = _virial_degrees_dict.get((_fw, _mol), (_virial_deg_a, _virial_deg_b))
                control.plot_lnP_vs_loading_from_virial(
                    RASPA_data=data_points_virial, framework=_fw, molecule=_mol,
                    temperatures=selection['temp'],
                    deg_a=_da_ln, deg_b=_db_ln,
                    min_points=VIRIAL_MIN_POINTS, n_points=config['n_loadings'],
                    selected_frameworks=[_fw],
                    selected_molecules=[_mol],
                    selected_temperatures=selection['temp'], out_dir=str(_combo_dir), show=False,
                    p_min=config['P_MIN'])
            except Exception as e:
                print(f"Virial lnP-vs-loading: skipping {_fw},{_mol}: {e}")



