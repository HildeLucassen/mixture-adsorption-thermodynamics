from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from math import isclose
import hashlib

from pandas._libs.tslibs.offsets import LastWeekOfMonth
import Initialize as init
import math

R = 8.31446261815324  # universal gas constant [J/mol/K]

# Unified plotting style settings
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
UNIFIED_FIGSIZE = (9, 8)
LINEWIDTH = 6
AXIS_LW = 3
AXIS_LEFT   = 0.19   # space for y-label + ticks (large AXIS_LABEL_FONTSIZE)
AXIS_BOTTOM = 0.15   # space for x-label + ticks
AXIS_RIGHT  = 0.97   # almost no content here
AXIS_TOP    = 0.97   # small space for title
AXIS_LEGEND_SIZE = 26
TICK_SIZE = 26
TICK_SIZE_X = 26
# Tick marks on axis spines: only major ticks are drawn (where labels are); minor length 0.
AXIS_TICK_WIDTH = 2
AXIS_TICK_LENGTH_MAJOR = 10
ALPHA = 1
ALPHA_GRID = 0.5
MARKER_SIZE = 14
AXIS_S_SIZE = MARKER_SIZE*MARKER_SIZE
AXIS_LABEL_FONTSIZE = 34  # Unified font size for all axis labels (x and y)
# tick_params(pad=): distance tick marks → tick labels [points]
AXIS_TICK_PAD = 8
# Distance tick labels → axis title text [points] (see also set_label_padding)
AXIS_LABELPAD_X = 8
AXIS_LABELPAD_Y = 14
# When auto-expanding margins (see ``apply_unified_axes_layout`` tightbbox pass): require at
# least this inset from the figure edge [fraction], up to AXIS_MARGIN_MAX_*.
AXIS_MARGIN_AUTOPAD_FRAC = 0.028
AXIS_MARGIN_MAX_BOTTOM = 0.36
AXIS_MARGIN_MAX_LEFT = 0.40
# Lower bound for ``subplots_adjust(..., right=…)`` when narrowing for overflowing x ticks.
AXIS_MARGIN_MIN_SUBPLOT_RIGHT = 0.82
# Log x: modest cap (tightbbox under-reports math-text superscripts on the last tick).
AXIS_RIGHT_LOG_X_CAP = 0.945

# Storage-density 3D layout / save tuning (colormap + color limits stay in design.in via Input).
STORAGE_DENSITY_3D_DEFAULTS = {
    'storage_density_3d_figwidth_pct_extra': 22.0,
    'storage_density_3d_subplot_left': 0.0,
    'storage_density_3d_subplot_bottom': 0.065,
    'storage_density_3d_subplot_top': 0.996,
    'storage_density_3d_subplot_right': 0.91,
    'storage_density_3d_axes_dist': 8.75,
    'storage_density_3d_box_scale': 0.84,
    'storage_density_3d_box_halign': 0.0,
    'storage_density_3d_box_valign': 0.5,
    'storage_density_3d_labelpad_x': 30.0,
    'storage_density_3d_labelpad_y': 34.0,
    'storage_density_3d_labelpad_z': 32.0,
    'storage_density_3d_colorbar_gap': 0.022,
    'storage_density_3d_cbar_label_labelpad': 12.0,
    'storage_density_3d_cbar_label_y': 0.58,
    'storage_density_3d_colorbar_width': 0.034,
    'storage_density_3d_colorbar_shrink': 0.6,
    'storage_density_3d_colorbar_ticks': None,
    'storage_density_3d_colorbar_nticks': 8,
    'storage_density_3d_save_bbox_pad': 0.10,
    'storage_density_3d_save_dpi': 250,
}

# Global marker map - set from Input.py via set_marker_map()
_MARKER_MAP = None
# Structure/framework line styles - set from Input.py via set_structure_linestyle_map()
LINESTYLE_STYLES = ['-', '--', '-.', ':']
_STRUCTURE_LINESTYLE_MAP = None
_AUTO_LINESTYLE_ASSIGNMENTS = {}
# Allow overriding the auto palettes from Input/design.in
_STRUCTURE_LINESTYLE_PALETTE = LINESTYLE_STYLES
_FIT_TYPE_LINESTYLE_PALETTE = LINESTYLE_STYLES
_MOLECULE_MARKER_PALETTE = MARKER_STYLES

# Molecule linestyles (optional)
_MOLECULE_LINESTYLE_MAP = None
_AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS = {}
_MOLECULE_LINESTYLE_PALETTE = LINESTYLE_STYLES
# RETMAP points: always use unfilled (open) markers for mmol/loading points

# Optional per-structure colors (set from Input via a small helper)
_STRUCTURE_COLOR_MAP = None
_AUTO_STRUCTURE_COLOR_ASSIGNMENTS = {}

# Optional per-molecule colors (adsorbate colors)
_MOLECULE_COLOR_MAP = None
_AUTO_MOLECULE_COLOR_ASSIGNMENTS = {}

def set_structure_linestyle_map(linestyle_map):
    """Set mapping from structure/framework names to line styles. Call from Main with Input.structure_linestyle_mapping."""
    global _STRUCTURE_LINESTYLE_MAP
    _STRUCTURE_LINESTYLE_MAP = linestyle_map if linestyle_map else {}


def set_structure_color_map(color_map):
    """Set mapping from structure/framework names to colors."""
    global _STRUCTURE_COLOR_MAP, _AUTO_STRUCTURE_COLOR_ASSIGNMENTS
    _STRUCTURE_COLOR_MAP = color_map if color_map else {}
    _AUTO_STRUCTURE_COLOR_ASSIGNMENTS = {}


def set_molecule_color_map(color_map):
    """Set mapping from molecule/adsorbate names to colors."""
    global _MOLECULE_COLOR_MAP, _AUTO_MOLECULE_COLOR_ASSIGNMENTS
    _MOLECULE_COLOR_MAP = color_map if color_map else {}
    _AUTO_MOLECULE_COLOR_ASSIGNMENTS = {}


def set_structure_linestyle_palette(palette):
    """Override the auto-assigned framework line style palette."""
    global _STRUCTURE_LINESTYLE_PALETTE, _AUTO_LINESTYLE_ASSIGNMENTS
    _STRUCTURE_LINESTYLE_PALETTE = list(palette) if palette else LINESTYLE_STYLES
    _AUTO_LINESTYLE_ASSIGNMENTS = {}

def set_fit_type_linestyle_map(linestyle_map):
    """Set mapping from fit type names to line styles. Call from Main with Input.fit_type_linestyle_mapping."""
    global _FIT_TYPE_LINESTYLE_MAP
    _FIT_TYPE_LINESTYLE_MAP = linestyle_map if linestyle_map else {}

_FIT_TYPE_LINESTYLE_MAP = None
_AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS = {}


def set_fit_type_linestyle_palette(palette):
    """Override the auto-assigned fit-type line style palette."""
    global _FIT_TYPE_LINESTYLE_PALETTE, _AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS
    _FIT_TYPE_LINESTYLE_PALETTE = list(palette) if palette else LINESTYLE_STYLES
    _AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS = {}

def get_linestyle_for_structure(structure_name):
    """Return line style for a structure/framework. Uses structure_linestyle_mapping if set, else assigns by order."""
    if _STRUCTURE_LINESTYLE_MAP and structure_name in _STRUCTURE_LINESTYLE_MAP:
        return _STRUCTURE_LINESTYLE_MAP[structure_name]
    if structure_name in _AUTO_LINESTYLE_ASSIGNMENTS:
        return _AUTO_LINESTYLE_ASSIGNMENTS[structure_name]
    idx = len(_AUTO_LINESTYLE_ASSIGNMENTS) % len(_STRUCTURE_LINESTYLE_PALETTE)
    ls = _STRUCTURE_LINESTYLE_PALETTE[idx]
    _AUTO_LINESTYLE_ASSIGNMENTS[structure_name] = ls
    return ls


def get_color_for_structure(structure_name):
    """
    Return color for a structure/framework.
    Uses explicit mapping if set, otherwise assigns from Matplotlib default cycle.
    """
    import matplotlib.pyplot as _plt
    if structure_name is None:
        return None
    if _STRUCTURE_COLOR_MAP and structure_name in _STRUCTURE_COLOR_MAP:
        return _STRUCTURE_COLOR_MAP[structure_name]
    if structure_name in _AUTO_STRUCTURE_COLOR_ASSIGNMENTS:
        return _AUTO_STRUCTURE_COLOR_ASSIGNMENTS[structure_name]
    try:
        palette = _plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
        idx = len(_AUTO_STRUCTURE_COLOR_ASSIGNMENTS) % len(palette)
        color = palette[idx]
    except Exception:
        color = 'C0'
    _AUTO_STRUCTURE_COLOR_ASSIGNMENTS[structure_name] = color
    return color


def get_color_for_molecule(molecule_name):
    """
    Return color for a molecule/adsorbate.
    Uses explicit mapping if set, otherwise assigns from Matplotlib default cycle.
    """
    import matplotlib.pyplot as _plt
    if molecule_name is None:
        return None
    if _MOLECULE_COLOR_MAP and molecule_name in _MOLECULE_COLOR_MAP:
        return _MOLECULE_COLOR_MAP[molecule_name]
    if molecule_name in _AUTO_MOLECULE_COLOR_ASSIGNMENTS:
        return _AUTO_MOLECULE_COLOR_ASSIGNMENTS[molecule_name]
    try:
        palette = _plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
        idx = len(_AUTO_MOLECULE_COLOR_ASSIGNMENTS) % len(palette)
        color = palette[idx]
    except Exception:
        color = 'C0'
    _AUTO_MOLECULE_COLOR_ASSIGNMENTS[molecule_name] = color
    return color


def resolve_series_style(
    fw,
    mol,
    T,
    *,
    vary_fw: bool,
    vary_mol: bool,
    vary_T: bool,
    plot_kind: str = "generic",
    combo_colors=None,
    method: str | None = None,
    method_linestyles=None,
):
    """
    Central style resolver.

    Rules (as requested):
    - If T>1: color encodes temperature.
    - If ADSORBATE>1 and T>1: marker encodes adsorbate.
    - If ADSORBATE>1 and T=1: color encodes adsorbate.
    - If ADSORBENT>1 and not already encoded: linestyle encodes adsorbent (only when T>1; if T=1, line is solid and color encodes adsorbent).
    - If ADSORBENT>1 & ADSORBATE>1 & T>1: Color=Temperature, Marker=Adsorbate, Linestyle=Adsorbent.
    - If ADSORBENT>1 & ADSORBATE=1 on isotherms: marker is the molecule's marker (same for every framework;
      avoids per-material auto-markers that change when frameworks are renamed).
    - Heat of Adsorption: linestyle encodes method, color encodes adsorbate.
    """
    kind = str(plot_kind or "generic").strip().lower()

    # --- color ---
    if kind in {"hoa", "heat_of_adsorption", "qst"}:
        color = get_color_for_molecule(mol)
    else:
        if vary_T and combo_colors is not None:
            # Expect combo_colors[(fw, mol, T)] to already be temperature-driven
            color = combo_colors.get((fw, mol, T))
        elif (not vary_T) and vary_mol:
            color = get_color_for_molecule(mol)
        elif (not vary_T) and (not vary_mol) and vary_fw:
            color = get_color_for_structure(fw)
        else:
            color = (combo_colors.get((fw, mol, T)) if combo_colors is not None else None) or get_color_for_molecule(mol) or get_color_for_structure(fw)

    # --- marker ---
    marker = ""
    if vary_mol:
        # Molecule marker from design only when comparing multiple adsorbates.
        marker = get_marker_for_molecule(mol) if mol is not None else ""
    elif (
        kind == "isotherm"
        and mol is not None
        and vary_fw
        and not vary_mol
    ):
        # One adsorbate, several frameworks: same marker shape everywhere (from molecule mapping),
        # not get_marker_for_material(fw) which depends on framework name / assignment order.
        marker = get_marker_for_molecule(mol)

    # --- linestyle ---
    if kind in {"hoa", "heat_of_adsorption", "qst"}:
        if method is not None and method_linestyles is not None:
            linestyle = method_linestyles.get(method, "-")
        else:
            linestyle = "-"
    else:
        # Multiple frameworks at a single temperature: use color only; keep curves solid.
        if vary_fw and vary_T:
            linestyle = get_linestyle_for_structure(fw)
        else:
            linestyle = "-"

    return {"color": color, "marker": marker, "linestyle": linestyle}


def build_series_label(
    fw,
    mol,
    T,
    *,
    vary_fw: bool,
    vary_mol: bool,
    vary_T: bool,
    prefix: str | None = None,
    suffix: str | None = None,
) -> str:
    """
    Build a concise legend label from framework, molecule, and temperature.

    Rules:
    - Only include framework if vary_fw is True.
    - Only include molecule if vary_mol is True.
    - Only include temperature if vary_T is True.
    - If nothing varies, fall back to molecule then framework.
    - Optional prefix/suffix for method/role text (e.g. 'Clausius Clapeyron').
    """
    parts: list[str] = []

    if vary_fw and fw is not None:
        parts.append(clean_material_name(fw))

    if vary_mol and mol is not None:
        parts.append(get_molecule_display_name(mol))

    if vary_T and T is not None:
        try:
            T_val = int(round(float(T)))
            parts.append(f"{T_val}K")
        except Exception:
            parts.append(str(T))

    # If nothing was added but we still need a label, fall back sensibly.
    if not parts:
        if mol is not None:
            parts.append(get_molecule_display_name(mol))
        elif fw is not None:
            parts.append(clean_material_name(fw))

    label = ", ".join(parts)

    if prefix:
        label = f"{prefix} {label}"
    if suffix:
        # Use an en dash separator for readability
        label = f"{label} – {suffix}"

    return label


def choose_isotherm_fitting_proxy_linestyle_mode(
    selected_frameworks, selected_fit_types, selected_temperatures=None
):
    """
    Decide which linestyle encoding to show in isotherm fitting legends.

    Returns one of: "structure", "fit_type", "none".
    At a single temperature with multiple frameworks, curves differ by color only; return "none"
    (see build_isotherm_fitting_proxy_legend for colored framework entries).
    """
    n_fw = len(selected_frameworks or [])
    n_ft = len(selected_fit_types or [])
    n_T = len(selected_temperatures or []) if selected_temperatures is not None else 0
    single_T = n_T <= 1
    if single_T and n_fw > 1 and n_ft <= 1:
        return "none"
    if n_fw > 1 and n_ft <= 1:
        return "structure"
    if n_ft > 1 and n_fw <= 1:
        return "fit_type"
    return "none"


def get_combo_color_temperature(combo_colors, fw, mol, temp):
    """
    Robustly resolve color for (fw, mol, temp) from combo_colors across
    mixed temp key types (raw/float/int/string).
    """
    if not combo_colors:
        return None

    keys = [(fw, mol, temp)]
    try:
        t_float = float(temp)
        keys.append((fw, mol, t_float))
        keys.append((fw, mol, int(round(t_float))))
    except Exception:
        pass
    keys.append((fw, mol, str(temp)))

    for key in keys:
        if key in combo_colors:
            return combo_colors.get(key)
    return None


def build_isotherm_fitting_proxy_legend(
    ax,
    selected_temperatures,
    selected_frameworks,
    selected_molecules,
    selected_fit_types,
    combo_colors,
    *,
    ref_fw_for_temp,
    ref_mol_for_temp,
    ref_temp_for_linestyle,
    vary_mol: bool = False,
    fontsize="14",
    loc="best",
):
    """
    Build one combined proxy legend for isotherm fitting plots:
    - Temperature entries (color) when multiple temperatures vary.
    - One temperature, multiple molecules: molecule entries (color + linestyle), and framework
      names in the label when several frameworks are plotted.
    - One temperature, multiple frameworks, single adsorbate: framework entries (color, solid line).
    - Otherwise: structure entries (black + linestyle) OR fit-type entries (linestyle) based on
      choose_isotherm_fitting_proxy_linestyle_mode() — but not when multiple temperatures are
      already listed (color encodes T; framework / material names are omitted from the legend).
    """
    proxy_handles = []
    proxy_labels = []

    vary_T = len(selected_temperatures or []) > 1
    mode = choose_isotherm_fitting_proxy_linestyle_mode(
        selected_frameworks, selected_fit_types, selected_temperatures
    )

    if vary_T and ref_fw_for_temp is not None and ref_mol_for_temp is not None:
        for temp in selected_temperatures or []:
            color_tmp = get_combo_color_temperature(
                combo_colors, ref_fw_for_temp, ref_mol_for_temp, temp
            ) or "gray"
            proxy_handles.append(Line2D([0], [0], color=color_tmp, lw=LINEWIDTH))
            try:
                temp_float = float(temp)
                if isclose(temp_float, round(temp_float), abs_tol=1e-9):
                    t_label = f"{int(round(temp_float))}K"
                else:
                    t_label = f"{temp}K"
            except Exception:
                t_label = f"{temp}K"
            proxy_labels.append(t_label)

    # Single temperature, multiple molecules: legend identifies each adsorbate (matches curve colors).
    if not vary_T and vary_mol and ref_temp_for_linestyle is not None:
        n_fw = len(selected_frameworks or [])
        for fw in selected_frameworks or []:
            for mol in selected_molecules or []:
                style_m = resolve_series_style(
                    fw,
                    mol,
                    ref_temp_for_linestyle,
                    vary_fw=(n_fw > 1),
                    vary_mol=True,
                    vary_T=False,
                    plot_kind="isotherm",
                    combo_colors=combo_colors,
                )
                c_m = style_m.get("color") or "gray"
                ls_m = style_m.get("linestyle") or "-"
                proxy_handles.append(Line2D([0], [0], color=c_m, lw=LINEWIDTH, linestyle=ls_m))
                proxy_labels.append(
                    build_series_label(
                        fw,
                        mol,
                        ref_temp_for_linestyle,
                        vary_fw=(n_fw > 1),
                        vary_mol=True,
                        vary_T=False,
                    )
                )

    # Single temperature, one adsorbate, multiple frameworks: legend = colored solid lines (no linestyle encoding).
    if (
        not vary_T
        and not vary_mol
        and len(selected_frameworks or []) > 1
        and len(selected_fit_types or []) <= 1
        and ref_mol_for_temp is not None
        and ref_temp_for_linestyle is not None
    ):
        for fw in selected_frameworks or []:
            style_fw = resolve_series_style(
                fw,
                ref_mol_for_temp,
                ref_temp_for_linestyle,
                vary_fw=True,
                vary_mol=False,
                vary_T=False,
                plot_kind="isotherm",
                combo_colors=combo_colors,
            )
            c_fw = style_fw.get("color") or "gray"
            proxy_handles.append(Line2D([0], [0], color=c_fw, lw=LINEWIDTH, linestyle="-"))
            proxy_labels.append(clean_material_name(fw))

    # Do not stack structure/fit-type proxies when the molecule-specific legend already applies,
    # or when multiple temperatures are shown (legend lists T only; frameworks stay linestyle on curves).
    if mode == "structure" and not (not vary_T and vary_mol) and not vary_T:
        if ref_temp_for_linestyle is not None and ref_mol_for_temp is not None:
            for fw in selected_frameworks or []:
                style_fw = resolve_series_style(
                    fw, ref_mol_for_temp, ref_temp_for_linestyle,
                    vary_fw=True, vary_mol=False, vary_T=False,
                    plot_kind="isotherm",
                    combo_colors=combo_colors,
                )
                ls_fw = style_fw.get("linestyle") or "-"
                proxy_handles.append(Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle=ls_fw))
                proxy_labels.append(clean_material_name(fw))
    elif mode == "fit_type" and not (not vary_T and vary_mol) and not vary_T:
        for ft_type in selected_fit_types or []:
            ls_ft = get_linestyle_for_fit_type(ft_type)
            proxy_handles.append(Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle=ls_ft))
            proxy_labels.append(str(ft_type).replace("_", " "))

    if proxy_handles:
        ax.legend(proxy_handles, proxy_labels, fontsize=fontsize, loc=loc)


def choose_hoa_proxy_linestyle_mode(methods_present):
    """
    Decide HoA linestyle encoding mode.

    Returns:
    - "method": when multiple HoA methods are present
    - "structure": otherwise
    """
    n_methods = len(methods_present or [])
    return "method" if n_methods > 1 else "structure"


def hoa_method_legend_display_name(method, overrides=None):
    """
    Legend label for an HoA method key (clausius_clapeyron, virial, hoa_file).
    CC and Virial are always those strings; only hoa_file is configurable (Input.config hoa_legend_file).
    """
    if overrides and method in overrides and overrides[method] is not None:
        s = str(overrides[method]).strip()
        if s:
            return s
    if method == 'clausius_clapeyron':
        return 'CC'
    if method == 'virial':
        return 'Virial'
    if method == 'hoa_file':
        try:
            import Input
            cfg = getattr(Input, 'config', None) or {}
            v = cfg.get('hoa_legend_file')
            if v is not None and str(v).strip() != '':
                return str(v).strip()
        except Exception:
            pass
        return 'Data'
    return str(method).replace('_', ' ')


def get_hoa_linestyle(fw, method, mode, method_linestyles=None):
    """
    Resolve HoA line style from selected legend encoding mode.
    """
    if mode == "method":
        if method_linestyles:
            return method_linestyles.get(method, "-")
        return "-"
    return get_linestyle_for_structure(fw) if fw is not None else "-"


def build_hoa_proxy_legend(
    ax,
    *,
    molecules_present,
    frameworks_present,
    methods_present,
    method_linestyles=None,
    fontsize="14",
    loc="best",
    method_legend_labels=None,
):
    """
    Build one combined HoA proxy legend:
    - colors encode molecules
    - linestyles encode structures (single-method) OR methods (multi-method)

    method_legend_labels: optional dict overriding any method key (rare).
    CC and Virial are fixed unless overridden here; hoa_file uses config HOA_LEGEND_FILE (default Data).
    """
    proxy_handles = []
    proxy_labels = []

    mol_seen = set()
    for mol in molecules_present or []:
        if mol in mol_seen:
            continue
        mol_seen.add(mol)
        color = get_color_for_molecule(mol) or "gray"
        proxy_handles.append(Line2D([0], [0], color=color, lw=LINEWIDTH))
        proxy_labels.append(get_molecule_display_name(mol))

    mode = choose_hoa_proxy_linestyle_mode(methods_present)
    if mode == "method":
        method_seen = set()
        for method in methods_present or []:
            if method in method_seen:
                continue
            method_seen.add(method)
            ls = get_hoa_linestyle(None, method, mode, method_linestyles=method_linestyles)
            proxy_handles.append(Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle=ls))
            pretty = hoa_method_legend_display_name(method, overrides=method_legend_labels)
            proxy_labels.append(pretty)
    else:
        # Only add a structure/framework row when there are multiple frameworks —
        # a single structure has one linestyle and adding it to the legend gives no information.
        unique_fws = list(dict.fromkeys(frameworks_present or []))
        if len(unique_fws) > 1:
            for fw in unique_fws:
                ls = get_hoa_linestyle(fw, None, mode, method_linestyles=method_linestyles)
                proxy_handles.append(Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle=ls))
                proxy_labels.append(clean_material_name(fw))

    if proxy_handles:
        ax.legend(proxy_handles, proxy_labels, fontsize=AXIS_LEGEND_SIZE, loc=loc)


def get_linestyle_for_fit_type(fit_type):
    """Return line style for an isotherm fit type (e.g. Sips, Langmuir_Freundlich). Uses fit_type_linestyle_mapping if set, else assigns by order. Normalizes hyphen/underscore for lookup."""
    if fit_type is None:
        return '-'
    ft_norm = _normalize_fit_type(fit_type)
    if _FIT_TYPE_LINESTYLE_MAP:
        if fit_type in _FIT_TYPE_LINESTYLE_MAP:
            return _FIT_TYPE_LINESTYLE_MAP[fit_type]
        if ft_norm in _FIT_TYPE_LINESTYLE_MAP:
            return _FIT_TYPE_LINESTYLE_MAP[ft_norm]
        for k, v in _FIT_TYPE_LINESTYLE_MAP.items():
            if _normalize_fit_type(k) == ft_norm:
                return v
    if ft_norm in _AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS:
        return _AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS[ft_norm]
    idx = len(_AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS) % len(_FIT_TYPE_LINESTYLE_PALETTE)
    ls = _FIT_TYPE_LINESTYLE_PALETTE[idx]
    _AUTO_FIT_TYPE_LINESTYLE_ASSIGNMENTS[ft_norm] = ls
    return ls


# Molecule marker map - set from Input.py via set_molecule_marker_map()
_MOLECULE_MARKER_MAP = None
_AUTO_MOLECULE_MARKER_ASSIGNMENTS = {}

def set_molecule_marker_map(marker_map):
    """Set mapping from molecule names to marker styles. Call from Main with Input.molecule_marker_mapping."""
    global _MOLECULE_MARKER_MAP
    _MOLECULE_MARKER_MAP = marker_map if marker_map else {}


def set_molecule_marker_palette(palette):
    """Override the auto-assigned molecule marker palette."""
    global _MOLECULE_MARKER_PALETTE, _AUTO_MOLECULE_MARKER_ASSIGNMENTS
    _MOLECULE_MARKER_PALETTE = list(palette) if palette else MARKER_STYLES
    _AUTO_MOLECULE_MARKER_ASSIGNMENTS = {}

def get_marker_for_molecule(molecule_name):
    """Return marker style for a molecule (for data points). Uses molecule_marker_mapping if set, else assigns by order."""
    if _MOLECULE_MARKER_MAP and molecule_name in _MOLECULE_MARKER_MAP:
        return _MOLECULE_MARKER_MAP[molecule_name]
    if molecule_name in _AUTO_MOLECULE_MARKER_ASSIGNMENTS:
        return _AUTO_MOLECULE_MARKER_ASSIGNMENTS[molecule_name]
    idx = len(_AUTO_MOLECULE_MARKER_ASSIGNMENTS) % len(_MOLECULE_MARKER_PALETTE)
    m = _MOLECULE_MARKER_PALETTE[idx]
    _AUTO_MOLECULE_MARKER_ASSIGNMENTS[molecule_name] = m
    return m


def set_molecule_linestyle_map(linestyle_map):
    """Set mapping from molecule names to line styles (for line plots)."""
    global _MOLECULE_LINESTYLE_MAP
    _MOLECULE_LINESTYLE_MAP = linestyle_map if linestyle_map else {}


def set_molecule_linestyle_palette(palette):
    """Override the auto-assigned molecule line style palette."""
    global _MOLECULE_LINESTYLE_PALETTE, _AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS
    _MOLECULE_LINESTYLE_PALETTE = list(palette) if palette else LINESTYLE_STYLES
    _AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS = {}


def get_linestyle_for_molecule(molecule_name):
    """Return line style for a molecule. Uses molecule_linestyle_mapping if set, else assigns by order."""
    if molecule_name is None:
        return '-'
    if _MOLECULE_LINESTYLE_MAP and molecule_name in _MOLECULE_LINESTYLE_MAP:
        return _MOLECULE_LINESTYLE_MAP[molecule_name]
    if molecule_name in _AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS:
        return _AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS[molecule_name]
    idx = len(_AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS) % len(_MOLECULE_LINESTYLE_PALETTE)
    ls = _MOLECULE_LINESTYLE_PALETTE[idx]
    _AUTO_MOLECULE_LINESTYLE_ASSIGNMENTS[molecule_name] = ls
    return ls

# Track automatic assignments for consistency
_AUTO_MARKER_ASSIGNMENTS = {}

def set_marker_map(marker_map):
    """Set a custom mapping from material names to marker styles.
    
    Args:
        marker_map: Dictionary mapping material names to marker styles.
                   Example: {'MOF-74': 'o', 'Bathia_01': 's', 'TAMOF_1': '^'}
    """
    global _MARKER_MAP
    _MARKER_MAP = marker_map if marker_map else {}

def isotherm_raspa_points_marker_axis(n_frameworks: int, n_molecules: int):
    """
    For Basic_Data isotherm RASPA scatter in ``DATA_SOURCE=points`` mode: when more
    than one framework or more than one molecule (adsorbates / mixture components)
    are present, return which dimension encodes marker shape (``'framework'`` or
    ``'molecule'``). If both exceed one, frameworks take priority.
    """
    n_fw = int(n_frameworks or 0)
    n_mol = int(n_molecules or 0)
    fw_many = n_fw > 1
    mol_many = n_mol > 1
    if fw_many:
        return "framework"
    if mol_many:
        return "molecule"
    return None


def get_marker_for_material(material_name):
    """Return a consistent marker style for a given material name.
    
    Priority:
    1. Use custom marker mapping if provided via set_marker_map()
    2. Use previously assigned automatic marker (for consistency within session)
    3. Assign new marker from available styles
    
    Args:
        material_name: Name of the material
        
    Returns:
        Marker style string (e.g., 'o', 's', '^', etc.)
    """
    # Check if custom marker map is defined and has this material
    if _MARKER_MAP is not None and material_name in _MARKER_MAP:
        return _MARKER_MAP[material_name]
    
    # Use previously assigned marker if available (for consistency within session)
    if material_name in _AUTO_MARKER_ASSIGNMENTS:
        return _AUTO_MARKER_ASSIGNMENTS[material_name]
    
    # Assign new marker: use a deterministic method based on material name
    # Sort all materials we've seen to get consistent ordering
    # Use a simple hash but store the result for consistency
    marker_idx = len(_AUTO_MARKER_ASSIGNMENTS) % len(MARKER_STYLES)
    marker = MARKER_STYLES[marker_idx]
    _AUTO_MARKER_ASSIGNMENTS[material_name] = marker
    return marker

def clean_material_name(name):
    """Remove underscores from material names for display."""
    if name is None:
        return None
    return str(name).replace('_', ' ')

# Molecule display name override (e.g. butane -> R600). Set from Input via set_molecule_display_name_override().
_MOLECULE_DISPLAY_NAME_OVERRIDE = {}

def set_molecule_display_name_override(mapping):
    """Set mapping from molecule name to display label. E.g. {'butane': 'R600'}. Keys are case-insensitive."""
    global _MOLECULE_DISPLAY_NAME_OVERRIDE
    if not mapping:
        _MOLECULE_DISPLAY_NAME_OVERRIDE = {}
        return
    _MOLECULE_DISPLAY_NAME_OVERRIDE = {str(k).strip().lower(): v for k, v in mapping.items()}

def get_molecule_display_name(mol):
    """Return display name for molecule (e.g. R600 for butane). Use for labels and titles."""
    if mol is None:
        return None
    key = str(mol).strip().lower()
    if key in _MOLECULE_DISPLAY_NAME_OVERRIDE:
        return _MOLECULE_DISPLAY_NAME_OVERRIDE[key]
    return clean_material_name(mol)

def set_axis_limits_nice(ax, pad_fraction=0.02):
    """
    Adjust axis limits to have:
    - Highest values rounded up to whole numbers
    - Lowest values just above/right of the data minimum
    - Improved readability

    For **linear** axes, if the current lower limit is already non-negative, the
    padded lower limit is clamped to **0** so the plot does not show a false gap
    at the origin (loading / Qst style figures).

    Note: Skips adjustment for log-scaled axes to avoid invalid limits.
    """
    # Get current data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Check if axes are log-scaled
    x_is_log = ax.get_xscale() == 'log'
    y_is_log = ax.get_yscale() == 'log'

    # X-axis: only adjust if linear scale
    if not x_is_log:
        x_min = xlim[0]
        x_max = math.ceil(xlim[1])
        x_range = x_max - x_min
        x_min_adj = x_min - x_range * pad_fraction
        if x_min >= 0.0:
            x_min_adj = max(0.0, x_min_adj)
        ax.set_xlim(left=x_min_adj, right=x_max)

    # Y-axis: only adjust if linear scale
    if not y_is_log:
        y_min = ylim[0]
        y_max = math.ceil(ylim[1])
        y_range = y_max - y_min
        y_min_adj = y_min - y_range * pad_fraction
        if y_min >= 0.0:
            y_min_adj = max(0.0, y_min_adj)
        ax.set_ylim(bottom=y_min_adj, top=y_max)

def set_label_padding(ax, xlabel_pad=None, ylabel_pad=None):
    """Set axis-label padding (points). Defaults: AXIS_LABELPAD_X / AXIS_LABELPAD_Y."""
    ax.xaxis.labelpad = AXIS_LABELPAD_X if xlabel_pad is None else xlabel_pad
    ax.yaxis.labelpad = AXIS_LABELPAD_Y if ylabel_pad is None else ylabel_pad


def resolve_storage_density_3d_colormap(spec=None):
    """Colormap for storage-density ``plot_surface`` (design.in name or list of colors)."""
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if spec is None:
        import Input as _inp
        spec = getattr(_inp, 'storage_density_3d_colormap', 'viridis')
    fallback = 'viridis'

    def _named(name):
        reg = getattr(mpl, 'colormaps', None)
        if reg is not None:
            try:
                return reg[name]
            except Exception:
                try:
                    return reg[fallback]
                except Exception:
                    pass
        try:
            return plt.cm.get_cmap(name)
        except Exception:
            return plt.cm.get_cmap(fallback)

    if isinstance(spec, str):
        return _named(spec)
    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        try:
            return LinearSegmentedColormap.from_list('storage_density_3d', list(spec), N=256)
        except Exception:
            return _named(fallback)
    return _named(fallback)


def resolve_storage_density_3d_boundries():
    """``(vmin, vmax)`` for ``plot_surface``; ``(None, None)`` = data range (design.in: ``storage_density_3d_boundries``)."""
    import Input as _inp
    spec = getattr(_inp, 'storage_density_3d_boundries', None)
    if spec is None:
        return None, None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = spec[0], spec[1]
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        return lo, hi
    return None, None


def resolve_storage_density_3d_figsize():
    """``(width, height)`` in inches for storage-density **3D** figures.

    By default, width is ``UNIFIED_FIGSIZE[0] * (1 + pct_extra/100)`` and height matches
    ``UNIFIED_FIGSIZE[1]`` (same as 2D). If ``Input.storage_density_3d_figsize`` is a
    ``(w, h)`` tuple/list, that overrides the percentage rule.
    """
    import Input as _inp
    spec = getattr(_inp, 'storage_density_3d_figsize', None)
    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        try:
            w, h = float(spec[0]), float(spec[1])
            return (max(3.0, w), max(3.0, h))
        except (TypeError, ValueError):
            pass
    uw, uh = UNIFIED_FIGSIZE
    _d = STORAGE_DENSITY_3D_DEFAULTS
    try:
        pct = float(getattr(_inp, 'storage_density_3d_figwidth_pct_extra', _d['storage_density_3d_figwidth_pct_extra']))
    except (TypeError, ValueError):
        pct = float(_d['storage_density_3d_figwidth_pct_extra'])
    pct = max(-80.0, min(pct, 300.0))
    w = float(uw) * (1.0 + pct / 100.0)
    return (max(3.0, w), float(uh))


def resolve_storage_density_3d_subplot_margins():
    """Outer figure margins and the **right edge of the 3D axes** in figure coordinates.

    The colorbar is placed in a separate axes to the right of ``right`` (see
    ``resolve_storage_density_3d_colorbar_kwargs``); ``fig.colorbar(..., ax=ax)``
    is not used, so the 3D axes are not auto-shrunk by Matplotlib.
    """
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS

    def _bound(name, lo, hi):
        default = _d[name]
        try:
            v = float(getattr(_inp, name, default))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(v, hi))

    left = _bound('storage_density_3d_subplot_left', 0.0, 0.45)
    bottom = _bound('storage_density_3d_subplot_bottom', 0.0, 0.45)
    top = _bound('storage_density_3d_subplot_top', 0.55, 1.0)
    right = _bound('storage_density_3d_subplot_right', 0.55, 0.995)
    if top <= bottom + 0.08:
        top = min(1.0, bottom + 0.90)
    return {'left': left, 'right': right, 'bottom': bottom, 'top': top}


def resolve_storage_density_3d_axes_dist():
    """Matplotlib 3D ``Axes3D.dist`` (camera distance); lower = zoomed-in, larger surface in frame."""
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS
    try:
        d = float(getattr(_inp, 'storage_density_3d_axes_dist', _d['storage_density_3d_axes_dist']))
    except (TypeError, ValueError):
        d = float(_d['storage_density_3d_axes_dist'])
    return max(4.0, min(d, 20.0))


def resolve_storage_density_3d_box_scale():
    """Scale the 3D ``Axes`` position (width/height) inside the allocated slot; <1 shrinks the visible box.

    Unlike ``axes_dist``, this always changes the on-page size of the 3D subplot rectangle
    (same figure size). Horizontal placement in the slot uses ``resolve_storage_density_3d_box_halign``.
    """
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS
    try:
        s = float(getattr(_inp, 'storage_density_3d_box_scale', _d['storage_density_3d_box_scale']))
    except (TypeError, ValueError):
        s = float(_d['storage_density_3d_box_scale'])
    return max(0.35, min(s, 1.0))


def resolve_storage_density_3d_box_halign():
    """Where the scaled 3D axes sit horizontally in ``[left, left+w]``: 0 = flush left, 0.5 = centered, 1 = right."""
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS
    try:
        h = float(getattr(_inp, 'storage_density_3d_box_halign', _d['storage_density_3d_box_halign']))
    except (TypeError, ValueError):
        h = float(_d['storage_density_3d_box_halign'])
    return max(0.0, min(h, 1.0))


def resolve_storage_density_3d_box_valign():
    """Vertical analogue of ``box_halign`` for ``[bottom, bottom+h]`` (0 = bottom, 0.5 = center)."""
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS
    try:
        v = float(getattr(_inp, 'storage_density_3d_box_valign', _d['storage_density_3d_box_valign']))
    except (TypeError, ValueError):
        v = float(_d['storage_density_3d_box_valign'])
    return max(0.0, min(v, 1.0))


def resolve_storage_density_3d_axis_labelpads():
    """Axis title distance from the 3D spines (points); larger reduces overlap with tick labels."""
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS

    def _f(name, lo, hi):
        default = _d[name]
        try:
            x = float(getattr(_inp, name, default))
        except (TypeError, ValueError):
            x = default
        return max(lo, min(x, hi))

    return {
        'x': _f('storage_density_3d_labelpad_x', 0.0, 80.0),
        'y': _f('storage_density_3d_labelpad_y', 0.0, 80.0),
        'z': _f('storage_density_3d_labelpad_z', 0.0, 100.0),
    }


def resolve_storage_density_3d_colorbar_kwargs():
    """Layout for a **dedicated** colorbar axes (``cax``): gap, strip width, ``shrink``, ``ticks``.

    ``gap_frac`` is horizontal space between the 3D axes and the colorbar (both sit left of the reserved strip).
    ``width_frac`` is the colorbar thickness as a fraction of figure width.
    ``cbar_label_labelpad`` / ``cbar_label_y`` tune the SD title on the vertical colorbar.
    ``nticks`` (when ``ticks`` is unset) selects roughly how many colorbar tick intervals to show.
    """
    import Input as _inp
    _d = STORAGE_DENSITY_3D_DEFAULTS

    def _f(name, lo, hi):
        default = _d[name]
        try:
            v = float(getattr(_inp, name, default))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(v, hi))

    gap = _f('storage_density_3d_colorbar_gap', 0.0, 0.15)
    width = _f('storage_density_3d_colorbar_width', 0.010, 0.12)
    try:
        shrink = float(getattr(_inp, 'storage_density_3d_colorbar_shrink', _d['storage_density_3d_colorbar_shrink']))
    except (TypeError, ValueError):
        shrink = float(_d['storage_density_3d_colorbar_shrink'])
    shrink = max(0.15, min(shrink, 1.0))
    ticks_spec = getattr(_inp, 'storage_density_3d_colorbar_ticks', _d['storage_density_3d_colorbar_ticks'])
    ticks = None
    if isinstance(ticks_spec, (list, tuple)) and len(ticks_spec) > 0:
        try:
            ticks = [float(t) for t in ticks_spec]
        except (TypeError, ValueError):
            ticks = None
    try:
        nticks = int(getattr(_inp, 'storage_density_3d_colorbar_nticks', _d['storage_density_3d_colorbar_nticks']))
    except (TypeError, ValueError):
        nticks = int(_d['storage_density_3d_colorbar_nticks'])
    nticks = max(0, min(nticks, 24))
    cbar_label_labelpad = _f('storage_density_3d_cbar_label_labelpad', 0.0, 60.0)
    cbar_label_y = _f('storage_density_3d_cbar_label_y', 0.2, 0.85)
    return {
        'gap_frac': gap,
        'width_frac': width,
        'shrink': shrink,
        'ticks': ticks,
        'nticks': nticks,
        'cbar_label_labelpad': cbar_label_labelpad,
        'cbar_label_y': cbar_label_y,
    }


def apply_unified_axes_layout(fig, ax):
    """Apply shared figure margins, axis ticks, spines, and label padding.

    Uses module-level constants: ``AXIS_LEFT`` / ``RIGHT`` / ``BOTTOM`` / ``TOP``,
    ``AXIS_TICK_*``, ``AXIS_LABELPAD_*``, ``TICK_SIZE`` / ``TICK_SIZE_X``,
    ``AXIS_LW``. Intended for any single-``Axes`` figure that should match the
    project's default look (e.g. isotherms). Call after ``set_xlabel`` /
    ``set_ylabel`` on ``ax``.

    If the axes tight bbox (tick labels, axis titles) still crosses the figure
    edge, bottom/left are increased up to ``AXIS_MARGIN_MAX_*`` so labels are
    not clipped (e.g. log-scale ticks). If the bbox extends past the **right**
    edge, ``right`` is decreased down to ``AXIS_MARGIN_MIN_SUBPLOT_RIGHT``.
    For log x, ``right`` is then capped by ``AXIS_RIGHT_LOG_X_CAP`` (light inset;
    bbox often misses superscripts on the last tick).
    """
    left = AXIS_LEFT
    right = AXIS_RIGHT
    bottom = AXIS_BOTTOM
    top = AXIS_TOP
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    for axis, labelsize in (('x', TICK_SIZE_X), ('y', TICK_SIZE)):
        ax.tick_params(
            axis=axis,
            which='major',
            labelsize=labelsize,
            width=AXIS_TICK_WIDTH,
            length=AXIS_TICK_LENGTH_MAJOR,
            pad=AXIS_TICK_PAD,
        )
        # No minor tick dashes on the spines (minor grid lines are unchanged).
        ax.tick_params(axis=axis, which='minor', length=0, width=0)
    ax.xaxis.labelpad = AXIS_LABELPAD_X
    ax.yaxis.labelpad = AXIS_LABELPAD_Y
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LW)
        spine.set_edgecolor('black')

    try:
        target = AXIS_MARGIN_AUTOPAD_FRAC
        # Multi-pass: one subplots_adjust can move tick/label bbox; y-axis label needs room.
        for _ in range(4):
            if hasattr(fig.canvas, "draw_without_rendering"):
                fig.canvas.draw_without_rendering()
            else:
                fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            changed = False
            if bb.y0 < target:
                new_bottom = min(
                    AXIS_MARGIN_MAX_BOTTOM,
                    bottom + (target - bb.y0),
                )
                if new_bottom > bottom + 1e-5:
                    bottom = new_bottom
                    changed = True
            if bb.x0 < target:
                new_left = min(
                    AXIS_MARGIN_MAX_LEFT,
                    left + (target - bb.x0),
                )
                if new_left > left + 1e-5:
                    left = new_left
                    changed = True
            if bb.x1 > 1.0 - target:
                excess = bb.x1 - (1.0 - target)
                new_right = max(AXIS_MARGIN_MIN_SUBPLOT_RIGHT, right - excess)
                if new_right < right - 1e-5:
                    right = new_right
                    changed = True
            if not changed:
                break
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    except Exception:
        pass

    try:
        if ax.get_xscale() == 'log':
            pars = fig.subplotpars
            new_r = min(float(pars.right), float(AXIS_RIGHT_LOG_X_CAP))
            if new_r < float(pars.right) - 1e-5:
                fig.subplots_adjust(
                    left=pars.left,
                    bottom=pars.bottom,
                    right=new_r,
                    top=pars.top,
                )
    except Exception:
        pass


def _save_plot(prefix, out_subdir, selected_frameworks, selected_molecules, selected_temperatures, fig=None, out_dir=None, temp_label_override=None, fw_label_override=None, mol_label_override=None, filename_suffix=None, *, tight_bbox=False, bbox_extra_artists=None):
    """Save a matplotlib plot. Default location is under ``Output/``.

    If *tight_bbox* is True, save with ``bbox_inches='tight'`` (used for storage-density 3D PNGs).
    Pass *bbox_extra_artists* so mplot3d axis titles and ticks are not clipped by tight layout.
    """

    # Use repo root as base (go up two levels from code/functions/)
    base_dir = Path(__file__).resolve().parents[2]

    # helper to create safe joined strings
    def _safe_join(lst):
        return "-".join([str(x).replace(" ", "_") for x in lst]) if lst else "all"

    # prepare filename parts using overrides or selections
    fw_part = fw_label_override if fw_label_override is not None else _safe_join(selected_frameworks)
    mol_part = mol_label_override if mol_label_override is not None else _safe_join(selected_molecules)
    temp_part = temp_label_override if temp_label_override is not None else _safe_join(selected_temperatures)

    # Mapping from out_subdir to user-friendly folder names
    subfolder_map = {
        'plot_storage_density': 'Storage_Density',
        'plot_heat_capacity': 'Heat_Capacity',
        'plot_virial': 'Heat_of_Adsorption',
        'plot_clausius_clapeyron': 'Heat_of_Adsorption',
        'plot_controls': 'Heat_of_Adsorption',
        'plot_isotherm_fitting': 'Basic_Data',
        'plot_enthalpy_vs_loading': 'Basic_Data',
        'experiment_simulation': 'Compare_Qst',
        'plot_storage_density_components': 'Storage_Density_components'
    }
    
    # Handle out_subdir with suffixes (e.g., 'plot_isotherm_fitting_langmuir')
    subfolder_name = None
    if out_subdir:
        # Try exact match first
        if out_subdir in subfolder_map:
            subfolder_name = subfolder_map[out_subdir]
        else:
            # Try prefix match for cases like 'plot_isotherm_fitting_langmuir'
            for key in subfolder_map:
                if out_subdir.startswith(key):
                    subfolder_name = subfolder_map[key]
                    break
    
    # If no match found, use out_subdir as-is (backwards compatibility)
    if subfolder_name is None:
        subfolder_name = out_subdir if out_subdir else 'Other'
    
    # Keep output path below common Windows path limits (OneDrive paths are deep).
    _max_path_chars = 260
    _run_key = f"{fw_part}|{mol_part}|{temp_part}"
    _run_hash = hashlib.sha1(_run_key.encode("utf-8")).hexdigest()[:10]
    _file_hash = hashlib.sha1(f"{prefix}|{_run_key}".encode("utf-8")).hexdigest()[:10]

    # If an explicit out_dir is provided, use it (create if needed). Otherwise
    # create a deterministic run folder under base_dir/Output using the parts.
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        plots_root = base_dir / 'Output'
        # Use a deterministic folder name based on the selection parts so that
        # all plots from the same selection are stored together. If the folder
        # already exists reuse it; otherwise create it.
        run_folder_name = f"{fw_part}_{mol_part}_{temp_part}"
        probe_name = f"{prefix}_{fw_part}__{mol_part}__{temp_part}.png"
        probe_path = plots_root / run_folder_name / subfolder_name / probe_name
        if len(str(probe_path)) > _max_path_chars:
            run_folder_name = f"run_{_run_hash}"
        out_dir = plots_root / run_folder_name / subfolder_name
        out_dir.mkdir(parents=True, exist_ok=True)

    # Filename without timestamp to enable overwriting.
    # If filename_suffix is set (e.g. single-structure name), use prefix_suffix.png only.
    if filename_suffix is not None:
        safe_suffix = str(filename_suffix).replace(" ", "_")
        filename = f"{prefix}_{safe_suffix}.png"
    else:
        filename = f"{prefix}_{fw_part}__{mol_part}__{temp_part}.png"
    out_path = out_dir / filename
    if len(str(out_path)) > _max_path_chars:
        filename = f"{prefix}_{_file_hash}.png"
        out_path = out_dir / filename
    try:
        if fig is None:
            plt.savefig(out_path, dpi=300, format='png')
        else:
            kw = {'dpi': 300, 'format': 'png'}
            if tight_bbox:
                import Input as _inp
                _d = STORAGE_DENSITY_3D_DEFAULTS
                try:
                    pad = float(getattr(_inp, 'storage_density_3d_save_bbox_pad', _d['storage_density_3d_save_bbox_pad']))
                except (TypeError, ValueError):
                    pad = float(_d['storage_density_3d_save_bbox_pad'])
                pad = max(0.0, min(pad, 0.5))
                try:
                    save_dpi = int(getattr(_inp, 'storage_density_3d_save_dpi', _d['storage_density_3d_save_dpi']))
                except (TypeError, ValueError):
                    save_dpi = int(_d['storage_density_3d_save_dpi'])
                kw['dpi'] = max(72, min(save_dpi, 600))
                kw['bbox_inches'] = 'tight'
                kw['pad_inches'] = pad
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
                if bbox_extra_artists:
                    kw['bbox_extra_artists'] = bbox_extra_artists
            fig.savefig(out_path, **kw)
    except Exception as e:
        print(f"Warning: failed to save {prefix} plot: {e}")
    return out_path


def _normalize_fit_type(ft):
    """Treat Langmuir-Freundlich and Langmuir_Freundlich as the same for matching."""
    if ft is None:
        return None
    return str(ft).replace('-', '_').strip()


# --- Calculations Plots ---
def build_fit_cache(fittings, frameworks, molecules, temperatures, fit_types, num_of_isotherm=None):
    """
    Build a cache of fitting parameters.
    
    Args:
        fittings: List of fitting data dictionaries
        frameworks: List of framework names
        molecules: List of molecule names
        temperatures: List of temperatures
        fit_types: List of fit types
        num_of_isotherm: Number of isotherm sites. Can be:
            - None: use any available (first match)
            - Single value (e.g., 2): only use fits with that many sites (2=6 params, 3=9 params)
            - List (e.g., [2, 3]): use fits matching any of the specified site counts
    
    Returns:
        Dictionary mapping (framework, molecule, temp, fit_type) to (params, fit_type)
        When num_of_isotherm is a list with multiple values, keys include 5th element: (fw, mol, temp, ft, num_params)
    """
    cache = {}
    
    # Normalize num_of_isotherm to a list for easier processing
    if num_of_isotherm is None:
        allowed_param_counts = None  # Allow any
        use_extended_key = False
    elif isinstance(num_of_isotherm, list):
        allowed_param_counts = [n * 3 for n in num_of_isotherm]
        use_extended_key = len(num_of_isotherm) > 1  # Use 5-tuple key only if comparing multiple
    else:
        allowed_param_counts = [num_of_isotherm * 3]
        use_extended_key = False
    
    for fw in frameworks:
        for mol in molecules:
            for temp in temperatures:
                for ft in fit_types:
                    # Normalize fit_type so Langmuir-Freundlich matches Langmuir_Freundlich in data
                    ft_norm = _normalize_fit_type(ft)
                    matching_fits = [fit for fit in fittings if
                                    fit.get("framework") == fw and
                                    str(fit.get("molecule", "")).strip().lower() == str(mol).strip().lower() and
                                    fit.get("temperature") is not None and
                                    abs(float(fit.get("temperature")) - float(temp)) < 0.01 and
                                    _normalize_fit_type(fit.get("fit_type", ft)) == ft_norm]
                    
                    # Filter by number of parameters if specified
                    if allowed_param_counts is not None:
                        matching_fits = [fit for fit in matching_fits if len(fit["params"]) in allowed_param_counts]
                    
                    # Store matches - use extended key format if comparing multiple parameter counts
                    if use_extended_key:
                        # Multiple parameter counts requested - store each with 5-tuple key
                        for fit_row in matching_fits:
                            num_params = len(fit_row["params"])
                            cache[(fw, mol, float(temp), ft, num_params)] = (fit_row["params"], ft)
                    else:
                        # Single or no parameter count filter - use 4-tuple key for backward compatibility
                        fit_row = matching_fits[0] if matching_fits else None
                        if fit_row:
                            cache[(fw, mol, float(temp), ft)] = (fit_row["params"], ft)
    
    return cache

def iter_cache_entries(fit_cache, fw, mol, temp, ft):
    """
    Iterator to get all cache entries for a given combination.
    Handles both 4-tuple and 5-tuple (with num_params) key formats.
    
    Yields tuples of (params, ft_type, num_params) for each matching entry.
    """
    # Try 4-tuple key first (backward compatible)
    key_4 = (fw, mol, float(temp), ft)
    if key_4 in fit_cache:
        params, ft_type = fit_cache[key_4]
        yield (params, ft_type, len(params))
    else:
        # Try 5-tuple keys (for multiple parameter count comparison)
        found_any = False
        for key in fit_cache:
            if len(key) == 5 and key[0] == fw and key[1] == mol and \
               isclose(key[2], float(temp), abs_tol=0.01) and key[3] == ft:
                params, ft_type = fit_cache[key]
                num_params = key[4]
                yield (params, ft_type, num_params)
                found_any = True
        
        if not found_any:
            # No entries found
            return

def get_filtered_data(RASPA_data, fw, mol, temp):
    return [d for d in RASPA_data if
            d["framework"] == fw and
            d["molecule"].strip().lower() == mol.strip().lower() and
            isclose(d["temperature"], float(temp), abs_tol=1)]

def filter_raspa_data(RASPA_data, frameworks=None, molecules=None, temperatures=None, tol=1.0,
                      only_pure_adsorption=False):
    """
    General-purpose filter for RASPA_data.

    Parameters
    - RASPA_data: iterable of dicts (each row)
    - frameworks: None or list of framework names to keep (exact string match)
    - molecules: None or list of molecule names to keep (case-insensitive)
    - temperatures: None or list of temperatures to keep (numeric; tolerant comparison)
    - tol: tolerance in K for matching temperatures (default 1.0)
    - only_pure_adsorption: if True, drop rows whose ``mixture_pure`` field is present
      and not ``pure`` (case-insensitive). Rows without ``mixture_pure`` (e.g. synthetic
      points) are kept. Use for pure-component Virial/CC/DataSelection on files that also
      contain mixture-component rows for the same (framework, molecule).

    Returns a list of matching rows (may be empty).
    """
    if RASPA_data is None:
        return []

    fw_set = set(str(x) for x in frameworks) if frameworks is not None else None
    mol_set = set(str(x).strip().lower() for x in molecules) if molecules is not None else None
    temps = [float(t) for t in temperatures] if temperatures is not None else None

    out = []
    for d in RASPA_data:
        try:
            if fw_set is not None and d.get('framework') not in fw_set:
                continue
            if mol_set is not None:
                m = d.get('molecule')
                if m is None or m.strip().lower() not in mol_set:
                    continue
            if temps is not None:
                t = d.get('temperature')
                if t is None:
                    continue
                # tolerant numeric comparison
                if not any(abs(float(t) - tt) <= float(tol) for tt in temps):
                    continue
            if only_pure_adsorption:
                mp = d.get('mixture_pure')
                if mp is not None and str(mp).strip() != '' and str(mp).strip().lower() != 'pure':
                    continue
            out.append(d)
        except Exception:
            # skip rows with unexpected/missing fields
            continue
    return out


# Reserved ``molecule`` in saved ``mixture_isotherm_log_*.txt`` for mixture-total (P, q_tot) rows.
MIXTURE_TOTAL_ISOTHERM_MOLECULE = "__MIXTURE_TOTAL__"


def safe_join_plot_labels(lst):
    """Join list of labels into a filesystem-safe segment (same as IsothermFittingPlot._safe_join)."""
    return "-".join([str(x).replace(" ", "_") for x in lst]) if lst else "all"


def mixture_isotherm_saved_log_path(repo_root, selected_frameworks, selected_molecules, selected_temperatures,
                                    prefix="mixture_isotherm_log"):
    """Path to the saved mixture isotherm log TSV (log-scale export naming)."""
    fw_part = safe_join_plot_labels(selected_frameworks)
    mol_part = safe_join_plot_labels(selected_molecules)
    temp_part = safe_join_plot_labels(selected_temperatures)
    run_folder = f"{fw_part}_{mol_part}_{temp_part}"
    return (
        Path(repo_root) / "Output" / run_folder / "Basic_Data" / "saved" /
        f"{prefix}_{fw_part}__{mol_part}__{temp_part}.txt"
    )


def mixture_total_pq_tuples(mixture_data, framework, temperature, components,
                            p_min=None, p_max=None, require_positive_pressure=False):
    """
    Total loading q_tot(P) as sum of component loadings at each pressure key.

    Matches ``mixture_isotherm_total_*`` in IsothermFittingPlot (sum-at-P, not strict
    same-P intersection across species).
    """
    p_min_use = max(float(p_min), 1e-8) if p_min is not None else None
    p_max_use = float(p_max) if p_max is not None else None
    sum_by_pressure = {}
    for comp in components:
        pts = filter_raspa_data(
            mixture_data, frameworks=[framework], molecules=[comp], temperatures=[temperature]
        )
        for d in pts:
            try:
                P = float(d["pressure"])
                q = float(d["loading"])
            except (TypeError, ValueError):
                continue
            if require_positive_pressure and P <= 0:
                continue
            if not (np.isfinite(P) and np.isfinite(q)):
                continue
            if p_min_use is not None and P < p_min_use:
                continue
            if p_max_use is not None and P > p_max_use:
                continue
            sum_by_pressure[P] = sum_by_pressure.get(P, 0.0) + q
    if not sum_by_pressure:
        return []
    return sorted(((float(P), float(q)) for P, q in sum_by_pressure.items()), key=lambda x: x[0])


def load_mixture_total_from_isotherm_log(path, framework, p_min=None, p_max=None, mixture_name=None):
    """
    Read ``mixture_isotherm_log_*.txt`` and return ``{temperature: [(P, q_tot), ...]}``
    for rows whose ``molecule`` is ``mixture_name`` (case-insensitive) or the legacy
    ``MIXTURE_TOTAL_ISOTHERM_MOLECULE`` tag, with matching framework.

    Pressures are optionally clipped to ``[p_min, p_max]`` (same window as mixture CC).
    """
    path = Path(path)
    if not path.is_file():
        return {}
    p_lo = max(float(p_min), 1e-8) if p_min is not None else None
    p_hi = float(p_max) if p_max is not None else None
    fw_need = str(framework).strip().lower()
    by_T = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.readline()  # header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                fw, mol, t_s, p_s, q_s = parts[0], parts[1], parts[2], parts[3], parts[4]
                mol_st = mol.strip()
                is_total_row = mol_st == MIXTURE_TOTAL_ISOTHERM_MOLECULE or (
                    mixture_name is not None
                    and mol_st.lower() == str(mixture_name).strip().lower()
                )
                if str(fw).strip().lower() != fw_need or not is_total_row:
                    continue
                try:
                    T = float(t_s)
                    P = float(p_s)
                    q = float(q_s)
                except ValueError:
                    continue
                if not (np.isfinite(P) and np.isfinite(q)):
                    continue
                if p_lo is not None and P < p_lo:
                    continue
                if p_hi is not None and P > p_hi:
                    continue
                by_T.setdefault(T, []).append((P, q))
    except OSError:
        return {}
    out = {}
    for T, lst in by_T.items():
        per_p = {}
        for P, q in lst:
            per_p[float(P)] = float(q)
        out[T] = sorted(((P, per_p[P]) for P in sorted(per_p)), key=lambda x: x[0])
    return out


def mixture_total_log_covers_temperatures(loaded_by_T, temperatures, tol=1.0, min_points=2):
    """True if ``loaded_by_T`` has at least ``min_points`` (P,q) for every ``temperatures`` (tolerant T match)."""
    if not loaded_by_T:
        return False
    keys = list(loaded_by_T.keys())

    def _match(T_need):
        for k in keys:
            if abs(float(k) - float(T_need)) <= float(tol):
                return k
        return None

    for T in temperatures:
        k = _match(T)
        if k is None or len(loaded_by_T[k]) < int(min_points):
            return False
    return True


def plot_raspa_points(data, fw, mol, temp, color, show_in_legend=True, marker=None, s=AXIS_S_SIZE, zorder=50, ax=None):
    """Plot RASPA data points.
    
    Args:
        data: RASPA data
        fw: framework name
        mol: molecule name
        temp: temperature
        color: point color
        show_in_legend: whether to include points in legend
        marker: marker style (if None, uses material-specific marker)
        s: marker area in points^2 (``matplotlib.axes.Axes.scatter``), same meaning as ``s=`` there.
        zorder: draw order for markers; higher values render above lines.
        ax: target matplotlib axes (uses current axes when None).
    """
    x = np.array([d["pressure"] for d in data])
    y = np.array([d["loading"] for d in data])
    if x.size and y.size:
        fw_display = clean_material_name(fw)
        mol_display = get_molecule_display_name(mol)
        if marker is None:
            marker = get_marker_for_material(fw)
        label = f"Data: {fw_display}, {mol_display}, {int(float(temp))}K" if show_in_legend else None
        target = ax if ax is not None else plt.gca()
        target.scatter(
            x, y, label=label, color=color, alpha=ALPHA, marker=marker, s=s, zorder=zorder,
            edgecolors='black', linewidths=1
        )

def plot_fit_curve(x_fit, params, ft_type, label, color, linestyle=None, ax=None):
    """Plot a fit curve.
    
    Args:
        x_fit: x values for the fit curve
        params: fitting parameters
        ft_type: fit type
        label: legend label
        color: line color
        linestyle: line style (if None, uses fit-type specific style)
        ax: matplotlib axes to plot on (if None, uses current axes)
    """
    y_fit = init.formula_fitting(x_fit, params, ft_type)
    target = ax if ax is not None else plt.gca()
    # Use specified linestyle, or default based on fit type
    if linestyle is not None:
        target.plot(x_fit, y_fit, label=label, color=color, lw=LINEWIDTH, linestyle=linestyle)
    else:
        # Use different line styles to visually distinguish fit types
        if ft_type == "Langmuir_Freundlich" or ft_type == "Langmuir-Freundlich":
            target.plot(x_fit, y_fit, label=label, color=color, lw=LINEWIDTH, linestyle='--')
        elif ft_type == "fitting_Sips" or ft_type == "Sips":
            target.plot(x_fit, y_fit, label=label, color=color, lw=LINEWIDTH, linestyle='-')
        elif ft_type == "fitting_toth" or ft_type == "Toth":
            target.plot(x_fit, y_fit, label=label, color=color, lw=LINEWIDTH, linestyle='-.')
        else:
            target.plot(x_fit, y_fit, label=label, color=color, lw=LINEWIDTH)

def evaluate_fit(p, params, ft_type):
    """
    Evaluate fitting formula for given pressure and parameters.
    Uses the unified dispatcher function from Initialize.
    """
    return init.formula_fitting(p, params, ft_type)

def inverse_fit(q_target, params, ft_type, p_min=1e-6, p_max=1e9):
    """
    Numerically find the pressure P that corresponds to a given loading q_target
    for a specific fitting formula. Uses Brent's method for root finding.
    
    This is the inverse of evaluate_fit: given q, find P such that q = f(P).
    
    Parameters:
    - q_target: target loading value(s) - can be scalar or array
    - params: fitting parameters for the isotherm model
    - ft_type: fitting type ('Sips', 'Langmuir_Freundlich', 'Toth', etc.)
    - p_min: minimum pressure bound for root finding
    - p_max: maximum pressure bound for root finding
    
    Returns:
    - P value(s) corresponding to q_target. Returns NaN if no solution found.
    """
    from scipy.optimize import brentq
    
    q_target = np.atleast_1d(q_target)
    p_results = np.full(len(q_target), np.nan)
    
    for i, q in enumerate(q_target):
        if not np.isfinite(q) or q < 0:
            continue
        
        # Define the function to find root of: f(P) - q = 0
        def residual(p):
            q_eval = evaluate_fit(p, params, ft_type)
            if hasattr(q_eval, '__len__'):
                q_eval = q_eval[0]
            return q_eval - q
        
        try:
            # Check if solution exists within bounds
            f_min = residual(p_min)
            f_max = residual(p_max)
            
            # If signs are the same, no root in interval
            if f_min * f_max > 0:
                # Try to find bounds where sign changes
                # Expand search if needed
                found_bounds = False
                for p_test_min in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                    for p_test_max in [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
                        try:
                            f_test_min = residual(p_test_min)
                            f_test_max = residual(p_test_max)
                            if f_test_min * f_test_max < 0:
                                p_min_use = p_test_min
                                p_max_use = p_test_max
                                found_bounds = True
                                break
                        except:
                            continue
                    if found_bounds:
                        break
                
                if not found_bounds:
                    continue
            else:
                p_min_use = p_min
                p_max_use = p_max
            
            # Find root using Brent's method
            p_solution = brentq(residual, p_min_use, p_max_use, xtol=1e-12, rtol=1e-10)
            p_results[i] = p_solution
            
        except (ValueError, RuntimeError):
            # No solution found
            continue
    
    # Return scalar if input was scalar
    if len(p_results) == 1:
        return p_results[0]
    return p_results

def interpolate_raspa_data(RASPA_data, frameworks, molecules, temperatures, n_points=40):
    """
    Augment RASPA data by keeping original points and adding interpolated points in between.
    
    Strategy:
    1. Keep ALL original RASPA points (including low-loading points)
    2. Add n_points additional interpolated points between existing points for smoother fitting
    3. Use PCHIP interpolation for physical, monotonic curves
    
    Parameters:
    - RASPA_data: list of RASPA data dictionaries
    - frameworks: list of framework names
    - molecules: list of molecule names  
    - temperatures: list of temperatures (K)
    - n_points: number of ADDITIONAL interpolated points per temperature (added to originals)
    
    Returns: list of original + interpolated data points in RASPA format
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    
    if RASPA_data is None or len(RASPA_data) == 0:
        return []
    
    # Filter data for selected frameworks/molecules/temperatures
    filtered_data = filter_raspa_data(RASPA_data, frameworks, molecules, temperatures)
    
    if len(filtered_data) == 0:
        print(f"Warning: No RASPA data found for interpolation")
        return []
    
    # Group by temperature
    data_by_temp = {}
    for pt in filtered_data:
        T = float(pt['temperature'])
        if T not in data_by_temp:
            data_by_temp[T] = []
        data_by_temp[T].append(pt)
    
    # Collect original + interpolated points
    all_points = []
    original_count = 0
    interpolated_count = 0
    
    for T in sorted(data_by_temp.keys()):
        points = data_by_temp[T]
        
        # Extract loadings and pressures, sort by loading
        data_list = [(float(p['loading']), float(p['pressure']), p) for p in points 
                     if p.get('loading') is not None and p.get('pressure') is not None 
                     and float(p['loading']) > 0 and float(p['pressure']) > 0]
        
        if len(data_list) < 2:
            print(f"Warning: Not enough data points for T={T}K (need at least 2)")
            continue
        
        data_list.sort(key=lambda x: x[0])  # Sort by loading
        loadings_T = np.array([x[0] for x in data_list])
        pressures_T = np.array([x[1] for x in data_list])
        
        # Add all ORIGINAL points first
        for _, _, original_pt in data_list:
            all_points.append(original_pt)
            original_count += 1
        
        # Now add INTERPOLATED points between originals
        if n_points > 0 and len(loadings_T) >= 2:
            try:
                # Use PCHIP for monotonic, smooth interpolation
                interpolator = PchipInterpolator(loadings_T, pressures_T, extrapolate=False)
                
                # Create additional points between min and max loading
                n_min = float(np.min(loadings_T))
                n_max = float(np.max(loadings_T))
                
                # Generate n_points evenly-spaced loadings
                loading_grid = np.linspace(n_min, n_max, n_points)
                
                # Interpolate pressures
                pressures_interp = interpolator(loading_grid)
                
                # Add interpolated points (skip if too close to original points)
                min_spacing = (n_max - n_min) / (len(loadings_T) * 10)  # 10% of average spacing
                for n, P in zip(loading_grid, pressures_interp):
                    # Skip if too close to any original point
                    if np.any(np.abs(loadings_T - n) < min_spacing):
                        continue
                    
                    if np.isfinite(P) and P > 0:
                        all_points.append({
                            'framework': frameworks[0] if frameworks else points[0]['framework'],
                            'molecule': molecules[0] if molecules else points[0]['molecule'],
                            'temperature': T,
                            'pressure': float(P),
                            'loading': float(n),
                            'enthalpy': 0.0,  # Not used in virial
                            'heat_capacity': 0.0,
                            'internal_energy': 0.0,
                            'enthalpy_correction': 0.0
                        })
                        interpolated_count += 1
            except Exception as e:
                print(f"Warning: Interpolation failed for T={T}K: {e}")
                continue
    
    print(f"Augmented RASPA data: {original_count} original + {interpolated_count} interpolated = {len(all_points)} total points")
    return all_points

def filter_pressures(x_fit, P_des_max):
    return np.array([p for p in x_fit if P_des_max is None or p <= P_des_max], dtype=float)

# storage density enthalpy + cc helpers
def format_storage_plot(ax, title, p_min=None, p_max=None, P_des_max=None,
                        global_p_min=None, subtitle=None, scale='log'):
    import numpy as np  # Import at function start to avoid UnboundLocalError
    if str(scale).strip().lower() == 'linear':
        ax.set_xscale('linear')
    else:
        ax.set_xscale('log')
    ax.set_xlabel('Desorption Pressure [Pa]', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('SD [kJ/kg]', fontsize=AXIS_LABEL_FONTSIZE)
    # Set title with optional subtitle (like heat capacity heatmap)
    if subtitle:
        # ax.set_title(f'{title}\n{subtitle}')
        pass
    else:
        # ax.set_title(title)
        pass
    ax.grid(True, which='both', ls='--', alpha=ALPHA_GRID)
    set_label_padding(ax)
    
    # Set x-axis limits: use p_min for left, auto-scale right based on plotted data
    # This allows the plot to extend naturally until data runs out (not limited by P_des_max)
    left = p_min if p_min is not None else global_p_min
    
    # Find minimum maximum pressure from all plotted lines (shortest line endpoint)
    # Use this to set right limit, but don't exceed P_des_max
    # This ensures lines extend to P_des_max if they have data, otherwise to shortest line
    right = None
    if len(ax.lines) > 0:
        x_data_max_list = []
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) > 0:
                line_max = np.nanmax(xdata)
                if np.isfinite(line_max):
                    x_data_max_list.append(line_max)
        
        if x_data_max_list:
            # Find the minimum of maximums (shortest line endpoint)
            shortest_line_end = min(x_data_max_list)
            # Use shortest line endpoint, but cap at P_des_max
            if P_des_max is not None:
                right = min(shortest_line_end, P_des_max)
            else:
                right = shortest_line_end
    
    # If no data yet, use p_max or P_des_max as fallback
    if right is None:
        right = P_des_max if P_des_max is not None else p_max
    
    if left is not None or right is not None:
        ax.set_xlim(left=left, right=right)
        ax.set_autoscale_on(False)
    
    # Only adjust y-axis limits nicely, skip x-axis if we set explicit limits
    # This prevents set_axis_limits_nice from overriding our explicit x-axis limits
    if left is None and right is None:
        set_axis_limits_nice(ax)
    else:
        # Only adjust y-axis, preserve x-axis limits we just set
        ylim = ax.get_ylim()
        y_is_log = ax.get_yscale() == 'log'
        if not y_is_log and len(ax.lines) > 0:
            # Get y data range from all plotted lines
            y_data_min = float('inf')
            y_data_max = float('-inf')
            for line in ax.lines:
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    y_data_min = min(y_data_min, np.nanmin(ydata))
                    y_data_max = max(y_data_max, np.nanmax(ydata))
            
            if y_data_min != float('inf') and y_data_max != float('-inf'):
                ax.set_ylim(bottom=0, top=y_data_max * 1.05)