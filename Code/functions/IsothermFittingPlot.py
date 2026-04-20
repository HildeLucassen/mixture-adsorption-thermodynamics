import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import PlotHelpers as phelp
import Initialize as init
import Input  # to access config['out_dir'] flag from config.in


def _safe_join(lst):
    """Join list of labels into a filesystem-safe segment."""
    return "-".join([str(x).replace(" ", "_") for x in lst]) if lst else "all"


def _save_isotherm_rows_to_run_folder(rows, fw_part, mol_part, temp_part, prefix, base_dir=None):
    """Helper to save isotherm-style rows to Output/<run>/Basic_Data/saved.

    Columns: framework, molecule, temperature_K, pressure_Pa, loading_mol_per_kg
    """
    if not rows:
        return
    try:
        base_dir = base_dir or init.get_pipeline_run_root()
        plots_root = base_dir / 'Output'

        run_folder_name = f"{fw_part}_{mol_part}_{temp_part}"
        saved_dir = plots_root / run_folder_name / 'Basic_Data' / 'saved'
        saved_dir.mkdir(parents=True, exist_ok=True)

        # Same idea as ``_save_plot`` PNGs: run folder already encodes fw/mol/temps.
        data_path = saved_dir / f"{prefix}.txt"
        with data_path.open('w', encoding='utf-8') as f:
            f.write(
                "framework\tmolecule\ttemperature_K\tpressure_Pa\tloading_mol_per_kg\n"
            )
            for r in rows:
                f.write(
                    f"{r['framework']}\t{r['molecule']}\t{r['temperature']}\t"
                    f"{r['pressure']}\t{r['loading']}\n"
                )
    except Exception as e:
        print(f"Warning: failed to write isotherm data file: {e}")

def plot_isotherm_fitting(selected_frameworks, selected_molecules, selected_temperatures,
                          selected_fit_types, fittings, RASPA_data, combo_colors,
                          x_fit, p_min=None, p_max=None, plot_RASPA=True, num_of_isotherm=None,
                          pressure_scale='both', save_data=False):
    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules,
        selected_temperatures, selected_fit_types,
        num_of_isotherm=num_of_isotherm,
    )

    # ------------------------------------------------------------------
    # Collect scale-independent raw data rows once (RASPA points).
    # These do not depend on whether the axis is log or linear.
    # ------------------------------------------------------------------
    data_rows = []
    if plot_RASPA:
        for fw in selected_frameworks:
            for mol in selected_molecules:
                for temp in selected_temperatures:
                    data = phelp.filter_raspa_data(
                        RASPA_data, frameworks=[fw], molecules=[mol], temperatures=[temp],
                        only_pure_adsorption=True,
                    )
                    for d in data:
                        try:
                            data_rows.append({
                                'framework': fw,
                                'molecule': mol,
                                'temperature': float(temp),
                                'pressure': float(d.get('pressure')),
                                'loading': float(d.get('loading')),
                            })
                        except Exception:
                            continue

    # Fit-curve rows will also be collected in a scale-independent way.
    fit_rows = []

    def _plot_for_scale(scale_name, suffix):
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
        vary_T = len(selected_temperatures or []) > 1
        vary_mol = len(selected_molecules or []) > 1
        vary_fw = len(selected_frameworks or []) > 1
        linestyle_mode = phelp.choose_isotherm_fitting_proxy_linestyle_mode(
            selected_frameworks, selected_fit_types, selected_temperatures
        )

        _data_src_iso = str(getattr(Input, "config", {}).get("data_source", "fitting")).lower()
        _raspa_marker_axis = None
        if _data_src_iso == "points" and plot_RASPA:
            _raspa_marker_axis = phelp.isotherm_raspa_points_marker_axis(
                len(selected_frameworks or []), len(selected_molecules or []))

        # Reference series used only for proxy legend colors (temperature) and labels.
        ref_fw_for_temp = selected_frameworks[0] if selected_frameworks else None
        ref_mol_for_temp = selected_molecules[0] if selected_molecules else None
        ref_temp_for_proxy = selected_temperatures[0] if selected_temperatures else None

        for fw in selected_frameworks:
            for mol in selected_molecules:
                for temp in selected_temperatures:
                    combo = (fw, mol, temp)
                    style = phelp.resolve_series_style(
                        fw, mol, temp,
                        vary_fw=vary_fw, vary_mol=vary_mol, vary_T=vary_T,
                        plot_kind="isotherm",
                        combo_colors=combo_colors,
                    )
                    color = style.get("color") or "gray"
                    # use centralized filter helper with tolerant temperature matching
                    data = phelp.filter_raspa_data(
                        RASPA_data, frameworks=[fw], molecules=[mol], temperatures=[temp],
                        only_pure_adsorption=True,
                    )

                    # Set x-limits
                    if p_min is not None or p_max is not None:
                        ax.set_xlim(left=p_min, right=p_max)
                    else:
                        p_bounds = init.get_pressure_bounds(data)
                        if p_bounds[0] is not None and p_bounds[1] is not None:
                            ax.set_xlim(left=p_bounds[0], right=p_bounds[1])

                    for ft in selected_fit_types:
                        # Iterate over all matching cache entries (handles both single and multiple parameter counts)
                        fit_idx = 0
                        for params, ft_type, num_params in phelp.iter_cache_entries(fit_cache, fw, mol, temp, ft):
                            fit_idx += 1
                            p_bounds = init.get_pressure_bounds(data)
                            if p_bounds[0] is not None and p_bounds[1] is not None:
                                # take x_fit points inside bounds
                                xs = sorted([float(p) for p in x_fit if p_bounds[0] <= float(p) <= p_bounds[1]])
                                # include endpoints and measured pressures so the curve is
                                # evaluated at every observed pressure (avoids missing last point)
                                obs_pressures = sorted({float(p['pressure']) for p in data if p.get('pressure') is not None and p_bounds[0] <= float(p['pressure']) <= p_bounds[1]})
                                # include endpoints if not present
                                if not xs:
                                    xs = [float(p_bounds[0]), float(p_bounds[1])] + obs_pressures
                                else:
                                    if xs[0] > float(p_bounds[0]):
                                        xs.insert(0, float(p_bounds[0]))
                                    if xs[-1] < float(p_bounds[1]):
                                        xs.append(float(p_bounds[1]))
                                    # extend with observed pressures
                                    if obs_pressures:
                                        xs = list(xs) + list(obs_pressures)
                                x_plot = np.array(sorted(set(xs)), dtype=float)
                            else:
                                x_plot = np.array(x_fit, dtype=float)

                            # If plotting on a log x-scale, drop any non-positive x values
                            if scale_name == 'log':
                                try:
                                    x_plot = x_plot[x_plot > 0.0]
                                except Exception:
                                    x_plot = np.array([v for v in x_plot if float(v) > 0.0], dtype=float)
                                if x_plot.size == 0:
                                    print(f"Skipping log-scale plotting for {fw},{mol},{temp}: no strictly positive pressures available in x_plot")
                                    continue

                            # Suppress per-curve legend entries; we will draw a single proxy legend below.
                            label = "_nolegend_"

                            # Choose linestyle system based on what varies.
                            # - If structures vary and fit types do not: use structure linestyles.
                            # - If fit types vary and structures do not: use fit-type linestyles.
                            if linestyle_mode == "structure":
                                linestyle = style.get("linestyle") or "-"
                            elif linestyle_mode == "fit_type":
                                linestyle = phelp.get_linestyle_for_fit_type(ft_type)
                            else:
                                linestyle = "-"

                            # Evaluate fit for saving as well as plotting
                            y_plot = phelp.evaluate_fit(x_plot, params, ft_type)
                            phelp.plot_fit_curve(x_plot, params, ft_type, label, color, linestyle=linestyle, ax=ax)

                            # Record fitted curve points for optional data export
                            for xp, yp in zip(np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)):
                                if not np.isfinite(xp) or not np.isfinite(yp):
                                    continue
                                fit_rows.append({
                                    'framework': fw,
                                    'molecule': mol,
                                    'temperature': float(temp),
                                    'pressure': float(xp),
                                    'loading': float(yp),
                                })
                    if plot_RASPA:
                        # Draw markers after fitted curves to guarantee they stay on top.
                        marker = style.get("marker") or None
                        if _raspa_marker_axis == "framework":
                            marker = phelp.get_marker_for_material(fw)
                        elif _raspa_marker_axis == "molecule":
                            marker = phelp.get_marker_for_molecule(mol)
                        phelp.plot_raspa_points(
                            data, fw, mol, temp, color,
                            show_in_legend=False, marker=marker, zorder=50, ax=ax
                        )

        # Set x-axis scale: logarithmic for 'log', linear for 'linear'
        if scale_name == 'log':
            ax.set_xscale("log")
            title_scale = "(log x-scale)"
            # Explicitly set sensible log limits to avoid Matplotlib auto-scaling
            # edge cases when values span many decades.
            try:
                x_left = float(p_min) if p_min is not None else float(np.nanmin(np.asarray(x_fit, dtype=float)))
                x_right = float(p_max) if p_max is not None else float(np.nanmax(np.asarray(x_fit, dtype=float)))
                x_left = max(x_left, 1e-12)
                if np.isfinite(x_left) and np.isfinite(x_right) and x_right > x_left:
                    ax.set_xlim(left=x_left, right=x_right)
            except Exception:
                pass
        else:
            ax.set_xscale("linear")
            title_scale = "(linear x-scale)"

        ax.set_xlabel("Pressure [Pa]", fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
        ax.set_ylabel("Loading [mol/kg]", fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
        ax.grid(True, which="both", ls="--", alpha=phelp.ALPHA)
        phelp.set_axis_limits_nice(ax)
        # Set axes to start at 0 (only x-axis for linear scale, y-axis for both)
        if scale_name == 'linear':
            ax.set_xlim(left=0)  # Only set x-axis to 0 for linear scale (log scale can't start at 0)
        ax.set_ylim(bottom=0)  # Set y-axis to start at 0 for both scales

        phelp.build_isotherm_fitting_proxy_legend(
            ax,
            selected_temperatures,
            selected_frameworks,
            selected_molecules,
            selected_fit_types,
            combo_colors,
            ref_fw_for_temp=ref_fw_for_temp,
            ref_mol_for_temp=ref_mol_for_temp,
            ref_temp_for_linestyle=ref_temp_for_proxy,
            vary_mol=vary_mol,
            fontsize=phelp.AXIS_LEGEND_SIZE,
            loc='best',
        )

        phelp.apply_unified_axes_layout(fig, ax)
        out_path = phelp._save_plot(
            f'isotherm_{suffix}',
            'plot_isotherm_fitting',
            selected_frameworks,
            selected_molecules,
            selected_temperatures,
            fig=fig,
        )

        #
        # When DATA_SOURCE='points', this export captures the underlying RASPA points
        # (and any fitted-curve samples included in the plot).
        data_source = str(getattr(Input, "config", {}).get("data_source", "fitting")).lower()
        all_rows = data_rows + fit_rows
        if all_rows and save_data and data_source != "fitting":
            try:
                base = Path(out_path)
                # Create a 'saved' subfolder under the Basic_Data folder (or
                # wherever the PNG lives) and store the text file there.
                saved_dir = base.parent / "saved"
                saved_dir.mkdir(parents=True, exist_ok=True)
                data_path = saved_dir / (base.stem + '_data.txt')
                with data_path.open('w', encoding='utf-8') as f:
                    # Header line (minimal columns: framework, molecule, T, P, loading)
                    f.write(
                        "framework\tmolecule\ttemperature_K\tpressure_Pa\tloading_mol_per_kg\n"
                    )
                    for r in all_rows:
                        f.write(
                            f"{r['framework']}\t{r['molecule']}\t{r['temperature']}\t"
                            f"{r['pressure']}\t{r['loading']}\n"
                        )
            except Exception as e:
                print(f"Warning: failed to write isotherm data text file next to {out_path}: {e}")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Optional: when multiple molecules are selected, also generate
        # molecule-separated PNGs in the *same* run folder, but do NOT
        # write any extra export text files.
        # ------------------------------------------------------------------
        if len(selected_molecules or []) > 1:
            for mol_single in selected_molecules:
                fig_m, ax_m = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
                vary_T_m = len(selected_temperatures or []) > 1
                vary_mol_m = False  # exactly one molecule per separated plot
                vary_fw_m = len(selected_frameworks or []) > 1
                _split_src = str(getattr(Input, "config", {}).get("data_source", "fitting")).lower()
                _raspa_axis_m = None
                if _split_src == "points" and plot_RASPA:
                    _raspa_axis_m = phelp.isotherm_raspa_points_marker_axis(
                        len(selected_frameworks or []), 1)

                for fw in selected_frameworks:
                    for temp in selected_temperatures:
                        style_m = phelp.resolve_series_style(
                            fw, mol_single, temp,
                            vary_fw=vary_fw_m, vary_mol=vary_mol_m, vary_T=vary_T_m,
                            plot_kind="isotherm",
                            combo_colors=combo_colors,
                        )
                        color_m = style_m.get("color") or "gray"

                        # use centralized filter helper with tolerant temperature matching
                        data_m = phelp.filter_raspa_data(
                            RASPA_data, frameworks=[fw], molecules=[mol_single], temperatures=[temp]
                        )

                        # Set x-limits
                        if p_min is not None or p_max is not None:
                            ax_m.set_xlim(left=p_min, right=p_max)
                        else:
                            p_bounds_m = init.get_pressure_bounds(data_m)
                            if p_bounds_m[0] is not None and p_bounds_m[1] is not None:
                                ax_m.set_xlim(left=p_bounds_m[0], right=p_bounds_m[1])

                        for ft in selected_fit_types:
                            fit_idx = 0
                            for params, ft_type, num_params in phelp.iter_cache_entries(
                                fit_cache, fw, mol_single, temp, ft
                            ):
                                fit_idx += 1
                                p_bounds_m = init.get_pressure_bounds(data_m)
                                if p_bounds_m[0] is not None and p_bounds_m[1] is not None:
                                    xs = sorted([float(p) for p in x_fit if p_bounds_m[0] <= float(p) <= p_bounds_m[1]])
                                    obs_pressures = sorted({
                                        float(p['pressure'])
                                        for p in data_m
                                        if p.get('pressure') is not None and p_bounds_m[0] <= float(p['pressure']) <= p_bounds_m[1]
                                    })
                                    if not xs:
                                        xs = [float(p_bounds_m[0]), float(p_bounds_m[1])] + obs_pressures
                                    else:
                                        if xs[0] > float(p_bounds_m[0]):
                                            xs.insert(0, float(p_bounds_m[0]))
                                        if xs[-1] < float(p_bounds_m[1]):
                                            xs.append(float(p_bounds_m[1]))
                                        if obs_pressures:
                                            xs = list(xs) + list(obs_pressures)
                                    x_plot = np.array(sorted(set(xs)), dtype=float)
                                else:
                                    x_plot = np.array(x_fit, dtype=float)

                                if scale_name == 'log':
                                    try:
                                        x_plot = x_plot[x_plot > 0.0]
                                    except Exception:
                                        x_plot = np.array([v for v in x_plot if float(v) > 0.0], dtype=float)
                                    if x_plot.size == 0:
                                        continue

                                # Suppress per-curve legend entries; we will draw a single proxy legend below.
                                label_m = "_nolegend_"

                                if linestyle_mode == "structure":
                                    linestyle_m = style_m.get("linestyle") or "-"
                                elif linestyle_mode == "fit_type":
                                    linestyle_m = phelp.get_linestyle_for_fit_type(ft_type)
                                else:
                                    linestyle_m = "-"

                                y_plot_m = phelp.evaluate_fit(x_plot, params, ft_type)
                                phelp.plot_fit_curve(
                                    x_plot, params, ft_type,
                                    label_m, color_m, linestyle=linestyle_m, ax=ax_m
                                )
                        if plot_RASPA:
                            # Draw markers after fitted curves to guarantee they stay on top.
                            marker_m = style_m.get("marker") or None
                            if _raspa_axis_m == "framework":
                                marker_m = phelp.get_marker_for_material(fw)
                            elif _raspa_axis_m == "molecule":
                                marker_m = phelp.get_marker_for_molecule(mol_single)
                            phelp.plot_raspa_points(
                                data_m, fw, mol_single, temp, color_m,
                                show_in_legend=False, marker=marker_m, zorder=50, ax=ax_m
                            )

                # Set x-axis scale
                if scale_name == 'log':
                    ax_m.set_xscale("log")
                    try:
                        x_left_m = float(p_min) if p_min is not None else float(np.nanmin(np.asarray(x_fit, dtype=float)))
                        x_right_m = float(p_max) if p_max is not None else float(np.nanmax(np.asarray(x_fit, dtype=float)))
                        x_left_m = max(x_left_m, 1e-12)
                        if np.isfinite(x_left_m) and np.isfinite(x_right_m) and x_right_m > x_left_m:
                            ax_m.set_xlim(left=x_left_m, right=x_right_m)
                    except Exception:
                        pass
                else:
                    ax_m.set_xscale("linear")

                ax_m.set_xlabel("Pressure [Pa]", fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
                ax_m.set_ylabel("Loading [mol/kg]", fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
                ax_m.grid(True, which="both", ls="--", alpha=phelp.ALPHA)
                phelp.set_axis_limits_nice(ax_m)
                if scale_name == 'linear':
                    ax_m.set_xlim(left=0)
                ax_m.set_ylim(bottom=0)

                phelp.build_isotherm_fitting_proxy_legend(
                    ax_m,
                    selected_temperatures,
                    selected_frameworks,
                    selected_molecules,
                    selected_fit_types,
                    combo_colors,
                    ref_fw_for_temp=ref_fw_for_temp,
                    ref_mol_for_temp=mol_single,
                    ref_temp_for_linestyle=ref_temp_for_proxy,
                    vary_mol=False,
                    fontsize=phelp.AXIS_LEGEND_SIZE,
                    loc='best',
                )

                phelp.apply_unified_axes_layout(fig_m, ax_m)

                # Save to the same run folder as the combined plot, but avoid
                # overwriting by encoding the single molecule in the prefix.
                mol_prefix = str(mol_single).replace(" ", "_")
                phelp._save_plot(
                    f'isotherm_{suffix}_{mol_prefix}',
                    'plot_isotherm_fitting',
                    selected_frameworks,
                    selected_molecules,      # keep original list to preserve folder
                    selected_temperatures,
                    fig=fig_m,
                )
                plt.close(fig_m)

    scale_flag = str(pressure_scale).strip().lower()
    if scale_flag in ('log', 'both'):
        _plot_for_scale('log', 'log')
    if scale_flag in ('linear', 'both'):
        _plot_for_scale('linear', 'linear')



def synthesize_points_from_fittings(fittings, selected_frameworks, selected_molecules,
                                    selected_temperatures, selected_fit_types,
                                    n_loadings=20, p_min=None, p_max=None, p_grid=None,
                                    formula_fit_types=None, num_of_isotherm=None,
                                    pressure_scale='log', save_data=False,
                                    folder_molecule_label=None):
    """Build synthetic (pressure, loading) points from fittings. Returns list of dicts with
    framework, molecule, temperature, pressure, loading for use as RASPA-like data."""
    synth = []
    formula_fit_types = formula_fit_types or selected_fit_types

    if p_grid is not None:
        p_vals = np.asarray(p_grid, dtype=float)
    else:
        if p_min is None or p_max is None:
            raise ValueError("p_min and p_max required when p_grid is not provided")
        p_min = max(1.0, float(p_min))
        p_max = float(p_max)
        if p_max <= p_min:
            raise ValueError("p_max must be greater than p_min")
        scale_flag = str(pressure_scale).strip().lower()
        if scale_flag == 'linear':
            p_vals = np.linspace(p_min, p_max, int(n_loadings))
        else:
            # default and 'log': logarithmic spacing in pressure
            p_vals = np.logspace(np.log10(p_min), np.log10(p_max), int(n_loadings))

    fit_cache = phelp.build_fit_cache(
        fittings, selected_frameworks, selected_molecules, selected_temperatures,
        selected_fit_types, num_of_isotherm=num_of_isotherm)

    for fw in selected_frameworks:
        for mol in selected_molecules:
            for temp in selected_temperatures:
                for ft in selected_fit_types:
                    for params, ft_type, _ in phelp.iter_cache_entries(fit_cache, fw, mol, temp, ft):
                        if ft_type not in formula_fit_types:
                            continue
                        try:
                            q_vals = phelp.evaluate_fit(p_vals, params, ft_type)
                        except Exception:
                            continue
                        for p_val, q_val in zip(p_vals, np.asarray(q_vals, dtype=float)):
                            if not np.isnan(q_val):
                                synth.append({
                                    'framework': fw,
                                    'molecule': mol,
                                    'temperature': float(temp),
                                    'pressure': float(p_val),
                                    'loading': float(q_val),
                                })

    # Optional: when requested by caller (Main), also save these synthetic points to a
    # text file with the same column format as the isotherm data exports.
    if synth and save_data:
        # Allow caller to override the folder molecule label so that, for
        # mixtures, synthetic pure-component points can still be stored
        # under the mixture run folder (e.g. R407F) instead of a combined
        # name like R125-R134a-R32.
        fw_part = _safe_join(selected_frameworks)
        if folder_molecule_label is not None:
            mol_part = str(folder_molecule_label).replace(" ", "_")
        else:
            mol_part = _safe_join(selected_molecules)
        temp_part = _safe_join(selected_temperatures)

        prefix = f"fit_isotherm_{pressure_scale}"
        _save_isotherm_rows_to_run_folder(synth, fw_part, mol_part, temp_part, prefix)

    return synth

def plot_mixture_isotherms(selected_frameworks, selected_molecules, selected_temperatures,
                           mixture_data, combo_colors,
                           p_min=None, p_max=None, out_dir=None,
                           pressure_scale='both', show_points=True, save_data=False):
    """
    Plot per-component mixture isotherms: one figure per component, all selected temperatures.

    Also writes **per-temperature overlay** figures: for each framework and each
    temperature, one axes with every component isotherm plus the summed overall loading.
    Component colours match :func:`MolFraction.plot_mol_fraction_vs_pressure`
    (``get_color_for_molecule`` per component; **solid** lines on these axes). Mixture total uses
    ``get_color_for_molecule(mixture_name)``. Files: ``mixture_isotherm_per_T_*.png``.
    """
    if not mixture_data:
        print("plot_mixture_isotherms: no mixture data provided, skipping.")
        return

    components = sorted({d['molecule'] for d in mixture_data})
    mixture_name = selected_molecules[0] if selected_molecules else 'mixture'

    _mix_data_src = str(getattr(Input, "config", {}).get("data_source", "fitting")).lower()
    _mix_raspa_axis = None
    if _mix_data_src == "points" and show_points:
        _mix_raspa_axis = phelp.isotherm_raspa_points_marker_axis(
            len(selected_frameworks or []), len(components))

    def _mix_pt_marker(fw, comp):
        if _mix_raspa_axis == "framework":
            return phelp.get_marker_for_material(fw)
        if _mix_raspa_axis == "molecule":
            return phelp.get_marker_for_molecule(comp)
        return phelp.get_marker_for_molecule(comp)

    # Collect export rows per scale outside the plotting helper so that file
    # writing happens once, after all figures are generated.
    export_rows_log = []
    export_rows_linear = []

    def _plot_per_temperature_overlay(scale_name, suffix):
        """One figure per (framework, temperature): all components + total loading vs pressure."""
        # Same component colour dict as MolFraction.plot_mol_fraction_vs_pressure
        comp_colors = {c: (phelp.get_color_for_molecule(c) or 'C0') for c in components}

        for fw in selected_frameworks:
            for temp in selected_temperatures:
                fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
                any_line = False
                sum_by_pressure = {}

                for comp in components:
                    pts = phelp.filter_raspa_data(
                        mixture_data, frameworks=[fw], molecules=[comp], temperatures=[temp]
                    )
                    if not pts:
                        continue

                    p_arr = np.array([d['pressure'] for d in pts], dtype=float)
                    q_arr = np.array([d['loading'] for d in pts], dtype=float)
                    if scale_name == 'log':
                        mask = p_arr > 0
                        p_arr, q_arr = p_arr[mask], q_arr[mask]
                    if p_arr.size == 0:
                        continue

                    color = comp_colors[comp]
                    marker = _mix_pt_marker(fw, comp)
                    comp_display = phelp.get_molecule_display_name(comp)
                    label = comp_display

                    order = np.argsort(p_arr)
                    p_sorted = p_arr[order]
                    q_sorted = q_arr[order]

                    if show_points:
                        ax.plot(
                            p_sorted, q_sorted,
                            color=color, linestyle='-', lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label=label,
                        )
                        ax.scatter(
                            p_arr, q_arr,
                            color=color, marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                            label='_nolegend_',
                        )
                    else:
                        ax.plot(
                            p_sorted, q_sorted,
                            color=color, linestyle='-', lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label=label,
                        )

                    for P, q in zip(p_arr, q_arr):
                        if not np.isfinite(P) or not np.isfinite(q):
                            continue
                        if scale_name == 'log' and P <= 0:
                            continue
                        sum_by_pressure[float(P)] = sum_by_pressure.get(float(P), 0.0) + float(q)

                    any_line = True

                if sum_by_pressure:
                    p_tot = np.array(sorted(sum_by_pressure.keys()), dtype=float)
                    q_tot = np.array([sum_by_pressure[P] for P in p_tot], dtype=float)
                    if scale_name == 'log':
                        mask = p_tot > 0
                        p_tot, q_tot = p_tot[mask], q_tot[mask]
                    if p_tot.size > 0:
                        mix = selected_molecules[0] if selected_molecules else None
                        mix_color = phelp.get_color_for_molecule(mixture_name)
                        if not mix_color and mix is not None:
                            mix_color = combo_colors.get((fw, mix, temp)) or combo_colors.get(
                                (fw, mix, float(temp))
                            )
                        if not mix_color:
                            mix_color = 'C0'
                        mix_marker = _mix_pt_marker(fw, mixture_name)
                        order_tot = np.argsort(p_tot)
                        p_tot_s = p_tot[order_tot]
                        q_tot_s = q_tot[order_tot]
                        mix_display = phelp.get_molecule_display_name(mixture_name)
                        mix_label = mix_display
                        ax.plot(
                            p_tot_s, q_tot_s,
                            color=mix_color, linestyle='-', lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, zorder=5,
                            label=mix_label,
                        )
                        if show_points:
                            ax.scatter(
                                p_tot, q_tot,
                                color=mix_color, marker=mix_marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                                zorder=10, edgecolors='none',
                                label='_nolegend_',
                            )
                        any_line = True

                if not any_line:
                    plt.close(fig)
                    continue

                if scale_name == 'log':
                    ax.set_xscale('log')
                else:
                    ax.set_xscale('linear')
                    ax.set_xlim(left=0)

                if p_min is not None:
                    ax.set_xlim(left=p_min)
                if p_max is not None:
                    ax.set_xlim(right=p_max)
                ax.set_ylim(bottom=0)
                ax.set_xlabel('Pressure [Pa]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
                ax.set_ylabel('Loading [mol/kg]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
                t_lab = int(round(float(temp)))
                ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
                _h_leg, _lab_leg = ax.get_legend_handles_labels()
                if _lab_leg:
                    ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
                phelp.apply_unified_axes_layout(fig, ax)

                safe_fw = str(fw).replace(' ', '_')
                phelp._save_plot(
                    'mixture_isotherm_per_T',
                    'Basic_data',
                    selected_frameworks, selected_molecules, selected_temperatures,
                    fig=fig, out_dir=out_dir,
                    filename_suffix=f'{safe_fw}_{t_lab}K_{suffix}',
                )
                plt.close(fig)

    def _plot_for_scale(scale_name, suffix):
        fw_first = selected_frameworks[0] if selected_frameworks else None
        for comp in components:
            fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
            n_fw_leg = len(selected_frameworks or [])
            n_T_leg = len(selected_temperatures or [])
            mix_lm = selected_molecules[0] if selected_molecules else None
            ref_fw_leg = selected_frameworks[0] if selected_frameworks else None
            ref_temp_leg = selected_temperatures[0] if selected_temperatures else None
            want_temp_fw_proxy = (n_T_leg > 1) or (n_fw_leg > 1)
            use_proxy_legend = want_temp_fw_proxy

            for fw in selected_frameworks:
                for temp in selected_temperatures:
                    pts = phelp.filter_raspa_data(
                        mixture_data, frameworks=[fw], molecules=[comp], temperatures=[temp]
                    )
                    if not pts:
                        continue

                    p_arr = np.array([d['pressure'] for d in pts], dtype=float)
                    q_arr = np.array([d['loading'] for d in pts], dtype=float)
                    if scale_name == 'log':
                        mask = p_arr > 0
                        p_arr, q_arr = p_arr[mask], q_arr[mask]
                    if p_arr.size == 0:
                        continue

                    # Reuse the existing combo_colors map built in Main.
                    # For mixtures the "molecule" key is the mixture name.
                    mix = selected_molecules[0] if selected_molecules else None
                    color = None
                    if mix is not None:
                        color = (
                            combo_colors.get((fw, mix, temp))
                            or combo_colors.get((fw, mix, float(temp)))
                            or phelp.get_combo_color_temperature(combo_colors, fw, mix, temp)
                        )
                    marker = _mix_pt_marker(fw, comp)
                    # Legend: component is fixed within this figure, temperatures vary
                    vary_fw = len(selected_frameworks or []) > 1
                    vary_mol = False  # this figure is for a single component
                    vary_T = len(selected_temperatures or []) > 1
                    style = phelp.resolve_series_style(
                        fw, comp, temp,
                        vary_fw=vary_fw, vary_mol=vary_mol, vary_T=vary_T,
                        plot_kind="isotherm",
                        combo_colors=combo_colors,
                    )
                    if use_proxy_legend:
                        label = "_nolegend_"
                    elif fw == fw_first:
                        label = phelp.build_series_label(
                            fw, comp, temp,
                            vary_fw=False, vary_mol=False, vary_T=vary_T,
                        )
                    else:
                        label = "_nolegend_"

                    if show_points:
                        order = np.argsort(p_arr)
                        p_sorted = p_arr[order]
                        q_sorted = q_arr[order]
                        ax.plot(
                            p_sorted, q_sorted,
                            color=color, linestyle=style.get("linestyle", "-"), lw=phelp.LINEWIDTH, label=label
                        )
                        ax.scatter(
                            p_arr, q_arr,
                            color=color, marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA, label="_nolegend_"
                        )
                    else:
                        # Plot a line through the points without markers
                        order = np.argsort(p_arr)
                        p_sorted = p_arr[order]
                        q_sorted = q_arr[order]
                        ax.plot(
                            p_sorted, q_sorted,
                            color=color, linestyle=style.get("linestyle", "-"), lw=phelp.LINEWIDTH, label=label
                        )
                    # Collect rows for optional data export, one per point.
                    target_rows = export_rows_log if scale_name == 'log' else export_rows_linear
                    for P, q in zip(p_arr, q_arr):
                        if not np.isfinite(P) or not np.isfinite(q):
                            continue
                        target_rows.append({
                            'framework': fw,
                            'molecule': comp,
                            'temperature': float(temp),
                            'pressure': float(P),
                            'loading': float(q),
                        })

            if scale_name == 'log':
                ax.set_xscale('log')
                # title_scale = '(log x-scale)'
            else:
                ax.set_xscale('linear')
                ax.set_xlim(left=0)
                title_scale = '(linear x-scale)'

            if p_min is not None:
                ax.set_xlim(left=p_min)
            if p_max is not None:
                ax.set_xlim(right=p_max)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Pressure [Pa]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
            ax.set_ylabel('Loading [mol/kg]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
            ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
            if use_proxy_legend:
                proxy_handles = []
                proxy_labels = []
                if ref_fw_leg is not None and mix_lm is not None:
                    if n_T_leg > 1:
                        for temp in selected_temperatures or []:
                            col_t = phelp.get_combo_color_temperature(
                                combo_colors, ref_fw_leg, mix_lm, temp
                            ) or 'gray'
                            proxy_handles.append(
                                Line2D(
                                    [0], [0],
                                    color=col_t,
                                    linestyle='-',
                                    lw=phelp.LINEWIDTH,
                                    marker=None,
                                )
                            )
                            try:
                                proxy_labels.append(f"{int(round(float(temp)))}K")
                            except Exception:
                                proxy_labels.append(f"{temp}K")
                if proxy_handles:
                    ax.legend(
                        proxy_handles,
                        proxy_labels,
                        fontsize=phelp.AXIS_LEGEND_SIZE,
                        loc='best',
                    )
                else:
                    # Multiple frameworks at one temperature: no temperature proxy rows —
                    # use line handles from the axes (frameworks encoded by linestyle on curves).
                    _h_c, _lab_c = ax.get_legend_handles_labels()
                    if _lab_c:
                        if show_points:
                            _h_f = [h for h in _h_c if isinstance(h, Line2D)]
                            _lab_f = [lb for h, lb in zip(_h_c, _lab_c) if isinstance(h, Line2D)]
                            if _h_f:
                                ax.legend(_h_f, _lab_f, fontsize=phelp.AXIS_LEGEND_SIZE)
                            else:
                                ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE)
                        else:
                            ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE)
            else:
                _h_c, _lab_c = ax.get_legend_handles_labels()
                if _lab_c:
                    if show_points:
                        _h_f = [h for h in _h_c if isinstance(h, Line2D)]
                        _lab_f = [lb for h, lb in zip(_h_c, _lab_c) if isinstance(h, Line2D)]
                        if _h_f:
                            ax.legend(_h_f, _lab_f, fontsize=phelp.AXIS_LEGEND_SIZE)
                        else:
                            ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE)
                    else:
                        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE)
            phelp.apply_unified_axes_layout(fig, ax)
            # Use the overall mixture name in the folder path so all component
            # figures live under the same mixture directory; distinguish components
            # only via the filename
            out_path = phelp._save_plot(
                f'mixture_isotherm_{comp}_{suffix}',
                'Basic_data',
                selected_frameworks, selected_molecules, selected_temperatures,
                fig=fig, out_dir=out_dir,
            )
            plt.close(fig)

        # ---- Total (summation) curve over all components ----
        fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)

        for fw in selected_frameworks:
            for temp in selected_temperatures:
                pq = phelp.mixture_total_pq_tuples(
                    mixture_data, fw, float(temp), components,
                    p_min=p_min, p_max=p_max,
                    require_positive_pressure=(scale_name == 'log'),
                )
                if not pq:
                    continue
                p_arr = np.array([x[0] for x in pq], dtype=float)
                q_arr = np.array([x[1] for x in pq], dtype=float)
                target_rows = export_rows_log if scale_name == 'log' else export_rows_linear
                for P, q in pq:
                    if not np.isfinite(P) or not np.isfinite(q):
                        continue
                    target_rows.append({
                        'framework': fw,
                        'molecule': mixture_name,
                        'temperature': float(temp),
                        'pressure': float(P),
                        'loading': float(q),
                    })

                # Reuse the existing combo_colors map built in Main.
                # For mixtures the "molecule" key is the mixture name.
                mix = selected_molecules[0] if selected_molecules else None
                color = None
                if mix is not None:
                    color = (
                        combo_colors.get((fw, mix, temp))
                        or combo_colors.get((fw, mix, float(temp)))
                        or phelp.get_combo_color_temperature(combo_colors, fw, mix, temp)
                    )

                marker = _mix_pt_marker(fw, mixture_name)

                vary_fw = len(selected_frameworks or []) > 1
                vary_mol = False  # this figure is for a single total curve
                vary_T = len(selected_temperatures or []) > 1
                style = phelp.resolve_series_style(
                    fw, mixture_name, temp,
                    vary_fw=vary_fw, vary_mol=vary_mol, vary_T=vary_T,
                    plot_kind="isotherm",
                    combo_colors=combo_colors,
                )
                if fw == fw_first:
                    label = phelp.build_series_label(
                        fw, mixture_name, temp,
                        vary_fw=False, vary_mol=False, vary_T=vary_T,
                    )
                else:
                    label = "_nolegend_"

                if show_points:
                    ax.plot(
                        p_arr, q_arr,
                        color=color, linestyle=style.get("linestyle", "-"), lw=phelp.LINEWIDTH, label=label
                    )
                    ax.scatter(
                        p_arr, q_arr,
                        color=color, marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA, label="_nolegend_"
                    )
                else:
                    ax.plot(
                        p_arr, q_arr,
                        color=color, linestyle=style.get("linestyle", "-"), lw=phelp.LINEWIDTH, label=label
                    )

        if scale_name == 'log':
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
            ax.set_xlim(left=0)

        if p_min is not None:
            ax.set_xlim(left=p_min)
        if p_max is not None:
            ax.set_xlim(right=p_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Pressure [Pa]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
        ax.set_ylabel('Loading [mol/kg]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
        ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE)
        phelp.apply_unified_axes_layout(fig, ax)

        out_path = phelp._save_plot(
            f'mixture_isotherm_total_{suffix}',
            'Basic_data',
            selected_frameworks, selected_molecules, selected_temperatures,
            fig=fig, out_dir=out_dir,
        )
        plt.close(fig)

    # Generate plots according to the configured pressure_scale, but only write
    # data files once afterwards (outside the plotting helper).
    ps = (str(pressure_scale) or 'both').strip().lower()
    if ps in ('log', 'both'):
        _plot_for_scale('log', 'log')
    if ps in ('linear', 'both'):
        _plot_for_scale('linear', 'linear')

    # Per temperature: all components + total on one axes (same scales as above).
    if ps in ('log', 'both'):
        _plot_per_temperature_overlay('log', 'log')
    if ps in ('linear', 'both'):
        _plot_per_temperature_overlay('linear', 'linear')

    # Save underlying points next to the PNGs when requested, using the same
    # column format as the other isotherm data exports.
    if save_data and out_dir is None:
        fw_part = _safe_join(selected_frameworks)
        mix_part = _safe_join(selected_molecules)
        temp_part = _safe_join(selected_temperatures)

        # One file for log-scale data (if any)
        if export_rows_log:
            _save_isotherm_rows_to_run_folder(
                export_rows_log, fw_part, mix_part, temp_part,
                prefix="mixture_isotherm_log"
            )

        # One file for linear-scale data (if any)
        if export_rows_linear:
            _save_isotherm_rows_to_run_folder(
                export_rows_linear, fw_part, mix_part, temp_part,
                prefix="mixture_isotherm_linear"
            )
    # Figures are already closed per-plot; avoid plt.show()/close('all') here (Agg backend,
    # and close('all' can confuse viewers / downstream plotting in the same process).
