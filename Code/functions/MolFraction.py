from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import PlotHelpers as phelp
import Initialize as init


def _safe_join(lst):
    return "-".join([str(x).replace(" ", "_") for x in lst]) if lst else "all"


def _save_mol_fraction_rows_to_run_folder(rows, fw_part, mix_part, temp_part, prefix):
    """Save mol-fraction rows to Output/<run>/Basic_Data/saved.

    Columns: framework, molecule, temperature_K, pressure_Pa, mole_fraction
    """
    if not rows:
        return
    try:
        base_dir = init.get_pipeline_run_root()
        plots_root = base_dir / 'Output'
        run_folder_name = f"{fw_part}_{mix_part}_{temp_part}"
        saved_dir = plots_root / run_folder_name / 'Basic_Data' / 'saved'
        saved_dir.mkdir(parents=True, exist_ok=True)

        data_path = saved_dir / f"{prefix}.txt"
        with data_path.open('w', encoding='utf-8') as f:
            f.write("framework\tmolecule\ttemperature_K\tpressure_Pa\tmole_fraction\n")
            for r in rows:
                f.write(
                    f"{r['framework']}\t{r['molecule']}\t{r['temperature']}\t"
                    f"{r['pressure']}\t{r['mole_fraction']}\n"
                )
    except Exception as e:
        print(f"Warning: failed to write mol-fraction data file: {e}")


def plot_mol_fraction_vs_pressure(mixture_data, selected_frameworks, selected_molecules,
                                   selected_temperatures, combo_colors,
                                   p_min=None, p_max=None, out_dir=None,
                                   scale='both', show_points=True, save_data=False):
    """
    Plot adsorbed-phase mol-fraction x_i = q_i / q_total vs pressure.
    One figure per temperature (both log and linear x-scale).
    Components and total loading are derived from the RASPA mixture data.

    Parameters
    - mixture_data: list of RASPA-like dicts already filtered to this mixture
    - selected_frameworks / selected_temperatures: filter lists from selection
    - selected_molecules: list with the mixture name (e.g. ['R407F']); used for folder/title
    - combo_colors: not used for component coloring (each component gets a fixed color),
      but passed for API consistency
    - p_min / p_max: optional axis limits
    """
    if not mixture_data:
        print("plot_mol_fraction_vs_pressure: no mixture data provided, skipping.")
        return

    components = sorted({d['molecule'] for d in mixture_data})
    n_comp = len(components)
    if n_comp < 2:
        print("plot_mol_fraction_vs_pressure: need at least 2 components, skipping.")
        return

    mixture_name = selected_molecules[0] if selected_molecules else 'mixture'
    # Use global molecule color mapping / fallback cycle for components
    comp_colors = {c: (phelp.get_color_for_molecule(c) or 'C0') for c in components}

    # Collect rows per scale for optional export
    export_rows_log = []
    export_rows_linear = []

    def _plot_for_scale(scale_name, suffix):
        fw_first = selected_frameworks[0] if selected_frameworks else None
        n_fw_leg = len(selected_frameworks or [])
        for temp in selected_temperatures:
            fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
            any_plotted = False

            for fw in selected_frameworks:
                # Build {pressure: {comp: loading}} for this (fw, temp)
                pressure_data = defaultdict(dict)
                for comp in components:
                    pts = phelp.filter_raspa_data(
                        mixture_data, frameworks=[fw], molecules=[comp], temperatures=[temp]
                    )
                    for pt in pts:
                        pressure_data[float(pt['pressure'])][comp] = float(pt['loading'])

                # Keep only pressure points where every component has a value
                shared_ps = sorted(
                    p for p, d in pressure_data.items() if len(d) == n_comp
                )
                if len(shared_ps) < 2:
                    print(f"mol_fraction: not enough shared pressure points for "
                          f"{fw}, T={int(temp)}K — skipping.")
                    continue

                p_arr = np.array(shared_ps, dtype=float)
                q_total = np.array(
                    [sum(pressure_data[p][c] for c in components) for p in shared_ps],
                    dtype=float,
                )

                for comp in components:
                    q_comp = np.array([pressure_data[p][comp] for p in shared_ps], dtype=float)
                    x_i = np.where(q_total > 0, q_comp / q_total, np.nan)

                    valid = np.isfinite(x_i) & np.isfinite(p_arr)
                    if scale_name == 'log':
                        valid = valid & (p_arr > 0)
                    if not np.any(valid):
                        continue

                    # Legend should only show component name; structures are
                    # distinguished by linestyle.
                    comp_display = phelp.get_molecule_display_name(comp)
                    marker = phelp.get_marker_for_molecule(comp)
                    x_plot = p_arr[valid]
                    y_plot = x_i[valid]

                    # Sort by pressure so a connecting line is monotonic
                    order = np.argsort(x_plot)
                    x_plot = x_plot[order]
                    y_plot = y_plot[order]

                    ls_fw = phelp.get_linestyle_for_structure(fw)
                    if n_fw_leg > 1:
                        label = "_nolegend_"
                    elif fw == fw_first:
                        label = comp_display
                    else:
                        label = "_nolegend_"

                    # Always draw styled lines so structure linestyles are visible.
                    ax.plot(
                        x_plot, y_plot,
                        color=comp_colors[comp],
                        lw=phelp.LINEWIDTH,
                        linestyle=ls_fw,
                        alpha=phelp.ALPHA,
                        label=label,
                    )
                    if show_points:
                        ax.scatter(
                            x_plot, y_plot,
                            color=comp_colors[comp], marker=marker, s=phelp.AXIS_S_SIZE, alpha=phelp.ALPHA,
                            label="_nolegend_",
                        )
                    any_plotted = True

                # Collect export rows: one per valid (P, x_i, component)
                target_rows = export_rows_log if scale_name == 'log' else export_rows_linear
                for comp in components:
                    q_comp = np.array([pressure_data[p][comp] for p in shared_ps], dtype=float)
                    x_i = np.where(q_total > 0, q_comp / q_total, np.nan)
                    valid = np.isfinite(x_i) & np.isfinite(p_arr)
                    if scale_name == 'log':
                        valid = valid & (p_arr > 0)
                    for P, xi_val in zip(p_arr[valid], x_i[valid]):
                        target_rows.append({
                            'framework': fw,
                            'molecule': comp,
                            'temperature': float(temp),
                            'pressure': float(P),
                            'mole_fraction': float(xi_val),
                        })

            if not any_plotted:
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
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Pressure [Pa]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
            ax.set_ylabel(r'Molar fraction [-]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium')
            ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
            if n_fw_leg > 1:
                proxy_handles = []
                proxy_labels = []
                for comp in components:
                    proxy_handles.append(
                        Line2D(
                            [0], [0],
                            color=comp_colors[comp],
                            linestyle='-',
                            lw=phelp.LINEWIDTH,
                            marker=None,
                        )
                    )
                    proxy_labels.append(phelp.get_molecule_display_name(comp))
                ax.legend(
                    proxy_handles,
                    proxy_labels,
                    fontsize=phelp.AXIS_LEGEND_SIZE,
                    loc='best',
                )
            else:
                _h_mf, _lab_mf = ax.get_legend_handles_labels()
                if _lab_mf:
                    if show_points:
                        _h_f = [h for h in _h_mf if isinstance(h, Line2D)]
                        _lab_f = [lb for h, lb in zip(_h_mf, _lab_mf) if isinstance(h, Line2D)]
                        if _h_f:
                            ax.legend(_h_f, _lab_f, fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
                        else:
                            ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
                    else:
                        ax.legend(fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
            phelp.apply_unified_axes_layout(fig, ax)
            phelp._save_plot(
                f'mol_fraction_{int(temp)}K_{suffix}',
                'Basic_data',
                selected_frameworks, selected_molecules, selected_temperatures,
                fig=fig, out_dir=out_dir,
            )
            plt.close(fig)

    scale_flag = str(scale).strip().lower()
    if scale_flag in ('log', 'both'):
        _plot_for_scale('log', 'log')
    if scale_flag in ('linear', 'both'):
        _plot_for_scale('linear', 'linear')

    # Export data files once per scale after plotting (OUT_DIR gating via save_data)
    if save_data and out_dir is None:
        fw_part = _safe_join(selected_frameworks)
        mix_part = _safe_join(selected_molecules)
        temp_part = _safe_join(selected_temperatures)

        if export_rows_log:
            _save_mol_fraction_rows_to_run_folder(
                export_rows_log, fw_part, mix_part, temp_part,
                prefix=f"mol_fraction_log"
            )
        if export_rows_linear:
            _save_mol_fraction_rows_to_run_folder(
                export_rows_linear, fw_part, mix_part, temp_part,
                prefix=f"mol_fraction_linear"
            )
    # Do not plt.show()/close('all') — figures already plt.close(fig); Agg headless run.