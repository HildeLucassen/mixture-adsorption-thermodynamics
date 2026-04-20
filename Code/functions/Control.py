import numpy as np
import matplotlib.pyplot as plt
from math import isclose
import os
from pathlib import Path
import ClausiusClapeyron as cc
import PlotHelpers as phelp
import Initialize as init
import Virial as virial
import DataSelection as ds


def plot_clausius_clapeyron_with_virial(
    selected_frameworks, selected_molecules, selected_temperatures, selected_fit_types,
    fittings, RASPA_data=None, x_fit=None, loadings=None, n_loadings=60,
    p_min=None, p_max=None, r2_min=0.95, out_dir=None,
    deg_a=2, deg_b=2, plot_virial=True, virial_plot=None,
    smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5, method_linestyles=None, show_markers=True,
    degrees_per_combo=None, qst_cache_file=None):
    """
    Plot Clausius-Clapeyron Qst and Virial Qst on the same axes (no RASPA enthalpy overlay).
    Virial uses separate degrees deg_a (1/T terms) and deg_b (T-independent terms).

    If several frameworks are selected, one figure is saved per framework in the same output
    folder, named ``heat_of_adsorption_combined_<framework>.png``.

    Parameters:
    - RASPA_data: Used only to fit Virial coefficients (pressure/loading points); no enthalpy plotted.
    - smooth: If True, apply smoothing to the CC Qst curve
    - use_direct_interpolation: If True, use direct interpolation for CC instead of integration
    - smoothing_sigma: Standard deviation for Gaussian smoothing (if smooth=True)
    """
    if len(selected_frameworks) > 1:
        joined_fw_for_folder = "-".join(str(x).replace(" ", "_") for x in selected_frameworks)
        for fw in selected_frameworks:
            _plot_clausius_clapeyron_with_virial_once(
                [fw], selected_molecules, selected_temperatures, selected_fit_types,
                fittings, RASPA_data=RASPA_data, x_fit=x_fit, loadings=loadings, n_loadings=n_loadings,
                p_min=p_min, p_max=p_max, r2_min=r2_min, out_dir=out_dir,
                deg_a=deg_a, deg_b=deg_b, plot_virial=plot_virial, virial_plot=virial_plot,
                smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                smoothing_sigma=smoothing_sigma, method_linestyles=method_linestyles,
                show_markers=show_markers, degrees_per_combo=degrees_per_combo,
                qst_cache_file=qst_cache_file,
                filename_suffix=str(fw).replace(" ", "_"),
                fw_label_override_for_folder=joined_fw_for_folder,
            )
        return

    _plot_clausius_clapeyron_with_virial_once(
        selected_frameworks, selected_molecules, selected_temperatures, selected_fit_types,
        fittings, RASPA_data=RASPA_data, x_fit=x_fit, loadings=loadings, n_loadings=n_loadings,
        p_min=p_min, p_max=p_max, r2_min=r2_min, out_dir=out_dir,
        deg_a=deg_a, deg_b=deg_b, plot_virial=plot_virial, virial_plot=virial_plot,
        smooth=smooth, use_direct_interpolation=use_direct_interpolation,
        smoothing_sigma=smoothing_sigma, method_linestyles=method_linestyles,
        show_markers=show_markers, degrees_per_combo=degrees_per_combo,
        qst_cache_file=qst_cache_file,
        filename_suffix=None,
        fw_label_override_for_folder=None,
    )


def _plot_clausius_clapeyron_with_virial_once(
    selected_frameworks, selected_molecules, selected_temperatures, selected_fit_types,
    fittings, RASPA_data=None, x_fit=None, loadings=None, n_loadings=60,
    p_min=None, p_max=None, r2_min=0.95, out_dir=None,
    deg_a=2, deg_b=2, plot_virial=True, virial_plot=None,
    smooth=False, use_direct_interpolation=False, smoothing_sigma=1.5, method_linestyles=None, show_markers=True,
    degrees_per_combo=None, qst_cache_file=None,
    filename_suffix=None, fw_label_override_for_folder=None):
    fig, ax = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    any_plotted = False
    global_max_loading = 0.0  # Track maximum loading across all lines
    global_min_qst = np.inf  # Track minimum Qst across all lines
    global_max_qst = -np.inf  # Track maximum Qst across all lines
    molecules_present = []
    frameworks_present = []
    methods_present = []
    
    # Store data for export
    export_data = []  # List of dicts: {'type': 'Virial'/'Clausius_Clapeyron'/'Enthalpy', 'fw': fw, 'mol': mol, 'loading': array, 'qst': array, 'p_min': p_min, 'p_max': p_max, 'deg_a': deg_a, 'deg_b': deg_b}

    # ensure the same pressure-grid default as the canonical CC plot
    if x_fit is None:
        x_fit = np.logspace(np.log10(0.5), 7, 300)

    for fw in selected_frameworks:
        for mol in selected_molecules:
            try:
                data = cc.compute_isosteric_heat(
                    fw, mol, selected_temperatures, selected_fit_types, fittings,
                    RASPA_data=RASPA_data, loadings=loadings, n_loadings=n_loadings,
                    p_grid=x_fit, p_min=p_min, p_max=p_max, r2_min=r2_min,
                    smooth=smooth, use_direct_interpolation=use_direct_interpolation,
                    smoothing_sigma=smoothing_sigma
                )
            except Exception as e:
                print(f"Clausius-Clapeyron: skipping {fw},{mol}: {e}")
                continue

            loading = data.get('loading')
            # Use smoothed Qst if smoothing is enabled, otherwise use original Qst
            if smooth and data.get('Qst_kJmol_smoothed') is not None:
                Qst = data.get('Qst_kJmol_smoothed')
            else:
                Qst = data.get('Qst_kJmol')
            slopes_arr = data.get('slope')
            r2_arr = data.get('r2')
            valid_counts = data.get('valid_counts')
            if loading is None or Qst is None:
                print(f"Clausius-Clapeyron: no data for {fw},{mol} - skipping")
                continue

            mask = np.isfinite(loading) & np.isfinite(Qst)
            if np.any(mask):
                # Clean names for display
                fw_display = phelp.clean_material_name(fw)
                mol_display = phelp.get_molecule_display_name(mol)
                # HoA rule: color encodes adsorbate (molecule)
                cc_color = phelp.get_color_for_molecule(mol) or 'C0'
                cc_ls = phelp.get_hoa_linestyle(
                    fw, 'clausius_clapeyron', 'method', method_linestyles=method_linestyles
                )
                # Get marker for this framework from marker mapping
                marker = phelp.get_marker_for_material(fw) if show_markers else ''
                ax.plot(
                    loading[mask], Qst[mask], marker=marker, linestyle=cc_ls, color=cc_color,
                    lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label="_nolegend_",
                    markersize=phelp.MARKER_SIZE if show_markers else 0,
                )
                molecules_present.append(mol)
                frameworks_present.append(fw)
                if 'clausius_clapeyron' not in methods_present:
                    methods_present.append('clausius_clapeyron')
                # Track maximum loading and Qst range from CC line
                if len(loading[mask]) > 0:
                    global_max_loading = max(global_max_loading, np.max(loading[mask]))
                    global_min_qst = min(global_min_qst, np.min(Qst[mask]))
                    global_max_qst = max(global_max_qst, np.max(Qst[mask]))
                any_plotted = True
                
                # Store Clausius-Clapeyron data for export
                export_data.append({
                    'type': 'Clausius_Clapeyron',
                    'fw': fw,
                    'mol': mol,
                    'loading': loading[mask],
                    'qst': Qst[mask],
                    'p_min': p_min,
                    'p_max': p_max,
                    'deg_a': None,
                    'deg_b': None
                })

            # overlay Virial-derived Qst using the same logic as plot_Qst
            # Always compute virial internally for each framework to ensure all are plotted
            if plot_virial:
                try:
                    vn = None
                    vQ = None
                    
                    # Always compute virial internally for this specific framework/molecule
                    if RASPA_data is None:
                        raise ValueError('No RASPA_data available to compute Virial Qst')

                    # Use the same logic as plot_Qst to ensure identical results
                    # Gather points for this specific fw/mol
                    pts_all = phelp.filter_raspa_data(
                        RASPA_data, frameworks=[fw], molecules=[mol], temperatures=selected_temperatures,
                        only_pure_adsorption=True,
                    )
                    
                    # Filter by P_min first (before loading range logic) - same as plot_Qst
                    if p_min is not None:
                        p_min_val = float(p_min)
                        pts_all = [p for p in pts_all if p.get('pressure') is not None and float(p['pressure']) >= p_min_val]
                    
                    pts_all = [p for p in pts_all if p.get('pressure') is not None and p.get('loading') is not None 
                              and p.get('temperature') is not None and float(p['pressure']) > 0 and float(p['loading']) > 0]
                    
                    if not pts_all:
                        raise ValueError(f"No valid data for {fw},{mol}")
                    
                    # Per-combo degree override
                    deg_a_use = degrees_per_combo[(fw, mol)][0] if degrees_per_combo and (fw, mol) in degrees_per_combo else deg_a
                    deg_b_use = degrees_per_combo[(fw, mol)][1] if degrees_per_combo and (fw, mol) in degrees_per_combo else deg_b

                    # Intersection range
                    n_min_val, n_max_val = ds._intersection_loading_range(pts_all)

                    # Filter to intersection range, then remove outliers
                    min_points = 3
                    pts_in_range = [p for p in pts_all
                                    if n_min_val <= float(p['loading']) <= n_max_val]
                    if len(pts_in_range) < min_points:
                        pts_in_range = pts_all

                    keep = virial._outlier_mask(pts_in_range, deg_a_use, deg_b_use)
                    pts_for_fit = [p for p, k in zip(pts_in_range, keep) if k]
                    if len(pts_for_fit) < min_points:
                        pts_for_fit = pts_in_range

                    # Fit and evaluate Qst curve
                    results_direct = virial.compute_lnP_per_temperature_separate(
                        pts_for_fit, fw, mol,
                        deg_a=deg_a_use, deg_b=deg_b_use, min_points=min_points,
                        temperatures=selected_temperatures, verbose=False, p_min=None
                    )
                    coeffs_a = np.asarray(results_direct['coeffs_a'], float)
                    n_grid = np.linspace(n_min_val, n_max_val, int(max(2, n_loadings)))
                    Q_plot = -float(virial.R) * virial._eval_poly(coeffs_a, n_grid) / 1000.0  # kJ/mol

                    vn = n_grid
                    vQ = Q_plot

                    if vn is not None and vQ is not None:
                        maskv = np.isfinite(vn) & np.isfinite(vQ)
                        if np.any(maskv):
                            # Clean names for display
                            clean_fw = phelp.clean_material_name(fw)
                            clean_mol = phelp.get_molecule_display_name(mol)
                            # HoA rule: color encodes adsorbate (molecule)
                            virial_color = phelp.get_color_for_molecule(mol) or 'C1'
                            virial_ls = phelp.get_hoa_linestyle(
                                fw, 'virial', 'method', method_linestyles=method_linestyles
                            )
                            # Get marker for this framework from marker mapping
                            marker = phelp.get_marker_for_material(fw) if show_markers else ''
                            ax.plot(
                                vn[maskv], vQ[maskv],
                                marker=marker, linestyle=virial_ls, color=virial_color,
                                lw=phelp.LINEWIDTH, alpha=phelp.ALPHA,
                                label="_nolegend_", markersize=phelp.MARKER_SIZE if show_markers else 0,
                            )
                            molecules_present.append(mol)
                            frameworks_present.append(fw)
                            if 'virial' not in methods_present:
                                methods_present.append('virial')
                            # Track maximum loading and Qst range from Virial line
                            if len(vn[maskv]) > 0:
                                global_max_loading = max(global_max_loading, np.max(vn[maskv]))
                                global_min_qst = min(global_min_qst, np.min(vQ[maskv]))
                                global_max_qst = max(global_max_qst, np.max(vQ[maskv]))
                            
                            # Store Virial data for export
                            export_data.append({
                                'type': 'Virial',
                                'fw': fw,
                                'mol': mol,
                                'loading': vn[maskv],
                                'qst': vQ[maskv],
                                'p_min': p_min,
                                'p_max': p_max,
                                'deg_a': deg_a,
                                'deg_b': deg_b
                            })
                except Exception as e:
                    print(f"Virial Qst overlay: failed for {fw},{mol}: {e}")

            # Optional: overlay HoA-from-file curve if available in qst_cache_file
            if qst_cache_file and (fw, mol) in qst_cache_file:
                try:
                    loads_file, qst_file = qst_cache_file[(fw, mol)]
                    loads_file = np.asarray(loads_file, dtype=float)
                    qst_file = np.asarray(qst_file, dtype=float)
                    maskf = np.isfinite(loads_file) & np.isfinite(qst_file)
                    if np.any(maskf):
                        fw_display = phelp.clean_material_name(fw)
                        mol_display = phelp.get_molecule_display_name(mol)
                        hoa_color = phelp.get_color_for_molecule(mol) or 'C2'
                        hoa_ls = phelp.get_hoa_linestyle(
                            fw, 'hoa_file', 'method', method_linestyles=method_linestyles
                        )
                        ax.plot(
                            loads_file[maskf], qst_file[maskf],
                            linestyle=hoa_ls, color=hoa_color,
                            lw=phelp.LINEWIDTH, alpha=phelp.ALPHA,
                            label="_nolegend_",
                        )
                        molecules_present.append(mol)
                        frameworks_present.append(fw)
                        if 'hoa_file' not in methods_present:
                            methods_present.append('hoa_file')
                        # update global ranges
                        global_max_loading = max(global_max_loading, np.max(loads_file[maskf]))
                        global_min_qst = min(global_min_qst, np.min(qst_file[maskf]))
                        global_max_qst = max(global_max_qst, np.max(qst_file[maskf]))
                        any_plotted = True
                        export_data.append({
                            'type': 'HoA_file',
                            'fw': fw,
                            'mol': mol,
                            'loading': loads_file[maskf],
                            'qst': qst_file[maskf],
                            'p_min': p_min,
                            'p_max': p_max,
                            'deg_a': None,
                            'deg_b': None,
                        })
                except Exception as e:
                    print(f"HoA-file overlay: failed for {fw},{mol}: {e}")

    if not any_plotted:
        print("Clausius-Clapeyron: nothing plotted (no valid fits)")
        return

    # Axes and styling (match Basic_data isotherms / ``plot_clausius_clapeyron``)
    ax.set_xlabel(
        'Loading [mol/kg]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium',
    )
    ax.set_ylabel(
        'Qst [kJ/mol]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium',
    )
    ax.set_ylim(top=global_max_qst * 1.05)
    ax.set_xlim(left=0, right=global_max_loading * 1.05)
    ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
    phelp.build_hoa_proxy_legend(
        ax,
        molecules_present=molecules_present if molecules_present else selected_molecules,
        frameworks_present=frameworks_present if frameworks_present else selected_frameworks,
        methods_present=methods_present,
        method_linestyles=method_linestyles,
        fontsize=phelp.AXIS_LEGEND_SIZE,
        loc='best',
    )
    phelp.apply_unified_axes_layout(fig, ax)

    plot_path = phelp._save_plot(
        'heat_of_adsorption_combined', 'plot_clausius_clapeyron',
        selected_frameworks, selected_molecules, selected_temperatures, fig=fig, out_dir=out_dir,
        fw_label_override=fw_label_override_for_folder,
        filename_suffix=filename_suffix,
    )
    plt.show()
    plt.close(fig)

def plot_lnP_vs_loading_from_virial(RASPA_data=None, framework=None, molecule=None, temperatures=None,
                                    deg_a=2, deg_b=2, min_points=3, n_points=200,
                                    selected_frameworks=None, selected_molecules=None, selected_temperatures=None,
                                    out_dir=None, show=True, p_min=None):
    """
    Plot ln(Pressure) vs loading using Virial coefficients computed across temperatures.

    Uses :func:`virial.compute_lnP_per_temperature_separate` for coefficients, then
    :func:`virial.compute_lnP_from_coeffs` on a q-grid per temperature. Styling matches
    other Heat-of-Adsorption figures (``UNIFIED_FIGSIZE``, axis labels, ticks, spines,
    legend). Two figures (log and linear loading axis); both are saved when
    ``out_dir`` / default Output path is used.
    """
    coeff_report = (Path(out_dir) / "virial_coefficients.txt") if out_dir else None
    results = virial.compute_lnP_per_temperature_separate(
        RASPA_data,
        framework,
        molecule,
        deg_a=deg_a,
        deg_b=deg_b,
        min_points=min_points,
        temperatures=temperatures,
        verbose=False,
        p_min=p_min,
        coefficient_report_path=str(coeff_report) if coeff_report is not None else None,
        report_preamble=getattr(virial, 'DEGREE_SEARCH_REPORT_TEXT', None),
    )
    coeffs_a = np.asarray(results['coeffs_a'], dtype=float)
    coeffs_b = np.asarray(results['coeffs_b'], dtype=float)

    pts = phelp.filter_raspa_data(
        RASPA_data, frameworks=[framework], molecules=[molecule], temperatures=temperatures,
        only_pure_adsorption=True,
    )
    pts = [p for p in pts if p.get('pressure') is not None and p.get('loading') is not None and p.get('temperature') is not None and float(p.get('pressure')) > 0 and float(p.get('loading')) > 0]
    if p_min is not None:
        _pm = float(p_min)
        pts = [p for p in pts if float(p['pressure']) >= _pm]
    if not pts:
        raise ValueError(f"No valid adsorption data found for {framework}, {molecule}")

    temps = sorted(set([float(p['temperature']) for p in pts]))
    colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2', 'C3'])

    fig_log, ax_log = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)
    ax_log.set_xscale('log')
    fig_lin, ax_lin = plt.subplots(figsize=phelp.UNIFIED_FIGSIZE)

    for idx, T in enumerate(temps):
        color = colors[idx % len(colors)]
        subset = [p for p in pts if abs(float(p['temperature']) - T) < 0.5]
        q = np.array([p['loading'] for p in subset], dtype=float)
        P = np.array([p['pressure'] for p in subset], dtype=float)
        if q.size == 0:
            continue

        # observed ln(P) where P is in Pa (consistent with Virial fitting)
        y_obs = np.log(P)

        if q.size >= 2:
            q_grid = np.linspace(np.min(q), np.max(q), int(max(50, n_points)))
        else:
            q_grid = np.linspace(q[0] * 0.9, q[0] * 1.1, int(max(50, n_points)))

        try:
            lnP_grid, _ = virial.compute_lnP_from_coeffs(coeffs_a, coeffs_b, q_grid, T)
        except Exception as e:
            print(f"plot_lnP_vs_loading_from_virial: failed for T={T}: {e}")
            continue

        # Legend: temperature only (R² is written to the coefficient report file).
        marker = phelp.get_marker_for_material(framework)
        label_fit = f'{int(T)}K'

        for ax in (ax_log, ax_lin):
            ax.scatter(q, y_obs, color=color, alpha=phelp.ALPHA, marker=marker, s=phelp.AXIS_S_SIZE, zorder=6)
            ax.plot(q_grid, lnP_grid, color=color, lw=phelp.LINEWIDTH, alpha=phelp.ALPHA, label=label_fit)

    for fig, ax in ((fig_log, ax_log), (fig_lin, ax_lin)):
        ax.set_xlabel(
            'Loading [mol/kg]', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium',
        )
        ax.set_ylabel(
            'ln(Pressure)', fontsize=phelp.AXIS_LABEL_FONTSIZE, fontweight='medium',
        )
        ax.grid(True, which='both', ls='--', alpha=phelp.ALPHA_GRID)
        phelp.set_axis_limits_nice(ax)
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), fontsize=phelp.AXIS_LEGEND_SIZE, loc='best')
        phelp.apply_unified_axes_layout(fig, ax)

    fw_list = selected_frameworks if selected_frameworks is not None else [framework]
    mol_list = selected_molecules if selected_molecules is not None else [molecule]
    temp_list = selected_temperatures if selected_temperatures is not None else (temperatures if temperatures is not None else [f'virial_deg_a{deg_a}_b{deg_b}'])
    # When saving into a per-combo folder (``.../virial_control/<fw>_<mol>/``), use only
    # ``prefix.png`` like other HOA figures—suffix would repeat fw/mol and can push the
    # full path past the Windows limit. When *out_dir* is None, keep a disambiguating suffix
    # under the flat ``Heat_of_Adsorption`` folder.
    _save_name_suffix = None
    if out_dir is None and len(fw_list) == 1 and len(mol_list) == 1:
        _save_name_suffix = (
            f"{str(fw_list[0]).replace(' ', '_')}__{str(mol_list[0]).replace(' ', '_')}"
        )
    # Default: save under Heat_of_Adsorption (Main.py passes an explicit subfolder for Virial HOA diagnostics).
    save_out_dir = out_dir
    if save_out_dir is None:
        try:
            base_dir = str(init.get_pipeline_run_root())
            run_folder_name = f"{'-'.join([str(x).replace(' ', '_') for x in fw_list])}_{'-'.join([str(x).replace(' ', '_') for x in mol_list])}_{'-'.join([str(x).replace(' ', '_') for x in temp_list])}"
            save_out_dir = os.path.join(base_dir, "Output", run_folder_name, "Heat_of_Adsorption")
            os.makedirs(save_out_dir, exist_ok=True)
        except Exception:
            save_out_dir = None
    try:
        phelp._save_plot(
            'virial_lnP_vs_loading_log', 'plot_controls',
            fw_list, mol_list, temp_list, fig=fig_log, out_dir=save_out_dir,
            filename_suffix=_save_name_suffix,
        )
    except Exception:
        pass
    try:
        phelp._save_plot(
            'virial_lnP_vs_loading_linear', 'plot_controls',
            fw_list, mol_list, temp_list, fig=fig_lin, out_dir=save_out_dir,
            filename_suffix=_save_name_suffix,
        )
    except Exception:
        pass
    if show:
        plt.show()
    plt.close(fig_log)
    plt.close(fig_lin)

    return fig_log, fig_lin
