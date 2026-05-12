# Mixture adsorption thermodynamics

Python workflow for **mixed-gas adsorption**: adsorption isotherm (Toth, Langmuir-Freundlich, Sips), **heat of adsorption** (Virial and Clausius–Clapeyron equation, optional HOA input files), and **storage density** (Clausius–Clapeyron equation for mixture, and linear mixing rule) with 2D and 3D views. 

The pipeline for the code, as described in the figure below

![Workflow](Workflow.png)


##  Configuration

| File | Role |
|------|------|
| **`config.in`** | Defines the adsorbent/adsorbate system, temperatures, pressure range, isotherm model settings, data sources, HOA and storage‑density modes, virial configuration, and output options. |
| **`design.in`** | Plot styling: colours, line styles, markers, colormaps for structures, molecules, temperatures, and 3D storage‑density figures. Default available|

Important keys in `config.in`
### **Data & system definition**
- **`ADSORBENT` / `ADSORBATE`** — names of the adsorbent–adsorbate system, matching the identifiers in the data input. Automatically detects pure vs. mixture adsorbates.  
- **`TEMPERATURE`** — temperatures at which adsorption isotherm data are provided. Heat of Adsorption needs minimum of three temperatures. 
- **`DATA_SOURCE`** — `fitting` (isotherm parameters) or `points` (raw equilibrium data).  
- **`DATA_FILE_FITTING`**, **`DATA_FILE_POINTS`**, **`DATA_FILE_HOA`** — paths to input tables (relative to repo root).

### **Isotherm model settings**
- **`FIT_TYPE` / `NUM_ISOTHERM_SITE`** — selected isotherm model (Toth, Langmuir‑Freundlich, Sips) and number of adsorption sites.  
- **`P_MIN` / `P_MAX` / `PRESSURE_UNIT`** — pressure boundaries and unit.  
- **`PRESSURE_SCALE`** — scaling of the pressure axis in the output.

### **Heat of adsorption (HOA)**
- **`HEAT_OF_ADSORPTION`** — HOA mode for pure components:  
  `virial`, `cc`, `file`, or `both` (virial + cc).  
- **`HEAT_OF_ADSORPTION_MIX`** — HOA mode for mixtures:  
  - `cc` — Clausius–Clapeyron mixture rules  
  - `hoa_pure_cc` — linear mixing using pure‑component CC HOA  
  - `hoa_pure_virial` — linear mixing using pure‑component virial HOA  
  - `hoa_pure_file` — linear mixing using pure‑component file HOA  
  - `both` — cc + pure‑virial + pure‑cc

### **Storage‑density settings**
- **`STORAGE_DENSITY`** — which thermodynamic route is used for the isosteric heat inside the storage‑density integral: `cc` (Clausius–Clapeyron), `virial`, `file` (from HOA file), or `both`. This is separate from the *swing scenario* below, which is determined by how each figure is built.  
- **`STORAGE_DENSITY_DIMENSION`** — `2D` or `3D`.  
- **`STORAGE_DENSITY_PARAMETERS`** — additional parameters for storage‑density calculations.

**Output filenames (2D / 3D under `Output/.../Storage_Density/`)** use short stems so the swing scenario is visible in the name. The middle token is the method (`cc`, `virial`, or shortened tokens such as `hpv` / `hpc` for some mixture HOA paths):

| Stem (examples) | Swing scenario |
|-----------------|----------------|
| **`sd_PS_Teq_*`** | **Pressure swing**, fixed adsorption pressure \(P_\mathrm{ads}\), with **\(T_\mathrm{ads} = T_\mathrm{des}\)** along each curve; desorption pressure is swept. |
| **`sd_TS_Peq_*`** | **Temperature swing**, **\(P_\mathrm{ads} = P_\mathrm{des}\)** at each abscissa; fixed \(T_\mathrm{ads}\) with several \(T_\mathrm{des}\) (and pressure) on the plot. |
| **`sd_PTS_fixedPads_*`** | **Combined pressure–temperature swing**, fixed **\(P_\mathrm{ads}\)** (and fixed \(T_\mathrm{ads}\) for adsorption loading) while **\(T_\mathrm{des}\)** and **\(P_\mathrm{des}\)** vary. |
| **`sd_Tads_Tdes_*_3d`** (3D only) | Surface over **\(T_\mathrm{ads}\) × \(T_\mathrm{des}\)** at fixed pressures — not one of the three swing stems above, but the same folder tree. |

3D surfaces append **`_3d`** before the framework/molecule suffix in the filename (e.g. `sd_PS_Teq_cc_3d_...png`).

### **Virial configuration**
- **`SUGGESTION_VIRIAL`** — `yes`/`no`; if `yes`, selects polynomial degrees automatically based on highest rounded‑off R² for lower degrees.  
- **`VIRIAL_DEGREE`** — `(a, b)` polynomial degrees for the virial equation.  
- **`VIRIAL_DEGREE_COMBO`** — list of combinations:  
  `ADSORBENT  ADSORBATE  a  b`  
  Allows defining multiple virial degree sets.

### **Output**
- **`OUT_DIR`** — directory under `Output/` where figures and summaries are written when enabled.
    
## Input data

Place (or point `config.in` to) text files under **`Input/`**, for example:

- Isotherm parameters or pressure–loading points for fitting and thermodynamic analysis.  
- Optional dedicated **heat-of-adsorption** tables when using file-based HOA modes.

Formats are defined by the reader logic in `Code/Input.py` and the rest of the pipeline; keep column layouts consistent with your existing example files.

The workflow can read adsorption data in two ways (`DATA_SOURCE` in `config.in`):

- **`fitting`**: use parameter tables (for isotherm-model based calculations).
- **`points`**: use raw pressure-loading points directly.

If `DATA_SOURCE` is not explicitly set or the fitting file is unavailable, the
code can fall back to points-based input (depending on the run configuration).

### Format templates

Example input-format files are available in **`template/Input/`** (e.g.
`data_points.txt`, `data_heat_of_adsorption.txt`, and related files). Use these
as the reference column structure when preparing your own datasets.

### Converting non-matching raw files

If your raw data files are not yet in the required format, use the
**`Data_formatting/`** tool.

## Outputs

With output options turned on in `config.in`, results are written under **`Output/<case_label>/`** (naming depends on adsorbent, adsorbate, temperatures, and similar selections). Typical artefacts include figures (PNG/PDF as configured) and summary text files.



## Requirements

- **Python** 3.9 or newer (3.9 through 3.13 and later are expected to work; use a matching `py -3.x` on Windows).  
- **Packages:** `numpy`, `scipy`, `matplotlib`, `pandas` (see note below)

Install dependencies (from the repository root):

```bash
pip install numpy scipy matplotlib pandas
```

`pandas` is imported by the plotting helpers; keep it installed even if you only use NumPy-style data elsewhere.

## How to run

### Python interpreter (Windows)

If you use the [Windows Python launcher](https://docs.python.org/3/using/windows.html#python-launcher-for-windows), pin the version explicitly, for example:

```bash
py -3.13 run.py
```

That selects Python 3.13 for this run. The same pattern works for **`py -3.12`**, **`py -3.11`**, … down to **`py -3.9`**, as long as that runtime is installed and dependencies are available. On Linux or macOS, use `python3.13 run.py`, `python3.9 run.py`, etc. (or whatever `python3` points to) instead of `py -3.13`.

### Where to run from

| Layout | What to do |
|--------|------------|
| **Repository root** (`Code_V5/`, next to the top-level `config.in`, `run.py`, `Input/`, `Code/`) | Open a shell **in that folder** (the workspace / clone root), then run e.g. `py -3.13 run.py` or `py -3.9 run.py`. `config.in` paths such as `Input\data_parameters.txt` are resolved relative to this directory. |
| **An example folder** (`Examples/<example_name>/`, with its own `config.in` and `Input/`) | Either **change directory** into that example and run e.g. `py -3.13 run.py`, or stay at the repo root and run `py -3.13 Examples/<example_name>/run.py` (same idea with `py -3.9`, …). Each example’s `run.py` sets the process working directory to that example folder and sets `PIPELINE_REPO_ROOT` so `config.in` and `Output/` stay with the example. |

`run.py` starts `Code/Main.py` with the correct working directory for the chosen layout. Plots use Matplotlib’s **Agg** backend (non-interactive, suitable for servers and batch runs).


## Project layout

```
Code/
  Main.py           # Pipeline entry (orchestration, plotting, exports)
  Input.py          # Loads config.in and design.in
  functions/        # Virial, Clausius–Clapeyron, storage density, mol fractions, plots, …
Input/              # User data (tracked examples or your own)
Output/             # Generated plots and summary text (when enabled)
config.in           # Run configuration
design.in           # Visual design mappings
run.py              # Wrapper: run Main.py from repo root
Workflow.png        # High-level workflow diagram for the README
```

## License

This project is licensed under the **MIT License**. See [`LICENSE.txt`](LICENSE.txt) in the repository root for the full text.

## Citing

If you use this software in academic work, please cite it as described by GitHub:

- Use the repository page **“Cite this repository”** control (filled from [`CITATION.cff`](CITATION.cff)), or  
- Read the citation metadata directly in **`CITATION.cff`** at the repo root.

## Authorship

Authors: H.A. Lucassen [1], A. Luna-Triguero [1, 2], and J. M. Vicent-Luna [2]

[1] Energy Technology, Department of Mechanical Engineering, Eindhoven University of Technology

[2] Eindhoven Institute for Renewable Energy Systems (EIRES), Eindhoven University of Technology

---


