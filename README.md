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
- **`TEMPERATURE`** — temperatures at which adsorption isotherm data are provided.  
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

## Outputs

With output options turned on in `config.in`, results are written under **`Output/<case_label>/`** (naming depends on adsorbent, adsorbate, temperatures, and similar selections). Typical artefacts include figures (PNG/PDF as configured) and summary text files.



## Requirements

- **Python** 3.10 or newer  
- **Packages:** `numpy`, `scipy`, `matplotlib`, `pandas` (see note below)

Install dependencies (from the repository root):

```bash
pip install numpy scipy matplotlib pandas
```

`pandas` is imported by the plotting helpers; keep it installed even if you only use NumPy-style data elsewhere.

## How to run

From the repository root (so paths in `config.in` resolve correctly):

```bash
python run.py
```

`run.py` executes `Code/Main.py` with the working directory set to the project root. Plots use Matplotlib’s **Agg** backend (non-interactive, suitable for servers and batch runs).


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


---


