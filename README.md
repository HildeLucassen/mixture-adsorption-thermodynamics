# Mixture adsorption thermodynamics

Python workflow for **mixed-gas adsorption**: adsorption isotherm (Toth, Langmuir-Freundlich, Sips), **heat of adsorption** (Virial and Clausius–Clapeyron equation, optional HOA input files), and **storage density** (Clausius–Clapeyron equation for mixture, and linear mixing rule) with 2D and 3D views. 

The pipeline for the code, as described in the figure below, is as followed:
- Configuration
  _config.in file_: This file defines all settings required to generate the intended output.
  > ADSORBENT / ADSORBATE — names of the adsorbent–adsorbate system, matching the identifiers used in the data input. Automatically detects if the adsorbate is a mixture or a pure component. 
  
  > TEMPERATURE — temperatures at which adsorption isotherm data are provided.
  
  > FIT_TYPE / NUM_ISOTHERM_SITE — selected isotherm model (Toth, Langmuir–Freundlich, Sips) and the number of adsorption sites.
  
  > P_MIN / P_MAX / PRESSURE_UNIT — pressure boundaries and the corresponding pressure unit.
  
  > PRESSURE_SCALE — scaling of the pressure axis in the output.
    
  > DATA_FILE_FITTING / POINTS / HOA — file locations for the input data (raw points, number of equilibrium points, or heat of adsorption data).
  
  > DATA_SOURCE — specifies whether the input is points (raw equilibrium data) or fitting (model parameters for the isotherm equations) of pure components. 

  > HEAT_OF_ADSORPTION: the intended output for Heat of adsorption incase of a pure component. Options: virial (Virial equations), cc (Clausius-Clapeyron), file (data_file) or both (virial and cc)  

  > HEAT_OF_ADSORPTION_MIX: the intended out put for Heat of adsorption incase of a multi-component mixture. Options, cc (Clausius-Clapeyron mixture rules), hoa_pure_cc (linear mixing rule using the heat of adsorption of the pure component calculated with the clausius-clapeyron), hoa_pure_virial (linear mixing rule using the heat of adsorption of the pure component calculated with the virial), hoa_pure_file (linear mixing rule using the heat of adsorption of the pure component from the data_file) or both (cc, hoa_pure_virial and hoa_pure_cc)
  
  > STORAGE_DENSITY: Specifies the type of storage‑density calculation: pressure‑swing, temperature‑swing, or pressure–temperature swing. Options: virial (Virial equations), cc (Clausius-Clapeyron), file (data_file) or both (virial and cc)
  
  > STORAGE_DENSITY_DIMENSION: 2D or 3D dimensions
  >
  > SUGGESTION_VIRIAL: yes or no, if yes selects polunominal degrees automatically based of highest rounded off (0.xx) R^2 value for the lowest degrees
  >
  > VIRIAL_DEGREE: (a,b) polynominal degrees used for the virial equation
  >
  > VIRIAL_DEGREE_COMBO: ADSORBENT ADSORBATE a b, able to create a list of multiple virial combinations
       

![Workflow](Workflow.png)


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

## Configuration

| File | Role |
|------|------|
| **`config.in`** | Adsorbent/adsorbate selection, temperatures, pressure range, data sources, which HOA and storage-density modes to run, virial settings, output flags, etc. |
| **`design.in`** | Plot styling: colours, line styles, markers, colormaps for structures, molecules, temperatures, and storage-density 3D figures. |

Important keys in `config.in` (see comments in the file for full detail):

- **`DATA_SOURCE`:** `fitting` (isotherm parameters) or `points` (simulation / tabulated points).  
- **`DATA_FILE_FITTING`**, **`DATA_FILE_POINTS`**, **`DATA_FILE_HOA`:** paths to input tables (relative to repo root).  
- **`HEAT_OF_ADSORPTION`**, **`HEAT_OF_ADSORPTION_MIX`**, **`STORAGE_DENSITY`:** enable virial, Clausius–Clapeyron, file-based HOA, or combinations.  
- **`OUT_DIR`:** write figures and summaries under **`Output/`** when enabled.

## Input data

Place (or point `config.in` to) text files under **`Input/`**, for example:

- Isotherm parameters or pressure–loading points for fitting and thermodynamic analysis.  
- Optional dedicated **heat-of-adsorption** tables when using file-based HOA modes.

Formats are defined by the reader logic in `Code/Input.py` and the rest of the pipeline; keep column layouts consistent with your existing example files.

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

## Outputs

With output options turned on in `config.in`, results are written under **`Output/<case_label>/`** (naming depends on adsorbent, adsorbate, temperatures, and similar selections). Typical artefacts include figures (PNG/PDF as configured) and summary text files.

---


