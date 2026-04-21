# Examples (simple → advanced)

These folders are **self-contained runs**: each has its own `config.in`, `Input/`, `run.py`, and (after a run) `Output/`. The table below goes **01 → 08** from simpler to more options (data mode, HOA paths, mixture vs pure, storage density, per-component SD, …); you can still open and run **any** folder on its own.

**Run any example:** `cd` into the folder (or call `py -3.13 path/to/Examples/<name>/run.py` from anywhere). See the repository **README.md** for Python version and path details.

| # | Folder | What it adds (keep in mind) |
|---|--------|-----------------------------|
| **01** | `01_pure_fitting_simple` | **Pure** (e.g. R125 on Bhathia). **`DATA_SOURCE=fitting`** with **`DATA_FILE_FITTING`** only — no equilibrium points file required for the main path. **`HEAT_OF_ADSORPTION=cc`**; no Virial block, no storage density. Good baseline before turning on points, Virial, or SD. |
| **02** | `02_pure_points_HoA` | **Pure** from **`DATA_SOURCE=points`** and **`DATA_FILE_POINTS`**. **`HEAT_OF_ADSORPTION=both`** so you get **Virial + CC** HOA together; **`VIRIAL_DEGREES`** fixed (e.g. `(4, 3)`) with **`SUGGESTION_VIRIAL=no`**. Use when isotherms come straight from simulation/tabulated points rather than pre-fitted parameters. |
| **03** | `03_pure_fitting_SD` | **Pure** with **`DATA_SOURCE=fitting`** but **`DATA_FILE_POINTS`** still present for Virial. Adds **`STORAGE_DENSITY`** / **`STORAGE_DENSITY_DIMENSION`** (here **`both`** for 2D+3D style SD plots) plus **`P_DES_*`**, **`T_ADS`**, **`P_ADS`**, **`T_DES`**, **`P_ADS_TT`**, **`P_DES_TT`**. **`SHOW_POINTS=yes`** so point data can be overlaid where relevant. |
| **04** | `04_mixture_simple` | **Mixture** (e.g. **R407F**) using **`DATA_FILE_POINTS`** for RASPA-style mixture loadings. **`HEAT_OF_ADSORPTION_MIX=cc`** only — mixture isotherms, mole fractions, and **mixture CC** HOA; no file-based HOA and no Virial “pure component” HOA branch. Smallest mixture-focused example. |
| **05** | `05_mixture_HoA_pure` | **Mixture** with **`DATA_SOURCE=fitting`** plus **`DATA_FILE_POINTS`**. **`HEAT_OF_ADSORPTION_MIX=both`** pulls in **mixture CC** and **pure CC / pure Virial** HOA paths. **`VIRIAL_DEGREES_COMBO`** lines pin **(deg_a, deg_b)** per **adsorbent + pure adsorbate** (constituents of the blend, not only the blend name). Heavier setup; use when you need Virial-based pure-component HOA alongside mixture CC. |
| **06** | `06_mixture_HoA_file` | **Mixture** with **`DATA_FILE_POINTS`** and **`DATA_FILE_HOA`** supplying tabulated **pure-component** \(Q_\mathrm{st}\) curves. **`HEAT_OF_ADSORPTION_MIX=hoa_file`** builds mixture HOA from **linear mixing** of those file curves. Use when you trust external HOA data more than in-code CC/Virial for the pure gases. |
| **07** | `07_mixture_SD_per_component` | **Mixture** + **`STORAGE_DENSITY`** with **`STORAGE_DENSITY_DIMENSION=per_component`** (in this template, SD is tied to **`HEAT_OF_ADSORPTION_MIX=cc`** as noted in `config.in`). Same swing parameters as other SD examples. **Outputs** land under **`Output/<run>/Storage_Density_components/<mixture>/`** with PNG stems like **`sd_PTS_components_*`** — one SD family **per mixture constituent**, not only the bulk mixture SD. |
| **08** | `08_mixture_every_option` | **Mixture** “all switches”: fitting + points + **`DATA_FILE_HOA`**; **`HEAT_OF_ADSORPTION_MIX`** / **`STORAGE_DENSITY`** as comma lists (e.g. **`cc, both`**). **`STORAGE_DENSITY_DIMENSION=per_component, both`** → per-component SD plus bulk **2D+3D** SD (`Input.py` joins comma-split dimensions for token parsing). |

**Tips for new users**

- Copy an example folder, rename it, then edit **`config.in`** in small steps rather than starting from the repo root `config.in`.
- If something fails, compare your **`DATA_SOURCE`**, file paths, and **`HEAT_OF_ADSORPTION` / `HEAT_OF_ADSORPTION_MIX` / `STORAGE_DENSITY`** lines to the nearest example above.
- New examples will continue this pattern: **same run mechanics**, **more `config.in` switches** as you go down the list.
