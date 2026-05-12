# Data Formatting Tool

Batch tool that converts raw RASPA-style text files into the two input formats
used by the adsorption-thermodynamics pipeline:

| Output file | Used for |
|---|---|
| `data_points.txt` | Adsorption isotherms (loading vs. pressure) |
| `data_heat_of_adsorption.txt` | Isosteric heat of adsorption data |

All paths, globs, column mapping, and per-file options are set in
`Data_formatting/config.txt` (same idea as `KEY  value  # comment` in
`template/config.in`). There is no interactive mode.

## Quick start

**1. Edit `config.txt`**

- Put shared settings first: `DATA_PATH`, `DATA_GLOB`, and any defaults
  (`STRUCTURE`, `MOLECULE`, `MIXTURE_PURE`, `PRESSURE_UNIT`, `PRESSURE`, `LOADING`, …).
- For each raw file you want to process, add a `FILE` line (basename or path under
  `DATA_PATH`, or a glob with `*` / `?`), then the options for that file on the
  lines below.

Comments: anything after `#` on a line is ignored (full-line `# …` blocks or
trailing `# comment` after the value).

Example (see the checked-in `config.txt` for a full template):

```
DATA_PATH               Data                    # folder under Data_formatting/
DATA_GLOB               **/*                    # which files are scanned

STRUCTURE               Bhatia_03
MOLECULE                R32
MIXTURE_PURE            pure
PRESSURE_UNIT           Pa                      # used in the points-file header
PRESSURE                1
LOADING                 2

FILE                    sim-…-333K.load
TEMPERATURE             333
KIND                    pts

FILE                    sim-…-333K-heat.load
KIND                    hoa
DELTAH                  7
```

**2. Run the tool**

```
python run.py
```

(from `Data_formatting/`, or run `python Code/formatting_tool.py` from the same
folder.)

The tool scans every file under `DATA_PATH` that matches `DATA_GLOB`. If a file
has no matching `FILE` block, it is skipped with a warning.

**3. Check the output**

Each run **replaces** (does not append to) the aggregate files:

```
Data_formatting/Output/data_points.txt
Data_formatting/Output/data_heat_of_adsorption.txt
```

Copy these into the `Input/` folder of your pipeline example to use them.

---

## Configuration reference

| Key | Where | Meaning |
|---|---|---|
| `DATA_PATH` | Top | Folder with raw files. Absolute path, or relative to `Data_formatting/`. Default if omitted: `Example/Data`. |
| `DATA_GLOB` | Top | Glob passed to `Path.glob` under `DATA_PATH` (default `**/*`). |
| `FILE` | Starts a block | Basename (`sim-….load`), relative path (`sub/out.load`), or pattern (`**/sim-*.load`). |
| `KIND` | Default or block | `pts` (isotherm), `hoa` (heat of adsorption), or `skip`. Default `pts` when unset. |
| `STRUCTURE` | Default / block | Adsorbent name. |
| `MOLECULE` | Default / block | Adsorbate name. |
| `TEMPERATURE` | Usually block | **Required for `KIND pts`.** Temperature in kelvin (`KIND hoa` does not use it). |
| `MIXTURE_PURE` | Default / block | Stored in the points file; use `pure` or your mixture label (`pure` is normalized to lowercase `pure`). |
| `PRESSURE_UNIT` | Default | Label for the pressure column in the points header (default `Pa`). |
| `PRESSURE` | Default / block | 1-based column index for pressure (`PRESSURE_COLUMN` is an alias). |
| `LOADING` | Default / block | 1-based column index for loading (`LOADING_COLUMN` is an alias). |
| `DELTAH` | Default / block | 1-based column index for ΔH (`DELTAH_COLUMN` is an alias). Required for `KIND hoa`. |

Settings above the first `FILE` line apply to all blocks unless overridden in a
block. If several `FILE` patterns match the same disk file, later blocks in the
file win when merging options.

Rows in raw files must start with a digit, `-`, or `.` to count as data; other
lines are treated as headers and skipped.

---

## Example raw file format

The tool reads whitespace- or tab-separated text files.

**Isotherm file** (`sim-R32-aCarbon-333K.load`):

```
#[1][Pa]  [2]mol/Kg  [3]molec/uc  [4]cm3/g  ...
1.0e+00   0.0000   0.0001   ...
3.0e+00   0.0001   0.0003   ...
```

Set `PRESSURE 1`, `LOADING 2`, `KIND pts`, and `TEMPERATURE` for that file.

**Heat-of-adsorption file** (`sim-R32-aCarbon-333K-heat.load`):

```
#[1][Pa]  [2]mol/Kg  ...  [7]deltaH  [8]error-deltaH
1.0e+00   0.0000   ...   -nan         nan
3.0e+00   0.0000   ...   -14.896      7.321
```

Typical mapping: `LOADING 2`, `DELTAH 7`, `KIND hoa`. If every ΔH value is
negative, the tool stores the negated values as positive magnitudes.

---

## Output file formats

**`data_points.txt`** (header uses `PRESSURE_UNIT`, e.g. `Pa`):

```
Structure	Molecule	Mixture/Pure	Temperature [K]	Pressure [Pa]	Loading [mol/kg]
Bhatia_03	R32	pure	333	1.0	0.0000
```

**`data_heat_of_adsorption.txt`**

```
#Structure	Molecule	Loading [mol/kg]	Heat of Adsorption [kJ/mol]
Bhatia_03	R32	0.0000	14.896
```

---

## Notes

- The configuration file **`config.txt` must exist**; there is no silent
  fallback when it is missing. If `DATA_PATH` is omitted, the tool defaults to
  `Example/Data` under `Data_formatting/`.
- At least one `FILE` block is required; otherwise the tool exits with an error.
- **`formatting_tool.py`** reads only `config.txt`; you do not need to
  edit Python to change folders or columns.
