# Data Formatting Tool

Interactive tool that converts output files into the two
input formats used by the adsorption-thermodynamics pipeline:

| Output file | Used for |
|---|---|
| `data_points.txt` | Adsorption isotherms (loading vs. pressure) |
| `data_heat_of_adsorption.txt` | Isosteric heat of adsorption data |

## Quick start

**1. Set the input data folder**

Open `Data_formatting/input_data_dir.txt` and replace the path with the folder
that contains your raw RASPA files.  You can use either an absolute or a
relative path (relative to `Data_formatting/`).

```
# input_data_dir.txt
# Absolute path example:
C:\Users\you\simulations\my_run\

# Relative path example (relative to Data_formatting/):
Example/Data
```

Lines starting with `#` are ignored.  If the file is missing or empty the tool
falls back to `Data_formatting/Example/Data`.

**2. Run the tool**

```
python run.py
```

The tool walks through every file in the input folder, shows you the header of
each file, and asks a short series of questions to map the columns.

**3. Check the output**

Formatted rows are appended to:

```
Data_formatting/Example/Output/data_points.txt
Data_formatting/Example/Output/data_heat_of_adsorption.txt
```

Copy these files into the `Input/` folder of your pipeline example to use them.

---

## Interactive questions

For each raw file you will be asked:

| Question | Options |
|---|---|
| File type | `1` = heat of adsorption, `2` = isotherm (data points), `s` = skip |
| Is the adsorbent name in the file? | yes / no |
| Column numbers for each quantity | 1-based integer |
| Temperature (if not in the file) | the tool tries to read it from the filename, e.g. `333K` |
| Pure or mixture? | `pure` / `mixture` |

---

## Example raw file format

The tool reads any whitespace- or tab-separated text file.  Lines that do not
start with a digit, `-`, or `.` are treated as header/comment lines and
skipped automatically.

**Isotherm file** (`sim-R32-aCarbon-333K.load`):

```
#[1][Pa]  [2]mol/Kg  [3]molec/uc  [4]cm3/g  ...
1.0e+00   0.0000   0.0001   ...
3.0e+00   0.0001   0.0003   ...
```

Column 1 = pressure [Pa], column 2 = loading [mol/kg].

**Heat-of-adsorption file** (`sim-R32-aCarbon-333K-heat.load`):

```
#[1][Pa]  [2]mol/Kg  ...  [7]deltaH  [8]error-deltaH
1.0e+00   0.0000   ...   -nan         nan
3.0e+00   0.0000   ...   -14.896      7.321
```

Column 2 = loading [mol/kg], column 7 = heat of adsorption [kJ/mol].
Negative values are automatically stored as positive (the tool detects the
sign convention and negates if all values are negative).

---

## Output file formats

**`data_points.txt`**

```
Structure    Molecule    Mixture/Pure    Temperature    Pressure    Loading mol/kg
Bhatia_01    R32         pure            333            1.0         0.0000
Bhatia_01    R32         pure            333            3.0         0.0001
```

**`data_heat_of_adsorption.txt`**

```
#Structure    Molecule    Loading (mol/kg)    Heat of Adsorption (kJ/mol)
Bhatia_01     R32         0.0000              14.896
```

---

## Notes

- The tool **appends** to existing output files — it does not overwrite them.
  Delete the output files before a fresh run if you want to start clean.
- The tool tries to read the temperature automatically from filenames
  containing a pattern like `333K`.  If it cannot, it will ask you.
- The `formatting_tool.py` inside `Code/` reads its source path from
  `input_data_dir.txt` automatically; you do not need to edit it directly.
