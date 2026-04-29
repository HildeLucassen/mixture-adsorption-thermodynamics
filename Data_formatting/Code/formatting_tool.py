"""
formatting_tool.py
==================
Interactive tool that walks through every raw data file under CO2_TAMOF,
asks column-mapping questions, and appends correctly formatted rows into
the project's Input files.

Run from the Working Code directory:
    python formatting_tool.py
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  *** edit these three lines to match your setup ***
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH_FILE = PROJECT_DIR / "input_data_dir.txt"

def _load_source_config() -> tuple[Path, str]:
    """Read source directory and glob pattern from a single line in input_data_dir.txt.

    Expected format (one non-comment line):
        <directory>/<glob pattern>
        e.g.  ../structures/*/*/*/sim-*K.load

    The path is split at the first path segment containing a wildcard (*/?).
    Falls back to Example/Data with pattern **/* if not configured.
    """
    default_dir = PROJECT_DIR / "Example" / "Data"
    default_pattern = "**/*"

    if INPUT_PATH_FILE.exists():
        for raw in INPUT_PATH_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = Path(line).parts
            # Split into fixed prefix (directory) and wildcard suffix (pattern)
            split = next((i for i, p in enumerate(parts) if "*" in p or "?" in p), None)
            if split is None:
                src_dir = PROJECT_DIR / line
                pattern = default_pattern
            else:
                dir_raw = str(Path(*parts[:split])) if split > 0 else "."
                p = Path(dir_raw)
                src_dir = p if p.is_absolute() else (PROJECT_DIR / p)
                pattern = str(Path(*parts[split:]))
            return src_dir, pattern

    return default_dir, default_pattern

SOURCE_DIR, SOURCE_PATTERN = _load_source_config()
# Output files are saved in Example/Output
OUTPUT_DIR = PROJECT_DIR / "Output"
TARGET_HOA = OUTPUT_DIR / "data_heat_of_adsorption.txt"
TARGET_PTS = OUTPUT_DIR / "data_points.txt"

HEADER_HOA = "#Structure\tMolecule\tLoading (mol/kg)\tHeat of Adsorption (kJ/mol)\n"
HEADER_PTS = "Structure\tMolecule\tMixture/Pure\tTemperature\tPressure\tLoading mol/kg\n"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, valid: list[str] | None = None) -> str:
    """Prompt until the user gives a non-empty (optionally validated) answer."""
    while True:
        ans = input(prompt).strip()
        if not ans:
            print("  (empty input — please try again)")
            continue
        if valid and ans.lower() not in [v.lower() for v in valid]:
            print(f"  Please enter one of: {', '.join(valid)}")
            continue
        return ans


def _ask_int(prompt: str, min_val: int = 1) -> int:
    """Prompt until the user gives an integer >= min_val."""
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if v >= min_val:
                return v
            print(f"  Please enter a number >= {min_val}.")
        except ValueError:
            print("  Not a valid number — please try again.")


def _try_parse_temp_from_filename(name: str) -> str | None:
    """Extract temperature like '293' from a filename containing '293K'."""
    m = re.search(r'(\d{3,4})K', name, re.IGNORECASE)
    return m.group(1) if m else None


def _ensure_header(path: Path, header: str) -> None:
    """Create target file with header if it does not already exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")
        print(f"  Created new file: {path.relative_to(SCRIPT_DIR)}")


def _is_data_line(line: str) -> bool:
    """Return True if the line looks like a data row (starts with a digit, '-', or '.')."""
    return bool(line) and (line[0].isdigit() or line[0] in "-.")


def _peek_header(filepath: Path) -> list[str]:
    """Return all leading header lines (lines that do NOT start with a digit, '-', or '.')."""
    headers: list[str] = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if _is_data_line(line):
                    break
                headers.append(line)
    except Exception:
        pass
    return headers


def _read_data_rows(filepath: Path, skip_header: bool = False) -> list[list[str]]:
    """
    Read whitespace/tab-separated rows from a file.
    Any line that does not start with a digit, '-', or '.' is treated as a header/comment and skipped.
    The skip_header parameter is kept for compatibility but is no longer needed.
    """
    rows: list[list[str]] = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if _is_data_line(line):
                rows.append(line.split())
    return rows


def _col(row: list[str], col_1based: int) -> str:
    """Return the value at 1-based column index, or '' if out of range."""
    idx = col_1based - 1
    return row[idx] if 0 <= idx < len(row) else ""


# ---------------------------------------------------------------------------
# Per-type processing
# ---------------------------------------------------------------------------

def _process_hoa(filepath: Path) -> int:
    """Ask HOA-specific questions and append rows to TARGET_HOA. Returns row count."""
    print()

    # Adsorbent
    if _ask("  Is the ADSORBENT name in the file? (yes/no): ", ["yes", "no"]).lower() == "yes":
        fw_col = _ask_int("  Which column number is the ADSORBENT? (1-based): ")
        fw_fixed = None
    else:
        fw_fixed = _ask("  Enter the ADSORBENT name: ")
        fw_col = None

    # Adsorbate
    if _ask("  Is the ADSORBATE name in the file? (yes/no): ", ["yes", "no"]).lower() == "yes":
        mol_col = _ask_int("  Which column number is the ADSORBATE? (1-based): ")
        mol_fixed = None
    else:
        mol_fixed = _ask("  Enter the ADSORBATE name: ")
        mol_col = None

    loading_col = _ask_int("  Which column is the Loading? (1-based): ")
    hoa_col     = _ask_int("  Which column is the Heat of Adsorption? (1-based): ")

    rows = _read_data_rows(filepath)
    if not rows:
        print("  No data rows found — skipping.")
        return 0

    # Detect sign: negate if all finite HOA values are negative
    hoa_vals = []
    for r in rows:
        try:
            hoa_vals.append(float(_col(r, hoa_col)))
        except (ValueError, IndexError):
            pass
    negate = bool(hoa_vals) and all(v < 0 for v in hoa_vals)
    if negate:
        print("  HOA values are all negative — will be stored as positive (negated).")

    _ensure_header(TARGET_HOA, HEADER_HOA)

    written = 0
    with open(TARGET_HOA, "a", encoding="utf-8") as fh:
        for r in rows:
            fw  = fw_fixed  if fw_fixed  is not None else _col(r, fw_col)
            mol = mol_fixed if mol_fixed is not None else _col(r, mol_col)
            try:
                loading = float(_col(r, loading_col))
                hoa     = float(_col(r, hoa_col))
            except (ValueError, IndexError):
                continue
            if negate:
                hoa = -hoa
            fh.write(f"{fw}\t{mol}\t{loading}\t{hoa}\n")
            written += 1

    return written


def _process_pts(filepath: Path) -> int:
    """Ask data_points-specific questions and append rows to TARGET_PTS. Returns row count."""
    print()

    # Framework
    fw_fixed = _ask("  Enter the framework name: ")
    fw_col = None

    # Molecule
    if _ask("  Is the molecule name in the file? (yes/no): ", ["yes", "no"]).lower() == "yes":
        mol_col = _ask_int("  Which column number is the molecule? (1-based): ")
        mol_fixed = None
    else:
        mol_fixed = _ask("  Enter the molecule name: ")
        mol_col = None

    # Temperature
    if _ask("  Is the temperature in the file? (yes/no): ", ["yes", "no"]).lower() == "yes":
        temp_col   = _ask_int("  Which column number is the temperature? (1-based): ")
        temp_fixed = None
    else:
        suggested = _try_parse_temp_from_filename(filepath.name)
        if suggested:
            raw = input(f"  Enter the temperature [K] (press Enter to accept '{suggested}'): ").strip()
            temp_fixed = raw if raw else suggested
        else:
            temp_fixed = _ask("  Enter the temperature [K]: ")
        temp_col = None

    # Mixture / pure
    mix_type = _ask("  Is it a pure or mixture file? (pure/mixture): ", ["pure", "mixture"]).lower()
    if mix_type == "mixture":
        mixture_label = _ask("  Enter the mixture name: ")
    else:
        mixture_label = "pure"

    pressure_col = _ask_int("  Which column is the Pressure? (1-based): ")
    loading_col  = _ask_int("  Which column is the Loading? (1-based): ")

    rows = _read_data_rows(filepath)
    if not rows:
        print("  No data rows found — skipping.")
        return 0

    _ensure_header(TARGET_PTS, HEADER_PTS)

    written = 0
    with open(TARGET_PTS, "a", encoding="utf-8") as fh:
        for r in rows:
            fw  = fw_fixed  if fw_fixed  is not None else _col(r, fw_col)
            mol = mol_fixed if mol_fixed is not None else _col(r, mol_col)
            T   = temp_fixed if temp_fixed is not None else _col(r, temp_col)
            try:
                pressure = float(_col(r, pressure_col))
                loading  = float(_col(r, loading_col))
            except (ValueError, IndexError):
                continue
            fh.write(f"{fw}\t{mol}\t{mixture_label}\t{T}\t{pressure}\t{loading}\n")
            written += 1

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source folder not found:\n  {SOURCE_DIR}")
        sys.exit(1)

    files = sorted(f for f in SOURCE_DIR.glob(SOURCE_PATTERN) if f.is_file())
    if not files:
        print(f"No files found for pattern '{SOURCE_PATTERN}' under {SOURCE_DIR}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) for pattern '{SOURCE_PATTERN}' under:\n  {SOURCE_DIR}\n")
    total_written = 0

    for filepath in files:
        print("=" * 60)
        print(f"File: {filepath.name}")
        print(f"Path: {filepath.relative_to(SOURCE_DIR.parent)}")
        print()

        file_type = _ask(
            "  What kind of file is this?\n"
            "    1 = data_heat_of_adsorption\n"
            "    2 = data_points (isotherm / RASPA results)\n"
            "    s = skip\n"
            "  Choice: ",
            valid=["1", "2", "s"],
        ).lower()

        if file_type == "s":
            print("  Skipped.\n")
            continue

        for h in _peek_header(filepath):
            print(f"  Header: {h}")

        if file_type == "1":
            n = _process_hoa(filepath)
        else:
            n = _process_pts(filepath)

        print(f"\n  Written {n} row(s) to Input file.")
        total_written += n
        print()

    print("=" * 60)
    print(f"Done. Total rows written: {total_written}")
    print(f"  HOA target : {TARGET_HOA}")
    print(f"  Points target: {TARGET_PTS}")


if __name__ == "__main__":
    main()
