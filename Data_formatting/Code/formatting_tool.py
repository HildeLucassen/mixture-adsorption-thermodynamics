"""
formatting_tool.py
==================
Batch-only: reads [`config.txt`](../config.txt) like ``template/config.in``,
with **one ``FILE …`` heading per raw file** (path under ``DATA_PATH`` or just the basename) and **all settings on the lines beneath it**.

Temperature is given explicitly per file via ``TEMPERATURE`` (Kelvin string or number)—no modes.

Each run recreates ``Output/data_points.txt`` and ``Output/data_heat_of_adsorption.txt``
(headers + this run's rows only — no accumulating duplicates).

Run from Data_formatting: ``python run.py`` or ``python Code/formatting_tool.py``.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH_FILE = PROJECT_DIR / "config.txt"

OUTPUT_DIR = PROJECT_DIR / "Output"
TARGET_HOA = OUTPUT_DIR / "data_heat_of_adsorption.txt"
TARGET_PTS = OUTPUT_DIR / "data_points.txt"

HEADER_HOA = "#Structure\tMolecule\tLoading [mol/kg]\tHeat of Adsorption [kJ/mol]\n"


def _header_pts_lines(pressure_unit: str) -> str:
    u = pressure_unit.strip() if pressure_unit else "Pa"
    if not u:
        u = "Pa"
    return (
        "Structure\tMolecule\tMixture/Pure\tTemperature [K]\tPressure ["
        + u
        + "]\tLoading [mol/kg]\n"
    )


class FormatConfig:
    __slots__ = ("source_dir", "glob_pattern", "defaults", "file_blocks")

    def __init__(
        self,
        source_dir: Path,
        glob_pattern: str,
        defaults: dict[str, Any],
        file_blocks: list[tuple[str, dict[str, Any]]],
    ) -> None:
        self.source_dir = source_dir
        self.glob_pattern = glob_pattern
        self.defaults = defaults
        self.file_blocks = file_blocks


class PtsPrepared(NamedTuple):
    structure: str
    molecule: str
    temperature_k: str
    mixture_label: str
    pressure_column: int
    loading_column: int


class HoaPrepared(NamedTuple):
    structure: str
    molecule: str
    loading_column: int
    deltah_column: int


def _to_int_maybe(s: str) -> Any:
    try:
        return int(float(s.strip()))
    except ValueError:
        return s.strip()


def _map_kv(uk: str, val: str) -> dict[str, Any]:
    """Map one KEY line (uppercase) to internal merge fragment."""
    s = val.strip()
    if not s:
        return {}
    key = uk.upper()
    if key == "STRUCTURE":
        return {"structure": s}
    if key == "MOLECULE":
        return {"molecule": s}
    if key == "TEMPERATURE":
        return {"temperature": s}
    if key == "KIND":
        return {"kind": s.lower()}
    if key == "MIXTURE_PURE":
        return {"mixture_pure": s}
    if key == "PRESSURE_UNIT":
        return {"pressure_unit": s}
    if key == "PRESSURE" or key == "PRESSURE_COLUMN":
        return {"pressure": _to_int_maybe(s)}
    if key == "LOADING" or key == "LOADING_COLUMN":
        return {"loading": _to_int_maybe(s)}
    if key == "DELTAH" or key == "DELTAH_COLUMN":
        return {"deltah": _to_int_maybe(s)}
    return {}


def _parse_input_data_dir(path: Path) -> tuple[dict[str, str], dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """
    Returns (meta, defaults, file_blocks).
    meta: data_path, data_glob (optional).
    defaults: keys before the first FILE line (optional shared settings).
    file_blocks: (FILE path/pattern, settings dict) in file order; later matching blocks win on merge.
    """
    meta: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    blocks: list[tuple[str, dict[str, Any]]] = []
    current_spec: str | None = None
    current: dict[str, Any] = {}

    def flush() -> None:
        nonlocal current_spec, current
        if current_spec is not None:
            blocks.append((current_spec, current))
            current = {}
            current_spec = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            uk, val = parts[0].upper(), parts[1].strip()
            if uk == "FILE":
                flush()
                current_spec = val.replace("\\", "/").lstrip("./")
                current = {}
                continue
            if uk == "DATA_PATH":
                meta["data_path"] = val.replace("\\", "/")
                continue
            if uk == "DATA_GLOB":
                meta["data_glob"] = val.strip()
                continue
            frag = _map_kv(uk, val)
            if not frag:
                continue
            if current_spec is None:
                defaults.update(frag)
            else:
                current.update(frag)
    flush()
    return meta, defaults, blocks


def load_format_config() -> FormatConfig:
    if not INPUT_PATH_FILE.exists():
        print(f"ERROR: Missing configuration file:\n  {INPUT_PATH_FILE}", file=sys.stderr)
        sys.exit(1)
    meta, defaults, file_blocks = _parse_input_data_dir(INPUT_PATH_FILE)

    if not file_blocks:
        print(
            "ERROR: No FILE blocks in config.txt. Use one block per data file, e.g.\n"
            "  FILE  sim-…333K.load\n"
            "  STRUCTURE …\n"
            "  MOLECULE …\n"
            "  TEMPERATURE 333\n"
            "  KIND pts\n"
            "  PRESSURE 1\n"
            "  LOADING 2\n",
            file=sys.stderr,
        )
        sys.exit(1)

    path_str = meta.get("data_path", "Example/Data").strip() or "Example/Data"
    glob_pat = meta.get("data_glob", "**/*").strip() or "**/*"
    pth = Path(path_str)
    source_dir = pth if pth.is_absolute() else (PROJECT_DIR / pth)

    if "kind" in defaults:
        defaults["kind"] = str(defaults["kind"]).strip().lower()

    return FormatConfig(source_dir, glob_pat, defaults, file_blocks)


def _spec_matches(rel_posix: str, basename: str, spec: str) -> bool:
    spec_clean = spec.replace("\\", "/").strip().lstrip("./")
    if not spec_clean:
        return False
    if "*" in spec_clean or "?" in spec_clean:
        return fnmatch.fnmatch(rel_posix, spec_clean) or fnmatch.fnmatch(basename, spec_clean)
    if "/" in spec_clean:
        return rel_posix == spec_clean
    return basename == spec_clean


def _merge_for_data_file(
    defaults: dict[str, Any],
    file_blocks: list[tuple[str, dict[str, Any]]],
    rel_posix: str,
    basename: str,
) -> tuple[dict[str, Any], bool]:
    merged = dict(defaults)
    hit = False
    for spec, patch in file_blocks:
        if _spec_matches(rel_posix, basename, spec):
            merged.update(patch)
            hit = True
    kin = merged.get("kind")
    if kin is None or str(kin).strip() == "":
        merged["kind"] = "pts"
    else:
        merged["kind"] = str(kin).strip().lower()
    return merged, hit


def _coerce_positive_int(val: Any, *, name: str, errs: list[str]) -> int | None:
    if val is None:
        errs.append(name)
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        n = val
    else:
        try:
            n = int(float(str(val).strip()))
        except ValueError:
            errs.append(f"{name} (not an integer)")
            return None
    if n < 1:
        errs.append(f"{name} (must be >= 1)")
        return None
    return n


def _validate_and_prepare_pts(_filepath: Path, merged: dict[str, Any]) -> tuple[str | None, PtsPrepared | None]:
    errs: list[str] = []

    st = merged.get("structure")
    fw = str(st).strip() if st is not None else ""
    if not fw:
        errs.append("STRUCTURE")

    mol_raw = merged.get("molecule")
    mol = str(mol_raw).strip() if mol_raw is not None else ""
    if not mol:
        errs.append("MOLECULE")

    t_raw = merged.get("temperature")
    if t_raw is None or str(t_raw).strip() == "":
        errs.append("TEMPERATURE (K, per FILE block)")
    temperature_k = str(t_raw).strip() if t_raw is not None else ""

    mp_raw = merged.get("mixture_pure", "pure")
    mp_str = str(mp_raw).strip() if mp_raw is not None else "pure"
    mixture_label = "pure" if mp_str.lower() == "pure" else mp_str

    pc = _coerce_positive_int(merged.get("pressure"), name="PRESSURE", errs=errs)
    lc = _coerce_positive_int(merged.get("loading"), name="LOADING", errs=errs)

    if errs:
        return "; ".join(errs), None
    assert pc is not None and lc is not None
    return None, PtsPrepared(
        structure=fw,
        molecule=mol,
        temperature_k=temperature_k,
        mixture_label=mixture_label,
        pressure_column=pc,
        loading_column=lc,
    )


def _validate_and_prepare_hoa(_filepath: Path, merged: dict[str, Any]) -> tuple[str | None, HoaPrepared | None]:
    errs: list[str] = []

    st = merged.get("structure")
    fw = str(st).strip() if st is not None else ""
    if not fw:
        errs.append("STRUCTURE")

    mol_raw = merged.get("molecule")
    mol = str(mol_raw).strip() if mol_raw is not None else ""
    if not mol:
        errs.append("MOLECULE")

    lc = _coerce_positive_int(merged.get("loading"), name="LOADING", errs=errs)
    dh = _coerce_positive_int(merged.get("deltah"), name="DELTAH", errs=errs)

    if errs:
        return "; ".join(errs), None
    assert lc is not None and dh is not None

    return None, HoaPrepared(
        structure=fw,
        molecule=mol,
        loading_column=lc,
        deltah_column=dh,
    )


def _init_output_txt_files(*, pts_header_lines: str) -> None:
    """Create Output/ if missing; replace points and HOA aggregates with headers only (no duplicate runs)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PTS.write_text(pts_header_lines, encoding="utf-8")
    TARGET_HOA.write_text(HEADER_HOA, encoding="utf-8")


def _is_data_line(line: str) -> bool:
    return bool(line) and (line[0].isdigit() or line[0] in "-.")


def _read_data_rows(filepath: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if _is_data_line(line):
                rows.append(line.split())
    return rows


def _col(row: list[str], col_1based: int) -> str:
    idx = col_1based - 1
    return row[idx] if 0 <= idx < len(row) else ""


def run_pts(filepath: Path, p: PtsPrepared) -> int:
    rows = _read_data_rows(filepath)
    if not rows:
        print("  No data rows found — skipping.")
        return 0

    written = 0
    with open(TARGET_PTS, "a", encoding="utf-8") as fh:
        for r in rows:
            try:
                pressure = float(_col(r, p.pressure_column))
                loading = float(_col(r, p.loading_column))
            except (ValueError, IndexError):
                continue
            fh.write(
                f"{p.structure}\t{p.molecule}\t{p.mixture_label}\t{p.temperature_k}\t{pressure}\t{loading}\n"
            )
            written += 1
    return written


def run_hoa(filepath: Path, h: HoaPrepared) -> int:
    rows = _read_data_rows(filepath)
    if not rows:
        print("  No data rows found — skipping.")
        return 0

    hoa_vals: list[float] = []
    for r in rows:
        try:
            hoa_vals.append(float(_col(r, h.deltah_column)))
        except (ValueError, IndexError):
            pass
    negate = bool(hoa_vals) and all(v < 0 for v in hoa_vals)
    if negate:
        print("  HOA values are all negative — will be stored as positive (negated).")

    written = 0
    with open(TARGET_HOA, "a", encoding="utf-8") as fh:
        for r in rows:
            try:
                loading = float(_col(r, h.loading_column))
                hoa_val = float(_col(r, h.deltah_column))
            except (ValueError, IndexError):
                continue
            if negate:
                hoa_val = -hoa_val
            fh.write(f"{h.structure}\t{h.molecule}\t{loading}\t{hoa_val}\n")
            written += 1
    return written


def main() -> None:
    cfg = load_format_config()

    if not cfg.source_dir.exists():
        print(f"ERROR: Source folder not found:\n  {cfg.source_dir}")
        sys.exit(1)

    files = sorted(f for f in cfg.source_dir.glob(cfg.glob_pattern) if f.is_file())
    if not files:
        print(f"No files for glob {cfg.glob_pattern!r} under\n  {cfg.source_dir}")
        sys.exit(0)

    print(f"Found {len(files)} file(s), glob={cfg.glob_pattern!r},\n  {cfg.source_dir}\n")

    pressure_unit_key = cfg.defaults.get("pressure_unit")
    pressure_unit_str = (
        str(pressure_unit_key).strip() if pressure_unit_key is not None else "Pa"
    ) or "Pa"
    pts_header_raw = _header_pts_lines(pressure_unit_str)
    _init_output_txt_files(pts_header_lines=pts_header_raw)
    print(
        "  Initialized Output (fresh files, headers only):\n"
        f"    {TARGET_PTS.relative_to(PROJECT_DIR)}\n"
        f"    {TARGET_HOA.relative_to(PROJECT_DIR)}\n"
    )

    total_written = 0
    for filepath in files:
        print("=" * 60)
        print(f"File: {filepath.name}")
        try:
            rel_parent = filepath.relative_to(cfg.source_dir.parent)
        except ValueError:
            rel_parent = filepath
        print(f"Path: {rel_parent}\n")

        rel_posix = filepath.relative_to(cfg.source_dir).as_posix()
        merged, cohort_hit = _merge_for_data_file(
            cfg.defaults, cfg.file_blocks, rel_posix, filepath.name
        )

        if not cohort_hit:
            print("WARN: No FILE … block matches this path — skipped.\n")
            continue

        kind = merged["kind"]

        if kind == "skip":
            print("  Skipped (KIND skip).\n")
            continue

        if kind == "hoa":
            err, hp = _validate_and_prepare_hoa(filepath, merged)
            if err or hp is None:
                print(
                    f"ERROR: HOA block invalid for {filepath.relative_to(PROJECT_DIR)}:\n  {err}"
                )
                sys.exit(1)
            n = run_hoa(filepath, hp)
        elif kind == "pts":
            err, pp = _validate_and_prepare_pts(filepath, merged)
            if err or pp is None:
                print(
                    f"ERROR: PTS block invalid for {filepath.relative_to(PROJECT_DIR)}:\n  {err}"
                )
                sys.exit(1)
            n = run_pts(filepath, pp)
        else:
            print(f"ERROR: Unknown KIND {kind!r}")
            sys.exit(1)

        print(f"\n  Written {n} row(s) to Output.")
        total_written += n
        print()

    print("=" * 60)
    print(f"Done. Total rows written: {total_written}")
    print(f"  HOA target : {TARGET_HOA}")
    print(f"  Points target: {TARGET_PTS}")


if __name__ == "__main__":
    main()
