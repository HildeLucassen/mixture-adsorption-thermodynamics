#!/usr/bin/env python3
"""Run the pipeline by executing Code/Main.py with cwd at the repo root."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    main_py = root / "Code" / "Main.py"
    if not main_py.is_file():
        print(f"error: missing {main_py}", file=sys.stderr)
        return 1
    return subprocess.call(
        [sys.executable, str(main_py), *sys.argv[1:]],
        cwd=str(root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
