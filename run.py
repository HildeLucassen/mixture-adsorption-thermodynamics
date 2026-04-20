#!/usr/bin/env python3
"""Run the pipeline by executing Code/Main.py.

The repository root is always the directory that contains this file (not your
shell's current working directory). You can run:

    python "C:/.../Code_V5/run.py"
    python "C:/.../Code_V5/run.py" --any-args-passed-to-main

from any folder; the child process cwd is set to that root so relative paths
in config.in (e.g. Input\\\\...) resolve correctly.
"""

import os
import subprocess
import sys
from pathlib import Path

# Home folder: directory where this script lives (never derived from os.getcwd()).
REPO_ROOT = Path(__file__).resolve().parent
MAIN_PY = REPO_ROOT / "Code" / "Main.py"


def main() -> int:
    if not MAIN_PY.is_file():
        print(f"error: missing {MAIN_PY}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["PIPELINE_REPO_ROOT"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(MAIN_PY), *sys.argv[1:]],
        cwd=REPO_ROOT,
        env=env,
    )
    return int(proc.returncode if proc.returncode is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
