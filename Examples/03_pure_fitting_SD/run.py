#!/usr/bin/env python3
"""Run the pipeline from this example folder using the shared ``Code_V5/Code`` package.

* This script's directory is the **run root**: ``config.in``, ``design.in``, and
  ``Output/`` are resolved here (via ``PIPELINE_REPO_ROOT``).
* ``Main.py`` is executed from the nearest ancestor directory that contains
  ``Code/Main.py`` (so it works whether the project folder is ``Code_V5``,
  ``Code_V5`` nested under ``CodeVersions``, etc.).

You can run from any shell working directory, for example:

    python path/to/Examples/01_pure_fitting_simple/run.py

The child process ``cwd`` is set to this example folder so relative paths in
``config.in`` resolve correctly.
"""

import os
import subprocess
import sys
from pathlib import Path

# Example folder (config.in, Output/, …) — never derived from os.getcwd().
RUN_ROOT = Path(__file__).resolve().parent


def _find_pipeline_code_main() -> Path | None:
    """Return ``…/Code/Main.py`` by walking up from *RUN_ROOT* until ``Code/Main.py`` exists."""
    for d in (RUN_ROOT, *RUN_ROOT.parents):
        candidate = d / "Code" / "Main.py"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    main_py = _find_pipeline_code_main()
    if main_py is None or not main_py.is_file():
        print(
            "error: could not find Code/Main.py (searched from "
            f"{RUN_ROOT} upward).",
            file=sys.stderr,
        )
        return 1
    env = os.environ.copy()
    env["PIPELINE_REPO_ROOT"] = str(RUN_ROOT)
    proc = subprocess.run(
        [sys.executable, str(main_py), *sys.argv[1:]],
        cwd=RUN_ROOT,
        env=env,
    )
    return int(proc.returncode if proc.returncode is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
