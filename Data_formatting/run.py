from pathlib import Path
import importlib.util


BASE_DIR = Path(__file__).resolve().parent
TOOL_PATH = BASE_DIR /"Code" / "formatting_tool.py"

def _load_tool(path: Path):
    spec = importlib.util.spec_from_file_location("formatting_tool", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load formatting tool from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tool = _load_tool(TOOL_PATH)
    tool.main()


if __name__ == "__main__":
    main()
