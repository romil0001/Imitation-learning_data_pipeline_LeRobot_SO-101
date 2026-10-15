"""Import user policies dropped in so101/algorithms/plugins/."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def discover_plugin_files(directory: Path) -> list[Path]:
    """Python files in ``directory`` whose names do not start with '_'."""
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.glob("*.py"))
        if path.name != "__init__.py" and not path.name.startswith("_")
    ]


def load_plugins(directory: Path | None = None) -> list[str]:
    directory = directory or Path(__file__).resolve().parent / "plugins"
    loaded: list[str] = []
    for path in discover_plugin_files(directory):
        module_name = f"so101_algo_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(path.stem)
    return loaded
