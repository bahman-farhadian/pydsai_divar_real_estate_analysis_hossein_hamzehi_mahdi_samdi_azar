"""Validate the runtime environment and required project inputs."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "pandas",
    "numpy",
    "pyarrow",
    "sklearn",
    "matplotlib",
    "seaborn",
    "torch",
    "jupytext",
    "nbconvert",
    "polars",
    "duckdb",
]


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Divar-Real-State-Ads").exists() and (path / "notebooks").exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def main() -> None:
    project_root = find_project_root(Path.cwd())
    print(f"Project root: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU threads: {__import__('os').cpu_count()}")

    missing = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"OK import: {module_name}")
        except ImportError:
            missing.append(module_name)
            print(f"MISSING import: {module_name}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA runtime: {torch.version.cuda}")
    except ImportError:
        pass

    required_files = [
        project_root / "Divar-Real-State-Ads" / "divar_real_estate_ads.csv.zst",
        project_root / "Divar-Real-State-Ads" / "divar_real_estate_ads.csv",
    ]
    for path in required_files:
        print(f"{'OK' if path.exists() else 'MISSING'} file: {path.relative_to(project_root)}")

    if missing:
        raise SystemExit(f"Missing Python packages: {', '.join(missing)}")


if __name__ == "__main__":
    main()
