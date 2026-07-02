"""Execute a # %% Python analysis file and export HTML without writing .ipynb files."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import jupytext
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Divar-Real-State-Ads").exists() and (path / "notebooks").exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def find_reports_root(output_path: Path) -> Path:
    output_path = output_path.resolve()
    if output_path.parent.name == "html":
        return output_path.parent.parent
    return output_path.parent


def sync_report_artifacts(project_root: Path, reports_root: Path) -> None:
    source_root = project_root / "notebooks" / "outputs"
    destination_root = reports_root / "notebooks" / "outputs"

    for name in ("figures", "models"):
        source = source_root / name
        destination = destination_root / name
        destination.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copytree(source, destination, dirs_exist_ok=True)


def export_html(input_path: Path, output_path: Path, timeout: int) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

    project_root = find_project_root(input_path)
    notebook = jupytext.read(input_path)

    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(project_root)}})

    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(notebook)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    sync_report_artifacts(project_root, find_reports_root(output_path))
    print(f"Exported: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a # %% Python analysis file and export an HTML report."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=-1, type=int)
    args = parser.parse_args()

    export_html(args.input, args.output, args.timeout)


if __name__ == "__main__":
    main()
