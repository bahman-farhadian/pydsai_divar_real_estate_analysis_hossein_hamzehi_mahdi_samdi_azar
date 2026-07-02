"""Execute a # %% Python analysis file and export HTML without writing .ipynb files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jupytext
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

INLINE_MATPLOTLIB_SETUP = """
import os
os.environ.pop("MPLBACKEND", None)
from IPython import get_ipython

_ipython = get_ipython()
if _ipython is not None:
    _ipython.run_line_magic("matplotlib", "inline")
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("png")
    except Exception:
        pass
"""


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Divar-Real-State-Ads").exists() and (path / "notebooks").exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def export_html(input_path: Path, output_path: Path, timeout: int) -> None:
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

    project_root = find_project_root(input_path)
    notebook = jupytext.read(input_path)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            INLINE_MATPLOTLIB_SETUP,
            metadata={"tags": ["injected-html-export-setup"]},
        ),
    )

    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(project_root)}})
    notebook.cells.pop(0)

    exporter = HTMLExporter()
    if "embed_images" in exporter.traits():
        exporter.embed_images = True
    body, _ = exporter.from_notebook_node(notebook)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
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
