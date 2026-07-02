"""Remove generated project outputs without touching source files or .venv."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Divar-Real-State-Ads").exists() and (path / "notebooks").exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    print(f"{'Would remove' if dry_run else 'Removing'}: {path}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated report outputs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-expanded-csv", action="store_true")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also remove old generated-output directories from pre-final layouts.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    targets = [
        project_root / "reports",
    ]
    if args.include_legacy:
        targets.extend(
            [
                project_root / "data",
                project_root / "notebooks" / "outputs",
            ]
        )
    if args.include_expanded_csv:
        targets.extend(
            [
                project_root / "Divar-Real-State-Ads" / "divar_real_estate_ads.csv",
                project_root / "Divar-Real-State-Ads" / "sampled_data.csv",
            ]
        )

    for target in targets:
        remove_path(target, args.dry_run)


if __name__ == "__main__":
    main()
