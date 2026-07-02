"""Run the project reports with dependency-aware parallel execution."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    input_path: Path
    output_path: Path
    runtime: str
    depends_on: tuple[str, ...] = ()


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Divar-Real-State-Ads").exists() and (path / "notebooks").exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def gpu_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return result.stdout.strip() == "True"


def display_command(project_root: Path, command: list[str]) -> str:
    display_parts = []
    for part in command:
        try:
            path = Path(part)
            if path.is_absolute():
                display_parts.append(str(path.relative_to(project_root)))
                continue
        except ValueError:
            pass
        display_parts.append(part)
    return " ".join(display_parts)


def sanitize_output(project_root: Path, value: str) -> str:
    return value.replace(str(project_root), ".")


def run_stage(project_root: Path, stage: Stage, timeout: int) -> dict[str, str]:
    start = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    command = [
        sys.executable,
        str(project_root / "scripts" / "export_html.py"),
        "--input",
        str(project_root / stage.input_path),
        "--output",
        str(project_root / stage.output_path),
        "--timeout",
        str(timeout),
    ]

    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        env={**os.environ, "PYDEVD_DISABLE_FILE_VALIDATION": "1"},
    )
    finished_at = datetime.now(timezone.utc)
    duration = time.perf_counter() - start
    status = "success" if result.returncode == 0 else "failed"

    log_dir = project_root / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    log_path.write_text(
        "\n".join(
            [
                f"command: {display_command(project_root, command)}",
                f"returncode: {result.returncode}",
                "",
                "[stdout]",
                sanitize_output(project_root, result.stdout),
                "",
                "[stderr]",
                sanitize_output(project_root, result.stderr),
            ]
        ),
        encoding="utf-8",
    )

    return {
        "stage": stage.name,
        "input": str(stage.input_path),
        "output": str(stage.output_path),
        "runtime": stage.runtime,
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": f"{duration:.3f}",
        "log": str(log_path.relative_to(project_root)),
    }


def stage_resource(stage: Stage) -> str:
    return "cuda" if stage.runtime == "cuda" else "cpu"


def run_stages(
    project_root: Path,
    stages: list[Stage],
    cpu_jobs: int,
    cuda_jobs: int,
    timeout: int,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    if not stages:
        return results

    limits = {"cpu": max(1, cpu_jobs), "cuda": max(0, cuda_jobs)}
    active = {"cpu": 0, "cuda": 0}
    stage_map = {stage.name: stage for stage in stages}
    unknown_dependencies = sorted(
        {
            dependency
            for stage in stages
            for dependency in stage.depends_on
            if dependency not in stage_map
        }
    )
    if unknown_dependencies:
        raise ValueError(f"Unknown stage dependencies: {', '.join(unknown_dependencies)}")

    pending = dict(stage_map)
    completed: set[str] = set()
    running = {}

    max_workers = max(1, min(sum(limits.values()), len(stages)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            ready = [
                stage
                for stage in pending.values()
                if all(dependency in completed for dependency in stage.depends_on)
            ]
            for stage in ready:
                resource = stage_resource(stage)
                if active[resource] >= limits[resource]:
                    continue
                future = executor.submit(run_stage, project_root, stage, timeout)
                running[future] = stage
                active[resource] += 1
                del pending[stage.name]

            if not running:
                blocked = ", ".join(sorted(pending))
                raise RuntimeError(f"Pipeline dependency deadlock. Pending stages: {blocked}")

            for future in as_completed(running):
                stage = running.pop(future)
                active[stage_resource(stage)] -= 1
                break

            result = future.result()
            results.append(result)
            print(f"{result['status'].upper()}: {result['stage']} ({result['duration_seconds']}s)")
            if result["status"] != "success":
                raise RuntimeError(f"Stage failed: {result['stage']} - see {result['log']}")
            completed.add(stage.name)

    return results


def write_runtime_summary(project_root: Path, rows: list[dict[str, str]]) -> None:
    summary_path = project_root / "reports" / "runtime_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "input",
        "output",
        "runtime",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "log",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Runtime summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Divar analysis report pipeline.")
    parser.add_argument("--timeout", type=int, default=-1)
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    (project_root / "reports" / "html").mkdir(parents=True, exist_ok=True)
    (project_root / "reports" / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (project_root / "reports" / "models").mkdir(parents=True, exist_ok=True)

    use_cuda = gpu_available()

    stages = [
        Stage("01_data_quality", Path("notebooks/01_data_quality.py"), Path("reports/html/01_data_quality.html"), "cpu"),
        Stage("02_eda", Path("notebooks/02_eda.py"), Path("reports/html/02_eda.html"), "cpu", ("01_data_quality",)),
        Stage(
            "02_eda_polars_duckdb",
            Path("notebooks/02_eda_polars_duckdb.py"),
            Path("reports/html/02_eda_polars_duckdb.html"),
            "cpu-parallel",
            ("01_data_quality",),
        ),
        Stage(
            "06_text_classification",
            Path("notebooks/06_text_classification.py"),
            Path("reports/html/06_text_classification.html"),
            "cpu",
            ("01_data_quality",),
        ),
        Stage(
            "03_market_analysis",
            Path("notebooks/03_market_analysis.py"),
            Path("reports/html/03_market_analysis.html"),
            "cpu",
            ("02_eda",),
        ),
        Stage(
            "04_clustering_MiniBatchKMeans",
            Path("notebooks/04_clustering_MiniBatchKMeans.py"),
            Path("reports/html/04_clustering_MiniBatchKMeans.html"),
            "cpu",
            ("02_eda",),
        ),
        Stage(
            "05_price_prediction",
            Path("notebooks/05_price_prediction.py"),
            Path("reports/html/05_price_prediction.html"),
            "cpu",
            ("02_eda",),
        ),
    ]

    if use_cuda:
        stages.append(
            Stage(
                "06_text_classification_TorchCUDA",
                Path("notebooks/06_text_classification_TorchCUDA.py"),
                Path("reports/html/06_text_classification_TorchCUDA.html"),
                "cuda",
                ("01_data_quality",),
            )
        )
        stages.extend(
            [
                Stage(
                    "04_clustering_TorchCUDAKMeans",
                    Path("notebooks/04_clustering_TorchCUDAKMeans.py"),
                    Path("reports/html/04_clustering_TorchCUDAKMeans.html"),
                    "cuda",
                    ("02_eda",),
                ),
                Stage(
                    "05_price_prediction_TorchCUDA",
                    Path("notebooks/05_price_prediction_TorchCUDA.py"),
                    Path("reports/html/05_price_prediction_TorchCUDA.html"),
                    "cuda",
                    ("02_eda",),
                ),
            ]
        )

    stages.append(
        Stage(
            "04_clustering_StandardKMeans",
            Path("notebooks/04_clustering_StandardKMeans.py"),
            Path("reports/html/04_clustering_StandardKMeans.html"),
            "cpu",
            ("02_eda",),
        )
    )

    all_results = run_stages(
        project_root,
        stages,
        cpu_jobs=max(1, os.cpu_count() or 1),
        cuda_jobs=1 if use_cuda else 0,
        timeout=args.timeout,
    )
    write_runtime_summary(project_root, all_results)


if __name__ == "__main__":
    main()
