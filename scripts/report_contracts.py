"""Shared artifact contracts for equivalent analysis implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


EDA_SUMMARY_COLUMNS = ["metric", "value"]
EDA_CITY_COLUMNS = [
    "city_slug",
    "price_per_sqm_median",
    "price_per_sqm_mean",
    "price_per_sqm_count",
    "building_size_median",
    "price_value_median",
]
TEXT_METRIC_COLUMNS = [
    "task",
    "implementation",
    "model_name",
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "train_rows",
    "test_rows",
]
TEXT_PREDICTION_COLUMNS = [
    "row_index",
    "predicted_user_type",
    "prediction_confidence",
]


def write_csv(frame: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    """Write a stable, explicitly ordered CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=list(columns)).to_csv(path, index=False)


def write_manifest(path: Path, implementation: str, artifacts: dict[str, Path]) -> None:
    """Record the artifact contract fulfilled by one implementation."""
    project_root = path.parents[2]
    payload = {
        "implementation": implementation,
        "artifacts": {
            name: str(value.relative_to(project_root))
            for name, value in artifacts.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_csv(path: Path, columns: Iterable[str]) -> list[str]:
    if not path.exists():
        return [f"missing artifact: {path}"]
    actual = list(pd.read_csv(path, nrows=0).columns)
    expected = list(columns)
    if actual != expected:
        return [f"schema mismatch: {path} (expected {expected}, found {actual})"]
    return []


def validate_report_contracts(project_root: Path, use_cuda: bool) -> list[str]:
    """Return all contract violations from a completed pipeline run."""
    data = project_root / "reports" / "data"
    figures = project_root / "reports" / "figures"
    errors: list[str] = []

    for implementation in ("pandas", "polars_duckdb"):
        prefix = f"eda_{implementation}"
        errors.extend(validate_csv(data / f"{prefix}_summary.csv", EDA_SUMMARY_COLUMNS))
        errors.extend(validate_csv(data / f"{prefix}_city_statistics.csv", EDA_CITY_COLUMNS))
        correlation_path = data / f"{prefix}_correlation_matrix.csv"
        if not correlation_path.exists():
            errors.append(f"missing artifact: {correlation_path}")
        for figure in (
            "price_distribution",
            "price_per_sqm_distribution",
            "building_size_distribution",
            "rooms_distribution",
            "city_distribution",
            "property_type_distribution",
            "user_type_distribution",
            "correlation_matrix",
            "scatter_plots",
            "price_by_city",
            "price_by_property_type",
            "price_by_user_type",
            "temporal_analysis",
            "rental_market",
        ):
            if not (figures / f"02_{implementation}_{figure}.png").exists():
                errors.append(f"missing figure: reports/figures/02_{implementation}_{figure}.png")

    implementations = ["cpu"] + (["torch_cuda"] if use_cuda else [])
    for implementation in implementations:
        errors.extend(
            validate_csv(
                data / f"text_classification_{implementation}_metrics.csv",
                TEXT_METRIC_COLUMNS,
            )
        )
        errors.extend(
            validate_csv(
                data / f"user_type_predictions_{implementation}.csv",
                TEXT_PREDICTION_COLUMNS,
            )
        )
        for figure in (
            "cat3_distribution",
            "cat3_model_comparison",
            "cat3_confusion_matrix",
            "user_type_distribution",
            "user_type_model_comparison",
            "user_type_confusion_matrix",
            "user_type_prediction_confidence",
        ):
            if not (figures / f"06_{implementation}_{figure}.png").exists():
                errors.append(f"missing figure: reports/figures/06_{implementation}_{figure}.png")

    return errors
