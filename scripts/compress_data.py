"""Compress CSV data files to Zstandard-compressed Parquet.

The analysis scripts currently read CSV files, but Parquet is much smaller for
storage and transfer. Keep the original CSV locally when running the existing
pipeline, and use this script to create compressed archives outside Git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compress_csv(input_path: Path, output_path: Path, compression_level: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="zstd",
        compression_level=compression_level,
    )

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = output_size / input_size if input_size else 0

    print(f"Input:  {input_path} ({input_size / 1024**2:.2f} MiB)")
    print(f"Output: {output_path} ({output_size / 1024**2:.2f} MiB)")
    print(f"Size ratio: {ratio:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress a CSV file into Parquet with aggressive Zstandard compression."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the compressed Parquet file to create.",
    )
    parser.add_argument(
        "--compression-level",
        default=19,
        type=int,
        help="Zstandard compression level passed to pyarrow. Default: 19.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    compress_csv(args.input, args.output, args.compression_level)


if __name__ == "__main__":
    main()
