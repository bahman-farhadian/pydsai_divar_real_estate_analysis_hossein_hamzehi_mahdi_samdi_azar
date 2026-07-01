"""Compress and decompress project CSV files with Zstandard.

The repository tracks `.csv.zst` archives and ignores expanded `.csv` files.
Compression uses the maximum Zstandard level because it is a one-time packaging
step, while decompression remains fast for normal project runs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from shutil import which

try:
    import zstandard as zstd
except ImportError:
    zstd = None


BUFFER_SIZE = 1024 * 1024
DEFAULT_LEVEL = 22


def _format_size(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def compress(input_path: Path, output_path: Path, level: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if zstd is None:
        _run_zstd_cli(["--ultra", f"-{level}", "-T0", "-f", str(input_path), "-o", str(output_path)])
        _print_ratio(input_path, output_path, "Compressed")
        return

    compressor = zstd.ZstdCompressor(level=level, threads=-1)

    with input_path.open("rb") as source, output_path.open("wb") as target:
        with compressor.stream_writer(target) as writer:
            shutil.copyfileobj(source, writer, length=BUFFER_SIZE)

    _print_ratio(input_path, output_path, "Compressed")


def _print_ratio(input_path: Path, output_path: Path, action: str) -> None:
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = output_size / input_size if input_size else 0

    print(f"{action}: {input_path} -> {output_path}")
    print(f"Input size:  {_format_size(input_size)}")
    print(f"Output size: {_format_size(output_size)}")
    print(f"Ratio:       {ratio:.2%}")


def decompress(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if zstd is None:
        _run_zstd_cli(["-d", "-f", str(input_path), "-o", str(output_path)])
        print(f"Decompressed: {input_path} -> {output_path}")
        print(f"Output size:  {_format_size(output_path.stat().st_size)}")
        return

    decompressor = zstd.ZstdDecompressor()

    with input_path.open("rb") as source, output_path.open("wb") as target:
        with decompressor.stream_reader(source) as reader:
            shutil.copyfileobj(reader, target, length=BUFFER_SIZE)

    print(f"Decompressed: {input_path} -> {output_path}")
    print(f"Output size:  {_format_size(output_path.stat().st_size)}")


def _run_zstd_cli(args: list[str]) -> None:
    if which("zstd") is None:
        raise RuntimeError(
            "Install the Python package `zstandard` from requirements.txt or install the `zstd` CLI."
        )
    subprocess.run(["zstd", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress or decompress project CSV files with Zstandard."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("--input", required=True, type=Path)
    compress_parser.add_argument("--output", required=True, type=Path)
    compress_parser.add_argument("--level", default=DEFAULT_LEVEL, type=int)

    decompress_parser = subparsers.add_parser("decompress")
    decompress_parser.add_argument("--input", required=True, type=Path)
    decompress_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    if args.command == "compress":
        compress(args.input, args.output, args.level)
    elif args.command == "decompress":
        decompress(args.input, args.output)


if __name__ == "__main__":
    main()
