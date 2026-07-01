# Divar Real Estate Analysis

This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.

The repository contains a Python-script version of a Divar real estate analysis workflow. The original notebooks are preserved for review, and each notebook has a matching `# %%`-formatted Python file that can be run in editors such as VS Code, PyCharm, Spyder, or from a terminal.

## What This Project Does

The project analyzes real estate advertisements from Divar and turns the raw listing data into data quality reports, market summaries, visualizations, clustering outputs, price prediction results, and text classification outputs.

The workflow is split into these stages:

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `notebooks/01_data_quality.py` | Loads the raw CSV, checks missing values and invalid records, removes unusable records, and creates cleaned datasets. |
| 2 | `notebooks/02_eda.py` | Performs exploratory analysis, adds engineered features, exports summary tables, and writes `cleaned_data_with_features.csv`. |
| 3 | `notebooks/03_market_analysis.py` | Builds stakeholder-focused market summaries such as city-level prices and amenity impact. |
| 4a | `notebooks/04_clustering_MiniBatchKMeans.py` | Runs the faster clustering workflow for practical iteration on large data. |
| 4b | `notebooks/04_clustering_StandardKMeans.py` | Runs the heavier standard K-Means workflow for final clustering comparison. |
| 5 | `notebooks/05_price_prediction.py` | Trains regression models for price-per-square-meter prediction and exports model artifacts. |
| 6 | `notebooks/06_text_classification.py` | Classifies listing text and predicts missing user-type labels where possible. |

Generated outputs are written under `data/processed/` and `notebooks/outputs/`. These paths are ignored by Git because the data and model artifacts are large and reproducible.

## Repository Layout

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   └── compress_data.py
└── notebooks/
    ├── 01_data_quality.ipynb
    ├── 01_data_quality.py
    ├── 02_eda.ipynb
    ├── 02_eda.py
    ├── 03_market_analysis.ipynb
    ├── 03_market_analysis.py
    ├── 04_clustering_MiniBatchKMeans.ipynb
    ├── 04_clustering_MiniBatchKMeans.py
    ├── 04_clustering_StandardKMeans.ipynb
    ├── 04_clustering_StandardKMeans.py
    ├── 05_price_prediction.ipynb
    ├── 05_price_prediction.py
    ├── 06_text_classification.ipynb
    └── 06_text_classification.py
```

Expected local data layout after pulling the repository:

```text
data/
├── raw/
│   └── divar_real_estate_ads.csv
├── processed/
└── compressed/
```

The `data/` directory is intentionally not tracked.

## Hardware Target

The intended full run target is the NVIDIA workstation:

```text
GPU: NVIDIA GeForce GTX 1080, 8 GB VRAM
Driver: 535.261.03
CUDA shown by nvidia-smi: 12.2
CPU: 20 cores
RAM: 31 GiB
```

Most current project code uses pandas and scikit-learn, so heavy stages still use CPU cores unless a later GPU-specific refactor is added. The environment file is CUDA-ready through the PyTorch CUDA wheel index for the workstation, while the existing scikit-learn models remain the implementation used by these scripts.

## Setup On The GPU Workstation

Use Python 3.10 or 3.11 on Linux.

```bash
git clone git@github.com:bahman-farhadian/pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar.git
cd pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional CUDA check:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Data Setup

Place the raw Divar CSV here:

```text
data/raw/divar_real_estate_ads.csv
```

The scripts expect this exact filename. The raw data is not included in Git because it is large and may have separate redistribution restrictions.

## Run The Python Workflow

Run the analysis files from inside the `notebooks/` directory. This matters because several converted scripts calculate the project root from the current working directory.

```bash
cd notebooks

python 01_data_quality.py
python 02_eda.py
python 03_market_analysis.py
python 04_clustering_MiniBatchKMeans.py
python 05_price_prediction.py
python 06_text_classification.py
```

Use the standard K-Means file only when you want the slower full clustering run:

```bash
python 04_clustering_StandardKMeans.py
```

Recommended order for the workstation:

1. Run `01_data_quality.py` first to create `data/processed/cleaned_data.csv`.
2. Run `02_eda.py` second to create `data/processed/cleaned_data_with_features.csv`.
3. Run `03_market_analysis.py`, clustering, price prediction, and text classification after the processed files exist.
4. Prefer `04_clustering_MiniBatchKMeans.py` for normal iteration. Use `04_clustering_StandardKMeans.py` for final validation if time allows.

## Compress The Large Data

The raw data is around hundreds of megabytes, so keep it outside Git and create a compressed copy for transfer or archival.

This repository includes an aggressive CSV-to-Parquet compressor:

```bash
python scripts/compress_data.py --input data/raw/divar_real_estate_ads.csv --output data/compressed/divar_real_estate_ads.parquet
```

The script uses Parquet with Zstandard compression level 19. This is usually much smaller than CSV and remains directly readable by pandas:

```python
import pandas as pd

df = pd.read_parquet("data/compressed/divar_real_estate_ads.parquet")
```

For a single archive to move between machines, compress the whole local data directory after generating processed outputs:

```bash
tar --zstd -cf divar_data_and_outputs.tar.zst data/
```

Keep the resulting archive outside the repository or in ignored local storage. Do not commit raw, processed, compressed, model, or figure artifacts.

## Outputs

Important generated files include:

| Path | Created By |
| --- | --- |
| `data/processed/cleaned_data.csv` | `01_data_quality.py` |
| `data/processed/data_for_price_prediction.csv` | `01_data_quality.py` |
| `data/processed/cleaned_data_with_features.csv` | `02_eda.py` |
| `data/processed/market_analysis_city_summary.csv` | `03_market_analysis.py` |
| `data/processed/clustering_assignments.csv` | clustering scripts |
| `data/processed/price_predictions.csv` | `05_price_prediction.py` |
| `data/processed/user_type_predictions.csv` | `06_text_classification.py` |
| `notebooks/outputs/figures/` | analysis scripts |
| `notebooks/outputs/models/` | model-training scripts |

## Notes For Review

The `.ipynb` files are preserved exactly as review artifacts. The `.py` files are the runnable version of the project and use `# %%` markers so each script can still be explored cell by cell in compatible editors.

## License

The source code and documentation are released under the MIT License. The Divar dataset is not part of this license and is not redistributed in this repository.
