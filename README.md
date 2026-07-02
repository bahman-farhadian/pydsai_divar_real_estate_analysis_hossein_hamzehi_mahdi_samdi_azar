# Divar Real Estate Analysis

This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.

This repository provides a reproducible real estate analysis pipeline for the Divar Real Estate Ads dataset. The project is organized around executable `# %%` Python reports, compressed source data, CUDA-aware modeling alternatives, dependency-aware server execution, and a single generated artifact root under `reports/`.

## Project Structure

```text
.
├── Divar-Real-State-Ads/
│   ├── README.md
│   ├── divar_real_estate_ads.csv.zst
│   └── sampled_data.csv.zst
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── scripts/
│   ├── check_environment.py
│   ├── clean_outputs.py
│   ├── compress_data.py
│   ├── export_html.py
│   └── run_pipeline.py
└── notebooks/
    ├── 01_data_quality.py
    ├── 02_eda.py
    ├── 02_eda_polars_duckdb.py
    ├── 03_market_analysis.py
    ├── 04_clustering_MiniBatchKMeans.py
    ├── 04_clustering_StandardKMeans.py
    ├── 04_clustering_TorchCUDAKMeans.py
    ├── 05_price_prediction.py
    ├── 05_price_prediction_TorchCUDA.py
    ├── 06_text_classification.py
    └── 06_text_classification_TorchCUDA.py
```

All generated project outputs are written directly under:

```text
reports/html/
reports/data/
reports/figures/
reports/models/
reports/logs/
reports/runtime_summary.csv
```

The source notebooks directory is only for executable report code. It is not used as an output directory.

## Runtime Target

Target CUDA version: `12.2`

Report-generation hardware:

| Component     | Value                        |
| ------------- | ---------------------------- |
| GPU           | NVIDIA GeForce GTX 1080, 8GB |
| NVIDIA driver | 535.261.03                   |
| CUDA          | 12.2                         |
| CPU           | 20 vCPU, AMD Ryzen 9 3900X   |
| RAM           | 32GB                         |

Modern RAPIDS/cuML is not pinned because current RAPIDS releases require newer GPU architecture than Pascal/GTX 1080. CUDA acceleration in this project is implemented with PyTorch CUDA.

## Dataset

The repository stores the dataset as maximum-compression Zstandard archives:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv.zst
Divar-Real-State-Ads/sampled_data.csv.zst
```

The main pipeline uses:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv
```

`sampled_data.csv.zst` was used during development for faster local checks. The final pipeline does not read it; it is retained only as a reference archive.

## Setup

Use Python 3.10 or 3.11. The Makefile is the primary command surface for setup, data extraction, execution, and cleanup.

```bash
git clone git@github.com:bahman-farhadian/pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar.git
```

```bash
cd pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar
```

```bash
make setup
```

Validate the runtime:

```bash
make check
```

## Data Preparation

Decompress the main dataset before running the pipeline:

```bash
make extract-data
```

Decompress the sampled dataset only when needed:

```bash
make extract-sample
```

Recreate compressed archives:

```bash
make compress-data
```

```bash
make compress-sample
```

## Pipeline Graph

```mermaid
flowchart TD
    raw["Divar CSV"] --> quality["01_data_quality.py<br/>Cleaned baseline"]

    quality --> eda["02_eda.py<br/>Feature dataset"]
    quality --> eda_fast["02_eda_polars_duckdb.py<br/>Polars/DuckDB EDA"]
    quality --> text_cpu["06_text_classification.py<br/>CPU text baseline"]
    quality --> text_cuda["06_text_classification_TorchCUDA.py<br/>CUDA text classifier"]

    eda --> market["03_market_analysis.py<br/>Market summaries"]
    eda --> cluster_cpu["04_clustering_MiniBatchKMeans.py<br/>CPU clustering baseline"]
    eda --> cluster_cuda["04_clustering_TorchCUDAKMeans.py<br/>CUDA clustering"]
    eda --> price_cpu["05_price_prediction.py<br/>CPU price baseline"]
    eda --> price_cuda["05_price_prediction_TorchCUDA.py<br/>CUDA price prediction"]
    eda --> cluster_validation["04_clustering_StandardKMeans.py<br/>Validation run"]
```

`01_data_quality.py` creates the cleaned baseline. `02_eda.py` creates the feature-enhanced dataset used by market analysis, clustering, and price prediction. Text classification can run after the cleaned baseline is available.

## Recommended Server Execution

Run the complete dependency-aware pipeline:

```bash
make run
```

The runner writes:

```text
reports/runtime_summary.csv
reports/logs/
reports/html/
reports/data/
reports/figures/
reports/models/
```

The scheduler runs `01_data_quality.py` first. After the cleaned baseline exists, it can run text classification and EDA work in parallel. Market analysis, clustering, and price prediction start as soon as `02_eda.py` finishes and do not wait for independent text-classification stages. CPU reports run concurrently; CUDA reports run one at a time to avoid contention on the single GPU.

Copy the complete report bundle from a headless server with:

```bash
scp -r reports/ USER@HOST:~
```

## Analysis Stages

| Stage | File | Runtime | Purpose |
| --- | --- | --- | --- |
| Data quality | `01_data_quality.py` | CPU | Validate raw records, inspect missingness, export cleaned datasets. |
| EDA | `02_eda.py` | CPU | Create engineered features, descriptive statistics, and exploratory figures. |
| Optimized EDA | `02_eda_polars_duckdb.py` | CPU parallel | Run Polars/DuckDB aggregations and Parquet feature export. |
| Market analysis | `03_market_analysis.py` | CPU | Produce stakeholder-oriented market summaries. |
| Clustering baseline | `04_clustering_MiniBatchKMeans.py` | CPU | Fast scikit-learn market segmentation baseline. |
| Clustering validation | `04_clustering_StandardKMeans.py` | CPU | Full StandardKMeans validation report. |
| CUDA clustering | `04_clustering_TorchCUDAKMeans.py` | CUDA | GPU K-Means, CUDA PCA, CUDA silhouette sampling, CUDA projection visualization. |
| Price prediction baseline | `05_price_prediction.py` | CPU | scikit-learn tabular regression baseline. |
| CUDA price prediction | `05_price_prediction_TorchCUDA.py` | CUDA | PyTorch CUDA tabular regression model. |
| Text classification baseline | `06_text_classification.py` | CPU | TF-IDF and scikit-learn text classification baseline. |
| CUDA text classification | `06_text_classification_TorchCUDA.py` | CUDA | PyTorch CUDA embedding-bag text classifiers. |

## Output Contract

| Path | Content |
| --- | --- |
| `reports/html/` | Executed HTML reports with embedded visualizations. |
| `reports/data/` | Generated CSV/Parquet datasets, predictions, metrics, and summaries created by `make run`. |
| `reports/figures/` | High-resolution generated figures. |
| `reports/models/` | Generated model artifacts created by `make run`. |
| `reports/logs/` | Per-stage stdout/stderr logs from `scripts/run_pipeline.py`. |
| `reports/runtime_summary.csv` | Runtime benchmark summary generated by `scripts/run_pipeline.py`. |

The repository ships the executed HTML reports, figures, logs, and runtime summary. Generated data tables and model artifacts are local run outputs; `make run` recreates them from the Zstandard source dataset.

```text
reports/data/
reports/models/
*.ipynb
.ipynb_checkpoints/
```

## Runtime Benchmark

`scripts/run_pipeline.py` records runtime measurements in `reports/runtime_summary.csv`. After a full server run, use that file as the authoritative benchmark table for the generated report set.

Latest full server run:

| Stage | Runtime |
| --- | ---: |
| `01_data_quality.py` | 95.079 seconds |
| `02_eda_polars_duckdb.py` | 28.319 seconds |
| `02_eda.py` | 56.355 seconds |
| `03_market_analysis.py` | 17.627 seconds |
| `06_text_classification_TorchCUDA.py` | 356.799 seconds |
| `04_clustering_TorchCUDAKMeans.py` | 46.771 seconds |
| `05_price_prediction_TorchCUDA.py` | 395.305 seconds |
| `05_price_prediction.py` | 3935.304 seconds |
| `06_text_classification.py` | 4895.791 seconds |
| `04_clustering_MiniBatchKMeans.py` | 7572.145 seconds |
| `04_clustering_StandardKMeans.py` | 7018.768 seconds |

## Cleaning Generated Outputs

Preview cleanup:

```bash
make clean-dry
```

Remove generated outputs without touching `.venv`, source files, Git history, or compressed dataset archives:

```bash
make clean
```

Also remove expanded CSV files when a fully fresh data extraction is required:

```bash
make clean-all
```

## License

The project code and documentation are released under the MIT License. Dataset usage is governed separately by the dataset terms documented in `Divar-Real-State-Ads/README.md`.
