# Divar Real Estate Analysis

This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.

This repository provides a complete, reproducible real estate analysis pipeline for the Divar Real Estate Ads dataset. It includes compressed source data, cell-structured Python analysis files, pinned runtime requirements, data compression utilities, and server-friendly HTML reporting.

## Project Structure

```text
.
├── Divar-Real-State-Ads/
│   ├── README.md
│   ├── divar_real_estate_ads.csv.zst
│   └── sampled_data.csv.zst
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   └── compress_data.py
└── notebooks/
    ├── 01_data_quality.py
    ├── 02_eda.py
    ├── 02_eda_polars_duckdb.py
    ├── 03_market_analysis.py
    ├── 04_clustering_MiniBatchKMeans.py
    ├── 04_clustering_StandardKMeans.py
    ├── 04_clustering_TorchCUDAKMeans.py
    ├── 05_price_prediction.py
    └── 06_text_classification.py
```

The analysis files use `# %%` cells so they can be run end to end from the terminal or interactively in editors that support Python cell execution.

Generated files are ignored by Git:

```text
Divar-Real-State-Ads/*.csv
data/processed/
notebooks/outputs/
*.ipynb
.ipynb_checkpoints/
```

## Runtime Target

Target CUDA version: `12.2`

The environment uses CUDA-enabled Python packages where applicable and keeps all project dependencies pinned in `requirements.txt`.

The baseline analysis files use pandas and scikit-learn. CSV loading uses the PyArrow engine where available, and the pipeline writes Parquet intermediates so later stages can avoid repeatedly parsing large CSV files.

The project also includes performance-oriented implementations:

```text
notebooks/02_eda_polars_duckdb.py
notebooks/04_clustering_TorchCUDAKMeans.py
```

`02_eda_polars_duckdb.py` provides a parallel Polars/DuckDB implementation for data engineering and aggregation. `04_clustering_TorchCUDAKMeans.py` provides a PyTorch CUDA implementation for K-Means, CUDA-based PCA, CUDA-based silhouette sampling, and CUDA projection visualizations on CUDA-capable NVIDIA GPUs.

Modern RAPIDS/cuML is not pinned as a dependency for this workstation target because current RAPIDS releases require newer GPU architecture than Pascal/GTX 1080. The CUDA path in this repository is therefore implemented with PyTorch CUDA, which supports the selected environment.

## Dataset

The repository stores the dataset as maximum-compression Zstandard archives:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv.zst
Divar-Real-State-Ads/sampled_data.csv.zst
```

The main workflow uses:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv
```

`sampled_data.csv` is included as an auxiliary dataset archive and is not required by the main analysis workflow.

## Setup

Use Python 3.10 or 3.11.

```bash
git clone git@github.com:bahman-farhadian/pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar.git
cd pydsai_divar_real_estate_analysis_hossein_hamzehi_mahdi_samdi_azar

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment already exists, refresh the pinned reporting stack before exporting HTML:

```bash
python -m pip install --force-reinstall -r requirements.txt
```

Verify CUDA availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Decompress Data

Decompress the main dataset before running the analysis:

```bash
python scripts/compress_data.py decompress --input Divar-Real-State-Ads/divar_real_estate_ads.csv.zst --output Divar-Real-State-Ads/divar_real_estate_ads.csv
```

Decompress the sampled dataset only when needed:

```bash
python scripts/compress_data.py decompress --input Divar-Real-State-Ads/sampled_data.csv.zst --output Divar-Real-State-Ads/sampled_data.csv
```

Recreate the compressed archives with maximum Zstandard compression:

```bash
python scripts/compress_data.py compress --input Divar-Real-State-Ads/divar_real_estate_ads.csv --output Divar-Real-State-Ads/divar_real_estate_ads.csv.zst
python scripts/compress_data.py compress --input Divar-Real-State-Ads/sampled_data.csv --output Divar-Real-State-Ads/sampled_data.csv.zst
```

## Analysis Stages

| Stage | File | Purpose |
| --- | --- | --- |
| Data quality | `01_data_quality.py` | Validate raw records, detect missing values and invalid entries, and export cleaned datasets. |
| EDA | `02_eda.py` | Build descriptive summaries, engineered features, correlations, and exploratory visualizations. |
| Optimized EDA | `02_eda_polars_duckdb.py` | Run a parallel Polars/DuckDB alternative and export Parquet/CSV summary artifacts. |
| Market analysis | `03_market_analysis.py` | Produce stakeholder-focused summaries for buyer and seller decisions. |
| Clustering | `04_clustering_MiniBatchKMeans.py` | Segment listings into interpretable market groups using the fast clustering workflow. |
| Clustering validation | `04_clustering_StandardKMeans.py` | Run the full K-Means comparison for final cluster validation. |
| CUDA clustering | `04_clustering_TorchCUDAKMeans.py` | Run the PyTorch CUDA clustering report with GPU-backed K-Means, PCA, silhouette sampling, and projection visualization. |
| Price prediction | `05_price_prediction.py` | Train price-per-square-meter models and identify over-valued and under-valued listings. |
| Text classification | `06_text_classification.py` | Classify listing text and infer missing user-type labels. |

## Execute And Export Reports

Run the analysis on a server without a graphical interface by exporting executed HTML reports. This is the recommended project execution path because it produces both the processed data files and reviewable report files.

```bash
export MPLBACKEND=Agg
mkdir -p reports/html
```

Export the main deliverable sequence. Run each command from the repository root and keep the order unchanged because later stages use files generated by earlier stages. These commands do not write `.ipynb` files.

```bash
python scripts/export_html.py --input notebooks/01_data_quality.py --output reports/html/01_data_quality.html
python scripts/export_html.py --input notebooks/02_eda.py --output reports/html/02_eda.html
python scripts/export_html.py --input notebooks/03_market_analysis.py --output reports/html/03_market_analysis.html
python scripts/export_html.py --input notebooks/04_clustering_MiniBatchKMeans.py --output reports/html/04_clustering_MiniBatchKMeans.html
python scripts/export_html.py --input notebooks/05_price_prediction.py --output reports/html/05_price_prediction.html
python scripts/export_html.py --input notebooks/06_text_classification.py --output reports/html/06_text_classification.html
```

Export the full StandardKMeans validation report:

```bash
python scripts/export_html.py --input notebooks/04_clustering_StandardKMeans.py --output reports/html/04_clustering_StandardKMeans.html
```

Export the performance-oriented reports:

```bash
python scripts/export_html.py --input notebooks/02_eda_polars_duckdb.py --output reports/html/02_eda_polars_duckdb.html
python scripts/export_html.py --input notebooks/04_clustering_TorchCUDAKMeans.py --output reports/html/04_clustering_TorchCUDAKMeans.html
```

The scripts also save high-resolution figures under:

```text
notebooks/outputs/figures/
```

## Outputs

| Path | Created By |
| --- | --- |
| `data/processed/cleaned_data.csv` | `01_data_quality.py` |
| `data/processed/cleaned_data.parquet` | `01_data_quality.py` |
| `data/processed/data_for_price_prediction.csv` | `01_data_quality.py` |
| `data/processed/data_for_price_prediction.parquet` | `01_data_quality.py` |
| `data/processed/cleaned_data_with_features.csv` | `02_eda.py` |
| `data/processed/cleaned_data_with_features.parquet` | `02_eda.py` |
| `data/processed/cleaned_data_with_features_polars.parquet` | `02_eda_polars_duckdb.py` |
| `data/processed/eda_city_statistics_duckdb.csv` | `02_eda_polars_duckdb.py` |
| `data/processed/eda_listing_summary_duckdb.csv` | `02_eda_polars_duckdb.py` |
| `data/processed/market_analysis_city_summary.csv` | `03_market_analysis.py` |
| `data/processed/clustering_assignments.csv` | clustering scripts |
| `data/processed/clustering_assignments_torch_cuda.csv` | `04_clustering_TorchCUDAKMeans.py` |
| `data/processed/clustering_assignments_torch_cuda.parquet` | `04_clustering_TorchCUDAKMeans.py` |
| `data/processed/cluster_profiles_torch_cuda.csv` | `04_clustering_TorchCUDAKMeans.py` |
| `data/processed/clustering_summary_torch_cuda.csv` | `04_clustering_TorchCUDAKMeans.py` |
| `data/processed/price_predictions.csv` | `05_price_prediction.py` |
| `data/processed/user_type_predictions.csv` | `06_text_classification.py` |
| `notebooks/outputs/figures/` | analysis scripts |
| `notebooks/outputs/models/` | modeling scripts |
| `reports/html/` | executed HTML reports |

## License

The project code and documentation are released under the MIT License. Dataset usage is governed separately by the dataset terms documented in `Divar-Real-State-Ads/README.md`.
