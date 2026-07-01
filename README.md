# Divar Real Estate Analysis

This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.

This repository is intended to be runnable after cloning. The source code is stored as regular Python files in `# %%` cell format, the dataset is stored as highly compressed CSV archives, and the only local rebuild step should be creating the Python virtual environment and decompressing the data files.

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
    ├── 03_market_analysis.py
    ├── 04_clustering_MiniBatchKMeans.py
    ├── 04_clustering_StandardKMeans.py
    ├── 05_price_prediction.py
    └── 06_text_classification.py
```

The repository no longer uses Jupyter Notebook files as source files. The `notebooks/` directory keeps the notebook-style workflow through `# %%` Python cells only.

Generated files are intentionally ignored by Git:

```text
Divar-Real-State-Ads/*.csv
data/processed/
notebooks/outputs/
reports/html/
```

## What The Project Does

The workflow analyzes Divar real estate advertisements from raw CSV data through final modeling and reporting artifacts:

| Step | File | Output |
| --- | --- | --- |
| 1 | `01_data_quality.py` | data validation, missing-value reports, cleaned datasets |
| 2 | `02_eda.py` | exploratory summaries, correlation analysis, engineered features |
| 3 | `03_market_analysis.py` | market summaries for buyers and sellers |
| 4 | `04_clustering_MiniBatchKMeans.py` | fast market segmentation |
| 4 optional | `04_clustering_StandardKMeans.py` | slower full K-Means comparison |
| 5 | `05_price_prediction.py` | price-per-square-meter models and value labels |
| 6 | `06_text_classification.py` | listing-text classification and user-type prediction |

The main dataset used by the workflow is:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv
```

`sampled_data.csv` is included only as a compressed supporting dataset. The current project workflow does not use it.

## Workstation Target

The full workflow is meant to run on the GPU workstation/server:

```text
GPU: NVIDIA GeForce GTX 1080, 8 GB VRAM
Driver: 535.261.03
CUDA shown by nvidia-smi: 12.2
CPU: 20 cores
RAM: 31 GiB
```

The current analysis code uses pandas and scikit-learn, so most computation is CPU-based. `requirements.txt` includes a CUDA-enabled PyTorch wheel for CUDA validation and future GPU-specific work, but the existing models are the scikit-learn implementations in the Python files.

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

Optional CUDA check:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Decompress The Data

The repository stores the CSV files as maximum-compression Zstandard archives:

```text
Divar-Real-State-Ads/divar_real_estate_ads.csv.zst
Divar-Real-State-Ads/sampled_data.csv.zst
```

Before running the analysis, decompress the main dataset:

```bash
python scripts/compress_data.py decompress --input Divar-Real-State-Ads/divar_real_estate_ads.csv.zst --output Divar-Real-State-Ads/divar_real_estate_ads.csv
```

The sampled file is not required, but it can be restored the same way:

```bash
python scripts/compress_data.py decompress --input Divar-Real-State-Ads/sampled_data.csv.zst --output Divar-Real-State-Ads/sampled_data.csv
```

To recompress either CSV at the highest configured level:

```bash
python scripts/compress_data.py compress --input Divar-Real-State-Ads/divar_real_estate_ads.csv --output Divar-Real-State-Ads/divar_real_estate_ads.csv.zst
python scripts/compress_data.py compress --input Divar-Real-State-Ads/sampled_data.csv --output Divar-Real-State-Ads/sampled_data.csv.zst
```

The script uses Zstandard level 22. Compression is slow by design; decompression is fast.

## Run The Workflow

Run from inside the `notebooks/` directory because the converted scripts use the current working directory to resolve the project root.

```bash
cd notebooks

python 01_data_quality.py
python 02_eda.py
python 03_market_analysis.py
python 04_clustering_MiniBatchKMeans.py
python 05_price_prediction.py
python 06_text_classification.py
```

For the slower full clustering comparison:

```bash
python 04_clustering_StandardKMeans.py
```

Recommended server run:

```bash
source .venv/bin/activate
python scripts/compress_data.py decompress --input Divar-Real-State-Ads/divar_real_estate_ads.csv.zst --output Divar-Real-State-Ads/divar_real_estate_ads.csv
cd notebooks
python 01_data_quality.py
python 02_eda.py
python 03_market_analysis.py
python 04_clustering_MiniBatchKMeans.py
python 05_price_prediction.py
python 06_text_classification.py
```

## Headless HTML Reports

On a server without a GUI, use Jupytext and nbconvert to execute the `# %%` Python files and export HTML notebooks.

Install dependencies from `requirements.txt`, then run:

```bash
mkdir -p reports/html

jupytext --to ipynb notebooks/01_data_quality.py --output reports/html/01_data_quality.ipynb
jupyter nbconvert \
  --to html \
  --execute reports/html/01_data_quality.ipynb \
  --output 01_data_quality.html \
  --output-dir reports/html \
  --ExecutePreprocessor.cwd=notebooks \
  --ExecutePreprocessor.timeout=-1
```

Repeat for the other Python files, or run this loop from the repository root:

```bash
mkdir -p reports/html
for file in notebooks/*.py; do
  name="$(basename "$file" .py)"
  jupytext --to ipynb "$file" --output "reports/html/${name}.ipynb"
  jupyter nbconvert \
    --to html \
    --execute "reports/html/${name}.ipynb" \
    --output "${name}.html" \
    --output-dir reports/html \
    --ExecutePreprocessor.cwd=notebooks \
    --ExecutePreprocessor.timeout=-1
done
```

If Persian text or plots render differently on the server, set a non-GUI backend before running:

```bash
export MPLBACKEND=Agg
```

The scripts also save figures under:

```text
notebooks/outputs/figures/
```

Bring back `reports/html/`, `data/processed/`, and `notebooks/outputs/` from the workstation when you want to review final results locally.

## Main Outputs

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
| `notebooks/outputs/models/` | modeling scripts |
| `reports/html/` | headless report export |

## License

The project code and documentation are released under the MIT License. The Divar dataset is distributed under its own dataset terms, documented in `Divar-Real-State-Ads/README.md`; those terms are separate from this project's source-code license.
