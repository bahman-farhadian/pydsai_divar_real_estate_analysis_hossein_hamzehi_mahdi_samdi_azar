PYTHON ?= python3
VENV ?= .venv
JOBS ?= 4

VPYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install setup check extract-data extract-sample compress-data compress-sample run run-standard run-cpu export clean-dry clean clean-legacy clean-all

help: ## Show available project commands.
	@awk 'BEGIN {FS = ":.*##"; printf "Available commands:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create the local Python virtual environment.
	$(PYTHON) -m venv $(VENV)

install: venv ## Install CUDA-oriented project dependencies into the virtual environment.
	$(VPYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

setup: install ## Create the virtual environment and install all dependencies.

check: ## Validate Python packages, CUDA visibility, and required dataset files.
	$(VPYTHON) scripts/check_environment.py

extract-data: ## Decompress the main dataset used by the final pipeline.
	$(VPYTHON) scripts/compress_data.py decompress --input Divar-Real-State-Ads/divar_real_estate_ads.csv.zst --output Divar-Real-State-Ads/divar_real_estate_ads.csv

extract-sample: ## Decompress the development sample dataset when needed for reference.
	$(VPYTHON) scripts/compress_data.py decompress --input Divar-Real-State-Ads/sampled_data.csv.zst --output Divar-Real-State-Ads/sampled_data.csv

compress-data: ## Recreate the compressed main dataset archive.
	$(VPYTHON) scripts/compress_data.py compress --input Divar-Real-State-Ads/divar_real_estate_ads.csv --output Divar-Real-State-Ads/divar_real_estate_ads.csv.zst

compress-sample: ## Recreate the compressed development sample archive.
	$(VPYTHON) scripts/compress_data.py compress --input Divar-Real-State-Ads/sampled_data.csv --output Divar-Real-State-Ads/sampled_data.csv.zst

run: ## Run the dependency-aware pipeline with CUDA stages when CUDA is available.
	$(VPYTHON) scripts/run_pipeline.py --jobs $(JOBS)

run-standard: ## Run the full pipeline including StandardKMeans validation.
	$(VPYTHON) scripts/run_pipeline.py --jobs $(JOBS) --include-standard-kmeans

run-cpu: ## Run the pipeline without CUDA alternatives.
	$(VPYTHON) scripts/run_pipeline.py --jobs $(JOBS) --skip-cuda

export: ## Export one report. Usage: make export INPUT=notebooks/01_data_quality.py OUTPUT=reports/html/01_data_quality.html
	@test -n "$(INPUT)" || (echo "INPUT is required"; exit 1)
	@test -n "$(OUTPUT)" || (echo "OUTPUT is required"; exit 1)
	$(VPYTHON) scripts/export_html.py --input $(INPUT) --output $(OUTPUT)

clean-dry: ## Preview generated report cleanup.
	$(PYTHON) scripts/clean_outputs.py --dry-run

clean: ## Remove generated report outputs from reports/.
	$(PYTHON) scripts/clean_outputs.py

clean-legacy: ## Remove reports/ plus old pre-final generated-output directories.
	$(PYTHON) scripts/clean_outputs.py --include-legacy

clean-all: ## Remove reports/, legacy output directories, and expanded CSV files.
	$(PYTHON) scripts/clean_outputs.py --include-legacy --include-expanded-csv
