# %% [markdown]
# # Phase 2 Alternative: Polars and DuckDB EDA
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records, cleaned)
#
# ---
#
# ## Objective
#
# Provide a parallel EDA implementation using Polars and DuckDB while keeping the pandas EDA file as the reference implementation.

# %%
import os
from pathlib import Path

THREAD_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault('POLARS_MAX_THREADS', THREAD_COUNT)

import duckdb
import polars as pl


def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / 'Divar-Real-State-Ads').exists() and (path / 'notebooks').exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


PROJECT_ROOT = find_project_root()
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

SOURCE_PARQUET = DATA_PROCESSED / 'cleaned_data.parquet'
SOURCE_CSV = DATA_PROCESSED / 'cleaned_data.csv'

if SOURCE_PARQUET.exists():
    scan = pl.scan_parquet(SOURCE_PARQUET)
    duckdb_source = str(SOURCE_PARQUET)
else:
    scan = pl.scan_csv(SOURCE_CSV, infer_schema_length=10000)
    duckdb_source = str(SOURCE_CSV)

print(f"Polars threads: {os.environ.get('POLARS_MAX_THREADS')}")
print(f"Source: {SOURCE_PARQUET if SOURCE_PARQUET.exists() else SOURCE_CSV}")

# %% [markdown]
# ## 1. Feature Engineering With Polars

# %%
df_features = (
    scan
    .with_columns([
        pl.col('price_value').cast(pl.Float64, strict=False),
        pl.col('building_size').cast(pl.Float64, strict=False),
        pl.col('rent_value').cast(pl.Float64, strict=False),
        pl.col('credit_value').cast(pl.Float64, strict=False),
    ])
    .with_columns([
        pl.when((pl.col('price_value') > 0) & (pl.col('building_size') > 0))
        .then(pl.col('price_value') / pl.col('building_size'))
        .otherwise(None)
        .alias('price_per_sqm')
    ])
    .with_columns([
        ((pl.col('price_per_sqm') < 5_000_000) | (pl.col('price_per_sqm') > 500_000_000))
        .fill_null(False)
        .alias('price_sqm_outlier')
    ])
)

df_features.sink_parquet(DATA_PROCESSED / 'cleaned_data_with_features_polars.parquet', compression='zstd')
print(f"Saved: {DATA_PROCESSED / 'cleaned_data_with_features_polars.parquet'}")

# %% [markdown]
# ## 2. Aggregations With DuckDB

# %%
connection = duckdb.connect()

if SOURCE_PARQUET.exists():
    relation_sql = f"read_parquet('{duckdb_source}')"
else:
    relation_sql = f"read_csv_auto('{duckdb_source}', sample_size=-1)"

city_stats = connection.execute(f"""
    WITH typed AS (
        SELECT
            city_slug,
            listing_type,
            TRY_CAST(price_value AS DOUBLE) AS price_value_num,
            TRY_CAST(building_size AS DOUBLE) AS building_size_num
        FROM {relation_sql}
    )
    SELECT
        city_slug,
        COUNT(*) AS listing_count,
        median(price_value_num / NULLIF(building_size_num, 0)) AS median_price_per_sqm,
        avg(building_size_num) AS avg_building_size
    FROM typed
    WHERE listing_type = 'sell'
      AND price_value_num > 0
      AND building_size_num > 0
    GROUP BY city_slug
    ORDER BY listing_count DESC
""").df()

city_stats.to_csv(DATA_PROCESSED / 'eda_city_statistics_duckdb.csv', index=False)
print(f"Saved: {DATA_PROCESSED / 'eda_city_statistics_duckdb.csv'}")

listing_summary = connection.execute(f"""
    SELECT
        listing_type,
        COUNT(*) AS listing_count,
        COUNT(DISTINCT city_slug) AS city_count
    FROM {relation_sql}
    GROUP BY listing_type
    ORDER BY listing_count DESC
""").df()

listing_summary.to_csv(DATA_PROCESSED / 'eda_listing_summary_duckdb.csv', index=False)
print(f"Saved: {DATA_PROCESSED / 'eda_listing_summary_duckdb.csv'}")
