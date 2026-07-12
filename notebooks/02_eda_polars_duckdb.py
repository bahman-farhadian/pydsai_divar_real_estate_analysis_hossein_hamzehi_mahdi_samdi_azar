# %% [markdown]
# # Phase 2 Alternative: Exploratory Data Analysis with Polars and DuckDB
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records, cleaned)
#
# ---
#
# ## Objective
#
# Reproduce the Phase 2 exploratory analysis with a columnar execution path. The pandas EDA remains the reference report; this file provides the same report-level outputs with Polars for feature engineering and DuckDB for parallel SQL aggregations.
#
# ## Analysis Scope
#
# 1. Load the cleaned Phase 1 dataset from `reports/data`.
# 2. Create analytical features and write a compressed Parquet feature dataset.
# 3. Generate numerical, categorical, segment, temporal, and rental summaries.
# 4. Save report-ready figures under `reports/figures`.
# 5. Save machine-readable summary tables under `reports/data`.

# %% [markdown]
# ## 1. Setup and Library Imports

# %%
import os
import warnings
from pathlib import Path

THREAD_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault('POLARS_MAX_THREADS', THREAD_COUNT)
os.environ.setdefault('OMP_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('MKL_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('NUMEXPR_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('ARROW_NUM_THREADS', THREAD_COUNT)

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import display
from scripts.report_contracts import EDA_CITY_COLUMNS, EDA_SUMMARY_COLUMNS, write_csv, write_manifest

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'sell': '#27ae60',
    'rent': '#3498db',
    'warning': '#f39c12',
    'purple': '#9b59b6',
}

ROOM_ALIASES = {
    'بدون اتاق': '0 rooms',
    'یک': '1 room',
    'دو': '2 rooms',
    'سه': '3 rooms',
    'چهار': '4 rooms',
    'پنج یا بیشتر': '5+ rooms',
}

USER_TYPE_ALIASES = {
    'مشاور املاک': 'Real Estate Agent',
    'شخصی': 'Private Seller',
}


def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / 'Divar-Real-State-Ads').exists() and (path / 'notebooks').exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def display_slug(value):
    if pd.isna(value):
        return 'Unknown'
    return str(value).replace('-', ' ').replace('_', ' ').title()


def display_room(value):
    if pd.isna(value):
        return 'Unknown'
    return ROOM_ALIASES.get(str(value), str(value))


def display_user_type(value):
    if pd.isna(value):
        return 'Unknown/NULL'
    return USER_TYPE_ALIASES.get(str(value), str(value))


def save_figure(*filenames):
    plt.tight_layout()
    for filename in filenames:
        output_path = FIGURES_PATH / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path.relative_to(PROJECT_ROOT)}")
    plt.show()


def save_table(frame, filename):
    output_path = DATA_PROCESSED / filename
    frame.to_csv(output_path, index=False)
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


def relation_sql(path):
    path_sql = path.as_posix().replace("'", "''")
    if path.suffix == '.parquet':
        return f"read_parquet('{path_sql}')"
    return f"read_csv_auto('{path_sql}', sample_size=-1)"


print("Libraries loaded successfully")
print(f"Polars threads: {os.environ.get('POLARS_MAX_THREADS')}")

# %% [markdown]
# ## 2. Project Structure

# %%
PROJECT_ROOT = find_project_root()
REPORTS_PATH = PROJECT_ROOT / 'reports'
DATA_PROCESSED = REPORTS_PATH / 'data'
FIGURES_PATH = REPORTS_PATH / 'figures'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

print("Project root: .")
print(f"Reports data path: {DATA_PROCESSED.relative_to(PROJECT_ROOT)}")
print(f"Figures path: {FIGURES_PATH.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 3. Load Cleaned Data

# %%
SOURCE_PARQUET = DATA_PROCESSED / 'cleaned_data.parquet'
SOURCE_CSV = DATA_PROCESSED / 'cleaned_data.csv'

if SOURCE_PARQUET.exists():
    SOURCE_FILE = SOURCE_PARQUET
    scan = pl.scan_parquet(SOURCE_FILE)
elif SOURCE_CSV.exists():
    SOURCE_FILE = SOURCE_CSV
    scan = pl.scan_csv(SOURCE_FILE, infer_schema_length=10000)
else:
    raise FileNotFoundError(
        "Phase 1 output is missing. Run notebooks/01_data_quality.py before this report."
    )

try:
    schema_names = scan.collect_schema().names()
except AttributeError:
    schema_names = list(scan.schema.keys())

print(f"Loading source: {SOURCE_FILE.relative_to(PROJECT_ROOT)}")
print(f"Columns: {len(schema_names)}")

row_count = scan.select(pl.len().alias('row_count')).collect().item()
print(f"Rows: {row_count:,}")

# %% [markdown]
# ## 4. Feature Engineering with Polars

# %%
numeric_columns = [
    'price_value',
    'building_size',
    'land_size',
    'rent_value',
    'credit_value',
    'floor',
    'total_floors_count',
    'construction_year',
]

cast_expressions = [
    pl.col(column).cast(pl.Float64, strict=False).alias(column)
    for column in numeric_columns
    if column in schema_names
]

features = scan.with_columns(cast_expressions) if cast_expressions else scan

if {'price_value', 'building_size'}.issubset(schema_names):
    features = features.with_columns(
        pl.when((pl.col('price_value') > 0) & (pl.col('building_size') > 0))
        .then(pl.col('price_value') / pl.col('building_size'))
        .otherwise(None)
        .alias('price_per_sqm')
    ).with_columns(
        ((pl.col('price_per_sqm') < 5_000_000) | (pl.col('price_per_sqm') > 500_000_000))
        .fill_null(False)
        .alias('price_sqm_outlier')
    ).with_columns(
        pl.when(pl.col('price_per_sqm').is_null())
        .then(pl.lit('Unknown'))
        .when(pl.col('price_per_sqm') < 30_000_000)
        .then(pl.lit('Budget (<30M)'))
        .when(pl.col('price_per_sqm') < 80_000_000)
        .then(pl.lit('Mid-range (30-80M)'))
        .when(pl.col('price_per_sqm') < 150_000_000)
        .then(pl.lit('Upper-mid (80-150M)'))
        .when(pl.col('price_per_sqm') < 300_000_000)
        .then(pl.lit('Premium (150-300M)'))
        .otherwise(pl.lit('Luxury (300M+)'))
        .alias('price_category')
    )

if 'created_at_month' in schema_names:
    features = features.with_columns(
        pl.col('created_at_month').cast(pl.Utf8).str.slice(0, 7).alias('year_month'),
        pl.col('created_at_month').cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32, strict=False).alias('year'),
        pl.col('created_at_month').cast(pl.Utf8).str.slice(5, 2).cast(pl.Int32, strict=False).alias('month'),
    )

FEATURE_PARQUET = DATA_PROCESSED / 'cleaned_data_with_features_polars.parquet'
features.sink_parquet(FEATURE_PARQUET, compression='zstd')
print(f"Saved: {FEATURE_PARQUET.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 5. Aggregations with DuckDB

# %%
connection = duckdb.connect()
connection.execute(f"PRAGMA threads={os.cpu_count() or 1}")
feature_sql = relation_sql(FEATURE_PARQUET)

listing_summary = connection.execute(f"""
    SELECT
        listing_type,
        COUNT(*) AS listing_count,
        COUNT(DISTINCT city_slug) AS city_count,
        COUNT(DISTINCT cat3_slug) AS property_type_count,
        median(price_value) AS median_price,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    GROUP BY listing_type
    ORDER BY listing_count DESC
""").df()
save_table(listing_summary, 'eda_polars_duckdb_listing_summary.csv')

price_summary = connection.execute(f"""
    SELECT
        COUNT(*) AS clean_sell_count,
        median(price_value) AS median_price,
        avg(price_value) AS avg_price,
        median(price_per_sqm) AS median_price_per_sqm,
        avg(price_per_sqm) AS avg_price_per_sqm,
        median(building_size) AS median_building_size,
        avg(building_size) AS avg_building_size
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
""").df()
save_table(price_summary, 'eda_polars_duckdb_price_summary.csv')

city_stats = connection.execute(f"""
    SELECT
        city_slug,
        COUNT(*) AS listing_count,
        median(price_value) AS median_price,
        avg(price_value) AS avg_price,
        median(price_per_sqm) AS median_price_per_sqm,
        avg(price_per_sqm) AS avg_price_per_sqm,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
    GROUP BY city_slug
    ORDER BY listing_count DESC
""").df()
save_table(city_stats, 'eda_polars_duckdb_city_statistics.csv')

property_stats = connection.execute(f"""
    SELECT
        cat3_slug,
        COUNT(*) AS listing_count,
        median(price_value) AS median_price,
        median(price_per_sqm) AS median_price_per_sqm,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
    GROUP BY cat3_slug
    ORDER BY listing_count DESC
""").df()
save_table(property_stats, 'eda_polars_duckdb_property_type_statistics.csv')

room_stats = connection.execute(f"""
    SELECT
        rooms_count,
        COUNT(*) AS listing_count,
        median(price_per_sqm) AS median_price_per_sqm,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
      AND rooms_count IS NOT NULL
    GROUP BY rooms_count
    ORDER BY listing_count DESC
""").df()
room_stats['rooms_label'] = room_stats['rooms_count'].map(display_room)
save_table(room_stats, 'eda_polars_duckdb_room_statistics.csv')

user_stats = connection.execute(f"""
    SELECT
        user_type,
        COUNT(*) AS listing_count,
        median(price_per_sqm) AS median_price_per_sqm,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
    GROUP BY user_type
    ORDER BY listing_count DESC
""").df()
user_stats['user_type_label'] = user_stats['user_type'].map(display_user_type)
save_table(user_stats, 'eda_polars_duckdb_user_type_statistics.csv')

rental_stats = connection.execute(f"""
    SELECT
        city_slug,
        COUNT(*) AS listing_count,
        median(rent_value) AS median_rent,
        median(credit_value) AS median_credit,
        median(building_size) AS median_building_size
    FROM {feature_sql}
    WHERE listing_type = 'rent'
      AND building_size > 0
      AND (rent_value > 0 OR credit_value > 0)
    GROUP BY city_slug
    ORDER BY listing_count DESC
""").df()
save_table(rental_stats, 'eda_polars_duckdb_rental_statistics.csv')

feature_columns = connection.execute(f"SELECT * FROM {feature_sql} LIMIT 0").df().columns.tolist()

if 'year_month' in feature_columns:
    monthly_stats = connection.execute(f"""
        SELECT
            year_month,
            listing_type,
            COUNT(*) AS listing_count,
            median(price_per_sqm) AS median_price_per_sqm,
            median(price_value) AS median_price
        FROM {feature_sql}
        WHERE year_month IS NOT NULL
          AND (
              listing_type <> 'sell'
              OR (
                  price_value > 0
                  AND building_size > 0
                  AND price_per_sqm IS NOT NULL
                  AND price_sqm_outlier = false
              )
          )
        GROUP BY year_month, listing_type
        ORDER BY year_month, listing_type
    """).df()
else:
    monthly_stats = pd.DataFrame()
save_table(monthly_stats, 'eda_polars_duckdb_monthly_statistics.csv')

print("Aggregation complete")

# %% [markdown]
# ## 6. Dataset Overview

# %%
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Total records: {row_count:,}")
print("\nListing type summary:")
display(listing_summary)

print("\nClean sell summary:")
display(price_summary)

availability_query = ", ".join(
    [
        f"SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) AS {column}_non_null"
        for column in ['price_value', 'building_size', 'rooms_count', 'city_slug', 'cat3_slug', 'user_type']
        if column in schema_names
    ]
)
if availability_query:
    availability = connection.execute(f"SELECT {availability_query} FROM {feature_sql}").df().T
    availability.columns = ['non_null']
    availability['availability_pct'] = availability['non_null'] / row_count * 100
    display(availability)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

listing_plot = listing_summary.copy()
listing_plot['listing_type_label'] = listing_plot['listing_type'].map(display_slug)
axes[0].bar(
    listing_plot['listing_type_label'],
    listing_plot['listing_count'],
    color=[COLORS['sell'], COLORS['rent']][:len(listing_plot)],
    edgecolor='white',
)
axes[0].set_title('Listing Type Distribution')
axes[0].set_ylabel('Number of Listings')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for index, value in enumerate(listing_plot['listing_count']):
    axes[0].text(index, value * 1.01, f'{int(value):,}', ha='center', va='bottom', fontsize=9)

axes[1].bar(
    listing_plot['listing_type_label'],
    listing_plot['city_count'],
    color=[COLORS['secondary'], COLORS['warning']][:len(listing_plot)],
    edgecolor='white',
)
axes[1].set_title('City Coverage by Listing Type')
axes[1].set_ylabel('Distinct Cities')
for index, value in enumerate(listing_plot['city_count']):
    axes[1].text(index, value * 1.01, f'{int(value):,}', ha='center', va='bottom', fontsize=9)

save_figure('02_polars_duckdb_listing_overview.png')

# %% [markdown]
# ## 7. Numerical Distributions

# %%
sell_sample = connection.execute(f"""
    SELECT
        price_value,
        building_size,
        price_per_sqm,
        price_category
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
    LIMIT 500000
""").df()

print(f"Sell sample for distributions: {len(sell_sample):,}")
print(sell_sample[['price_value', 'building_size', 'price_per_sqm']].describe())

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

price_values = sell_sample['price_value'].dropna()
axes[0, 0].hist(np.log10(price_values), bins=50, color=COLORS['sell'], edgecolor='white', alpha=0.85)
axes[0, 0].axvline(np.log10(price_values.median()), color=COLORS['accent'], linestyle='--', linewidth=2)
axes[0, 0].set_title('Sell Price Distribution (Log Scale)')
axes[0, 0].set_xlabel('Log10(Price in Tomans)')
axes[0, 0].set_ylabel('Frequency')

trimmed_price = price_values[price_values <= price_values.quantile(0.95)]
axes[0, 1].hist(trimmed_price / 1e9, bins=50, color=COLORS['sell'], edgecolor='white', alpha=0.85)
axes[0, 1].axvline(trimmed_price.median() / 1e9, color=COLORS['accent'], linestyle='--', linewidth=2)
axes[0, 1].set_title('Sell Price Distribution (<=95th Percentile)')
axes[0, 1].set_xlabel('Price (Billion Tomans)')
axes[0, 1].set_ylabel('Frequency')

price_per_sqm = sell_sample['price_per_sqm'].dropna()
axes[1, 0].hist(price_per_sqm / 1e6, bins=50, color=COLORS['purple'], edgecolor='white', alpha=0.85)
axes[1, 0].axvline(price_per_sqm.median() / 1e6, color=COLORS['accent'], linestyle='--', linewidth=2)
axes[1, 0].set_title('Price per Square Meter')
axes[1, 0].set_xlabel('Million Tomans per sqm')
axes[1, 0].set_ylabel('Frequency')

category_order = ['Budget (<30M)', 'Mid-range (30-80M)', 'Upper-mid (80-150M)', 'Premium (150-300M)', 'Luxury (300M+)']
category_counts = sell_sample['price_category'].value_counts().reindex(category_order).fillna(0)
bars = axes[1, 1].bar(
    range(len(category_counts)),
    category_counts.values,
    color=[COLORS['sell'], COLORS['secondary'], COLORS['warning'], COLORS['accent'], COLORS['purple']],
    edgecolor='white',
)
axes[1, 1].set_title('Listings by Price Category')
axes[1, 1].set_xticks(range(len(category_counts)))
axes[1, 1].set_xticklabels(['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury'], rotation=25, ha='right')
axes[1, 1].set_ylabel('Number of Listings')
max_category = max(category_counts.max(), 1)
for bar, value in zip(bars, category_counts.values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + max_category * 0.02, f'{int(value):,}', ha='center', va='bottom', fontsize=8)

save_figure('02_polars_duckdb_price_distribution.png', '02_polars_duckdb_price_per_sqm_distribution.png')

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

building_size = sell_sample['building_size'].dropna()
building_trimmed = building_size[building_size <= building_size.quantile(0.99)]
axes[0].hist(building_trimmed, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.85)
axes[0].axvline(building_trimmed.median(), color=COLORS['accent'], linestyle='--', linewidth=2)
axes[0].set_title('Building Size Distribution')
axes[0].set_xlabel('Building Size (sqm)')
axes[0].set_ylabel('Frequency')

scatter_sample = sell_sample.sample(min(len(sell_sample), 50000), random_state=42)
axes[1].scatter(
    scatter_sample['building_size'],
    scatter_sample['price_per_sqm'] / 1e6,
    s=6,
    alpha=0.2,
    color=COLORS['primary'],
)
axes[1].set_xlim(0, building_trimmed.quantile(0.99))
axes[1].set_title('Building Size vs Price per sqm')
axes[1].set_xlabel('Building Size (sqm)')
axes[1].set_ylabel('Price per sqm (Million Tomans)')

save_figure('02_polars_duckdb_building_size_distribution.png', '02_polars_duckdb_scatter_plots.png')

# %% [markdown]
# ## 8. Categorical Distributions

# %%
top_cities = city_stats.head(15).copy()
top_cities['city_label'] = top_cities['city_slug'].map(display_slug)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(top_cities['city_label'][::-1], top_cities['listing_count'][::-1], color=COLORS['secondary'])
axes[0].set_title('Top Cities by Clean Sell Listing Count')
axes[0].set_xlabel('Number of Listings')

axes[1].barh(top_cities['city_label'][::-1], (top_cities['median_price_per_sqm'][::-1] / 1e6), color=COLORS['sell'])
axes[1].set_title('Median Price per sqm by City')
axes[1].set_xlabel('Million Tomans per sqm')

save_figure('02_polars_duckdb_city_distribution.png')

# %%
top_properties = property_stats.head(15).copy()
top_properties['property_label'] = top_properties['cat3_slug'].map(display_slug)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(top_properties['property_label'][::-1], top_properties['listing_count'][::-1], color=COLORS['warning'])
axes[0].set_title('Top Property Types by Listing Count')
axes[0].set_xlabel('Number of Listings')

axes[1].barh(top_properties['property_label'][::-1], (top_properties['median_price_per_sqm'][::-1] / 1e6), color=COLORS['purple'])
axes[1].set_title('Median Price per sqm by Property Type')
axes[1].set_xlabel('Million Tomans per sqm')

save_figure('02_polars_duckdb_property_type_distribution.png')

# %%
room_order = ['0 rooms', '1 room', '2 rooms', '3 rooms', '4 rooms', '5+ rooms']
room_plot = room_stats.copy()
room_plot['room_sort'] = room_plot['rooms_label'].map({label: index for index, label in enumerate(room_order)})
room_plot = room_plot.sort_values(['room_sort', 'rooms_label'], na_position='last')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].bar(room_plot['rooms_label'], room_plot['listing_count'], color=COLORS['primary'], edgecolor='white')
axes[0].set_title('Room Count Distribution')
axes[0].set_xlabel('Rooms')
axes[0].set_ylabel('Number of Listings')
axes[0].tick_params(axis='x', rotation=25)

axes[1].bar(room_plot['rooms_label'], room_plot['median_price_per_sqm'] / 1e6, color=COLORS['accent'], edgecolor='white')
axes[1].set_title('Median Price per sqm by Room Count')
axes[1].set_xlabel('Rooms')
axes[1].set_ylabel('Million Tomans per sqm')
axes[1].tick_params(axis='x', rotation=25)

save_figure('02_polars_duckdb_rooms_distribution.png')

# %%
user_plot = user_stats.copy()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(user_plot['user_type_label'], user_plot['listing_count'], color=[COLORS['accent'], COLORS['neutral'], COLORS['secondary']][:len(user_plot)], edgecolor='white')
axes[0].set_title('User Type Distribution')
axes[0].set_ylabel('Number of Listings')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(user_plot['user_type_label'], user_plot['median_price_per_sqm'] / 1e6, color=[COLORS['accent'], COLORS['neutral'], COLORS['secondary']][:len(user_plot)], edgecolor='white')
axes[1].set_title('Median Price per sqm by User Type')
axes[1].set_ylabel('Million Tomans per sqm')
axes[1].tick_params(axis='x', rotation=15)

save_figure('02_polars_duckdb_user_type_distribution.png')

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(user_plot['user_type_label'], user_plot['median_price_per_sqm'] / 1e6, color=COLORS['accent'], edgecolor='white')
ax.set_title('Median Price per sqm by User Type')
ax.set_ylabel('Million Tomans per sqm')
ax.tick_params(axis='x', rotation=15)
save_figure('02_polars_duckdb_price_by_user_type.png')

# %% [markdown]
# ## 9. Correlations and Segment Prices

# %%
corr_data = connection.execute(f"""
    SELECT
        price_value,
        building_size,
        price_per_sqm,
        rent_value,
        credit_value,
        construction_year
    FROM {feature_sql}
    WHERE listing_type = 'sell'
      AND price_value > 0
      AND building_size > 0
      AND price_per_sqm IS NOT NULL
      AND price_sqm_outlier = false
    LIMIT 250000
""").df()

corr_cols = [column for column in corr_data.columns if corr_data[column].notna().sum() > 100]
corr_matrix = corr_data[corr_cols].corr()
corr_matrix.to_csv(DATA_PROCESSED / 'eda_polars_duckdb_correlation_matrix.csv')
display(corr_matrix)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    ax=ax,
)
ax.set_title('Correlation Matrix (Filtered Sell Listings)')
save_figure('02_polars_duckdb_correlation_matrix.png')

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

price_city = city_stats.sort_values('median_price_per_sqm', ascending=False).head(15).copy()
price_city['city_label'] = price_city['city_slug'].map(display_slug)
axes[0].barh(price_city['city_label'][::-1], price_city['median_price_per_sqm'][::-1] / 1e6, color=COLORS['sell'])
axes[0].set_title('Highest Median Price per sqm by City')
axes[0].set_xlabel('Million Tomans per sqm')

price_property = property_stats.sort_values('median_price_per_sqm', ascending=False).head(15).copy()
price_property['property_label'] = price_property['cat3_slug'].map(display_slug)
axes[1].barh(price_property['property_label'][::-1], price_property['median_price_per_sqm'][::-1] / 1e6, color=COLORS['purple'])
axes[1].set_title('Highest Median Price per sqm by Property Type')
axes[1].set_xlabel('Million Tomans per sqm')

save_figure('02_polars_duckdb_price_by_city.png', '02_polars_duckdb_price_by_property_type.png')

# %% [markdown]
# ## 10. Temporal and Rental Market Overview

# %%
if not monthly_stats.empty:
    sell_monthly = monthly_stats[monthly_stats['listing_type'] == 'sell'].copy()
    rent_monthly = monthly_stats[monthly_stats['listing_type'] == 'rent'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    if not sell_monthly.empty:
        axes[0].plot(sell_monthly['year_month'], sell_monthly['median_price_per_sqm'] / 1e6, marker='o', color=COLORS['sell'])
        axes[0].set_title('Monthly Median Sell Price per sqm')
        axes[0].set_ylabel('Million Tomans per sqm')
        axes[0].tick_params(axis='x', rotation=45)

    if not rent_monthly.empty:
        axes[1].plot(rent_monthly['year_month'], rent_monthly['listing_count'], marker='o', color=COLORS['rent'])
        axes[1].set_title('Monthly Rental Listing Count')
        axes[1].set_ylabel('Number of Listings')
        axes[1].tick_params(axis='x', rotation=45)

    save_figure('02_polars_duckdb_temporal_analysis.png')
else:
    print("Temporal analysis skipped because year_month is unavailable.")

# %%
if not rental_stats.empty:
    rental_plot = rental_stats.head(15).copy()
    rental_plot['city_label'] = rental_plot['city_slug'].map(display_slug)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(rental_plot['city_label'][::-1], rental_plot['median_rent'][::-1] / 1e6, color=COLORS['rent'])
    axes[0].set_title('Median Monthly Rent by City')
    axes[0].set_xlabel('Million Tomans')

    axes[1].barh(rental_plot['city_label'][::-1], rental_plot['median_credit'][::-1] / 1e6, color=COLORS['warning'])
    axes[1].set_title('Median Credit Deposit by City')
    axes[1].set_xlabel('Million Tomans')

    save_figure('02_polars_duckdb_rental_market.png')
else:
    print("Rental market analysis skipped because rental data is unavailable.")

# %% [markdown]
# ## 11. Export Summary

# %%
final_summary = pd.DataFrame(
    [
        {
            'metric': 'total_records',
            'value': int(row_count),
        },
        {
            'metric': 'clean_sell_records',
            'value': int(price_summary.loc[0, 'clean_sell_count']) if not price_summary.empty else 0,
        },
        {
            'metric': 'median_sell_price_billion_tomans',
            'value': float(price_summary.loc[0, 'median_price'] / 1e9) if not price_summary.empty else np.nan,
        },
        {
            'metric': 'median_sell_price_per_sqm_million_tomans',
            'value': float(price_summary.loc[0, 'median_price_per_sqm'] / 1e6) if not price_summary.empty else np.nan,
        },
        {
            'metric': 'feature_parquet',
            'value': str(FEATURE_PARQUET.relative_to(PROJECT_ROOT)),
        },
    ]
)
save_table(final_summary, 'eda_polars_duckdb_final_summary.csv')
write_csv(final_summary, DATA_PROCESSED / 'eda_polars_duckdb_summary.csv', EDA_SUMMARY_COLUMNS)
polars_city_statistics = city_stats.rename(columns={
    'listing_count': 'price_per_sqm_count',
    'median_price_per_sqm': 'price_per_sqm_median',
    'avg_price_per_sqm': 'price_per_sqm_mean',
    'median_building_size': 'building_size_median',
    'median_price': 'price_value_median',
})
write_csv(polars_city_statistics, DATA_PROCESSED / 'eda_polars_duckdb_city_statistics.csv', EDA_CITY_COLUMNS)
corr_matrix.to_csv(DATA_PROCESSED / 'eda_polars_duckdb_correlation_matrix.csv')
write_manifest(
    DATA_PROCESSED / 'eda_polars_duckdb_manifest.json',
    'polars_duckdb',
    {
        'summary': DATA_PROCESSED / 'eda_polars_duckdb_summary.csv',
        'city_statistics': DATA_PROCESSED / 'eda_polars_duckdb_city_statistics.csv',
        'correlation_matrix': DATA_PROCESSED / 'eda_polars_duckdb_correlation_matrix.csv',
    },
)
display(final_summary)

print("\nGenerated Polars/DuckDB EDA outputs:")
print(f"- Feature dataset: {FEATURE_PARQUET.relative_to(PROJECT_ROOT)}")
print("- Tables: reports/data/eda_polars_duckdb_*.csv")
print("- Figures: reports/figures/02_polars_duckdb_*.png")

# %% [markdown]
# ## 12. Summary
#
# The Polars/DuckDB report produces the same delivery class as the pandas EDA: engineered data, summary tables, and visualization outputs. It is optimized for parallel columnar execution while preserving the project-wide output contract under `reports/`.
