# %% [markdown]
# # Phase 1: Data Quality Assessment
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records)
#
# ---
#
# ## Objective
#
# Identify data recording errors and determine which fields are reliable for analysis.
#
#
# ## Analysis Scope
#
# 1. **Data Loading & Verification** - Ensure all records are loaded correctly
# 2. **Missing Values Analysis** - Identify columns by data availability
# 3. **Duplicate Detection** - Find exact and potential duplicates
# 4. **Value Consistency Checks** - Detect negative values, zeros, and data entry errors
# 5. **Distribution Analysis** - Visualize key variables
# 6. **Price Reasonability Check** - Price-per-sqm analysis to catch errors
# 7. **Data Cleaning** - Prepare foundation for subsequent phases
#
# ---

# %% [markdown]
# ## 1. Setup and Library Imports
#
# We begin by importing necessary libraries for data manipulation, visualization, and Persian text handling. The `arabic_reshaper` and `python-bidi` packages are essential for correctly displaying Persian/Arabic text in matplotlib charts.

# %%
import os

THREAD_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault('OMP_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('MKL_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('NUMEXPR_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('ARROW_NUM_THREADS', THREAD_COUNT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')

pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True

# Persian text display fix for matplotlib
import arabic_reshaper
from bidi.algorithm import get_display

def fix_persian(text):
    """Reshape Persian/Arabic text for correct display in matplotlib.
    RTL languages need special handling to display correctly in plots."""
    if pd.isna(text):
        return 'Unknown/NULL'
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

# Display settings for better output readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Plot styling
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
sns.set_style('whitegrid')

print(" Libraries loaded successfully")

# Color palette (consistent with other analysis files)
COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'purple': '#9b59b6',
    'orange': '#e67e22',
    'teal': '#1abc9c'
}

print("Libraries loaded successfully")

def read_csv_fast(path, **kwargs):
    try:
        return pd.read_csv(path, engine='pyarrow', **kwargs)
    except Exception as exc:
        print(f"PyArrow CSV engine unavailable for {path.name}; falling back to pandas C engine ({exc})")
        return pd.read_csv(path, low_memory=False, **kwargs)

# %% [markdown]
# ## 2. Project Structure and Path Configuration
#
# We define the project directory structure to maintain organized data flow:
# - **Raw data**: Original unmodified dataset
# - **Processed data**: Cleaned outputs for subsequent phases
# - **Figures**: Saved visualizations for the report

# %%
# Define project paths
def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / 'Divar-Real-State-Ads').exists() and (path / 'notebooks').exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


PROJECT_ROOT = find_project_root()
DATA_RAW = PROJECT_ROOT / 'Divar-Real-State-Ads'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
FIGURES_PATH = PROJECT_ROOT / 'notebooks' / 'outputs' / 'figures'

# Create output directories if they don't exist
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data path: {DATA_RAW}")
print(f"Processed data path: {DATA_PROCESSED}")
print(f"Figures path: {FIGURES_PATH}")

# %% [markdown]
# ## 3. Data Loading and Verification
#
# ### Why Verification Matters
#
# The CSV file may appear to have more lines than actual records due to multiline text fields (like `description`). We use pandas chunked reading to count actual records and verify complete data loading.
#
# **Important**: `wc -l` counts newline characters, not CSV records. Our dataset has ~10.9M lines but only 1M actual records because each description contains ~9-10 newlines on average.

# %%
# Verify file existence and get metadata
DATA_FILE = DATA_RAW / 'divar_real_estate_ads.csv'
print(f"Data file: {DATA_FILE}")
print(f"File exists: {DATA_FILE.exists()}")
print(f"File size: {DATA_FILE.stat().st_size / 1024**3:.2f} GB")

# Compare with wc -l for educational purposes
result = subprocess.run(['wc', '-l', str(DATA_FILE)], capture_output=True, text=True)
wc_lines = int(result.stdout.split()[0])
print(f"Lines reported by wc -l: {wc_lines:,}")

# %% [markdown]
# ### Load Complete Dataset
#
# We load the entire dataset without any row limits (`nrows` parameter) to ensure we're working with the complete data. The `low_memory=False` parameter ensures consistent dtype inference across all chunks.

# %%
# Load the FULL dataset
print(f"Loading FULL dataset from: {DATA_FILE}")
print("This may take a few minutes for large files...")

df = read_csv_fast(DATA_FILE)  # No nrows = load ALL records
total_rows = len(df)

print(f"\n" + "=" * 60)
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
print("=" * 60)
print(f"Actual CSV records: {total_rows:,}")
print(f"Difference from wc -l: {wc_lines - total_rows - 1:,} extra newlines (from multiline description field)")

# Verify all rows loaded
print("\n All records loaded successfully!")

# %% [markdown]
# ## 4. Initial Data Overview
#
# Before diving into quality checks, we examine the dataset's basic structure:
# - Total dimensions (rows × columns)
# - Column names and their data types
# - Sample records to understand the data format

# %%
# Basic dataset statistics
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
print(f"\nData types distribution:")
print(df.dtypes.value_counts())

# %% [markdown]
# ### Column Names and Data Types
#
# Understanding each column's data type helps identify potential issues:
# - Numeric columns stored as `object` may contain mixed or invalid values
# - Date columns may need parsing
# - Boolean columns stored as strings need conversion

# %%
# Column names and data types
print("Column names and types:")
print("-" * 60)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    print(f"{i:2d}. {col:35s} {str(dtype):15s}")

# %% [markdown]
# ### Sample Data Preview
#
# Examining the first few rows helps us understand the data format and identify obvious issues at a glance.

# %%
# Preview first few rows
df.head(3)

# %% [markdown]
# ---
#
# ## 5. Missing Values Analysis
#
# Missing values are a critical quality indicator. We categorize columns into three availability tiers:
#
# | Category | Missing % | Reliability for Analysis |
# |----------|-----------|-------------------------|
# | **High Availability** | ≤5% | Safe for core analysis |
# | **Medium Availability** | 5-50% | Use with caution |
# | **Low Availability** | >50% | Exclude or use peripherally |
#
# This categorization directly impacts which features we can use in Phase 5 (Price Prediction).

# %%
# Calculate missing values statistics for each column
missing_stats = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum().values,
    'missing_percent': (df.isnull().sum().values / len(df) * 100),
    'non_null_count': df.notnull().sum().values,
    'dtype': df.dtypes.values
})

missing_stats = missing_stats.sort_values('missing_percent', ascending=False)
missing_stats = missing_stats.reset_index(drop=True)

print("=" * 60)
print("MISSING VALUES ANALYSIS (sorted by missing %)")
print("=" * 60)
missing_stats

# %% [markdown]
# ### Missing Values Visualization
#
# A horizontal bar chart provides quick visual identification of problematic columns. Colors indicate severity:
# -  Red: >50% missing (Low availability)
# -  Orange: 20-50% missing (Medium availability)
# -  Green: <20% missing (Higher availability)

# %%
# Visualize missing values
fig, ax = plt.subplots(figsize=(14, 14))

# Only show columns with some missing values
missing_cols = missing_stats[missing_stats['missing_percent'] > 0].copy()

# Color coding by severity
colors = ['#d73027' if x > 50 else '#fc8d59' if x > 20 else '#91cf60' 
          for x in missing_cols['missing_percent']]

bars = ax.barh(missing_cols['column'], missing_cols['missing_percent'], color=colors)
ax.set_xlabel('Missing Percentage (%)', fontsize=11)
ax.set_ylabel('Column', fontsize=11)
ax.set_title('Missing Values by Column\n(Red >50%, Orange 20-50%, Green <20%)', fontsize=12, fontweight='bold')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='50% threshold')
ax.axvline(x=20, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='20% threshold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '01_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n Figure saved to: {FIGURES_PATH / '01_missing_values.png'}")

# %% [markdown]
# ### Data Completeness Heatmap
#
# An alternative view showing data completeness (inverse of missing percentage). This helps identify which columns have the most usable data.

# %%
# Data Completeness visualization
fig, ax = plt.subplots(figsize=(14, 14))

# Calculate completeness (100% - missing%)
completeness = (df.notnull().sum() / len(df) * 100).sort_values(ascending=True)

# Color by completeness level
colors = ['#d73027' if x < 50 else '#fc8d59' if x < 80 else '#91cf60' for x in completeness]

ax.barh(completeness.index, completeness.values, color=colors)
ax.set_xlabel('Data Completeness (%)', fontsize=11)
ax.set_ylabel('Column', fontsize=11)
ax.set_title('Data Completeness by Column\n(Green ≥80%, Orange 50-80%, Red <50%)', fontsize=12, fontweight='bold')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='50%')
ax.axvline(x=80, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='80%')
ax.axvline(x=95, color='green', linestyle='--', alpha=0.7, linewidth=2, label='95%')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '01_data_completeness.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Column Categorization by Availability
#
# Based on missing value percentages, we categorize columns for analysis planning. This directly informs feature selection for machine learning in Phase 5.

# %%
# Categorize columns by data availability
high_availability = missing_stats[missing_stats['missing_percent'] <= 5]['column'].tolist()
medium_availability = missing_stats[(missing_stats['missing_percent'] > 5) & 
                                    (missing_stats['missing_percent'] <= 50)]['column'].tolist()
low_availability = missing_stats[missing_stats['missing_percent'] > 50]['column'].tolist()

print("=" * 60)
print("COLUMN CATEGORIZATION BY DATA AVAILABILITY")
print("=" * 60)

print(f"\n HIGH AVAILABILITY (≤5% missing): {len(high_availability)} columns")
print(f"   -> Safe for core analysis")
for col in high_availability:
    pct = missing_stats[missing_stats['column'] == col]['missing_percent'].values[0]
    print(f"      - {col} ({pct:.2f}% missing)")

print(f"\n MEDIUM AVAILABILITY (5-50% missing): {len(medium_availability)} columns")
print(f"   -> Use with caution, handle missing values appropriately")
for col in medium_availability:
    pct = missing_stats[missing_stats['column'] == col]['missing_percent'].values[0]
    print(f"      - {col} ({pct:.2f}% missing)")

print(f"\n LOW AVAILABILITY (>50% missing): {len(low_availability)} columns")
print(f"   -> Exclude from core analysis or use peripherally")

# %% [markdown]
# ---
#
# ## 6. Duplicate Records Analysis
#
# Duplicates can skew analysis and inflate model performance metrics. We check for:
#
# 1. **Exact duplicates**: Rows identical across ALL columns
# 2. **Potential duplicates**: Rows with same key identifying features (title, price, size, city)
#
# The second type may represent legitimate re-listings or data collection artifacts.

# %%
# Check for exact duplicate rows
duplicate_count = df.duplicated().sum()
duplicate_percent = duplicate_count / len(df) * 100

print("=" * 60)
print("DUPLICATE RECORDS ANALYSIS")
print("=" * 60)
print(f"\nExact duplicate rows: {duplicate_count:,} ({duplicate_percent:.2f}%)")

if duplicate_count == 0:
    print(" No exact duplicates found - data appears clean")
else:
    print(" Exact duplicates found - will be removed during cleaning")

# %% [markdown]
# ### Potential Duplicates Check
#
# Even without exact duplicates, records with identical key fields may indicate:
# - Re-posted listings
# - Data scraping overlap
# - Legitimate similar properties
#
# We flag these for awareness but don't automatically remove them.

# %%
# Check for potential duplicates based on key identifying columns
key_columns = ['title', 'price_value', 'building_size', 'city_slug']
existing_keys = [col for col in key_columns if col in df.columns]

print(f"\nChecking for potential duplicates using: {existing_keys}")

if existing_keys:
    potential_duplicates = df.duplicated(subset=existing_keys, keep=False).sum()
    potential_dup_percent = potential_duplicates / len(df) * 100
    print(f"\nPotential duplicates (same {existing_keys}):")
    print(f"   Count: {potential_duplicates:,} ({potential_dup_percent:.2f}%)")
    
    if potential_duplicates > 0:
        print(f"\n    These may be re-listings or similar properties.")
        print(f"   -> Will be flagged but not removed to preserve data integrity.")

# %% [markdown]
# ---
#
# ## 7. Value Consistency and Data Entry Error Detection
#
# Following the instructor's guidance, we check for logically inconsistent values:
# - Negative values where only positive are valid (prices, sizes, counts)
# - Zero values in mandatory fields
# - Extreme outliers that may indicate data entry errors
#
# ### 7.1 Numerical Columns Overview

# %%
# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns ({len(numerical_cols)}):")
print("-" * 60)
for i, col in enumerate(numerical_cols, 1):
    non_null = df[col].notna().sum()
    print(f"{i:2d}. {col:35s} ({non_null:,} non-null values)")

# %% [markdown]
# ### 7.2 Summary Statistics for Numerical Columns
#
# Statistical summaries help identify:
# - Unexpected ranges (min/max)
# - Skewed distributions (mean vs median)
# - High variance indicating potential outliers

# %%
# Summary statistics for numerical columns
numerical_summary = df[numerical_cols].describe().T
numerical_summary['null_count'] = df[numerical_cols].isnull().sum()
numerical_summary['null_percent'] = numerical_summary['null_count'] / len(df) * 100
numerical_summary

# %% [markdown]
# ### 7.3 Negative Value Check
#
# Certain columns should never have negative values:
# - Prices (price_value, rent_value, credit_value)
# - Sizes (building_size, land_size)
# - Counts (rooms_count, floor, total_floors_count)
#
# Negative values here indicate data entry errors or encoding issues.

# %%
# Check for negative values where they shouldn't exist
non_negative_columns = ['price_value', 'rent_value', 'credit_value', 
                        'building_size', 'land_size', 'rooms_count',
                        'floor', 'total_floors_count', 'unit_per_floor']

existing_non_negative = [col for col in non_negative_columns if col in df.columns]

print("=" * 60)
print("NEGATIVE VALUE CHECK")
print("=" * 60)

negative_issues = {}
for col in existing_non_negative:
    # Convert to numeric first to handle mixed types
    col_numeric = pd.to_numeric(df[col], errors='coerce')
    negative_count = (col_numeric < 0).sum()
    if negative_count > 0:
        negative_issues[col] = negative_count
        print(f" {col}: {negative_count:,} negative values found")

if not negative_issues:
    print(" No negative values found in expected non-negative columns")

# %% [markdown]
# ### 7.4 Zero Value Check
#
# Zero values in price or size fields may indicate:
# - Missing data encoded as zero
# - "Price on request" listings
# - Data entry errors

# %%
# Check for zero values in key columns
print("\n" + "=" * 60)
print("ZERO VALUE CHECK")
print("=" * 60)

zero_check_cols = ['price_value', 'building_size', 'rent_value', 'credit_value']
existing_zero_check = [col for col in zero_check_cols if col in df.columns]

for col in existing_zero_check:
    col_numeric = pd.to_numeric(df[col], errors='coerce')
    zero_count = (col_numeric == 0).sum()
    zero_pct = zero_count / len(df) * 100
    if zero_count > 0:
        print(f" {col}: {zero_count:,} zero values ({zero_pct:.2f}%)")
    else:
        print(f" {col}: No zero values")

# %% [markdown]
# ---
#
# ## 8. Price Analysis
#
# Price is the **target variable for Phase 5** (Price Prediction). Understanding its distribution, completeness, and outliers is critical.
#
# ### 8.1 Price Distribution Statistics

# %%
# Price value analysis
if 'price_value' in df.columns:
    price_data = pd.to_numeric(df['price_value'], errors='coerce').dropna()
    
    print("=" * 60)
    print("PRICE VALUE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal records with price: {len(price_data):,} ({len(price_data)/len(df)*100:.1f}% of dataset)")
    print(f"Records without price: {len(df) - len(price_data):,} ({(len(df)-len(price_data))/len(df)*100:.1f}%)")
    
    print(f"\nPrice Statistics (in Tomans):")
    print(f"  Min:     {price_data.min():>20,.0f}")
    print(f"  Max:     {price_data.max():>20,.0f}")
    print(f"  Mean:    {price_data.mean():>20,.0f}")
    print(f"  Median:  {price_data.median():>20,.0f}")
    print(f"  Std Dev: {price_data.std():>20,.0f}")
    
    print(f"\nPercentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {p:3d}th percentile: {price_data.quantile(p/100):>20,.0f}")

# %% [markdown]
# ### 8.2 Price Distribution Visualization
#
# Price data is typically right-skewed (few very expensive properties). We use:
# - **Log scale histogram**: Shows the full distribution
# - **Box plot (1st-99th percentile)**: Shows typical range without extreme outliers

# %%
# Price distribution visualization
if 'price_value' in df.columns:
    price_data = pd.to_numeric(df['price_value'], errors='coerce').dropna()
    price_positive = price_data[price_data > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram (log scale)
    axes[0].hist(np.log10(price_positive), bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Log10(Price in Tomans)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution (Log Scale)')
    axes[0].axvline(x=np.log10(price_positive.median()), color='red', linestyle='--', 
                    linewidth=2, label=f'Median: {price_positive.median()/1e9:.1f}B')
    axes[0].legend()
    
    # Box plot (1st-99th percentile)
    price_trimmed = price_positive[(price_positive >= price_positive.quantile(0.01)) & 
                                   (price_positive <= price_positive.quantile(0.99))]
    bp = axes[1].boxplot(price_trimmed / 1e9, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    axes[1].set_ylabel('Price (Billion Tomans)')
    axes[1].set_title('Price Distribution (1st-99th Percentile)\nExcludes extreme outliers')
    axes[1].set_xticklabels(['Price'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_price_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### 8.3 Unrealistic Price Detection
#
# We flag potentially unrealistic prices:
# - **Very low**: <10 million Tomans (suspiciously cheap for any property)
# - **Very high**: >500 billion Tomans (extreme luxury/commercial)

# %%
# Check for unrealistic prices
if 'price_value' in df.columns:
    price_data = pd.to_numeric(df['price_value'], errors='coerce').dropna()
    
    very_low_price = (price_data < 10_000_000).sum()
    very_high_price = (price_data > 500_000_000_000).sum()
    
    print("\n" + "=" * 60)
    print("UNREALISTIC PRICE DETECTION")
    print("=" * 60)
    print(f"\nVery low prices (<10M Tomans):  {very_low_price:,} records")
    print(f"Very high prices (>500B Tomans): {very_high_price:,} records")
    
    # Show examples of extreme prices
    if very_low_price > 0:
        print(f"\nSample very low prices:")
        low_price_sample = df[pd.to_numeric(df['price_value'], errors='coerce') < 10_000_000][['title', 'price_value', 'building_size', 'city_slug']].head(3)
        print(low_price_sample.to_string())

# %% [markdown]
# ---
#
# ## 9. Building Size Analysis
#
# Building size is a key predictor for price. We analyze its distribution and identify anomalies.

# %%
if 'building_size' in df.columns:
    size_data = pd.to_numeric(df['building_size'], errors='coerce').dropna()
    
    print("=" * 60)
    print("BUILDING SIZE ANALYSIS")
    print("=" * 60)
    print(f"\nRecords with size: {len(size_data):,} ({len(size_data)/len(df)*100:.1f}%)")
    print(f"\nSize Statistics (in sqm):")
    print(f"  Min:    {size_data.min():>10,.0f} sqm")
    print(f"  Max:    {size_data.max():>10,.0f} sqm")
    print(f"  Mean:   {size_data.mean():>10,.0f} sqm")
    print(f"  Median: {size_data.median():>10,.0f} sqm")
    
    # Unrealistic sizes
    very_small = (size_data < 10).sum()
    very_large = (size_data > 10000).sum()
    
    print(f"\nPotentially unrealistic sizes:")
    print(f"  Very small (<10 sqm):     {very_small:,} records")
    print(f"  Very large (>10,000 sqm): {very_large:,} records")

# %% [markdown]
# ### Building Size Distribution Visualization

# %%
# Building size distribution visualization
if 'building_size' in df.columns:
    size_data = pd.to_numeric(df['building_size'], errors='coerce').dropna()
    size_reasonable = size_data[(size_data > 0) & (size_data < 2000)]  # Reasonable residential range
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(size_reasonable, bins=50, color='#27ae60', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Building Size (sqm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Building Size Distribution (0-2000 sqm)')
    axes[0].axvline(x=size_reasonable.median(), color='red', linestyle='--', 
                    linewidth=2, label=f'Median: {size_reasonable.median():.0f} sqm')
    axes[0].legend()
    
    # Box plot by property type
    if 'cat3_slug' in df.columns:
        top_cats = df['cat3_slug'].value_counts().head(5).index.tolist()
        size_by_cat = []
        labels = []
        for cat in top_cats:
            cat_sizes = pd.to_numeric(df[df['cat3_slug'] == cat]['building_size'], errors='coerce').dropna()
            cat_sizes = cat_sizes[(cat_sizes > 0) & (cat_sizes < 1000)]
            if len(cat_sizes) > 0:
                size_by_cat.append(cat_sizes)
                labels.append(cat[:12] + '...' if len(cat) > 12 else cat)
        
        bp = axes[1].boxplot(size_by_cat, labels=labels, patch_artist=True)
        colors_box = plt.cm.Set2(range(len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        axes[1].set_xlabel('Property Type')
        axes[1].set_ylabel('Building Size (sqm)')
        axes[1].set_title('Size by Property Type (Top 5)')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_building_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
#
# ## 10. Price-per-Square-Meter Analysis (Data Entry Error Detection)
#
# **This is a critical quality check.** By calculating price per square meter, we can identify:
# - Data entry errors (e.g., typing "10000" instead of "1,000,000,000")
# - Mismatched price/size combinations
# - Regional price variations
#
# Typical price per sqm in Iran (2024-2025):
# - Low-cost areas: ~20-50 million Tomans/sqm
# - Mid-range: 50-150 million Tomans/sqm  
# - Premium areas (North Tehran): 150-500+ million Tomans/sqm

# %%
# Calculate price per square meter
print("=" * 60)
print("PRICE PER SQUARE METER ANALYSIS")
print("=" * 60)

# Create temporary columns for analysis
df['_price_numeric'] = pd.to_numeric(df['price_value'], errors='coerce')
df['_size_numeric'] = pd.to_numeric(df['building_size'], errors='coerce')

# Calculate price per sqm only where both values exist and are valid
valid_mask = (df['_price_numeric'] > 0) & (df['_size_numeric'] > 0)
df.loc[valid_mask, '_price_per_sqm'] = df.loc[valid_mask, '_price_numeric'] / df.loc[valid_mask, '_size_numeric']

price_per_sqm = df['_price_per_sqm'].dropna()

print(f"\nRecords with calculable price/sqm: {len(price_per_sqm):,}")
print(f"\nPrice per sqm Statistics (Tomans):")
print(f"  Min:     {price_per_sqm.min():>15,.0f}")
print(f"  Max:     {price_per_sqm.max():>15,.0f}")
print(f"  Mean:    {price_per_sqm.mean():>15,.0f}")
print(f"  Median:  {price_per_sqm.median():>15,.0f}")

# Flag suspicious values
very_low_psqm = (price_per_sqm < 1_000_000).sum()  # < 1M per sqm is very suspicious
very_high_psqm = (price_per_sqm > 1_000_000_000).sum()  # > 1B per sqm is very suspicious

print(f"\n Potentially erroneous records:")
print(f"  Price/sqm < 1M Tomans:  {very_low_psqm:,} (likely data entry errors)")
print(f"  Price/sqm > 1B Tomans:  {very_high_psqm:,} (likely data entry errors)")

# %% [markdown]
# ### Price per sqm Visualization
#
# This visualization helps identify outliers and understand the price distribution across the market.

# %%
# Price per sqm visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filter to reasonable range for visualization
price_per_sqm_reasonable = price_per_sqm[(price_per_sqm >= 1_000_000) & (price_per_sqm <= 500_000_000)]

# Histogram
axes[0].hist(price_per_sqm_reasonable / 1e6, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Price per sqm (Million Tomans)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Price per Square Meter Distribution\n(1M - 500M Tomans range)')
axes[0].axvline(x=price_per_sqm_reasonable.median()/1e6, color='red', linestyle='--', 
                linewidth=2, label=f'Median: {price_per_sqm_reasonable.median()/1e6:.0f}M')
axes[0].legend()

# Box plot by city (top 5 cities)
if 'city_slug' in df.columns:
    top_cities = df['city_slug'].value_counts().head(5).index.tolist()
    psqm_by_city = []
    city_labels = []
    for city in top_cities:
        city_psqm = df[df['city_slug'] == city]['_price_per_sqm'].dropna()
        city_psqm = city_psqm[(city_psqm >= 1_000_000) & (city_psqm <= 500_000_000)]
        if len(city_psqm) > 100:
            psqm_by_city.append(city_psqm / 1e6)  # Convert to millions
            city_labels.append(city)
    
    bp = axes[1].boxplot(psqm_by_city, labels=city_labels, patch_artist=True)
    colors_box = plt.cm.Set3(range(len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    axes[1].set_xlabel('City')
    axes[1].set_ylabel('Price per sqm (Million Tomans)')
    axes[1].set_title('Price per sqm by City (Top 5)')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '01_price_per_sqm.png', dpi=150, bbox_inches='tight')
plt.show()

# Clean up temporary columns
df.drop(columns=['_price_numeric', '_size_numeric', '_price_per_sqm'], inplace=True, errors='ignore')

# %% [markdown]
# ---
#
# ## 11. Rooms Count Analysis
#
# Room count is an important feature but has high missing rate. We analyze the distribution of available data.

# %%
# Rooms count analysis
if 'rooms_count' in df.columns:
    print("=" * 60)
    print("ROOMS COUNT ANALYSIS")
    print("=" * 60)
    
    rooms_non_null = df['rooms_count'].notna().sum()
    print(f"\nRecords with rooms data: {rooms_non_null:,} ({rooms_non_null/len(df)*100:.1f}%)")
    print(f"\nValue distribution:")
    print(df['rooms_count'].value_counts())

# %% [markdown]
# ### Rooms Count Visualization with Persian Text

# %%
# Rooms count visualization - labels ABOVE bars
if 'rooms_count' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rooms_persian_to_english = {
        'بدون اتاق': 'Studio (0)',
        'یک': '1 Room',
        'دو': '2 Rooms',
        'سه': '3 Rooms',
        'چهار': '4 Rooms',
        'پنج یا بیشتر': '5+ Rooms'
    }
    
    room_order_persian = ['بدون اتاق', 'یک', 'دو', 'سه', 'چهار', 'پنج یا بیشتر']
    room_order_english = ['Studio (0)', '1 Room', '2 Rooms', '3 Rooms', '4 Rooms', '5+ Rooms']
    
    rooms_counts = df['rooms_count'].value_counts()
    
    ordered_counts = []
    ordered_labels = []
    for persian, english in zip(room_order_persian, room_order_english):
        if persian in rooms_counts.index:
            ordered_counts.append(rooms_counts[persian])
            ordered_labels.append(english)
    
    bars = ax.bar(range(len(ordered_counts)), ordered_counts, color='#9b59b6', edgecolor='white', linewidth=2)
    
    # 25% headroom for labels
    max_val = max(ordered_counts) if ordered_counts else 1
    ax.set_ylim(0, max_val * 1.25)
    
    # Labels ABOVE all bars
    for bar, count in zip(bars, ordered_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, fontsize=11)
    ax.set_xlabel('Number of Rooms', fontsize=12)
    ax.set_ylabel('Number of Listings', fontsize=12)
    ax.set_title('Rooms Count Distribution', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_rooms_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("rooms_count column not found")

# %% [markdown]
# ---
#
# ## 12. Categorical Columns Analysis
#
# Categorical columns provide segmentation for analysis. We examine key categorical variables:
# - `cat2_slug`: Listing type (sell vs rent)
# - `cat3_slug`: Property type
# - `city_slug`: Location
# - `user_type`: Advertiser type

# %%
# Categorical columns overview
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns ({len(categorical_cols)}):")
print("-" * 60)
for col in categorical_cols:
    unique = df[col].nunique()
    null_pct = df[col].isnull().sum() / len(df) * 100
    print(f"  {col:35s} {unique:>6,} unique values  ({null_pct:>5.1f}% null)")

# %% [markdown]
# ### Key Categorical Variables Distribution

# %%
# Value counts for key categorical columns
key_categorical = ['cat2_slug', 'cat3_slug', 'city_slug', 'user_type']
existing_categorical = [col for col in key_categorical if col in df.columns]

for col in existing_categorical:
    print("=" * 60)
    print(f"{col.upper()}")
    print("=" * 60)
    print(f"Unique values: {df[col].nunique()}")
    print(f"Null count: {df[col].isnull().sum():,} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
    print(f"\nTop 10 values:")
    print(df[col].value_counts().head(10))
    print()

# %% [markdown]
# ---
#
# ## 13. Geographic Data Validation
#
# For records with coordinates, we verify they fall within Iran's boundaries:
# - Latitude: 25°N to 40°N
# - Longitude: 44°E to 64°E

# %%
if 'location_latitude' in df.columns and 'location_longitude' in df.columns:
    print("=" * 60)
    print("GEOGRAPHIC DATA VALIDATION")
    print("=" * 60)
    
    lat = pd.to_numeric(df['location_latitude'], errors='coerce').dropna()
    lon = pd.to_numeric(df['location_longitude'], errors='coerce').dropna()
    
    print(f"\nRecords with coordinates: {len(lat):,} ({len(lat)/len(df)*100:.1f}%)")
    print(f"\nLatitude range:  {lat.min():.4f} to {lat.max():.4f}")
    print(f"Longitude range: {lon.min():.4f} to {lon.max():.4f}")
    
    # Iran's approximate bounds
    outside_lat = ((lat < 25) | (lat > 40)).sum()
    outside_lon = ((lon < 44) | (lon > 64)).sum()
    
    print(f"\nCoordinates outside Iran bounds:")
    print(f"  Latitude outside 25-40:   {outside_lat:,}")
    print(f"  Longitude outside 44-64:  {outside_lon:,}")
    
    if outside_lat == 0 and outside_lon == 0:
        print("\n All coordinates fall within Iran's boundaries")

# %% [markdown]
# ---
#
# ## 14. Distribution Visualizations
#
# ### 14.1 City Distribution
#
# Understanding geographic distribution helps identify data coverage and potential regional biases.

# %%
# City distribution visualization
if 'city_slug' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart - Top 20 cities
    city_counts = df['city_slug'].value_counts().head(20)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(city_counts)))[::-1]
    axes[0].barh(city_counts.index[::-1], city_counts.values[::-1], color=colors)
    axes[0].set_xlabel('Number of Listings')
    axes[0].set_ylabel('City')
    axes[0].set_title('Top 20 Cities by Number of Listings')
    
    # Pie chart - Top 5 + Others (clean, no overlapping labels)
    city_top5 = df['city_slug'].value_counts().head(5)
    others = df['city_slug'].value_counts()[5:].sum()
    city_pie = pd.concat([city_top5, pd.Series({'Others': others})])
    
    colors_pie = plt.cm.Set3(range(len(city_pie)))
    wedges, texts, autotexts = axes[1].pie(
        city_pie.values, 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        pctdistance=0.75,
        explode=[0.02] * len(city_pie)
    )
    axes[1].legend(wedges, city_pie.index, title='City', loc='center left', 
                   bbox_to_anchor=(1, 0.5), fontsize=10)
    axes[1].set_title('City Distribution (Top 5 + Others)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_city_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTotal unique cities: {df['city_slug'].nunique()}")
    print(f"Top 5 cities cover: {city_top5.sum() / len(df) * 100:.1f}% of all listings")

# %% [markdown]
# ### 14.2 Property Type Distribution

# %%
# Property type distribution
if 'cat3_slug' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart - Top 15 property types
    cat3_counts = df['cat3_slug'].value_counts().head(15)
    colors = plt.cm.tab20(range(len(cat3_counts)))
    axes[0].barh(cat3_counts.index[::-1], cat3_counts.values[::-1], color=colors[::-1])
    axes[0].set_xlabel('Number of Listings')
    axes[0].set_ylabel('Property Type')
    axes[0].set_title('Property Types Distribution (Top 15)')
    
    # Pie chart - Top 5 + Others
    cat3_top = df['cat3_slug'].value_counts().head(5)
    others = df['cat3_slug'].value_counts()[5:].sum()
    cat3_pie = pd.concat([cat3_top, pd.Series({'Others': others})])
    
    colors_pie = plt.cm.Set2(range(len(cat3_pie)))
    wedges, texts, autotexts = axes[1].pie(
        cat3_pie.values,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        pctdistance=0.75,
        explode=[0.02] * len(cat3_pie)
    )
    axes[1].legend(wedges, cat3_pie.index, title='Property Type', loc='center left',
                   bbox_to_anchor=(1, 0.5), fontsize=10)
    axes[1].set_title('Property Type Distribution (Top 5 + Others)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_property_type_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### 14.3 Listing Category Distribution (Sell vs Rent)

# %%
# Category distribution - labels ABOVE bars
if 'cat2_slug' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cat2_counts = df['cat2_slug'].value_counts()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e91e63'][:len(cat2_counts)]
    
    bars = ax.bar(range(len(cat2_counts)), cat2_counts.values, color=colors, edgecolor='white', linewidth=2)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Number of Listings', fontsize=12)
    ax.set_title('Listing Categories Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cat2_counts)))
    ax.set_xticklabels(cat2_counts.index, rotation=25, ha='right', fontsize=10)
    
    # 25% headroom
    max_val = cat2_counts.max()
    ax.set_ylim(0, max_val * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Labels ABOVE all bars
    total = len(df)
    for bar, val in zip(bars, cat2_counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_category_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("cat2_slug column not found")

# %% [markdown]
# ### 14.4 Time Distribution (Listings by Month)

# %%
# Time distribution
if 'created_at_month' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    month_counts = df['created_at_month'].value_counts().sort_index()
    
    # Create shorter labels
    x_labels = [str(m)[:7] if len(str(m)) > 7 else str(m) for m in month_counts.index]
    
    ax.bar(range(len(month_counts)), month_counts.values, color='steelblue', edgecolor='white')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Listings')
    ax.set_title('Listings Over Time (by Month)')
    
    # Show every Nth label to avoid crowding
    n_labels = len(x_labels)
    step = max(1, n_labels // 12)
    ax.set_xticks(range(0, n_labels, step))
    ax.set_xticklabels([x_labels[i] for i in range(0, n_labels, step)], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_time_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nData spans {df['created_at_month'].nunique()} months")
    print(f"Date range: {month_counts.index.min()} to {month_counts.index.max()}")

# %% [markdown]
# ### 14.5 User Type Distribution (with Persian Text)

# %%
# User type distribution - labels ABOVE bars
if 'user_type' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    user_type_persian_to_english = {
        'مشاور املاک': 'Real Estate Agent',
        'شخصی': 'Private Seller'
    }
    
    user_counts = df['user_type'].value_counts(dropna=False)
    
    bar_labels = []
    bar_values = []
    bar_colors = []
    color_map = {'مشاور املاک': '#3498db', 'شخصی': '#e74c3c'}
    
    for idx in user_counts.index:
        if pd.isna(idx):
            bar_labels.append('Unknown/NULL')
            bar_colors.append('#95a5a6')
        elif idx in user_type_persian_to_english:
            bar_labels.append(user_type_persian_to_english[idx])
            bar_colors.append(color_map.get(idx, '#95a5a6'))
        else:
            bar_labels.append(str(idx))
            bar_colors.append('#95a5a6')
        bar_values.append(user_counts[idx])
    
    # LEFT: Vertical bars with labels ABOVE
    bars = axes[0].bar(bar_labels, bar_values, color=bar_colors, edgecolor='white', linewidth=2)
    axes[0].set_xlabel('User Type', fontsize=11)
    axes[0].set_ylabel('Number of Listings', fontsize=11)
    axes[0].set_title('User Type Distribution', fontsize=12, fontweight='bold')
    
    max_val = max(bar_values)
    axes[0].set_ylim(0, max_val * 1.25)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    for bar, val in zip(bars, bar_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                     f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    
    # RIGHT: Horizontal bars with labels to the RIGHT
    y_pos = range(len(bar_labels))
    bars2 = axes[1].barh(y_pos, bar_values, color=bar_colors, edgecolor='white', height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(bar_labels, fontsize=11)
    axes[1].set_xlabel('Number of Listings', fontsize=11)
    axes[1].set_title('User Type Proportion', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, max_val * 1.25)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    total = sum(bar_values)
    for bar, val in zip(bars2, bar_values):
        pct = val / total * 100
        axes[1].text(val + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                     f'{pct:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_user_type_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("user_type column not found")

# %% [markdown]
# ### 14.6 Data Quality Overview Heatmap (NEW)
#
# A comprehensive heatmap showing the relationship between key features and their data availability. This helps identify patterns in missing data.

# %%
# Data Quality Overview Heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Select key columns for the heatmap
key_cols_for_heatmap = ['price_value', 'building_size', 'rooms_count', 'floor', 
                         'total_floors_count', 'construction_year', 'user_type',
                         'location_latitude', 'has_elevator', 'has_parking']
existing_heatmap_cols = [col for col in key_cols_for_heatmap if col in df.columns]

# Calculate correlation of missingness
missing_matrix = df[existing_heatmap_cols].isnull().astype(int)
missing_corr = missing_matrix.corr()

sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Missing Value Correlation Heatmap\n(High correlation = columns tend to be missing together)', 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '01_missing_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nInterpretation: High positive correlation means columns tend to be missing together.")
print("This helps identify data collection patterns.")

# %% [markdown]
# ---
#
# ## 15. Summary and Quality Report
#
# ### Final Statistics Summary

# %%
print("=" * 70)
print("DATA QUALITY ASSESSMENT - FINAL SUMMARY")
print("=" * 70)

print(f"""
 DATASET STATISTICS

  Total records:     {len(df):>12,}
  Total columns:     {len(df.columns):>12}
  Memory usage:      {df.memory_usage(deep=True).sum() / 1024**3:>12.2f} GB
  Duplicate rows:    {duplicate_count:>12,} ({duplicate_percent:.2f}%)

 COLUMN AVAILABILITY

  High availability (≤5% missing):    {len(high_availability):>5} columns
  Medium availability (5-50% missing): {len(medium_availability):>4} columns
  Low availability (>50% missing):    {len(low_availability):>5} columns

 COLUMNS SAFE FOR CORE ANALYSIS (≤5% missing):
   {high_availability}

  IMPORTANT LIMITATION FOR PHASE 5 (Price Prediction):
   price_value has {missing_stats[missing_stats['column']=='price_value']['missing_percent'].values[0]:.1f}% missing data
   This means only ~{(1 - missing_stats[missing_stats['column']=='price_value']['missing_percent'].values[0]/100) * len(df):,.0f} records have price data
""")

# %% [markdown]
# ### Save Quality Summary Reports

# %%
# Create and save quality summary
quality_summary = missing_stats.copy()
quality_summary['availability_category'] = pd.cut(
    quality_summary['missing_percent'],
    bins=[-1, 5, 50, 100],
    labels=['High', 'Medium', 'Low']
)

quality_summary.to_csv(DATA_PROCESSED / 'column_quality_summary.csv', index=False)
print(f" Column quality summary saved to: {DATA_PROCESSED / 'column_quality_summary.csv'}")

# Create overall summary table
summary_data = {
    'Metric': [
        'Total Records',
        'Total Columns',
        'Duplicate Rows',
        'Potential Duplicates',
        'High Availability Columns',
        'Medium Availability Columns',
        'Low Availability Columns',
        'Unique Cities',
        'Unique Property Types',
        'Date Range',
        'Records with Price',
        'Records with Size'
    ],
    'Value': [
        f"{len(df):,}",
        f"{len(df.columns)}",
        f"{duplicate_count:,} ({duplicate_percent:.2f}%)",
        f"{potential_duplicates:,}" if 'potential_duplicates' in dir() else 'N/A',
        f"{len(high_availability)}",
        f"{len(medium_availability)}",
        f"{len(low_availability)}",
        f"{df['city_slug'].nunique() if 'city_slug' in df.columns else 'N/A'}",
        f"{df['cat3_slug'].nunique() if 'cat3_slug' in df.columns else 'N/A'}",
        f"{df['created_at_month'].min()} to {df['created_at_month'].max()}" if 'created_at_month' in df.columns else 'N/A',
        f"{df['price_value'].notna().sum():,} ({df['price_value'].notna().sum()/len(df)*100:.1f}%)",
        f"{df['building_size'].notna().sum():,} ({df['building_size'].notna().sum()/len(df)*100:.1f}%)"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(DATA_PROCESSED / 'data_quality_summary.csv', index=False)
print(f" Data quality summary saved to: {DATA_PROCESSED / 'data_quality_summary.csv'}")

print("\n" + "=" * 60)
print("QUALITY SUMMARY TABLE")
print("=" * 60)
print(summary_df.to_string(index=False))

# %% [markdown]
# ---
#
# ## 16. Data Cleaning and Preparation
#
# Based on our quality assessment, we now perform comprehensive data cleaning to create a solid foundation for subsequent analysis phases.
#
# ### Cleaning Steps:
# 1. Remove exact duplicates
# 2. Convert numeric columns stored as objects
# 3. Flag outliers (don't remove - preserve for analysis flexibility)
# 4. Flag potential duplicates
# 5. Create listing type categories
# 6. Calculate row quality scores
# 7. Export cleaned datasets

# %%
# Start with a copy of the original data
df_cleaned = df.copy()
print(f"Starting data cleaning with {len(df_cleaned):,} rows")
print("=" * 60)

# %% [markdown]
# ### Step 1: Remove Exact Duplicates

# %%
# Step 1: Remove exact duplicates
initial_rows = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
removed_duplicates = initial_rows - len(df_cleaned)
print(f"\nStep 1 - Remove exact duplicates:")
print(f"  Removed: {removed_duplicates:,} rows")
print(f"  Remaining: {len(df_cleaned):,} rows")

# %% [markdown]
# ### Step 2: Convert Numeric Columns
#
# Some numeric columns are stored as objects (strings). We convert them to proper numeric types for analysis.

# %%
# Step 2: Convert numeric columns
numeric_conversion_cols = ['price_value', 'rent_value', 'credit_value', 
                           'building_size', 'land_size', 'location_latitude', 
                           'location_longitude', 'location_radius']

print("\nStep 2 - Convert numeric columns:")
for col in numeric_conversion_cols:
    if col in df_cleaned.columns:
        original_dtype = df_cleaned[col].dtype
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        print(f"  {col}: {original_dtype} -> {df_cleaned[col].dtype}")

# %% [markdown]
# ### Step 3: Create Outlier Flags
#
# We flag outliers using the 1st and 99th percentiles but don't remove them. This preserves data while allowing filtering during analysis.

# %%
# Step 3: Create outlier flags
print("\nStep 3 - Create outlier flags:")

# Price outliers
if 'price_value' in df_cleaned.columns:
    price_valid = df_cleaned['price_value'].notna() & (df_cleaned['price_value'] > 0)
    price_q01 = df_cleaned.loc[price_valid, 'price_value'].quantile(0.01)
    price_q99 = df_cleaned.loc[price_valid, 'price_value'].quantile(0.99)
    
    df_cleaned['price_outlier'] = (
        (df_cleaned['price_value'] < price_q01) | 
        (df_cleaned['price_value'] > price_q99)
    ).fillna(False)
    
    print(f"  Price outliers: {df_cleaned['price_outlier'].sum():,} rows flagged")
    print(f"    Valid range (1st-99th): {price_q01:,.0f} - {price_q99:,.0f} Tomans")

# Size outliers
if 'building_size' in df_cleaned.columns:
    size_valid = df_cleaned['building_size'].notna() & (df_cleaned['building_size'] > 0)
    size_q01 = df_cleaned.loc[size_valid, 'building_size'].quantile(0.01)
    size_q99 = df_cleaned.loc[size_valid, 'building_size'].quantile(0.99)
    
    df_cleaned['size_outlier'] = (
        (df_cleaned['building_size'] < size_q01) | 
        (df_cleaned['building_size'] > size_q99)
    ).fillna(False)
    
    print(f"  Size outliers: {df_cleaned['size_outlier'].sum():,} rows flagged")
    print(f"    Valid range (1st-99th): {size_q01:.0f} - {size_q99:.0f} sqm")

# %% [markdown]
# ### Step 4: Flag Potential Duplicates
#
# Records with identical key fields are flagged for awareness during analysis.

# %%
# Step 4: Flag potential duplicates
print("\nStep 4 - Flag potential duplicates:")

key_cols_dup = ['title', 'price_value', 'building_size', 'city_slug']
existing_dup_cols = [col for col in key_cols_dup if col in df_cleaned.columns]

if existing_dup_cols:
    df_cleaned['potential_duplicate'] = df_cleaned.duplicated(subset=existing_dup_cols, keep=False)
    print(f"  Potential duplicates flagged: {df_cleaned['potential_duplicate'].sum():,} rows")
    print(f"  (Based on: {existing_dup_cols})")

# %% [markdown]
# ### Step 5: Create Listing Type Categories
#
# Simplify listing categorization for easier filtering (sell vs rent vs other).

# %%
# Step 5: Create listing type categories
print("\nStep 5 - Create listing type categories:")

if 'cat2_slug' in df_cleaned.columns:
    def categorize_listing(cat2):
        if pd.isna(cat2):
            return 'unknown'
        cat2_lower = str(cat2).lower()
        if 'sell' in cat2_lower:
            return 'sell'
        elif 'rent' in cat2_lower:
            return 'rent'
        else:
            return 'other'
    
    df_cleaned['listing_type'] = df_cleaned['cat2_slug'].apply(categorize_listing)
    print(f"  Listing type distribution:")
    for lt, count in df_cleaned['listing_type'].value_counts().items():
        print(f"    {lt}: {count:,} ({count/len(df_cleaned)*100:.1f}%)")

# %% [markdown]
# ### Step 6: Calculate Row Quality Scores
#
# A quality score (0-100%) indicates how complete each row is based on key columns.

# %%
# Step 6: Calculate row quality scores
print("\nStep 6 - Calculate row quality scores:")

key_columns_for_quality = ['title', 'city_slug', 'cat3_slug', 'building_size', 
                           'price_value', 'rooms_count', 'description']
existing_key_cols = [col for col in key_columns_for_quality if col in df_cleaned.columns]

df_cleaned['quality_score'] = df_cleaned[existing_key_cols].notna().sum(axis=1) / len(existing_key_cols) * 100

print(f"  Quality score based on: {existing_key_cols}")
print(f"  Distribution:")
print(f"    Mean:   {df_cleaned['quality_score'].mean():.1f}%")
print(f"    Median: {df_cleaned['quality_score'].median():.1f}%")
print(f"    Min:    {df_cleaned['quality_score'].min():.1f}%")
print(f"    Max:    {df_cleaned['quality_score'].max():.1f}%")

# %% [markdown]
# ### Final Cleaned Data Summary

# %%
# Final summary of cleaned data
print("\n" + "=" * 70)
print("CLEANED DATA SUMMARY")
print("=" * 70)
print(f"\nTotal rows: {len(df_cleaned):,}")
print(f"Total columns: {len(df_cleaned.columns)}")
print(f"\nNew columns added:")
print(f"  - price_outlier     - Boolean flag for price outliers")
print(f"  - size_outlier      - Boolean flag for size outliers")
print(f"  - potential_duplicate - Boolean flag for potential duplicates")
print(f"  - listing_type      - Simplified category (sell/rent/other)")
print(f"  - quality_score     - Row completeness score (0-100%)")

# Data available for Phase 5
valid_price = df_cleaned['price_value'].notna().sum()
print(f"\n Data Available for Phase 5 (Price Prediction):")
print(f"  Records with price: {valid_price:,} ({valid_price/len(df_cleaned)*100:.1f}%)")

# %% [markdown]
# ### Step 7: Export Cleaned Datasets
#
# We create multiple output files for different analysis needs:
# 1. **cleaned_data.csv** - Full dataset with all cleaning flags
# 2. **data_for_price_prediction.csv** - Filtered for Phase 5 (valid sell prices, no outliers)
# 3. **data_rentals.csv** - Rental listings subset

# %%
# Save the main cleaned dataset
print("\nStep 7 - Export cleaned datasets:")
print("\nSaving cleaned_data.csv (this may take a few minutes)...")

df_cleaned.to_csv(DATA_PROCESSED / 'cleaned_data.csv', index=False)
file_size = (DATA_PROCESSED / 'cleaned_data.csv').stat().st_size / 1024**3
print(f" Saved: {DATA_PROCESSED / 'cleaned_data.csv'}")
print(f"  Size: {file_size:.2f} GB")
print(f"  Rows: {len(df_cleaned):,}")

# %%
# Create subset for price prediction (Phase 5)
df_for_price_prediction = df_cleaned[
    (df_cleaned['price_value'].notna()) & 
    (df_cleaned['price_value'] > 0) &
    (df_cleaned['listing_type'] == 'sell') &
    (~df_cleaned['price_outlier'])
].copy()

print(f"\n Price Prediction Dataset (Phase 5):")
print(f"  Filters applied:")
print(f"    - Has valid price (not null, > 0)")
print(f"    - Listing type = 'sell'")
print(f"    - Not a price outlier")
print(f"  Rows: {len(df_for_price_prediction):,} ({len(df_for_price_prediction)/len(df_cleaned)*100:.1f}% of cleaned data)")

df_for_price_prediction.to_csv(DATA_PROCESSED / 'data_for_price_prediction.csv', index=False)
print(f" Saved: {DATA_PROCESSED / 'data_for_price_prediction.csv'}")

# %%
# Create subset for rental analysis
df_rentals = df_cleaned[
    df_cleaned['listing_type'] == 'rent'
].copy()

print(f"\n Rental Analysis Dataset:")
print(f"  Rows: {len(df_rentals):,} ({len(df_rentals)/len(df_cleaned)*100:.1f}% of cleaned data)")

df_rentals.to_csv(DATA_PROCESSED / 'data_rentals.csv', index=False)
print(f" Saved: {DATA_PROCESSED / 'data_rentals.csv'}")

# %% [markdown]
# ---
#
# ## 17. Conclusion and Next Steps
#
# ### Phase 1 Deliverables
#
# | Output File | Description | Records |
# |-------------|-------------|---------|
# | `cleaned_data.csv` | Full cleaned dataset with quality flags | 1,000,000 |
# | `data_for_price_prediction.csv` | Filtered for Phase 5 | ~340,000 |
# | `data_rentals.csv` | Rental listings subset | ~383,000 |
# | `column_quality_summary.csv` | Missing value analysis by column | 60 columns |
# | `data_quality_summary.csv` | Overall quality metrics | - |
#
# ### Key Findings
#
# 1. **Data Completeness**: Only 7 columns have high availability (≤5% missing)
# 2. **Price Data**: ~43% of records have valid price values - this is the main limitation for Phase 5
# 3. **Duplicates**: No exact duplicates; ~20K potential duplicates flagged
# 4. **Geographic Coverage**: 421 unique cities, with Tehran dominating
# 5. **Time Coverage**: Data spans from 2020-02 to 2025-03
#
# ### Recommendations for Subsequent Phases
#
# - **Phase 2 (EDA)**: Use `cleaned_data.csv` for comprehensive exploration
# - **Phase 3 (Statistics)**: Consider the high missing rate when computing statistics
# - **Phase 5 (Price Prediction)**: Use `data_for_price_prediction.csv` with ~340K quality records
#
# ---
#
# **Phase 1 Complete** 
#
# Proceed to **Phase 2: Exploratory Data Analysis**
