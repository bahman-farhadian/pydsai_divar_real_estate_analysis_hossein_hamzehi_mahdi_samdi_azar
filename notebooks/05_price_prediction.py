# %% [markdown]
# # Phase 5: Price Prediction
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records)
#
# ---
#
# ## Objective
#
# Build price prediction models and identify over-valued and under-valued listings.
#
#
# ## Analysis Scope
#
# 1. **Data Loading & Preparation** - Load cleaned data and filter for modeling
# 2. **Feature Engineering** - Create and encode features for prediction
# 3. **Train/Test Split** - Prepare data for model training
# 4. **Model 1: Linear Regression** - Interpretable baseline
# 5. **Model 2: Random Forest** - Feature importance analysis
# 6. **Model 3: Gradient Boosting** - Best performance comparison
# 7. **Model Comparison** - Evaluate using R², RMSE, MAE
# 8. **Value Classification** - Identify over/under-valued listings
# 9. **Model Limitations** - Document where the model succeeds and fails
# 10. **Export Results** - Save predictions and classifications
#
# ---

# %% [markdown]
# ## 1. Setup and Library Imports

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
import warnings
warnings.filterwarnings('ignore')

pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Plot styling (consistent with previous phases)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
sns.set_style('whitegrid')

# Color palette
COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'purple': '#9b59b6',
    'orange': '#e67e22'
}

print("Libraries loaded successfully")

def read_csv_fast(path, **kwargs):
    parquet_path = path.with_suffix('.parquet')
    if parquet_path.exists():
        print(f"Loading Parquet: {parquet_path.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(parquet_path)
    try:
        return pd.read_csv(path, engine='pyarrow', **kwargs)
    except Exception as exc:
        print(f"PyArrow CSV engine unavailable for {path.name}; falling back to pandas C engine ({exc})")
        return pd.read_csv(path, low_memory=False, **kwargs)

# %% [markdown]
# ## 2. Project Structure and Data Loading

# %%
# Define project paths
def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / 'Divar-Real-State-Ads').exists() and (path / 'notebooks').exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


PROJECT_ROOT = find_project_root()
REPORTS_PATH = PROJECT_ROOT / 'reports'
DATA_PROCESSED = REPORTS_PATH / 'data'
FIGURES_PATH = REPORTS_PATH / 'figures'
MODELS_PATH = REPORTS_PATH / 'models'

# Create output directories
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

print("Project root: .")
print(f"Figures path: {FIGURES_PATH.relative_to(PROJECT_ROOT)}")
print(f"Models path: {MODELS_PATH.relative_to(PROJECT_ROOT)}")

# %%
# Load the enhanced dataset from Phase 2
DATA_FILE = DATA_PROCESSED / 'cleaned_data_with_features.csv'

print(f"Loading data from: {DATA_FILE.relative_to(PROJECT_ROOT)}")
df_full = read_csv_fast(DATA_FILE)
print(f"\n Full dataset: {len(df_full):,} rows, {len(df_full.columns)} columns")

# %% [markdown]
# ## 3. Data Preparation
#
# ### 3.1 Filter for Modeling
#
# We focus on sale listings with valid price per sqm, excluding outliers.

# %%
print("=" * 60)
print("DATA FILTERING FOR PRICE PREDICTION")
print("=" * 60)

# Ensure numeric types
df_full['price_value'] = pd.to_numeric(df_full['price_value'], errors='coerce')
df_full['building_size'] = pd.to_numeric(df_full['building_size'], errors='coerce')
df_full['price_per_sqm'] = pd.to_numeric(df_full['price_per_sqm'], errors='coerce')

# Debug: Check initial values
print("\nDebug - price_per_sqm initial stats:")
print(f"  Non-null: {df_full['price_per_sqm'].notna().sum():,}")
print(f"  Min: {df_full['price_per_sqm'].min()}")
print(f"  Max: {df_full['price_per_sqm'].max()}")
print(f"  Median: {df_full['price_per_sqm'].median()}")

# Filter criteria
print("\nApplying filters:")
print(f"  1. Starting records: {len(df_full):,}")

# Filter 1: Sale listings only
df = df_full[df_full['listing_type'] == 'sell'].copy()
print(f"  2. After 'sell' filter: {len(df):,}")

# Filter 2: Valid price_per_sqm (not null)
df = df[df['price_per_sqm'].notna()].copy()
print(f"  3. After price_per_sqm not null: {len(df):,}")

# Debug: Check price range before filtering
print(f"\nDebug - price_per_sqm range (sell only):")
print(f"  Min: {df['price_per_sqm'].min():,.0f}")
print(f"  Max: {df['price_per_sqm'].max():,.0f}")
print(f"  Median: {df['price_per_sqm'].median():,.0f}")

# Check how many fall in different ranges
print(f"\nDebug - price_per_sqm distribution:")
print(f"  < 1M: {(df['price_per_sqm'] < 1_000_000).sum():,}")
print(f"  1M - 5M: {((df['price_per_sqm'] >= 1_000_000) & (df['price_per_sqm'] < 5_000_000)).sum():,}")
print(f"  5M - 500M: {((df['price_per_sqm'] >= 5_000_000) & (df['price_per_sqm'] <= 500_000_000)).sum():,}")
print(f"  > 500M: {(df['price_per_sqm'] > 500_000_000).sum():,}")

# Filter 3: Reasonable price range
# Use percentile-based filtering instead of fixed thresholds
p1 = df['price_per_sqm'].quantile(0.01)
p99 = df['price_per_sqm'].quantile(0.99)
print(f"\nUsing percentile-based filtering (1st-99th percentile):")
print(f"  Lower bound (1%): {p1:,.0f}")
print(f"  Upper bound (99%): {p99:,.0f}")

df = df[(df['price_per_sqm'] >= p1) & (df['price_per_sqm'] <= p99)].copy()
print(f"  4. After outlier removal: {len(df):,}")

# Filter 4: Valid building size
df = df[(df['building_size'] > 0) & (df['building_size'] <= 2000)].copy()
print(f"  5. After size filter (0-2000 sqm): {len(df):,}")

print(f"\n Final dataset for modeling: {len(df):,} records")

if len(df) == 0:
    print("\nERROR: No data remaining after filtering!")
    print("Check your data quality and filtering criteria.")
else:
    print(f"\nSuccess: {len(df):,} records ready for modeling")

# %%
# Target variable summary
print("\n" + "=" * 60)
print("TARGET VARIABLE: price_per_sqm (Million Tomans)")
print("=" * 60)

target = df['price_per_sqm'] / 1e6  # Convert to millions for readability

print(f"\nStatistics:")
print(f"  Count:  {len(target):,}")
print(f"  Mean:   {target.mean():.1f}M Tomans/sqm")
print(f"  Median: {target.median():.1f}M Tomans/sqm")
print(f"  Std:    {target.std():.1f}M")
print(f"  Min:    {target.min():.1f}M")
print(f"  Max:    {target.max():.1f}M")
print(f"  Skewness: {target.skew():.2f}")

# %%
# Visualize target distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original distribution
axes[0].hist(target, bins=50, color=COLORS['primary'], edgecolor='white', alpha=0.8)
axes[0].axvline(x=target.median(), color='red', linestyle='--', lw=2, label=f'Median: {target.median():.0f}M')
axes[0].set_xlabel('Price per sqm (Million Tomans)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution (Original)')
axes[0].legend()

# Log-transformed distribution
log_target = np.log1p(target)
axes[1].hist(log_target, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
axes[1].axvline(x=log_target.median(), color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Log(Price per sqm)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Target Distribution (Log-transformed)')

# Box plot
bp = axes[2].boxplot(target, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor(COLORS['primary'])
axes[2].set_ylabel('Price per sqm (Million Tomans)')
axes[2].set_title('Target Box Plot')
axes[2].set_xticklabels(['Price/sqm'])

plt.suptitle('Target Variable Analysis: Price per Square Meter', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n Log transformation reduces skewness from {target.skew():.2f} to {log_target.skew():.2f}")

# %% [markdown]
# ## 4. Feature Engineering
#
# ### 4.1 Feature Selection and Creation

# %%
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Helper function to convert Persian numerals to English
def persian_to_english_num(text):
    """Convert Persian/Arabic numerals to English numerals"""
    if pd.isna(text):
        return np.nan
    persian_nums = '۰۱۲۳۴۵۶۷۸۹'
    arabic_nums = '٠١٢٣٤٥٦٧٨٩'
    english_nums = '0123456789'
    
    text = str(text)
    for p, a, e in zip(persian_nums, arabic_nums, english_nums):
        text = text.replace(p, e).replace(a, e)
    
    try:
        return int(text)
    except ValueError:
        return np.nan

# 1. Convert rooms_count to numeric
room_mapping = {
    'بدون اتاق': 0,
    'یک': 1,
    'دو': 2,
    'سه': 3,
    'چهار': 4,
    'پنج یا بیشتر': 5
}
df['rooms_numeric'] = df['rooms_count'].map(room_mapping)
print(f"rooms_numeric: {df['rooms_numeric'].notna().sum():,} valid values")

# 2. Building age (Iranian calendar year 1403)
# First convert Persian numerals to English
df['construction_year_numeric'] = df['construction_year'].apply(persian_to_english_num)
current_year = 1403
df['building_age'] = current_year - df['construction_year_numeric']
# Cap building age at reasonable range
df.loc[df['building_age'] < 0, 'building_age'] = np.nan
df.loc[df['building_age'] > 100, 'building_age'] = np.nan
print(f"building_age: {df['building_age'].notna().sum():,} valid values")

# 3. Convert amenities to binary
for col in ['has_elevator', 'has_parking', 'has_warehouse']:
    if col in df.columns:
        df[col + '_binary'] = df[col].apply(lambda x: 1 if x == True or x == 'True' or x == 1 else 0)
        print(f"{col}_binary: {df[col + '_binary'].sum():,} positive values")

# 4. Amenity score (sum of amenities)
amenity_cols = ['has_elevator_binary', 'has_parking_binary', 'has_warehouse_binary']
df['amenity_score'] = df[amenity_cols].sum(axis=1)
print(f"amenity_score: mean = {df['amenity_score'].mean():.2f}")

# 5. Log transform building size
df['log_building_size'] = np.log1p(df['building_size'])
print(f"log_building_size created")

# %% [markdown]
# ### 4.2 City Encoding (Target Encoding)
#
# With 400+ cities, we use target encoding - replacing city name with its median price per sqm.

# %%
# Target encoding for city
print("\n" + "=" * 60)
print("CITY TARGET ENCODING")
print("=" * 60)

# Calculate median price per sqm for each city
city_medians = df.groupby('city_slug')['price_per_sqm'].median()

# Encode city by its median price (normalized to millions)
df['city_price_level'] = df['city_slug'].map(city_medians) / 1e6

print(f"\nUnique cities: {df['city_slug'].nunique()}")
print(f"City price level range: {df['city_price_level'].min():.1f}M - {df['city_price_level'].max():.1f}M")

# Show top and bottom cities
print(f"\nTop 5 Most Expensive Cities:")
top_cities = city_medians.sort_values(ascending=False).head(5)
for city, price in top_cities.items():
    print(f"  {city:<20}: {price/1e6:.1f}M Tomans/sqm")

print(f"\nTop 5 Most Affordable Cities:")
bottom_cities = city_medians.sort_values().head(5)
for city, price in bottom_cities.items():
    print(f"  {city:<20}: {price/1e6:.1f}M Tomans/sqm")

# %% [markdown]
# ### 4.3 Property Type Encoding

# %%
# Target encoding for property type
print("\n" + "=" * 60)
print("PROPERTY TYPE ENCODING")
print("=" * 60)

cat3_medians = df.groupby('cat3_slug')['price_per_sqm'].median()
df['property_price_level'] = df['cat3_slug'].map(cat3_medians) / 1e6

print(f"\nProperty types by median price/sqm:")
for cat, price in cat3_medians.sort_values(ascending=False).items():
    count = (df['cat3_slug'] == cat).sum()
    print(f"  {cat:<35}: {price/1e6:>6.1f}M  (n={count:,})")

# %%
# ENHANCED FEATURE ENGINEERING (Additional features)
print("=" * 60)
print("ENHANCED FEATURE ENGINEERING")
print("=" * 60)

# Helper to handle '30+' type values
def clean_numeric(val):
    """Convert Persian numerals and handle '30+' type values"""
    if pd.isna(val):
        return np.nan
    val = str(val).replace('+', '').strip()
    return persian_to_english_num(val)

# 1. Floor features
df['floor_numeric'] = df['floor'].apply(clean_numeric)
df['total_floors_numeric'] = df['total_floors_count'].apply(clean_numeric)
df['floor_ratio'] = df['floor_numeric'] / df['total_floors_numeric'].replace(0, np.nan)
df['is_ground_floor'] = (df['floor_numeric'] == 0).astype(int)
print(f"floor features: {df['floor_numeric'].notna().sum():,} valid")

# 2. Size categories (non-linear relationship with price)
df['size_category'] = pd.cut(df['building_size'], 
                              bins=[0, 50, 80, 120, 180, 300, 2000],
                              labels=[1, 2, 3, 4, 5, 6])
df['size_category'] = pd.to_numeric(df['size_category'], errors='coerce')
print(f"size_category created")

# 3. Age categories
df['is_new_building'] = (df['building_age'] <= 3).astype(int)
print(f"is_new_building: {df['is_new_building'].sum():,} new buildings")

# 4. Rooms per sqm (density)
df['sqm_per_room'] = df['building_size'] / (df['rooms_numeric'].fillna(2) + 1)
print(f"sqm_per_room: mean = {df['sqm_per_room'].mean():.1f}")

# 5. Neighborhood encoding (if available)
if 'neighborhood_slug' in df.columns and df['neighborhood_slug'].notna().sum() > 10000:
    neighborhood_medians = df.groupby('neighborhood_slug')['price_per_sqm'].median()
    df['neighborhood_price_level'] = df['neighborhood_slug'].map(neighborhood_medians) / 1e6
    print(f"neighborhood_price_level: {df['neighborhood_price_level'].notna().sum():,} valid")
else:
    df['neighborhood_price_level'] = df['city_price_level']
    print("neighborhood_price_level: using city_price_level as fallback")

# 6. Interaction features
df['size_x_city'] = df['building_size'] * df['city_price_level']
print(f"size_x_city created")

print("\nEnhanced features complete!")

# %%
# ADVANCED FEATURE ENGINEERING FOR HIGHER ACCURACY
print("=" * 60)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# 1. Neighborhood-level price encoding (more granular than city)
if 'neighborhood_slug' in df.columns:
    # Only use neighborhoods with enough samples
    neighborhood_counts = df['neighborhood_slug'].value_counts()
    valid_neighborhoods = neighborhood_counts[neighborhood_counts >= 20].index
    
    neighborhood_medians = df[df['neighborhood_slug'].isin(valid_neighborhoods)].groupby('neighborhood_slug')['price_per_sqm'].median()
    df['neighborhood_price_level'] = df['neighborhood_slug'].map(neighborhood_medians) / 1e6
    
    # Fill missing with city level
    df['neighborhood_price_level'] = df['neighborhood_price_level'].fillna(df['city_price_level'])
    print(f"neighborhood_price_level: {df['neighborhood_price_level'].notna().sum():,} valid")

# 2. Price per room (important derived feature)
df['price_per_room'] = df['price_per_sqm'] / (df['rooms_numeric'].fillna(2) + 1)
print(f"price_per_room created")

# 3. Size squared (captures non-linear relationship)
df['building_size_sq'] = df['building_size'] ** 2
df['building_size_sqrt'] = np.sqrt(df['building_size'])
print(f"size transformations created")

# 4. Age-based features
df['age_squared'] = df['building_age'] ** 2
df['is_very_new'] = (df['building_age'] <= 2).astype(int)
df['is_old'] = (df['building_age'] >= 20).astype(int)
print(f"age features created")

# 5. Location quality score (combination of city and property type)
df['location_quality'] = df['city_price_level'] * df['property_price_level'] / df['property_price_level'].mean()
print(f"location_quality created")

# 6. Amenity interactions
df['elevator_x_floor'] = df['has_elevator_binary'] * df['floor_numeric'].fillna(0)
df['new_with_amenities'] = df['is_new_building'] * df['amenity_score']
print(f"amenity interactions created")

# 7. Size-location interaction
df['size_x_neighborhood'] = df['building_size'] * df['neighborhood_price_level']
print(f"size_x_neighborhood created")

print("\nAdvanced features complete!")

# %%
# COMPREHENSIVE FEATURE SELECTION (After all encoding is done)
print("=" * 60)
print("COMPREHENSIVE FEATURE SELECTION")
print("=" * 60)

# List all potential features - location features are now available!
all_features = [
    # Core features
    'building_size',
    'log_building_size',
    'building_size_sqrt',
    'rooms_numeric',
    'building_age',
    
    # Location features (MOST IMPORTANT - created in previous cells)
    'city_price_level',
    'property_price_level',
    'neighborhood_price_level',
    'location_quality',
    
    # Amenities
    'has_elevator_binary',
    'has_parking_binary', 
    'has_warehouse_binary',
    'amenity_score',
    
    # Floor features
    'floor_numeric',
    'floor_ratio',
    'is_ground_floor',
    
    # Age features
    'is_new_building',
    'is_very_new',
    'is_old',
    
    # Interactions
    'size_x_city',
    'size_x_neighborhood',
    'sqm_per_room',
    'elevator_x_floor',
    'new_with_amenities',
]

# Filter to available and valid columns
feature_cols = []
print("\nChecking feature availability:")
for f in all_features:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        valid_count = df[f].notna().sum()
        if valid_count > 1000:
            feature_cols.append(f)
            pct = valid_count / len(df) * 100
            print(f"  [OK] {f:<25}: {valid_count:>10,} ({pct:>5.1f}%)")
        else:
            print(f"  [SKIP] {f:<25}: only {valid_count} valid values")
    else:
        print(f"  [MISSING] {f}")

print(f"\n>>> Using {len(feature_cols)} features for modeling")

# %%
# Handle missing values
print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

# Strategy: Impute with median for numeric, 0 for binary
df_model = df.copy()

# Impute rooms_numeric with median
rooms_median = df_model['rooms_numeric'].median()
df_model['rooms_numeric'] = df_model['rooms_numeric'].fillna(rooms_median)
print(f" rooms_numeric: imputed missing with median ({rooms_median:.0f})")

# Impute building_age with median
age_median = df_model['building_age'].median()
df_model['building_age'] = df_model['building_age'].fillna(age_median)
print(f" building_age: imputed missing with median ({age_median:.0f} years)")

# Impute city_price_level with global median (for any missing)
global_price_median = df_model['price_per_sqm'].median() / 1e6
df_model['city_price_level'] = df_model['city_price_level'].fillna(global_price_median)
df_model['property_price_level'] = df_model['property_price_level'].fillna(global_price_median)
print(f" city/property price levels: imputed with global median ({global_price_median:.1f}M)")

# Check completeness
print(f"\nFinal dataset completeness:")
for feat in feature_cols:
    missing = df_model[feat].isna().sum()
    print(f"  {feat:<25}: {missing} missing")

# %% [markdown]
# ### 4.5 Feature Correlation Analysis

# %%
# Correlation analysis
print("\n" + "=" * 60)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 60)

# Create target in millions
df_model['target'] = df_model['price_per_sqm'] / 1e6

# Correlation with target
corr_cols = feature_cols + ['target']
corr_matrix = df_model[corr_cols].corr()

print("\nCorrelation with Target (price_per_sqm):")
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
for feat, corr in target_corr.items():
    strength = 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
    print(f"  {feat:<25}: {corr:>7.3f} ({strength})")

# %%
# Correlation heatmap - Enhanced readability
fig, ax = plt.subplots(figsize=(16, 14))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, square=True,
            annot_kws={'size': 9, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

# Improve tick labels readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_feature_correlation.png', dpi=200, bbox_inches='tight')
plt.show()

print("Key insight: city_price_level and neighborhood_price_level show strongest correlation with target")

# %% [markdown]
# ## 5. Train/Test Split

# %%
print("=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

# Prepare X and y
X = df_model[feature_cols].copy()
y = df_model['target'].copy()  # Already in millions

# Remove any remaining rows with NaN
valid_mask = X.notna().all(axis=1) & y.notna()
X = X[valid_mask]
y = y[valid_mask]

print(f"\nFinal samples for modeling: {len(X):,}")

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.0f}%)")

# Verify distributions are similar
print(f"\nTarget distribution check:")
print(f"  Train mean: {y_train.mean():.1f}M | Test mean: {y_test.mean():.1f}M")
print(f"  Train median: {y_train.median():.1f}M | Test median: {y_test.median():.1f}M")

# %%
# Scale features (important for Linear Regression and comparison)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Features scaled using StandardScaler")
print(f"  Train shape: {X_train_scaled.shape}")
print(f"  Test shape: {X_test_scaled.shape}")

# %% [markdown]
# ## 6. Model 1: Linear Regression (Baseline)
#
# Linear Regression provides an interpretable baseline with coefficient analysis.

# %%
print("=" * 60)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 60)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Metrics
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print(f"\n Performance Metrics:")
print(f"  R² Score:  {lr_r2:.4f} ({lr_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {lr_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {lr_mae:.2f}M Tomans/sqm")

# %%
# Coefficient analysis
print("\n Feature Coefficients (Standardized):")
print("-" * 50)

coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for _, row in coef_df.iterrows():
    direction = '^' if row['Coefficient'] > 0 else 'v'
    print(f"  {row['Feature']:<25}: {row['Coefficient']:>8.3f} {direction}")

print(f"\n  Intercept: {lr_model.intercept_:.2f}M")

# %%
# Visualize coefficients
fig, ax = plt.subplots(figsize=(10, 6))

colors = [COLORS['primary'] if c > 0 else COLORS['accent'] for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient (Standardized)')
ax.set_ylabel('Feature')
ax.set_title('Linear Regression Coefficients\n(Positive = Increases Price, Negative = Decreases Price)')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Model 2: Random Forest
#
# Random Forest for feature importance analysis and non-linear relationships.

# %%
# IMPROVED RANDOM FOREST WITH BETTER HYPERPARAMETERS
print("=" * 60)
print("MODEL 2: RANDOM FOREST (Tuned)")
print("=" * 60)

# Better hyperparameters for this dataset
rf_model = RandomForestRegressor(
    n_estimators=200,        # More trees
    max_depth=20,            # Deeper trees
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=3,      # Minimum samples in leaf
    max_features='sqrt',     # Features per split
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest (200 trees, max_depth=20)...")
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

print("[OK] Training complete")

# Metrics
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mape = np.mean(np.abs((y_test[y_test > 1] - y_pred_rf[y_test > 1]) / y_test[y_test > 1])) * 100  # Exclude near-zero

print(f"\nPerformance Metrics:")
print(f"  R2 Score:  {rf_r2:.4f} ({rf_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {rf_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {rf_mae:.2f}M Tomans/sqm")
print(f"  MAPE:      {rf_mape:.1f}% (Mean Absolute Percentage Error)")

# %%
# Feature importance
print("\n Feature Importance:")
print("-" * 50)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<25}: {row['Importance']:.3f} {bar}")

# %%
# Visualize feature importance
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))[::-1]
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='white')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
ax.set_title('Random Forest Feature Importance\n(Higher = More Important for Prediction)')

# Add headroom for labels (25% extra space on right)
max_importance = importance_df['Importance'].max()
ax.set_xlim(0, max_importance * 1.25)

# Add percentage labels to the RIGHT of bars
for i, (_, row) in enumerate(importance_df.iterrows()):
    ax.text(row['Importance'] + max_importance * 0.02, i, f"{row['Importance']*100:.1f}%", 
            va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_rf_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Model 3: Gradient Boosting
#
# Gradient Boosting for best performance comparison.

# %%
# IMPROVED GRADIENT BOOSTING
print("=" * 60)
print("MODEL 3: GRADIENT BOOSTING (Tuned)")
print("=" * 60)

gb_model = GradientBoostingRegressor(
    n_estimators=200,        # More iterations
    max_depth=6,             # Slightly deeper
    learning_rate=0.08,      # Slightly lower for better generalization
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,           # Stochastic gradient boosting
    random_state=42
)

print("Training Gradient Boosting (200 iterations)...")
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

print("[OK] Training complete")

# Metrics
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_mape = np.mean(np.abs((y_test[y_test > 1] - y_pred_gb[y_test > 1]) / y_test[y_test > 1])) * 100  # Exclude near-zero

print(f"\nPerformance Metrics:")
print(f"  R2 Score:  {gb_r2:.4f} ({gb_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {gb_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {gb_mae:.2f}M Tomans/sqm")
print(f"  MAPE:      {gb_mape:.1f}%")

# %%
# ADVANCED MODELS FOR HIGHER ACCURACY
print("=" * 60)
print("ADVANCED MODEL: HISTOGRAM GRADIENT BOOSTING")
print("=" * 60)

from sklearn.ensemble import HistGradientBoostingRegressor

# HistGradientBoosting is faster and often more accurate
hgb_model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=12,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42
)

print("Training HistGradientBoosting (500 iterations)...")
hgb_model.fit(X_train_scaled, y_train)
y_pred_hgb = hgb_model.predict(X_test_scaled)

hgb_r2 = r2_score(y_test, y_pred_hgb)
hgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
hgb_mae = mean_absolute_error(y_test, y_pred_hgb)

print(f"\nHistGradientBoosting Performance:")
print(f"  R2 Score:  {hgb_r2:.4f} ({hgb_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {hgb_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {hgb_mae:.2f}M Tomans/sqm")

# %%
# TUNED RANDOM FOREST FOR HIGHER ACCURACY
print("=" * 60)
print("MODEL: TUNED RANDOM FOREST")
print("=" * 60)

# More trees, deeper, optimized parameters
rf_tuned = RandomForestRegressor(
    n_estimators=300,          # More trees
    max_depth=25,              # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.5,          # Use 50% of features per split
    bootstrap=True,
    oob_score=True,            # Out-of-bag score for validation
    random_state=42,
    n_jobs=-1
)

print("Training Tuned Random Forest (300 trees, depth=25)...")
rf_tuned.fit(X_train_scaled, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test_scaled)

rf_tuned_r2 = r2_score(y_test, y_pred_rf_tuned)
rf_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
rf_tuned_mae = mean_absolute_error(y_test, y_pred_rf_tuned)

print(f"\nTuned Random Forest Performance:")
print(f"  R2 Score:  {rf_tuned_r2:.4f} ({rf_tuned_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {rf_tuned_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {rf_tuned_mae:.2f}M Tomans/sqm")
print(f"  OOB Score: {rf_tuned.oob_score_:.4f}")

# %%
# LOG-TRANSFORMED TARGET APPROACH
print("=" * 60)
print("LOG-TRANSFORMED TARGET MODELING")
print("=" * 60)

# Log transform reduces impact of outliers and skewness
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Train on log-transformed target
rf_log = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest on log-transformed target...")
rf_log.fit(X_train_scaled, y_train_log)
y_pred_log = rf_log.predict(X_test_scaled)

# Transform predictions back to original scale
y_pred_original = np.expm1(y_pred_log)

# Calculate metrics on original scale
rf_log_r2 = r2_score(y_test, y_pred_original)
rf_log_rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
rf_log_mae = mean_absolute_error(y_test, y_pred_original)

print(f"\nLog-Target Random Forest Performance:")
print(f"  R2 Score:  {rf_log_r2:.4f} ({rf_log_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {rf_log_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {rf_log_mae:.2f}M Tomans/sqm")

# %%
# ENSEMBLE MODEL (Combining multiple models)
print("=" * 60)
print("ENSEMBLE MODEL")
print("=" * 60)

# Simple averaging ensemble
y_pred_ensemble = (y_pred_rf_tuned + y_pred_hgb + y_pred_original) / 3

ensemble_r2 = r2_score(y_test, y_pred_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)

print(f"\nEnsemble (RF + HGB + LogRF) Performance:")
print(f"  R2 Score:  {ensemble_r2:.4f} ({ensemble_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {ensemble_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {ensemble_mae:.2f}M Tomans/sqm")

# Weighted ensemble (give more weight to better models)
weights = np.array([rf_tuned_r2, hgb_r2, rf_log_r2])
weights = weights / weights.sum()
y_pred_weighted = (weights[0] * y_pred_rf_tuned + 
                   weights[1] * y_pred_hgb + 
                   weights[2] * y_pred_original)

weighted_r2 = r2_score(y_test, y_pred_weighted)
weighted_rmse = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
weighted_mae = mean_absolute_error(y_test, y_pred_weighted)

print(f"\nWeighted Ensemble Performance:")
print(f"  R2 Score:  {weighted_r2:.4f} ({weighted_r2*100:.1f}% variance explained)")
print(f"  RMSE:      {weighted_rmse:.2f}M Tomans/sqm")
print(f"  MAE:       {weighted_mae:.2f}M Tomans/sqm")
print(f"  Weights: RF={weights[0]:.2f}, HGB={weights[1]:.2f}, LogRF={weights[2]:.2f}")

# %%
# FINAL MODEL COMPARISON
print("=" * 70)
print("FINAL MODEL COMPARISON - ALL APPROACHES")
print("=" * 70)

all_results = pd.DataFrame([
    {'Model': 'Linear Regression', 'R2': lr_r2, 'RMSE': lr_rmse, 'MAE': lr_mae},
    {'Model': 'Random Forest (Basic)', 'R2': rf_r2, 'RMSE': rf_rmse, 'MAE': rf_mae},
    {'Model': 'Gradient Boosting (Basic)', 'R2': gb_r2, 'RMSE': gb_rmse, 'MAE': gb_mae},
    {'Model': 'Random Forest (Tuned)', 'R2': rf_tuned_r2, 'RMSE': rf_tuned_rmse, 'MAE': rf_tuned_mae},
    {'Model': 'HistGradientBoosting', 'R2': hgb_r2, 'RMSE': hgb_rmse, 'MAE': hgb_mae},
    {'Model': 'Log-Target RF', 'R2': rf_log_r2, 'RMSE': rf_log_rmse, 'MAE': rf_log_mae},
    {'Model': 'Ensemble (Average)', 'R2': ensemble_r2, 'RMSE': ensemble_rmse, 'MAE': ensemble_mae},
    {'Model': 'Ensemble (Weighted)', 'R2': weighted_r2, 'RMSE': weighted_rmse, 'MAE': weighted_mae},
]).sort_values('R2', ascending=False)

print(f"\n{'Model':<25} {'R2':>10} {'RMSE':>12} {'MAE':>12}")
print("-" * 60)
for _, row in all_results.iterrows():
    marker = " <-- BEST" if row['R2'] == all_results['R2'].max() else ""
    print(f"{row['Model']:<25} {row['R2']:>10.4f} {row['RMSE']:>10.2f}M {row['MAE']:>10.2f}M{marker}")

best_r2 = all_results['R2'].max()
print(f"\n{'='*60}")
if best_r2 >= 0.80:
    print(f"SUCCESS! Best R2 = {best_r2:.4f} (>= 80%)")
elif best_r2 >= 0.70:
    print(f"GOOD! Best R2 = {best_r2:.4f} (70-80%)")
else:
    print(f"Best R2 = {best_r2:.4f} (<70%)")
    print("Note: Real estate prediction is inherently limited by unobserved factors")
print("=" * 60)

# %%
# Gradient Boosting feature importance
print("\n Gradient Boosting Feature Importance:")
print("-" * 50)

gb_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in gb_importance_df.iterrows():
    bar = '' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<25}: {row['Importance']:.3f} {bar}")

# %% [markdown]
# ## 9. Model Comparison

# %%
# CROSS-VALIDATION FOR ROBUST EVALUATION
print("=" * 60)
print("CROSS-VALIDATION (5-Fold)")
print("=" * 60)

from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nEvaluating models with 5-fold cross-validation...")
print("(This gives more reliable performance estimates)\n")

# Linear Regression CV
lr_cv_scores = cross_val_score(LinearRegression(), X_train_scaled, y_train, 
                                cv=kfold, scoring='r2', n_jobs=-1)
print(f"Linear Regression:")
print(f"  CV R2: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std()*2:.4f})")

# Random Forest CV  
rf_cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, 
                          random_state=42, n_jobs=-1),
    X_train_scaled, y_train, cv=kfold, scoring='r2', n_jobs=-1
)
print(f"\nRandom Forest:")
print(f"  CV R2: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

# Gradient Boosting CV
gb_cv_scores = cross_val_score(
    GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=42),
    X_train_scaled, y_train, cv=kfold, scoring='r2', n_jobs=-1
)
print(f"\nGradient Boosting:")
print(f"  CV R2: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std()*2:.4f})")

print("\n" + "-" * 40)
print("Cross-validation helps detect overfitting.")
print("If CV score << test score, model is overfitting.")

# %%
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Create comparison table
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'R² Score': [lr_r2, rf_r2, gb_r2],
    'RMSE (M Tomans)': [lr_rmse, rf_rmse, gb_rmse],
    'MAE (M Tomans)': [lr_mae, rf_mae, gb_mae]
})

print("\n Performance Comparison:")
print(results.to_string(index=False))

# Find best model
best_idx = results['R² Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
best_r2 = results.loc[best_idx, 'R² Score']

print(f"\n Best Model: {best_model_name} (R² = {best_r2:.4f})")

# %%
# COMPREHENSIVE MODEL EVALUATION
print("=" * 60)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)

def evaluate_model(name, y_true, y_pred):
    """Calculate comprehensive metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true[y_true > 1] - y_pred[y_true > 1]) / y_true[y_true > 1])) * 100  # Exclude near-zero
    
    # Percentage within 20% of actual
    within_20 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.20) * 100
    within_30 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.30) * 100
    
    return {
        'Model': name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Within_20pct': within_20,
        'Within_30pct': within_30
    }

# Evaluate all models
eval_results = []
eval_results.append(evaluate_model('Linear Regression', y_test, y_pred_lr))
eval_results.append(evaluate_model('Random Forest', y_test, y_pred_rf))
eval_results.append(evaluate_model('Gradient Boosting', y_test, y_pred_gb))

eval_df = pd.DataFrame(eval_results)

print("\nModel Comparison:")
print("=" * 80)
print(f"{'Model':<20} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'<20%':>8} {'<30%':>8}")
print("-" * 80)
for _, row in eval_df.iterrows():
    print(f"{row['Model']:<20} {row['R2']:>8.3f} {row['RMSE']:>8.1f} {row['MAE']:>8.1f} {row['MAPE']:>7.1f}% {row['Within_20pct']:>7.1f}% {row['Within_30pct']:>7.1f}%")

print("\n" + "=" * 80)
print("KEY METRICS EXPLAINED:")
print("  - R2: Variance explained (higher = better, 1.0 = perfect)")
print("  - RMSE: Root Mean Square Error in M Tomans/sqm")
print("  - MAE: Mean Absolute Error in M Tomans/sqm")  
print("  - MAPE: Mean Absolute Percentage Error")
print("  - <20%: Percentage of predictions within 20% of actual")
print("  - <30%: Percentage of predictions within 30% of actual")

# %%
# ERROR ANALYSIS - Where does the model fail?
print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# Use best model predictions
y_pred_best = y_pred_gb if gb_r2 > rf_r2 else y_pred_rf
best_name = "Gradient Boosting" if gb_r2 > rf_r2 else "Random Forest"

# Calculate errors
errors = y_test - y_pred_best
abs_errors = np.abs(errors)
pct_errors = np.abs(errors / y_test) * 100

# Create analysis dataframe
error_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_best,
    'error': errors,
    'abs_error': abs_errors,
    'pct_error': pct_errors
})

print(f"\nUsing {best_name} for error analysis\n")

# Error by price range
print("Error by Price Range:")
print("-" * 50)
price_bins = [(0, 30), (30, 50), (50, 80), (80, 120), (120, 200)]
for low, high in price_bins:
    mask = (error_df['actual'] >= low) & (error_df['actual'] < high)
    if mask.sum() > 0:
        subset = error_df[mask]
        print(f"  {low:>3}-{high:<3}M: MAE={subset['abs_error'].mean():>5.1f}M, MAPE={subset['pct_error'].mean():>5.1f}%, n={mask.sum():,}")

# Identify worst predictions
print("\nTop 10 Worst Predictions (highest % error):")
print("-" * 50)
worst = error_df.nlargest(10, 'pct_error')
for i, (idx, row) in enumerate(worst.iterrows(), 1):
    print(f"  {i}. Actual: {row['actual']:.1f}M, Pred: {row['predicted']:.1f}M, Error: {row['pct_error']:.0f}%")

# %%
# Model Performance Comparison - labels ABOVE bars
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
colors = [COLORS['secondary'], COLORS['primary'], COLORS['accent']]

# R2 comparison - labels ABOVE
r2_vals = [lr_r2, rf_r2, gb_r2]
bars1 = axes[0].bar(models, r2_vals, color=colors, edgecolor='white', linewidth=2)
axes[0].set_ylabel('R² Score', fontsize=11)
axes[0].set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1.15)
axes[0].tick_params(axis='x', rotation=15)

for bar, val in zip(bars1, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

# RMSE comparison - labels ABOVE
rmse_vals = [lr_rmse, rf_rmse, gb_rmse]
bars2 = axes[1].bar(models, rmse_vals, color=colors, edgecolor='white', linewidth=2)
axes[1].set_ylabel('RMSE (Million Tomans)', fontsize=11)
axes[1].set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
max_rmse = max(rmse_vals)
axes[1].set_ylim(0, max_rmse * 1.25)
axes[1].tick_params(axis='x', rotation=15)

for bar, val in zip(bars2, rmse_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_rmse * 0.02,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

# MAE comparison - labels ABOVE
mae_vals = [lr_mae, rf_mae, gb_mae]
bars3 = axes[2].bar(models, mae_vals, color=colors, edgecolor='white', linewidth=2)
axes[2].set_ylabel('MAE (Million Tomans)', fontsize=11)
axes[2].set_title('MAE (Lower is Better)', fontsize=12, fontweight='bold')
max_mae = max(mae_vals)
axes[2].set_ylim(0, max_mae * 1.25)
axes[2].tick_params(axis='x', rotation=15)

for bar, val in zip(bars3, mae_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_mae * 0.02,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest R²: {max(r2_vals):.4f} | Best RMSE: {min(rmse_vals):.2f}M")

# %%
# Actual vs Predicted scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

predictions = [
    (y_pred_lr, 'Linear Regression', COLORS['secondary']),
    (y_pred_rf, 'Random Forest', COLORS['primary']),
    (y_pred_gb, 'Gradient Boosting', COLORS['accent'])
]

for ax, (y_pred, name, color) in zip(axes, predictions):
    # Sample for visualization
    sample_idx = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    
    ax.scatter(y_test.iloc[sample_idx], y_pred[sample_idx], 
               alpha=0.3, s=10, c=color)
    
    # Perfect prediction line
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price/sqm (M Tomans)')
    ax.set_ylabel('Predicted Price/sqm (M Tomans)')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.3f}')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)

plt.suptitle('Actual vs Predicted Price per Square Meter', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Value Classification
#
# Categorize listings as over-valued, normal, or under-valued based on prediction residuals.

# %%
print("=" * 60)
print("VALUE CLASSIFICATION")
print("=" * 60)

# Use best model (based on results) for classification
# Select the best performing model
if rf_r2 >= gb_r2 and rf_r2 >= lr_r2:
    best_model = rf_model
    best_name = 'Random Forest'
    y_pred_best = y_pred_rf
elif gb_r2 >= rf_r2 and gb_r2 >= lr_r2:
    best_model = gb_model
    best_name = 'Gradient Boosting'
    y_pred_best = y_pred_gb
else:
    best_model = lr_model
    best_name = 'Linear Regression'
    y_pred_best = y_pred_lr

print(f"\nUsing {best_name} for value classification")

# Calculate residuals
residuals = y_test.values - y_pred_best
residual_pct = (residuals / y_pred_best) * 100

print(f"\nResidual Statistics:")
print(f"  Mean residual: {residuals.mean():.2f}M Tomans")
print(f"  Std residual:  {residuals.std():.2f}M Tomans")
print(f"  Mean % error:  {residual_pct.mean():.1f}%")

# %%
# Classification function
def classify_value(actual, predicted, threshold=15):
    """
    Classify listing value based on prediction residual.
    
    - Over-valued: Actual price > predicted + threshold%
    - Under-valued: Actual price < predicted - threshold%
    - Normal: Within threshold%
    """
    residual_pct = ((actual - predicted) / predicted) * 100
    
    if residual_pct > threshold:
        return 'Over-valued'
    elif residual_pct < -threshold:
        return 'Under-valued'
    else:
        return 'Normal'

# Apply classification
THRESHOLD = 15  # 15% threshold

value_categories = [classify_value(a, p, THRESHOLD) 
                   for a, p in zip(y_test.values, y_pred_best)]

# Count distribution
value_counts = pd.Series(value_categories).value_counts()

print(f"\n Value Classification (±{THRESHOLD}% threshold):")
print("-" * 50)
for cat, count in value_counts.items():
    pct = count / len(value_categories) * 100
    bar = '' * int(pct / 2)
    print(f"  {cat:<15}: {count:>8,} ({pct:>5.1f}%) {bar}")

# %%
# Visualize value classification
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Pie chart of categories
colors_pie = {'Over-valued': COLORS['accent'], 'Normal': COLORS['neutral'], 'Under-valued': COLORS['primary']}
wedges, _, autotexts = axes[0].pie(
    value_counts.values, 
    labels=None,
    autopct='%1.1f%%',
    colors=[colors_pie[c] for c in value_counts.index],
    startangle=90,
    explode=[0.02] * len(value_counts)
)
axes[0].legend(wedges, value_counts.index, title='Category', loc='center left', bbox_to_anchor=(1, 0.5))
axes[0].set_title(f'Value Classification\n(±{THRESHOLD}% threshold)')

# 2. Residual distribution
axes[1].hist(residual_pct, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
axes[1].axvline(x=THRESHOLD, color=COLORS['accent'], linestyle='--', lw=2, label=f'+{THRESHOLD}% (Over-valued)')
axes[1].axvline(x=-THRESHOLD, color=COLORS['primary'], linestyle='--', lw=2, label=f'-{THRESHOLD}% (Under-valued)')
axes[1].axvline(x=0, color='black', linestyle='-', lw=1)
axes[1].set_xlabel('Residual %')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Prediction Residuals')
axes[1].legend()
axes[1].set_xlim(-100, 100)

# 3. Box plot by category
test_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best,
    'Category': value_categories
})

cat_order = ['Under-valued', 'Normal', 'Over-valued']
box_data = [test_df[test_df['Category'] == cat]['Actual'] for cat in cat_order]
bp = axes[2].boxplot(box_data, labels=cat_order, patch_artist=True, showfliers=False)
for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(colors_pie[cat])
axes[2].set_ylabel('Actual Price/sqm (M Tomans)')
axes[2].set_title('Price Distribution by Value Category')

plt.suptitle('Listing Value Classification Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_value_classification.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Show sample under-valued and over-valued listings
print("\n" + "=" * 60)
print("SAMPLE LISTINGS BY VALUE CATEGORY")
print("=" * 60)

# Add predictions to test data
test_indices = y_test.index
sample_df = df_model.loc[test_indices].copy()
sample_df['predicted_price_sqm'] = y_pred_best
sample_df['value_category'] = value_categories
sample_df['residual_pct'] = residual_pct

# Show under-valued (opportunities)
print("\n TOP 5 UNDER-VALUED LISTINGS (Potential Opportunities):")
under_valued = sample_df[sample_df['value_category'] == 'Under-valued'].nsmallest(5, 'residual_pct')
for i, (_, row) in enumerate(under_valued.iterrows(), 1):
    print(f"\n  {i}. {row.get('city_slug', 'N/A')} - {row.get('cat3_slug', 'N/A')}")
    print(f"     Size: {row['building_size']:.0f} sqm | Rooms: {row['rooms_numeric']:.0f}")
    print(f"     Actual: {row['target']:.1f}M | Predicted: {row['predicted_price_sqm']:.1f}M")
    print(f"      {abs(row['residual_pct']):.0f}% BELOW predicted (good deal!)")

# Show over-valued (poor value)
print("\n TOP 5 OVER-VALUED LISTINGS (Poor Value):")
over_valued = sample_df[sample_df['value_category'] == 'Over-valued'].nlargest(5, 'residual_pct')
for i, (_, row) in enumerate(over_valued.iterrows(), 1):
    print(f"\n  {i}. {row.get('city_slug', 'N/A')} - {row.get('cat3_slug', 'N/A')}")
    print(f"     Size: {row['building_size']:.0f} sqm | Rooms: {row['rooms_numeric']:.0f}")
    print(f"     Actual: {row['target']:.1f}M | Predicted: {row['predicted_price_sqm']:.1f}M")
    print(f"      {row['residual_pct']:.0f}% ABOVE predicted (overpriced!)")

# %% [markdown]
# ## 11. Model Limitations Analysis
#

# %%
print("=" * 60)
print("MODEL LIMITATIONS ANALYSIS")
print("=" * 60)

# Add error metrics to sample_df
sample_df['abs_error'] = np.abs(sample_df['target'] - sample_df['predicted_price_sqm'])
sample_df['abs_pct_error'] = np.abs(sample_df['residual_pct'])

# 1. Performance by price segment
print("\n 1. PERFORMANCE BY PRICE SEGMENT:")
print("-" * 50)

price_bins = [0, 20, 40, 60, 100, 200, 500]
price_labels = ['0-20M', '20-40M', '40-60M', '60-100M', '100-200M', '200M+']
sample_df['price_segment'] = pd.cut(sample_df['target'], bins=price_bins, labels=price_labels)

segment_perf = sample_df.groupby('price_segment').agg({
    'abs_error': 'mean',
    'abs_pct_error': 'mean',
    'target': 'count'
}).rename(columns={'target': 'count'})

for seg, row in segment_perf.iterrows():
    status = '' if row['abs_pct_error'] < 20 else '' if row['abs_pct_error'] < 30 else ''
    print(f"  {seg:<12}: MAE={row['abs_error']:>5.1f}M | MAPE={row['abs_pct_error']:>5.1f}% | n={int(row['count']):,} {status}")

# %%
# 2. Performance by city size
print("\n 2. PERFORMANCE BY CITY SIZE:")
print("-" * 50)

city_counts = df_model['city_slug'].value_counts()
sample_df['city_size'] = sample_df['city_slug'].map(
    lambda x: 'Large (1000+)' if city_counts.get(x, 0) >= 1000 
    else 'Medium (100-1000)' if city_counts.get(x, 0) >= 100 
    else 'Small (<100)'
)

city_perf = sample_df.groupby('city_size').agg({
    'abs_error': 'mean',
    'abs_pct_error': 'mean',
    'target': 'count'
}).rename(columns={'target': 'count'})

for size in ['Large (1000+)', 'Medium (100-1000)', 'Small (<100)']:
    if size in city_perf.index:
        row = city_perf.loc[size]
        status = '' if row['abs_pct_error'] < 20 else '' if row['abs_pct_error'] < 30 else ''
        print(f"  {size:<20}: MAE={row['abs_error']:>5.1f}M | MAPE={row['abs_pct_error']:>5.1f}% | n={int(row['count']):,} {status}")

# %%
# 3. Performance by property type
print("\n 3. PERFORMANCE BY PROPERTY TYPE:")
print("-" * 50)

if 'cat3_slug' in sample_df.columns:
    cat_perf = sample_df.groupby('cat3_slug').agg({
        'abs_error': 'mean',
        'abs_pct_error': 'mean',
        'target': 'count'
    }).rename(columns={'target': 'count'})
    cat_perf = cat_perf[cat_perf['count'] >= 100].sort_values('count', ascending=False)
    
    for cat, row in cat_perf.iterrows():
        status = '' if row['abs_pct_error'] < 20 else '' if row['abs_pct_error'] < 30 else ''
        print(f"  {cat:<35}: MAPE={row['abs_pct_error']:>5.1f}% | n={int(row['count']):,} {status}")

# %%
# Visualize limitations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Error by price segment
segment_perf_plot = segment_perf.reset_index()
colors = [COLORS['primary'] if e < 20 else COLORS['orange'] if e < 30 else COLORS['accent'] 
          for e in segment_perf_plot['abs_pct_error']]
bars = axes[0].bar(segment_perf_plot['price_segment'], segment_perf_plot['abs_pct_error'], 
                   color=colors, edgecolor='white')
axes[0].axhline(y=20, color='green', linestyle='--', lw=2, label='Good (<20%)')
axes[0].axhline(y=30, color='orange', linestyle='--', lw=2, label='Acceptable (<30%)')
axes[0].set_xlabel('Price Segment (M Tomans/sqm)')
axes[0].set_ylabel('Mean Absolute % Error')
axes[0].set_title('Model Error by Price Segment')
axes[0].legend()
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. Residuals vs Predicted
sample_idx = np.random.choice(len(sample_df), size=min(5000, len(sample_df)), replace=False)
axes[1].scatter(sample_df.iloc[sample_idx]['predicted_price_sqm'], 
                sample_df.iloc[sample_idx]['residual_pct'],
                alpha=0.3, s=10, c=COLORS['secondary'])
axes[1].axhline(y=0, color='black', linestyle='-', lw=1)
axes[1].axhline(y=THRESHOLD, color=COLORS['accent'], linestyle='--', lw=2, alpha=0.7)
axes[1].axhline(y=-THRESHOLD, color=COLORS['primary'], linestyle='--', lw=2, alpha=0.7)
axes[1].set_xlabel('Predicted Price/sqm (M Tomans)')
axes[1].set_ylabel('Residual %')
axes[1].set_title('Residuals vs Predicted Price\n(Heteroscedasticity Check)')
axes[1].set_ylim(-100, 100)

plt.suptitle('Model Limitations Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_model_limitations.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Summary of limitations
print("\n" + "=" * 60)
print("MODEL LIMITATIONS SUMMARY")
print("=" * 60)

print("""
 WHERE THE MODEL WORKS WELL:

- Mid-range properties (20-60M Tomans/sqm): Best accuracy
- Large cities with many listings: Stable predictions
- Apartments and standard houses: Most training data
- Properties with complete feature information

 WHERE THE MODEL STRUGGLES:

- Luxury segment (>100M Tomans/sqm): High variance, few samples
- Small cities (<100 listings): Limited training data
- Commercial properties: Different pricing dynamics
- Properties with many missing features

 SUGGESTED IMPROVEMENTS:

- Separate models for different price segments
- Include neighborhood-level features
- Add temporal features (listing date trends)
- Collect more data for luxury segment
""")

# %% [markdown]
# ## 12. Export Results

# %%
print("=" * 60)
print("EXPORTING RESULTS")
print("=" * 60)

# 1. Save best model
model_path = MODELS_PATH / 'price_prediction_model.joblib'
joblib.dump(best_model, model_path)
print(f"\n Model saved: {model_path.relative_to(PROJECT_ROOT)}")

# 2. Save scaler
scaler_path = MODELS_PATH / 'feature_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f" Scaler saved: {scaler_path.relative_to(PROJECT_ROOT)}")

# 3. Save predictions for test set
predictions_df = sample_df[['city_slug', 'cat3_slug', 'building_size', 'rooms_numeric',
                            'target', 'predicted_price_sqm', 'value_category', 'residual_pct']].copy()
predictions_df.columns = ['city', 'property_type', 'size_sqm', 'rooms', 
                          'actual_price_per_sqm_M', 'predicted_price_per_sqm_M', 
                          'value_category', 'residual_pct']

pred_path = DATA_PROCESSED / 'price_predictions.csv'
predictions_df.to_csv(pred_path, index=False)
predictions_df.to_parquet(DATA_PROCESSED / 'price_predictions.parquet', index=False, compression='zstd')
print(f" Predictions saved: {pred_path.relative_to(PROJECT_ROOT)} ({len(predictions_df):,} rows)")
print(f" Predictions saved: {(DATA_PROCESSED / 'price_predictions.parquet').relative_to(PROJECT_ROOT)} ({len(predictions_df):,} rows)")

# 4. Save model comparison results
results_path = DATA_PROCESSED / 'model_comparison.csv'
results.to_csv(results_path, index=False)
print(f" Model comparison saved: {results_path.relative_to(PROJECT_ROOT)}")

# 5. Save feature importance
importance_path = DATA_PROCESSED / 'feature_importance.csv'
importance_df.to_csv(importance_path, index=False)
print(f" Feature importance saved: {importance_path.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 13. Conclusion

# %%
print("=" * 70)
print("PHASE 5 SUMMARY: PRICE PREDICTION")
print("=" * 70)

print(f"""
 DATA SUMMARY
{''*70}
- Training samples: {len(X_train):,}
- Test samples: {len(X_test):,}
- Features used: {len(feature_cols)}
- Target: Price per square meter (Million Tomans)

 MODEL PERFORMANCE
{''*70}
- Best Model: {best_name}
- R² Score: {rf_r2:.4f} ({rf_r2*100:.1f}% variance explained)
- RMSE: {rf_rmse:.2f}M Tomans/sqm
- MAE: {rf_mae:.2f}M Tomans/sqm

 TOP PREDICTORS (from Random Forest)
{''*70}
- 1. city_price_level: Location is the most important factor
- 2. property_price_level: Property type matters significantly  
- 3. building_size: Size affects total price but less price/sqm

 VALUE CLASSIFICATION (±{THRESHOLD}% threshold)
{''*70}
""")

for cat in ['Under-valued', 'Normal', 'Over-valued']:
    if cat in value_counts.index:
        count = value_counts[cat]
        pct = count / len(value_categories) * 100
        print(f"- {cat}: {count:,} listings ({pct:.1f}%)")

print(f"""
 KEY LIMITATIONS
{''*70}
- Luxury properties (>100M/sqm): Higher prediction error
- Small cities: Limited training data affects accuracy
- Missing features: Some listings lack important information

 OUTPUT FILES
{''*70}
- price_prediction_model.joblib: Trained model
- price_predictions.csv: All predictions with value categories
- model_comparison.csv: R², RMSE, MAE for all models
- feature_importance.csv: Feature importance rankings
- Figures: 05_*.png in reports/figures/

{'='*70}
Phase 5 Complete 
Next: Phase 6 - Text Classification
{'='*70}
""")
