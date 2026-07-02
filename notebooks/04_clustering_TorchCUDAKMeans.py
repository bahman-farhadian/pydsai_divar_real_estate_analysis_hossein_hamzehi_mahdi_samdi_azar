# %% [markdown]
# # Phase 4: Clustering Analysis (Torch CUDA K-Means)
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records)
#
# ---
#
# ## Objective
#
# Segment the real estate market into distinct, interpretable categories that provide business value beyond simple statistics.
#
#
# ## Analysis Scope
#
# 1. **Data Loading & Feature Selection** - Select relevant features for clustering
# 2. **Preprocessing** - Handle missing values, outliers, normalization
# 3. **Optimal K Selection** - Elbow and Silhouette methods
# 4. **Approach 1: Cluster Original Features** - K-Means with PCA visualization
# 5. **Approach 2: Cluster PCA Components** - Compare with Approach 1
# 6. **CUDA Projection Visualization** - Multiple runs to verify broad visual patterns
# 7. **Cluster Interpretation** - Assign business-meaningful names
# 8. **Results & Export** - Save cluster assignments and profiles
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

# Clustering and preprocessing
from sklearn.preprocessing import StandardScaler
import torch

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
    'orange': '#e67e22',
    'teal': '#1abc9c'
}

# Cluster colors (for consistent visualization) - extended to support up to 10 clusters
CLUSTER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', 
                  '#1abc9c', '#f39c12', '#34495e', '#e91e63', '#00bcd4']

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


class TorchKMeans:
    """Small K-Means estimator with a scikit-learn-like interface and CUDA support."""

    def __init__(self, n_clusters, random_state=42, n_init=3, batch_size=262144, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _assign_labels(self, x, centers):
        labels = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        inertia = 0.0
        for start in range(0, x.shape[0], self.batch_size):
            end = min(start + self.batch_size, x.shape[0])
            batch = x[start:end]
            distances = (
                batch.pow(2).sum(dim=1, keepdim=True)
                + centers.pow(2).sum(dim=1).unsqueeze(0)
                - 2 * batch @ centers.T
            ).clamp_min_(0)
            min_distances, batch_labels = distances.min(dim=1)
            labels[start:end] = batch_labels
            inertia += min_distances.sum().item()
        return labels, inertia

    def _single_fit(self, x, seed):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        initial_idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:self.n_clusters]
        centers = x[initial_idx].clone()

        for iteration in range(self.max_iter):
            labels, inertia = self._assign_labels(x, centers)
            raw_counts = torch.bincount(labels, minlength=self.n_clusters)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels[:, None].expand(-1, x.shape[1]), x)
            counts = raw_counts.clamp_min(1).to(x.dtype)
            new_centers = new_centers / counts[:, None]

            empty_clusters = torch.where(raw_counts == 0)[0]
            for cluster_id in empty_clusters.tolist():
                if not (labels == cluster_id).any():
                    replacement = torch.randint(0, x.shape[0], (1,), generator=generator, device=x.device)
                    new_centers[cluster_id] = x[replacement.item()]

            shift = torch.norm(new_centers - centers, dim=1).max().item()
            centers = new_centers
            if shift < self.tol:
                break

        labels, inertia = self._assign_labels(x, centers)
        return labels, centers, inertia, iteration + 1

    def fit(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        x = torch.from_numpy(X_np).to(self.device)
        print(f"Torch K-Means device: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        best = None
        for init_id in range(self.n_init):
            labels, centers, inertia, n_iter = self._single_fit(x, self.random_state + init_id)
            print(f"  init {init_id + 1}/{self.n_init}: inertia={inertia:,.0f}, iterations={n_iter}")
            if best is None or inertia < best[2]:
                best = (labels, centers, inertia, n_iter)

        labels, centers, inertia, n_iter = best
        self.labels_ = labels.detach().cpu().numpy()
        self.cluster_centers_ = centers.detach().cpu().numpy()
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("TorchKMeans must be fitted before predict().")
        X_np = np.asarray(X, dtype=np.float32)
        x = torch.from_numpy(X_np).to(self.device)
        centers = torch.from_numpy(self.cluster_centers_.astype(np.float32)).to(self.device)
        labels, _ = self._assign_labels(x, centers)
        return labels.detach().cpu().numpy()


def get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torch_pca_transform(X, n_components):
    device = get_torch_device()
    x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :n_components]
    transformed = centered @ components
    explained_ratio = eigenvalues / eigenvalues.sum()
    return (
        transformed.detach().cpu().numpy(),
        components.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        explained_ratio.detach().cpu().numpy(),
    )


def torch_pca_apply(X, mean, components):
    device = get_torch_device()
    x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    components_tensor = torch.as_tensor(components, dtype=torch.float32, device=device)
    return ((x - mean_tensor) @ components_tensor).detach().cpu().numpy()


def torch_silhouette_score(X, labels, sample_size=12000, batch_size=1024, random_state=42):
    device = get_torch_device()
    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, len(X))
    sample_idx = rng.choice(len(X), sample_size, replace=False)

    x = torch.as_tensor(np.asarray(X[sample_idx], dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels[sample_idx], dtype=np.int64), device=device)
    unique_labels = torch.unique(y, sorted=True)
    cluster_masks = [y == cluster_id for cluster_id in unique_labels]
    cluster_counts = torch.stack([mask.sum() for mask in cluster_masks]).to(torch.float32)

    scores = []
    for start in range(0, sample_size, batch_size):
        end = min(start + batch_size, sample_size)
        batch = x[start:end]
        distances = torch.cdist(batch, x)
        batch_labels = y[start:end]

        mean_distances = []
        for mask, count in zip(cluster_masks, cluster_counts):
            total_distance = distances[:, mask].sum(dim=1)
            same_cluster = batch_labels == unique_labels[len(mean_distances)]
            denominator = torch.where(same_cluster, count - 1, count).clamp_min(1)
            mean_distances.append(total_distance / denominator)

        mean_distances = torch.stack(mean_distances, dim=1)
        own_cluster_index = torch.searchsorted(unique_labels, batch_labels)
        a = mean_distances.gather(1, own_cluster_index[:, None]).squeeze(1)
        other_distances = mean_distances.clone()
        other_distances.scatter_(1, own_cluster_index[:, None], float('inf'))
        b = other_distances.min(dim=1).values
        score = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
        scores.append(score)

    return torch.cat(scores).mean().item()

# %% [markdown]
# ## 2. Project Structure and Data Loading

# %%
# Define project paths
import os

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

# Create directories
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

print("Project root: .")
print(f"Figures path: {FIGURES_PATH.relative_to(PROJECT_ROOT)}")
print(f"Figures path exists: {FIGURES_PATH.exists()}")

# %%
# Load the enhanced dataset from Phase 2
DATA_FILE = DATA_PROCESSED / 'cleaned_data_with_features.csv'

print(f"Loading data from: {DATA_FILE.relative_to(PROJECT_ROOT)}")
df_full = read_csv_fast(DATA_FILE)
print(f"Full dataset: {len(df_full):,} rows")

# Filter to sale listings only (clustering for sale market)
df = df_full[df_full['listing_type'] == 'sell'].copy()
print(f"Sale listings: {len(df):,} rows")

# %% [markdown]
# ## 3. Feature Selection and Engineering
#
# We select features that are meaningful for market segmentation:
# - **Price per sqm**: Core value metric
# - **Building size**: Property size
# - **Room count**: Property configuration
# - **Amenities**: Elevator, parking, storage
# - **City**: Location factor (target encoded by median price)

# %%
# Ensure numeric types
df['price_value'] = pd.to_numeric(df['price_value'], errors='coerce')
df['building_size'] = pd.to_numeric(df['building_size'], errors='coerce')
df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')

# Convert rooms_count to numeric
room_mapping = {
    'بدون اتاق': 0,
    'یک': 1,
    'دو': 2,
    'سه': 3,
    'چهار': 4,
    'پنج یا بیشتر': 5
}
df['rooms_numeric'] = df['rooms_count'].map(room_mapping)

# Convert amenities to binary (handle various formats)
for col in ['has_elevator', 'has_parking', 'has_warehouse']:
    if col in df.columns:
        # Convert to string first, then check for True/true/1
        df[col + '_binary'] = df[col].astype(str).str.lower().isin(['true', '1', '1.0']).astype(int)

print("Feature engineering complete")
print(f"\nFeature availability:")
print(f"  price_per_sqm: {df['price_per_sqm'].notna().sum():,} valid ({df['price_per_sqm'].notna().mean()*100:.1f}%)")
print(f"  building_size: {df['building_size'].notna().sum():,} valid ({df['building_size'].notna().mean()*100:.1f}%)")
print(f"  rooms_numeric: {df['rooms_numeric'].notna().sum():,} valid ({df['rooms_numeric'].notna().mean()*100:.1f}%)")

# %% [markdown]
# ### 3.1 City Target Encoding
#
# We encode city by its median price per sqm. This creates a single numeric feature that captures the "price level" of each location.

# %%
# Calculate median price per sqm for each city
city_price_median = df.groupby('city_slug')['price_per_sqm'].median()

# Target encode: replace city name with its median price
df['city_price_level'] = df['city_slug'].map(city_price_median)

# Show encoding for top cities
print("City Target Encoding (by median price/sqm):")
print("=" * 50)
top_cities = city_price_median.sort_values(ascending=False).head(10)
for city, price in top_cities.items():
    if pd.notna(price):
        city_name = city.replace('-', ' ').title()
        print(f"{city_name:<25} -> {price/1e6:.1f}M Tomans/sqm")

print(f"\nCity price level: {df['city_price_level'].notna().sum():,} valid")

# %% [markdown]
# ## 4. Data Preprocessing
#
# Following instructor guidance: normalize features with different scales.

# %% [markdown]
# ### 4.1 Select Features and Handle Missing Values

# %%
# Define clustering features (only features with good availability)
clustering_features = [
    'price_per_sqm',
    'building_size',
    'rooms_numeric',
    'has_elevator_binary',
    'has_parking_binary',
    'has_warehouse_binary',
    'city_price_level'
]

# Check availability
print("Feature Availability:")
print("=" * 50)
for feat in clustering_features:
    if feat in df.columns:
        valid = df[feat].notna().sum()
        pct = valid / len(df) * 100
        print(f"{feat:<25} {valid:>10,} ({pct:>5.1f}%)")
    else:
        print(f"{feat:<25} MISSING COLUMN")

# %%
# Create clustering dataset
# Start with essential features that have high availability
essential_features = ['price_per_sqm', 'building_size', 'city_price_level']

# Filter: must have essential features
df_cluster = df[df['price_per_sqm'].notna() & df['building_size'].notna()].copy()
print(f"After requiring price_per_sqm and building_size: {len(df_cluster):,} rows")

# Apply reasonable bounds (same as Phase 3)
df_cluster = df_cluster[
    (df_cluster['price_per_sqm'] >= 5_000_000) &
    (df_cluster['price_per_sqm'] <= 500_000_000) &
    (df_cluster['building_size'] >= 20) &
    (df_cluster['building_size'] <= 1000)
].copy()
print(f"After outlier removal: {len(df_cluster):,} rows")

# Impute missing values for other features with median
df_cluster['rooms_numeric'] = df_cluster['rooms_numeric'].fillna(df_cluster['rooms_numeric'].median())
df_cluster['city_price_level'] = df_cluster['city_price_level'].fillna(df_cluster['city_price_level'].median())

# Binary features: fill missing with 0 (assume no amenity if not specified)
for col in ['has_elevator_binary', 'has_parking_binary', 'has_warehouse_binary']:
    if col in df_cluster.columns:
        df_cluster[col] = df_cluster[col].fillna(0).astype(int)

print(f"\nFinal clustering dataset: {len(df_cluster):,} rows")

# %% [markdown]
# ### 4.2 Log Transformation for Skewed Variables

# %%
# Check skewness before transformation
print("Skewness before log transformation:")
print(f"  price_per_sqm: {df_cluster['price_per_sqm'].skew():.2f}")
print(f"  building_size: {df_cluster['building_size'].skew():.2f}")
print(f"  city_price_level: {df_cluster['city_price_level'].skew():.2f}")

# Apply log transformation to highly skewed variables
df_cluster['log_price_per_sqm'] = np.log1p(df_cluster['price_per_sqm'])
df_cluster['log_building_size'] = np.log1p(df_cluster['building_size'])
df_cluster['log_city_price_level'] = np.log1p(df_cluster['city_price_level'])

print("\nSkewness after log transformation:")
print(f"  log_price_per_sqm: {df_cluster['log_price_per_sqm'].skew():.2f}")
print(f"  log_building_size: {df_cluster['log_building_size'].skew():.2f}")
print(f"  log_city_price_level: {df_cluster['log_city_price_level'].skew():.2f}")

# %% [markdown]
# ### 4.3 Feature Normalization
#
# As per instructor guidance: normalize features with different scales.

# %%
# Final features for clustering (using log-transformed where applicable)
final_features = [
    'log_price_per_sqm',
    'log_building_size',
    'rooms_numeric',
    'has_elevator_binary',
    'has_parking_binary',
    'has_warehouse_binary',
    'log_city_price_level'
]

# Verify no missing values
print("Final feature check (should all be 0 missing):")
for feat in final_features:
    missing = df_cluster[feat].isna().sum()
    print(f"  {feat}: {missing} missing")

# Extract feature matrix
X = df_cluster[final_features].values
print(f"\nFeature matrix shape: {X.shape}")

# %%
# Normalize using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Normalization Complete")
print("=" * 50)
print(f"Shape: {X_scaled.shape}")
print(f"\nScaled feature statistics:")
print(f"  Mean (should be ~0): {X_scaled.mean(axis=0).round(2)}")
print(f"  Std (should be ~1): {X_scaled.std(axis=0).round(2)}")

# %% [markdown]
# ## 5. Optimal Number of Clusters (K)
#
# We use two methods to determine the optimal K:
# 1. **Elbow Method**: Look for the "elbow" in the inertia plot
# 2. **Silhouette Method**: Higher score = better cluster separation

# %%
# K Selection with Torch CUDA K-Means
# PyTorch executes the distance and centroid update steps on CUDA when a compatible GPU is available.

k_range = range(2, 11)
inertias = []
silhouette_scores = []

# Use GPU sampling for silhouette calculation (O(n^2) is too expensive on full data)
SAMPLE_SIZE = 12000
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), min(SAMPLE_SIZE, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

print("Evaluating K values with Torch CUDA K-Means...")
print(f"Using {SAMPLE_SIZE:,} samples for GPU silhouette approximation")
print("=" * 50)

for k in k_range:
    # Torch CUDA K-Means
    kmeans = TorchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=262144, max_iter=75)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    
    # Silhouette on a GPU sample only (full silhouette is O(n^2))
    sample_labels = kmeans.predict(X_sample)
    sil_score = torch_silhouette_score(X_sample, sample_labels, sample_size=len(X_sample), random_state=42)
    silhouette_scores.append(sil_score)
    
    print(f"  K={k}: Inertia={kmeans.inertia_:,.0f}, Silhouette={sil_score:.4f}")

print("\n[OK] K evaluation complete!")

# %%
# Visualize Elbow and Silhouette methods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Elbow Method
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
axes[0].set_title('Elbow Method for Optimal K')
axes[0].set_xticks(list(k_range))
axes[0].grid(True, alpha=0.3)

# Right: Silhouette Method
colors = [COLORS['accent'] if s == max(silhouette_scores) else COLORS['secondary'] for s in silhouette_scores]
bars = axes[1].bar(list(k_range), silhouette_scores, color=colors, edgecolor='white', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by K (Higher = Better)')
axes[1].set_xticks(list(k_range))
axes[1].grid(True, alpha=0.3, axis='y')

# Labels ABOVE with headroom
max_score = max(silhouette_scores)
axes[1].set_ylim(0, max_score * 1.25)
for bar, score in zip(bars, silhouette_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_score * 0.02, 
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_optimal_k_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# Determine optimal K
best_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nBest K by Silhouette Score: {best_k_silhouette}")
print(f"Silhouette Score: {max(silhouette_scores):.4f}")

# %%
# Select optimal K based on analysis
optimal_k = best_k_silhouette
print(f"Selected K = {optimal_k} for clustering")
print(f"\nThis creates {optimal_k} distinct market segments.")

# %% [markdown]
# ## 6. Approach 1: Cluster Original Features, Visualize with PCA
#
# First approach: Run K-Means on the normalized original features, then use PCA for 2D visualization.

# %%
# Fit final Torch CUDA K-Means model
print(f"Fitting Torch CUDA K-Means with K={optimal_k}...")

kmeans_original = TorchKMeans(n_clusters=optimal_k, random_state=42, n_init=5, batch_size=262144, max_iter=100)
cluster_labels_original = kmeans_original.fit_predict(X_scaled)

# Add cluster labels to dataframe
df_cluster['cluster_original'] = cluster_labels_original

# Check cluster sizes
print(f"\nCluster Sizes (Approach 1 - Torch CUDA K-Means):")
print("=" * 40)
cluster_counts = pd.Series(cluster_labels_original).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = count / len(cluster_labels_original) * 100
    print(f"  Cluster {cluster_id}: {count:>10,} ({pct:>5.1f}%)")

# %%
# CUDA PCA for visualization (reduce to 2D)
X_pca_2d, pca_components_2d, pca_mean_2d, pca_explained_2d = torch_pca_transform(X_scaled, 2)

print(f"PCA Explained Variance:")
print(f"  PC1: {pca_explained_2d[0]*100:.1f}%")
print(f"  PC2: {pca_explained_2d[1]*100:.1f}%")
print(f"  Total: {sum(pca_explained_2d[:2])*100:.1f}%")

# %%
# Visualize clusters with PCA
fig, ax = plt.subplots(figsize=(12, 8))

# Sample for visualization (too many points obscure patterns)
np.random.seed(42)
sample_size = min(20000, len(X_pca_2d))
sample_idx = np.random.choice(len(X_pca_2d), sample_size, replace=False)

for cluster_id in range(optimal_k):
    mask = (cluster_labels_original[sample_idx] == cluster_id)
    ax.scatter(
        X_pca_2d[sample_idx][mask, 0],
        X_pca_2d[sample_idx][mask, 1],
        c=CLUSTER_COLORS[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.5,
        s=20
    )

# Plot cluster centers
centers_pca = torch_pca_apply(kmeans_original.cluster_centers_, pca_mean_2d, pca_components_2d)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, 
           edgecolors='white', linewidths=2, label='Centroids')

ax.set_xlabel(f'PC1 ({pca_explained_2d[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca_explained_2d[1]*100:.1f}% variance)')
ax.set_title(f'Approach 1: K-Means Clusters on Original Features (K={optimal_k})')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_clusters_pca_approach1.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nNote: Showing {sample_size:,} sampled points for clarity.")

# %% [markdown]
# ## 7. Approach 2: Cluster on PCA Components
#
# As per instructor guidance: "Do dimensionality reduction with PCA first, then cluster on principal components."

# %%
# CUDA PCA with enough components to capture 90% variance
max_components = min(X_scaled.shape[1], 7)
X_pca_all, pca_components_all, pca_mean_all, pca_explained_all = torch_pca_transform(X_scaled, max_components)
cumsum = np.cumsum(pca_explained_all)

n_components_90 = int(np.searchsorted(cumsum, 0.90) + 1)
print(f"Components needed for 90% variance: {n_components_90}")

X_pca = X_pca_all[:, :n_components_90]

print(f"\nPCA for Clustering:")
print(f"  Original features: {X_scaled.shape[1]}")
print(f"  PCA components: {n_components_90}")
print(f"  Variance explained: {sum(pca_explained_all[:n_components_90])*100:.1f}%")

# %%
# Cluster on PCA components (Torch CUDA K-Means)
print(f"\nFitting Torch CUDA K-Means on PCA components (K={optimal_k})...")

kmeans_pca = TorchKMeans(n_clusters=optimal_k, random_state=42, n_init=5, batch_size=262144, max_iter=100)
cluster_labels_pca = kmeans_pca.fit_predict(X_pca)

df_cluster['cluster_pca'] = cluster_labels_pca

# Check cluster sizes
print(f"\nCluster Sizes (Approach 2 - PCA + Torch CUDA K-Means):")
print("=" * 40)
cluster_counts_pca = pd.Series(cluster_labels_pca).value_counts().sort_index()
for cluster_id, count in cluster_counts_pca.items():
    pct = count / len(cluster_labels_pca) * 100
    print(f"  Cluster {cluster_id}: {count:>10,} ({pct:>5.1f}%)")

# %%
# Visualize Approach 2 clusters
fig, ax = plt.subplots(figsize=(12, 8))

# Use first 2 PCA components for visualization
for cluster_id in range(optimal_k):
    mask = (cluster_labels_pca[sample_idx] == cluster_id)
    ax.scatter(
        X_pca[sample_idx, 0][mask],
        X_pca[sample_idx, 1][mask],
        c=CLUSTER_COLORS[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.5,
        s=20
    )

# Plot centroids (only first 2 components)
ax.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1], 
           c='black', marker='X', s=200, edgecolors='white', linewidths=2, label='Centroids')

ax.set_xlabel(f'PC1 ({pca_explained_all[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca_explained_all[1]*100:.1f}% variance)')
ax.set_title(f'Approach 2: K-Means Clusters on PCA Components (K={optimal_k})')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_clusters_pca_approach2.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 7.1 Compare Approach 1 vs Approach 2

# %%
# Compare cluster assignments between approaches
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(cluster_labels_original, cluster_labels_pca)
nmi = normalized_mutual_info_score(cluster_labels_original, cluster_labels_pca)

print("Comparison: Approach 1 (Original) vs Approach 2 (PCA)")
print("=" * 55)
print(f"  Adjusted Rand Index: {ari:.4f}")
print(f"  Normalized Mutual Information: {nmi:.4f}")
print(f"\nInterpretation:")
print(f"  ARI = 1.0 means identical clustering")
print(f"  ARI = 0.0 means random clustering")
if ari > 0.7:
    print(f"\n  Result: Both approaches produce SIMILAR clusters.")
elif ari > 0.4:
    print(f"\n  Result: Approaches have MODERATE agreement.")
else:
    print(f"\n  Result: Approaches produce DIFFERENT clusters - worth exploring both.")

# %%
# Side-by-side comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Approach 1
for cluster_id in range(optimal_k):
    mask = (cluster_labels_original[sample_idx] == cluster_id)
    axes[0].scatter(
        X_pca_2d[sample_idx][mask, 0],
        X_pca_2d[sample_idx][mask, 1],
        c=CLUSTER_COLORS[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.4,
        s=15
    )
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('Approach 1: Cluster Original Features')
axes[0].legend(loc='upper right', fontsize=8)

# Approach 2
for cluster_id in range(optimal_k):
    mask = (cluster_labels_pca[sample_idx] == cluster_id)
    axes[1].scatter(
        X_pca[sample_idx, 0][mask],
        X_pca[sample_idx, 1][mask],
        c=CLUSTER_COLORS[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.4,
        s=15
    )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Approach 2: Cluster PCA Components')
axes[1].legend(loc='upper right', fontsize=8)

plt.suptitle(f'Comparison of Clustering Approaches (ARI = {ari:.3f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_approach_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. CUDA Projection Stability Visualization
#
# Stochastic low-dimensional visualizations can shift across random seeds.
#
# This CUDA report uses repeated GPU random projections to verify whether the same broad cluster patterns remain visible across independent projections.

# %%
# CUDA random projections on a sample
projection_sample_size = 128000
np.random.seed(42)
projection_idx = np.random.choice(len(X_scaled), min(projection_sample_size, len(X_scaled)), replace=False)

X_projection_sample = X_scaled[projection_idx]
labels_projection_sample = cluster_labels_original[projection_idx]

print(f"Running CUDA random projections on {len(X_projection_sample):,} samples...")

# %%
# Run projections with 3 different random states to verify visual stability
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

random_states = [42, 123, 456]
device = get_torch_device()
x_projection_tensor = torch.as_tensor(np.asarray(X_projection_sample, dtype=np.float32), device=device)

for idx, rs in enumerate(random_states):
    print(f"Running CUDA projection with random_state={rs}...")
    generator = torch.Generator(device=device)
    generator.manual_seed(rs)
    projection = torch.randn((x_projection_tensor.shape[1], 2), generator=generator, device=device)
    projection, _ = torch.linalg.qr(projection, mode='reduced')
    X_projection = (x_projection_tensor @ projection).detach().cpu().numpy()
    
    # Plot
    for cluster_id in range(optimal_k):
        mask = (labels_projection_sample == cluster_id)
        axes[idx].scatter(
            X_projection[mask, 0],
            X_projection[mask, 1],
            c=CLUSTER_COLORS[cluster_id],
            label=f'Cluster {cluster_id}',
            alpha=0.5,
            s=15
        )
    axes[idx].set_xlabel('Projection 1')
    axes[idx].set_ylabel('Projection 2')
    axes[idx].set_title(f'CUDA Projection (seed={rs})')
    if idx == 0:
        axes[idx].legend(loc='upper right', fontsize=8)

plt.suptitle('CUDA Random Projection Visualization: Multiple Runs to Verify Broad Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_projection_multiple_runs.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nNote: Random projections are for visualization only. Clusters are defined by Torch CUDA K-Means.")
print("Patterns that appear consistently across projections are broad, stable visual signals.")

# %% [markdown]
# ## 9. Cluster Interpretation
#
# We analyze each cluster's characteristics and assign business-meaningful names.

# %%
# Use Approach 1 clusters for interpretation (original features)
df_cluster['cluster'] = df_cluster['cluster_original']

# Calculate cluster profiles using original (non-transformed) values
profile_features = [
    'price_per_sqm',
    'building_size',
    'rooms_numeric',
    'has_elevator_binary',
    'has_parking_binary',
    'has_warehouse_binary',
    'city_price_level'
]

cluster_profiles = df_cluster.groupby('cluster')[profile_features].mean()

print("Cluster Profiles (Mean Values)")
print("=" * 80)
print(cluster_profiles.round(2).to_string())

# %%
# Detailed cluster statistics
print("\nDetailed Cluster Statistics")
print("=" * 80)

for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    n = len(cluster_data)
    pct = n / len(df_cluster) * 100
    
    print(f"\nCluster {cluster_id} ({n:,} listings, {pct:.1f}% of market)")
    print("-" * 60)
    print(f"  Price/sqm:    Median = {cluster_data['price_per_sqm'].median()/1e6:.1f}M Tomans")
    print(f"  Size:         Median = {cluster_data['building_size'].median():.0f} sqm")
    print(f"  Rooms:        Median = {cluster_data['rooms_numeric'].median():.0f}")
    print(f"  Has Elevator: {cluster_data['has_elevator_binary'].mean()*100:.0f}%")
    print(f"  Has Parking:  {cluster_data['has_parking_binary'].mean()*100:.0f}%")
    print(f"  Has Storage:  {cluster_data['has_warehouse_binary'].mean()*100:.0f}%")

# %%
# Visualize cluster profiles - labels ABOVE bars
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Price per sqm by cluster
cluster_price = df_cluster.groupby('cluster')['price_per_sqm'].median() / 1e6
bars = axes[0, 0].bar(range(optimal_k), cluster_price, color=CLUSTER_COLORS[:optimal_k], edgecolor='white', linewidth=2)
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Median Price/sqm (Million Tomans)')
axes[0, 0].set_title('Price Level by Cluster')
axes[0, 0].set_xticks(range(optimal_k))
max_val = cluster_price.max()
axes[0, 0].set_ylim(0, max_val * 1.25)
for i, v in enumerate(cluster_price):
    axes[0, 0].text(i, v + max_val * 0.02, f'{v:.0f}M', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# 2. Building size by cluster
cluster_size = df_cluster.groupby('cluster')['building_size'].median()
bars = axes[0, 1].bar(range(optimal_k), cluster_size, color=CLUSTER_COLORS[:optimal_k], edgecolor='white', linewidth=2)
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Median Size (sqm)')
axes[0, 1].set_title('Property Size by Cluster')
axes[0, 1].set_xticks(range(optimal_k))
max_val = cluster_size.max()
axes[0, 1].set_ylim(0, max_val * 1.25)
for i, v in enumerate(cluster_size):
    axes[0, 1].text(i, v + max_val * 0.02, f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# 3. Rooms by cluster
cluster_rooms = df_cluster.groupby('cluster')['rooms_numeric'].median()
bars = axes[1, 0].bar(range(optimal_k), cluster_rooms, color=CLUSTER_COLORS[:optimal_k], edgecolor='white', linewidth=2)
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Median Rooms')
axes[1, 0].set_title('Room Count by Cluster')
axes[1, 0].set_xticks(range(optimal_k))
max_val = cluster_rooms.max()
axes[1, 0].set_ylim(0, max_val * 1.25)
for i, v in enumerate(cluster_rooms):
    axes[1, 0].text(i, v + max_val * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# 4. Amenities by cluster
amenity_cols = ['has_elevator_binary', 'has_parking_binary', 'has_warehouse_binary']
amenity_labels = ['Elevator', 'Parking', 'Storage']
x = np.arange(optimal_k)
width = 0.25

for i, (col, label) in enumerate(zip(amenity_cols, amenity_labels)):
    values = df_cluster.groupby('cluster')[col].mean() * 100
    axes[1, 1].bar(x + i*width, values, width, label=label, color=CLUSTER_COLORS[i], edgecolor='white', linewidth=1)

axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('% with Amenity')
axes[1, 1].set_title('Amenities by Cluster')
axes[1, 1].set_xticks(x + width)
axes[1, 1].set_xticklabels([f'C{i}' for i in range(optimal_k)])
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 115)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 9.1 City Distribution per Cluster

# %%
# Top cities in each cluster
print("Top 5 Cities in Each Cluster")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    top_cities = cluster_data['city_slug'].value_counts().head(5)
    total = len(cluster_data)
    
    print(f"\nCluster {cluster_id}:")
    for city, count in top_cities.items():
        city_name = city.replace('-', ' ').title()
        pct = count / total * 100
        print(f"  {city_name:<25} {count:>8,} ({pct:>5.1f}%)")

# %%
# Tehran share in each cluster
print("\nTehran Presence in Each Cluster")
print("=" * 50)

for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    tehran_count = (cluster_data['city_slug'] == 'tehran').sum()
    tehran_pct = tehran_count / len(cluster_data) * 100
    print(f"Cluster {cluster_id}: {tehran_pct:.1f}% Tehran listings")

# %% [markdown]
# ### 9.2 Assign Business-Meaningful Names

# %%
# Build cluster summary for naming
cluster_summary = pd.DataFrame({
    'price_per_sqm': df_cluster.groupby('cluster')['price_per_sqm'].median(),
    'building_size': df_cluster.groupby('cluster')['building_size'].median(),
    'rooms': df_cluster.groupby('cluster')['rooms_numeric'].median(),
    'elevator_pct': df_cluster.groupby('cluster')['has_elevator_binary'].mean() * 100,
    'count': df_cluster.groupby('cluster').size()
})

# Add Tehran percentage
tehran_pcts = []
for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    tehran_pct = (cluster_data['city_slug'] == 'tehran').sum() / len(cluster_data) * 100
    tehran_pcts.append(tehran_pct)
cluster_summary['tehran_pct'] = tehran_pcts

print("Cluster Summary for Naming:")
print(cluster_summary.round(1).to_string())

# %%
# Assign names based on characteristics
# Sort clusters by price to assign names logically
cluster_by_price = cluster_summary.sort_values('price_per_sqm', ascending=False).index.tolist()

cluster_names = {}
name_templates = [
    "Premium Properties",
    "Upper-Mid Market", 
    "Standard Family Homes",
    "Mid-Range Apartments",
    "Budget Apartments",
    "Entry-Level Units",
    "Economy Segment",
    "Value Properties",
    "Starter Homes",
    "Basic Units"
]

for rank, cluster_id in enumerate(cluster_by_price):
    row = cluster_summary.loc[cluster_id]
    
    # Customize name based on characteristics
    if row['tehran_pct'] > 60:
        location = "Tehran "
    elif row['tehran_pct'] < 20:
        location = "Provincial "
    else:
        location = ""
    
    if row['building_size'] > 120:
        size_desc = "Spacious "
    elif row['building_size'] < 70:
        size_desc = "Compact "
    else:
        size_desc = ""
    
    base_name = name_templates[min(rank, len(name_templates)-1)]
    cluster_names[cluster_id] = f"{location}{size_desc}{base_name}".strip()

print("\nCluster Names (Business Interpretation):")
print("=" * 60)
for cluster_id in range(optimal_k):
    count = len(df_cluster[df_cluster['cluster'] == cluster_id])
    pct = count / len(df_cluster) * 100
    price = cluster_summary.loc[cluster_id, 'price_per_sqm'] / 1e6
    print(f"Cluster {cluster_id}: {cluster_names[cluster_id]}")
    print(f"           ({count:,} listings, {pct:.1f}%, {price:.0f}M Tomans/sqm)")

# %%
# Add cluster names to dataframe
df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_names)

# Final summary table
print("\n" + "=" * 80)
print("FINAL MARKET SEGMENTATION SUMMARY")
print("=" * 80)

for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    n = len(cluster_data)
    pct = n / len(df_cluster) * 100
    
    print(f"\n{cluster_names[cluster_id]} (Cluster {cluster_id})")
    print(f"  Count: {n:,} ({pct:.1f}%)")
    print(f"  Price/sqm: {cluster_data['price_per_sqm'].median()/1e6:.1f}M Tomans")
    print(f"  Size: {cluster_data['building_size'].median():.0f} sqm")
    print(f"  Elevator: {cluster_data['has_elevator_binary'].mean()*100:.0f}%")

# %% [markdown]
# ## 10. Cluster Size Distribution

# %%
# Visualize cluster distribution - labels ABOVE bars
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cluster sizes
cluster_sizes = df_cluster['cluster'].value_counts().sort_index()

# Left: Bar chart with labels ABOVE
bars = axes[0].bar(range(optimal_k), cluster_sizes, color=CLUSTER_COLORS[:optimal_k], edgecolor='white', linewidth=2)
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Number of Listings')
axes[0].set_title('Cluster Sizes')
axes[0].set_xticks(range(optimal_k))
max_val = cluster_sizes.max()
axes[0].set_ylim(0, max_val * 1.25)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

for i, v in enumerate(cluster_sizes):
    axes[0].text(i, v + max_val * 0.02, f'{v:,}', ha='center', va='bottom', fontsize=9, 
                 fontweight='bold', color='#333333')

# Right: Pie chart with adaptive text colors for visibility
labels = [f"C{i}" for i in range(optimal_k)]
wedges, texts, autotexts = axes[1].pie(cluster_sizes, labels=labels, autopct='%1.1f%%', 
                                        colors=CLUSTER_COLORS[:optimal_k], startangle=90,
                                        explode=[0.03]*optimal_k)

# Function to determine if color is dark (needs white text)
def is_dark_color(color):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    # Calculate luminance
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance < 0.5

for i, autotext in enumerate(autotexts):
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
    # Use white text for dark backgrounds, dark text for light backgrounds
    if is_dark_color(CLUSTER_COLORS[i]):
        autotext.set_color('white')
    else:
        autotext.set_color('#333333')

axes[1].set_title('Market Share by Segment')

plt.tight_layout()
plt.savefig(FIGURES_PATH / '04_torch_cuda_cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Key Findings Summary

# %%
print("=" * 70)
print("PHASE 4: KEY FINDINGS SUMMARY")
print("=" * 70)

print(f"""
CLUSTERING RESULTS
------------------
  Total listings clustered: {len(df_cluster):,}
  Number of clusters: {optimal_k}
  Best silhouette score: {max(silhouette_scores):.4f}

METHODOLOGY
-----------
  Features: 7 (price, size, rooms, amenities, city)
  City encoding: Target encoding (by median price)
  Normalization: StandardScaler
  Log transform: price_per_sqm, building_size, city_price_level
  Algorithm: Torch CUDA K-Means

APPROACH COMPARISON
-------------------
  Approach 1 (Original Features): Primary results
  Approach 2 (PCA Components): {X_pca.shape[1]} components (90% variance)
  Adjusted Rand Index: {ari:.4f}
  Conclusion: {'Similar' if ari > 0.7 else 'Moderate agreement' if ari > 0.4 else 'Different'} cluster structures

MARKET SEGMENTS IDENTIFIED
--------------------------""")

for cluster_id in range(optimal_k):
    count = len(df_cluster[df_cluster['cluster'] == cluster_id])
    pct = count / len(df_cluster) * 100
    price = df_cluster[df_cluster['cluster'] == cluster_id]['price_per_sqm'].median() / 1e6
    print(f"  {cluster_id}. {cluster_names[cluster_id]}")
    print(f"     {count:,} listings ({pct:.1f}%), Median: {price:.0f}M Tomans/sqm")

print("\n" + "=" * 70)

# %% [markdown]
# ## 12. Export Results

# %%
# Save cluster assignments
export_cols = ['city_slug', 'price_per_sqm', 'building_size', 'rooms_numeric', 
               'has_elevator_binary', 'has_parking_binary', 
               'has_warehouse_binary', 'cluster', 'cluster_name']

df_cluster[export_cols].to_csv(DATA_PROCESSED / 'clustering_assignments_torch_cuda.csv', index=False)
df_cluster[export_cols].to_parquet(DATA_PROCESSED / 'clustering_assignments_torch_cuda.parquet', index=False, compression='zstd')
print(f"Saved: clustering_assignments_torch_cuda.csv ({len(df_cluster):,} rows)")
print(f"Saved: clustering_assignments_torch_cuda.parquet ({len(df_cluster):,} rows)")

# Save cluster profiles
cluster_profiles_export = df_cluster.groupby('cluster').agg({
    'price_per_sqm': ['median', 'mean', 'std'],
    'building_size': ['median', 'mean'],
    'rooms_numeric': ['median', 'mean'],
    'has_elevator_binary': 'mean',
    'has_parking_binary': 'mean',
    'has_warehouse_binary': 'mean',
    'city_slug': 'count'
}).round(2)

cluster_profiles_export.columns = ['_'.join(col).strip() for col in cluster_profiles_export.columns]
cluster_profiles_export['cluster_name'] = cluster_profiles_export.index.map(cluster_names)
cluster_profiles_export.to_csv(DATA_PROCESSED / 'cluster_profiles_torch_cuda.csv')
print(f"Saved: cluster_profiles_torch_cuda.csv")

# Save summary
summary_data = {
    'cluster': list(range(optimal_k)),
    'name': [cluster_names[i] for i in range(optimal_k)],
    'count': [len(df_cluster[df_cluster['cluster'] == i]) for i in range(optimal_k)],
    'median_price_per_sqm': [df_cluster[df_cluster['cluster'] == i]['price_per_sqm'].median() for i in range(optimal_k)],
    'median_size': [df_cluster[df_cluster['cluster'] == i]['building_size'].median() for i in range(optimal_k)]
}
pd.DataFrame(summary_data).to_csv(DATA_PROCESSED / 'clustering_summary_torch_cuda.csv', index=False)
print(f"Saved: clustering_summary_torch_cuda.csv")

print(f"\nAll figures saved to: {FIGURES_PATH.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ---
#
# ## 13. Conclusion
#
# ### Phase 4 Deliverables
#
# | Output | Description |
# |--------|-------------|
# | Optimal K selection | Elbow and Silhouette analysis |
# | Approach 1 clusters | K-Means on original features |
# | Approach 2 clusters | K-Means on PCA components |
# | Approach comparison | ARI and visual comparison |
# | CUDA projection visualization | Multiple runs for broad-pattern validation |
# | Cluster profiles | Detailed statistics per cluster |
# | Business names | Interpretable segment labels |
# | City distribution | Geographic patterns per cluster |
#
# ### Files Created
#
# - `clustering_assignments_torch_cuda.csv` - Full data with cluster labels
# - `clustering_assignments_torch_cuda.parquet` - Parquet export of cluster labels
# - `cluster_profiles_torch_cuda.csv` - Statistics per cluster
# - `clustering_summary_torch_cuda.csv` - Summary table
# - Figures: `04_torch_cuda_*.png` in figures directory
#
# ### Key Insights
#
# The real estate market naturally segments into distinct categories based on price, size, amenities, and location. These segments provide actionable intelligence:
#
# - **For Buyers**: Identify which market segment matches your budget and needs
# - **For Sellers**: Understand your competition within your segment
# - **For Investors**: Spot opportunities in underserved segments
#
# ---
#
# **Phase 4 Torch CUDA Complete**
#
# Proceed to **Phase 5: Price Prediction**
