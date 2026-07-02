# %% [markdown]
# # Phase 4: Clustering Analysis (Torch CUDA KMeans)
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements (1M records)
#
# ---
#
# ## Objective
#
# Provide an optional CUDA-backed K-Means clustering implementation for the GTX 1080 workstation using PyTorch.
#
# This file keeps the pandas/scikit-learn clustering files intact and adds a separate GPU-oriented implementation. It uses CUDA when PyTorch reports a CUDA device, otherwise it falls back to CPU with the same code path.

# %%
import os

THREAD_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault('OMP_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('MKL_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('NUMEXPR_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('ARROW_NUM_THREADS', THREAD_COUNT)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True


def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / 'Divar-Real-State-Ads').exists() and (path / 'notebooks').exists():
            return path
    raise FileNotFoundError("Could not locate project root.")


def read_csv_fast(path, **kwargs):
    parquet_path = path.with_suffix('.parquet')
    if parquet_path.exists():
        print(f"Loading Parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    try:
        return pd.read_csv(path, engine='pyarrow', **kwargs)
    except Exception as exc:
        print(f"PyArrow CSV engine unavailable for {path.name}; falling back to pandas C engine ({exc})")
        return pd.read_csv(path, low_memory=False, **kwargs)


def torch_kmeans(x, n_clusters, max_iter=100, batch_size=65536, seed=42, tol=1e-4):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    initial_idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:n_clusters]
    centers = x[initial_idx].clone()
    labels = torch.empty(x.shape[0], dtype=torch.long, device=x.device)

    for iteration in range(max_iter):
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            distances = torch.cdist(x[start:end], centers)
            labels[start:end] = distances.argmin(dim=1)

        new_centers = torch.empty_like(centers)
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                new_centers[cluster_id] = x[mask].mean(dim=0)
            else:
                replacement = torch.randint(0, x.shape[0], (1,), generator=generator, device=x.device)
                new_centers[cluster_id] = x[replacement.item()]

        shift = torch.norm(new_centers - centers, dim=1).max().item()
        centers = new_centers
        print(f"Iteration {iteration + 1:03d}: center shift={shift:.6f}")
        if shift < tol:
            break

    inertia = 0.0
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        distances = torch.cdist(x[start:end], centers)
        inertia += distances.min(dim=1).values.pow(2).sum().item()

    return labels.detach().cpu().numpy(), centers.detach().cpu().numpy(), inertia


# %% [markdown]
# ## 1. Load Feature Dataset

# %%
PROJECT_ROOT = find_project_root()
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
FIGURES_PATH = PROJECT_ROOT / 'notebooks' / 'outputs' / 'figures'
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_PROCESSED / 'cleaned_data_with_features.csv'
df_full = read_csv_fast(DATA_FILE)
df = df_full[df_full['listing_type'] == 'sell'].copy()

print(f"Project root: {PROJECT_ROOT}")
print(f"Full rows: {len(df_full):,}")
print(f"Sale rows: {len(df):,}")

# %% [markdown]
# ## 2. Feature Engineering

# %%
df['price_value'] = pd.to_numeric(df['price_value'], errors='coerce')
df['building_size'] = pd.to_numeric(df['building_size'], errors='coerce')
df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')

room_mapping = {
    'بدون اتاق': 0,
    'یک': 1,
    'دو': 2,
    'سه': 3,
    'چهار': 4,
    'پنج یا بیشتر': 5,
}
df['rooms_numeric'] = df['rooms_count'].map(room_mapping)

for col in ['has_elevator', 'has_parking', 'has_warehouse']:
    if col in df.columns:
        df[col + '_binary'] = df[col].astype(str).str.lower().isin(['true', '1', '1.0']).astype(int)

city_price_median = df.groupby('city_slug')['price_per_sqm'].median()
df['city_price_level'] = df['city_slug'].map(city_price_median)

feature_cols = [
    'price_per_sqm',
    'building_size',
    'rooms_numeric',
    'city_price_level',
    'has_elevator_binary',
    'has_parking_binary',
    'has_warehouse_binary',
]

df_cluster = df.dropna(subset=feature_cols).copy()
df_cluster = df_cluster[
    (df_cluster['price_per_sqm'] > 5_000_000)
    & (df_cluster['price_per_sqm'] < 500_000_000)
    & (df_cluster['building_size'] > 10)
    & (df_cluster['building_size'] < 1000)
].copy()

X = df_cluster[feature_cols].to_numpy(dtype=np.float32)
X_scaled = StandardScaler().fit_transform(X).astype(np.float32)

print(f"Clustering rows: {len(df_cluster):,}")
print(f"Features: {feature_cols}")

# %% [markdown]
# ## 3. CUDA K-Means

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Torch device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

x_tensor = torch.from_numpy(X_scaled).to(device)
k_range = range(2, 11)
sample_size = min(100_000, len(X_scaled))
sample_idx = np.random.default_rng(42).choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

results = []
for k in k_range:
    labels, centers, inertia = torch_kmeans(x_tensor, n_clusters=k, max_iter=75, seed=42)
    sample_score = silhouette_score(X_sample, labels[sample_idx])
    results.append({'k': k, 'inertia': inertia, 'silhouette': sample_score})
    print(f"K={k}: inertia={inertia:,.0f}, silhouette={sample_score:.4f}")

results_df = pd.DataFrame(results)
optimal_k = int(results_df.sort_values('silhouette', ascending=False).iloc[0]['k'])
print(f"Selected K={optimal_k}")

labels, centers, inertia = torch_kmeans(x_tensor, n_clusters=optimal_k, max_iter=100, seed=42)
df_cluster['cluster_torch_cuda'] = labels

# %% [markdown]
# ## 4. Export Results

# %%
export_cols = ['city_slug', 'cat3_slug', 'price_per_sqm', 'building_size', 'rooms_numeric', 'cluster_torch_cuda']
df_cluster[export_cols].to_csv(DATA_PROCESSED / 'clustering_assignments_torch_cuda.csv', index=False)
df_cluster[export_cols].to_parquet(DATA_PROCESSED / 'clustering_assignments_torch_cuda.parquet', index=False, compression='zstd')

profile = df_cluster.groupby('cluster_torch_cuda')[feature_cols].median()
profile['count'] = df_cluster.groupby('cluster_torch_cuda').size()
profile.to_csv(DATA_PROCESSED / 'cluster_profiles_torch_cuda.csv')

results_df.to_csv(DATA_PROCESSED / 'clustering_torch_cuda_k_selection.csv', index=False)

pca = PCA(n_components=2, random_state=42)
pca_points = pca.fit_transform(X_scaled[sample_idx])
pca_export = pd.DataFrame({
    'pc1': pca_points[:, 0],
    'pc2': pca_points[:, 1],
    'cluster': labels[sample_idx],
})
pca_export.to_csv(DATA_PROCESSED / 'clustering_torch_cuda_pca_sample.csv', index=False)

print("Exported Torch CUDA clustering outputs.")
