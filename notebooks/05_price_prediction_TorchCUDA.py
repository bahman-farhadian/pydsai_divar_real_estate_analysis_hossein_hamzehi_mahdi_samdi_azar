# %% [markdown]
# # Phase 5: Price Prediction (Torch CUDA)
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements
#
# ---
#
# ## Objective
#
# Train a CUDA-backed tabular regression model for price-per-square-meter prediction while keeping the scikit-learn price prediction report as the CPU baseline.

# %%
import json
import os
from datetime import datetime, timezone
from pathlib import Path

THREAD_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault('OMP_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('MKL_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('NUMEXPR_NUM_THREADS', THREAD_COUNT)
os.environ.setdefault('ARROW_NUM_THREADS', THREAD_COUNT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


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


class PriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# %% [markdown]
# ## 1. Paths and Runtime

# %%
PROJECT_ROOT = find_project_root()
REPORTS_PATH = PROJECT_ROOT / 'reports'
DATA_PROCESSED = REPORTS_PATH / 'data'
FIGURES_PATH = REPORTS_PATH / 'figures'
MODELS_PATH = REPORTS_PATH / 'models'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Project root: {PROJECT_ROOT}")
print(f"Reports data path: {DATA_PROCESSED}")
print(f"Torch device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Load and Prepare Modeling Data

# %%
DATA_FILE = DATA_PROCESSED / 'cleaned_data_with_features.csv'
df_full = read_csv_fast(DATA_FILE)
print(f"Loaded: {len(df_full):,} rows, {len(df_full.columns)} columns")

df = df_full[df_full['listing_type'] == 'sell'].copy()
for col in ['price_value', 'building_size', 'price_per_sqm']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

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

df = df[
    df['price_per_sqm'].notna()
    & df['building_size'].notna()
    & (df['price_per_sqm'] > 0)
    & (df['building_size'] > 0)
].copy()

p1 = df['price_per_sqm'].quantile(0.01)
p99 = df['price_per_sqm'].quantile(0.99)
df = df[(df['price_per_sqm'] >= p1) & (df['price_per_sqm'] <= p99)].copy()
df = df[(df['building_size'] >= 20) & (df['building_size'] <= 1000)].copy()

print(f"Modeling rows after filtering: {len(df):,}")

# %% [markdown]
# ## 3. Feature Matrix

# %%
numeric_features = [
    'building_size',
    'rooms_numeric',
    'has_elevator_binary',
    'has_parking_binary',
    'has_warehouse_binary',
]
for col in numeric_features:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

top_cities = df['city_slug'].value_counts().head(25).index
top_types = df['cat3_slug'].value_counts().head(20).index
df['city_model'] = np.where(df['city_slug'].isin(top_cities), df['city_slug'], 'other_city')
df['type_model'] = np.where(df['cat3_slug'].isin(top_types), df['cat3_slug'], 'other_type')

feature_frame = pd.concat(
    [
        df[numeric_features].reset_index(drop=True),
        pd.get_dummies(df[['city_model', 'type_model']], dtype=np.float32).reset_index(drop=True),
    ],
    axis=1,
)
feature_columns = feature_frame.columns.tolist()

X = feature_frame.to_numpy(dtype=np.float32)
y = np.log1p(df['price_per_sqm'].to_numpy(dtype=np.float32))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index.to_numpy(), test_size=0.2, random_state=SEED
)

mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std[std == 0] = 1
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

print(f"Feature matrix: {X.shape}")
print(f"Train rows: {len(X_train_scaled):,}")
print(f"Test rows: {len(X_test_scaled):,}")

# %% [markdown]
# ## 4. Train Torch CUDA Regression Model

# %%
train_ds = TensorDataset(
    torch.from_numpy(X_train_scaled.astype(np.float32)),
    torch.from_numpy(y_train.astype(np.float32)),
)
test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32)).to(device)
train_loader = DataLoader(train_ds, batch_size=8192, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')

model = PriceMLP(X_train_scaled.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()

history = []
best_loss = float('inf')
best_state = None
patience = 8
patience_used = 0

for epoch in range(1, 101):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_pred = model(test_tensor)
        val_loss = loss_fn(val_pred, torch.from_numpy(y_test.astype(np.float32)).to(device)).item()

    train_loss = float(np.mean(epoch_losses))
    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
    print(f"Epoch {epoch:03d}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        patience_used = 0
    else:
        patience_used += 1
        if patience_used >= patience:
            print("Early stopping triggered.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# %% [markdown]
# ## 5. Evaluation

# %%
model.eval()
with torch.no_grad():
    pred_log = model(test_tensor).detach().cpu().numpy()

y_true = np.expm1(y_test)
y_pred = np.expm1(pred_log)

r2 = r2_score(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

metrics = {
    'model': 'Torch CUDA MLP',
    'device': str(device),
    'rows_train': int(len(X_train_scaled)),
    'rows_test': int(len(X_test_scaled)),
    'features': int(X_train_scaled.shape[1]),
    'r2': float(r2),
    'rmse': float(rmse),
    'mae': float(mae),
    'mape': float(mape),
}
print(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 6. Visual Diagnostics

# %%
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train')
axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Smooth L1 Loss')
axes[0].set_title('Torch CUDA Training Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sample_size = min(25000, len(y_true))
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(y_true), sample_size, replace=False)
axes[1].scatter(y_true[sample_idx] / 1e6, y_pred[sample_idx] / 1e6, alpha=0.25, s=12)
max_axis = max(np.percentile(y_true, 99), np.percentile(y_pred, 99)) / 1e6
axes[1].plot([0, max_axis], [0, max_axis], color='red', linestyle='--')
axes[1].set_xlabel('Actual price/sqm (Million Tomans)')
axes[1].set_ylabel('Predicted price/sqm (Million Tomans)')
axes[1].set_title(f'Actual vs Predicted (R2={r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_torch_cuda_training_and_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
residual_pct = (y_pred - y_true) / np.maximum(y_true, 1) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(residual_pct[np.isfinite(residual_pct)], bins=80, color='#3498db', edgecolor='white')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_xlim(-100, 100)
axes[0].set_xlabel('Prediction residual (%)')
axes[0].set_ylabel('Count')
axes[0].set_title('Residual Distribution')

value_category = np.where(residual_pct < -20, 'Under-valued', np.where(residual_pct > 20, 'Over-valued', 'Fair'))
pd.Series(value_category).value_counts().reindex(['Under-valued', 'Fair', 'Over-valued']).plot(
    kind='bar', ax=axes[1], color=['#2ecc71', '#95a5a6', '#e74c3c'], edgecolor='white'
)
axes[1].set_xlabel('Value category')
axes[1].set_ylabel('Listings')
axes[1].set_title('Value Classification')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '05_torch_cuda_residuals_and_value_segments.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Export Results

# %%
predictions_df = df.loc[idx_test, ['city_slug', 'cat3_slug', 'building_size']].copy()
predictions_df['actual_price_per_sqm'] = y_true
predictions_df['predicted_price_per_sqm'] = y_pred
predictions_df['residual_pct'] = residual_pct
predictions_df['value_category'] = value_category

predictions_df.to_csv(DATA_PROCESSED / 'price_predictions_torch_cuda.csv', index=False)
predictions_df.to_parquet(DATA_PROCESSED / 'price_predictions_torch_cuda.parquet', index=False, compression='zstd')
pd.DataFrame([metrics]).to_csv(DATA_PROCESSED / 'price_prediction_torch_cuda_metrics.csv', index=False)
history_df.to_csv(DATA_PROCESSED / 'price_prediction_torch_cuda_history.csv', index=False)

model_path = MODELS_PATH / 'price_prediction_torch_cuda.pt'
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'feature_columns': feature_columns,
        'feature_mean': mean.astype(np.float32),
        'feature_std': std.astype(np.float32),
        'metrics': metrics,
    },
    model_path,
)

metadata = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'target': 'log1p(price_per_sqm)',
    'feature_columns': feature_columns,
    'metrics': metrics,
    'model_file': str(model_path.relative_to(PROJECT_ROOT)),
}
(MODELS_PATH / 'price_prediction_torch_cuda_metadata.json').write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print(f"Saved predictions: {DATA_PROCESSED / 'price_predictions_torch_cuda.csv'}")
print(f"Saved model: {model_path}")

# %% [markdown]
# ## 8. Summary
#
# The CUDA model provides a neural-network price prediction path that complements the scikit-learn CPU baseline. The exported artifacts are written directly to `reports/data`, `reports/figures`, and `reports/models`.
