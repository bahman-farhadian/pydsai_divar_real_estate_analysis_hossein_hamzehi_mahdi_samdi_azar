# %% [markdown]
# # Phase 6: Text Classification (Torch CUDA)
#
# This project was done by Hossein Hamzehei and Mahdi Samdi Azar for the course named Data Science & AI Introductory Course with Python, conducted by the Department of Mathematical Sciences, Sharif University of Technology.
#
# **Dataset**: Divar Real Estate Advertisements
#
# ---
#
# ## Objective
#
# Train CUDA-backed text classifiers for listing category and user type prediction while keeping the scikit-learn TF-IDF report as the CPU baseline.

# %%
import json
import os
import re
from collections import Counter
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scripts.report_contracts import TEXT_METRIC_COLUMNS, TEXT_PREDICTION_COLUMNS, write_csv, write_manifest

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
        print(f"Loading Parquet: {parquet_path.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(parquet_path)
    try:
        return pd.read_csv(path, engine='pyarrow', **kwargs)
    except Exception as exc:
        print(f"PyArrow CSV engine unavailable for {path.name}; falling back to pandas C engine ({exc})")
        return pd.read_csv(path, low_memory=False, **kwargs)


def normalize_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.replace('\u0643', '\u06A9').replace('\u064A', '\u06CC')
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'09\d{9}|\+98\d{10}', ' ', text)
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\w\s]', ' ', text)
    return ' '.join(text.lower().split())


def tokenize(text):
    return normalize_text(text).split()


USER_TYPE_DISPLAY = {
    'مشاور املاک': 'Real Estate Agent',
    'شخصی': 'Private Seller',
}


def display_class_label(value):
    if pd.isna(value):
        return 'Unknown'
    text = str(value)
    if text in USER_TYPE_DISPLAY:
        return USER_TYPE_DISPLAY[text]
    return text.replace('-', ' ').replace('_', ' ').title()


def display_class_labels(values):
    return [display_class_label(value) for value in values]


def build_vocab(texts, max_vocab=60000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, count in counter.most_common(max_vocab - len(vocab)):
        if count < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab, max_tokens=120):
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_tokens]]
    return ids or [1]


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.encoded = [encode_text(text, vocab) for text in texts]
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.encoded[index], int(self.labels[index])


def collate_batch(batch):
    token_ids = []
    offsets = [0]
    labels = []
    for tokens, label in batch:
        token_ids.extend(tokens)
        offsets.append(offsets[-1] + len(tokens))
        labels.append(label)
    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(offsets[:-1], dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class EmbeddingBagClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean', sparse=False)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, token_ids, offsets):
        embedded = self.embedding(token_ids, offsets)
        return self.classifier(embedded)


def train_text_model(task_name, texts, labels, max_vocab=60000, epochs=20):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )
    vocab = build_vocab(train_texts, max_vocab=max_vocab)

    train_ds = TextDataset(train_texts, y_train, vocab)
    test_ds = TextDataset(test_texts, y_test, vocab)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, collate_fn=collate_batch)

    model = EmbeddingBagClassifier(len(vocab), len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for token_ids, offsets, yb in train_loader:
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, offsets)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        y_true, y_pred = predict_loader(model, test_loader)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'accuracy': acc, 'macro_f1': macro_f1})
        print(f"{task_name} epoch {epoch:02d}: loss={np.mean(losses):.4f}, accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict_loader(model, test_loader)
    metrics = {
        'task': task_name,
        'device': str(device),
        'rows_train': int(len(train_ds)),
        'rows_test': int(len(test_ds)),
        'vocab_size': int(len(vocab)),
        'classes': int(len(label_encoder.classes_)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
    }
    return model, label_encoder, vocab, pd.DataFrame(history), metrics, y_true, y_pred


def predict_loader(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for token_ids, offsets, labels in loader:
            logits = model(token_ids.to(device), offsets.to(device))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def predict_texts(model, texts, vocab, label_encoder):
    model.eval()
    encoded = [encode_text(text, vocab) for text in texts]
    predictions = []
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(encoded), 4096):
            batch = [(tokens, 0) for tokens in encoded[start:start + 4096]]
            token_ids, offsets, _ = collate_batch(batch)
            logits = model(token_ids.to(device), offsets.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            predictions.extend(pred.detach().cpu().numpy().tolist())
            probabilities.extend(confidence.detach().cpu().numpy().tolist())
    return label_encoder.inverse_transform(np.asarray(predictions)), np.asarray(probabilities)


# %% [markdown]
# ## 1. Paths and Runtime

# %%
PROJECT_ROOT = find_project_root()
DATA_RAW = PROJECT_ROOT / 'Divar-Real-State-Ads'
REPORTS_PATH = PROJECT_ROOT / 'reports'
DATA_PROCESSED = REPORTS_PATH / 'data'
FIGURES_PATH = REPORTS_PATH / 'figures'
MODELS_PATH = REPORTS_PATH / 'models'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Project root: .")
print(f"Reports data path: {DATA_PROCESSED.relative_to(PROJECT_ROOT)}")
print(f"Torch device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Load Text Dataset

# %%
CLEANED_FILE = DATA_PROCESSED / 'cleaned_data.csv'
RAW_FILE = DATA_RAW / 'divar_real_estate_ads.csv'
df = read_csv_fast(CLEANED_FILE if CLEANED_FILE.exists() else RAW_FILE)

text_columns = []
if 'title' in df.columns:
    text_columns.append('title')
for col in ['desc', 'description', 'text', 'content', 'body']:
    if col in df.columns:
        text_columns.append(col)
        break
if not text_columns:
    raise ValueError("No text columns were found.")

df['model_text'] = df[text_columns].fillna('').agg(' '.join, axis=1).map(normalize_text)
df = df[df['model_text'].str.len() > 0].copy()
print(f"Rows with usable text: {len(df):,}")
print(f"Text columns: {text_columns}")

# %% [markdown]
# ## 3. Category Classification

# %%
cat_df = df[df['cat3_slug'].notna()].copy()
cat_counts = cat_df['cat3_slug'].value_counts()
valid_cat = cat_counts[cat_counts >= 200].index
cat_df = cat_df[cat_df['cat3_slug'].isin(valid_cat)].copy()

cat_model, cat_encoder, cat_vocab, cat_history, cat_metrics, cat_true, cat_pred = train_text_model(
    'cat3_slug', cat_df['model_text'].tolist(), cat_df['cat3_slug'].astype(str).tolist(), epochs=18
)
print(json.dumps(cat_metrics, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 4. User Type Classification

# %%
user_df = df[df['user_type'].notna()].copy()
user_counts = user_df['user_type'].value_counts()
valid_user = user_counts[user_counts >= 200].index
user_df = user_df[user_df['user_type'].isin(valid_user)].copy()

user_model, user_encoder, user_vocab, user_history, user_metrics, user_true, user_pred = train_text_model(
    'user_type', user_df['model_text'].tolist(), user_df['user_type'].astype(str).tolist(), epochs=18
)
print(json.dumps(user_metrics, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 5. Evaluation Figures

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(cat_history['epoch'], cat_history['macro_f1'], label='cat3 macro F1')
axes[0].plot(user_history['epoch'], user_history['macro_f1'], label='user_type macro F1')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Macro F1')
axes[0].set_title('Torch CUDA Text Classification Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

metric_frame = pd.DataFrame([cat_metrics, user_metrics])
axes[1].bar(metric_frame['task'], metric_frame['accuracy'], color=['#3498db', '#2ecc71'], edgecolor='white')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Final Accuracy by Task')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '06_torch_cuda_training_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
for task_name, history, color in [
    ('cat3_slug', cat_history, '#3498db'),
    ('user_type', user_history, '#2ecc71'),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history['epoch'], history['macro_f1'], label='Macro F1', color=color)
    ax.plot(history['epoch'], history['accuracy'], label='Accuracy', color='#e74c3c')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'{task_name} CUDA Training Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'06_torch_cuda_{task_name.replace("cat3_slug", "cat3")}_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_true, y_pred, encoder, title in [
    (axes[0], cat_true, cat_pred, cat_encoder, 'cat3_slug Confusion Matrix'),
    (axes[1], user_true, user_pred, user_encoder, 'user_type Confusion Matrix'),
]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(encoder.classes_)))
    if matrix.shape[0] > 12:
        top_indices = np.argsort(matrix.sum(axis=1))[-12:]
        matrix = matrix[np.ix_(top_indices, top_indices)]
        labels = display_class_labels(encoder.classes_[top_indices])
    else:
        labels = display_class_labels(encoder.classes_)
    sns.heatmap(matrix, ax=ax, cmap='Blues', cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_PATH / '06_torch_cuda_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
for task_name, y_true, y_pred, encoder in [
    ('cat3', cat_true, cat_pred, cat_encoder),
    ('user_type', user_true, user_pred, user_encoder),
]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(encoder.classes_)))
    indices = np.argsort(matrix.sum(axis=1))[-12:] if matrix.shape[0] > 12 else np.arange(len(encoder.classes_))
    matrix = matrix[np.ix_(indices, indices)]
    labels = display_class_labels(encoder.classes_[indices])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, ax=ax, cmap='Blues', cbar=False)
    ax.set_title(f'{task_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'06_torch_cuda_{task_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
cat_distribution = cat_df['cat3_slug'].value_counts().head(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(display_class_labels(cat_distribution.index), cat_distribution.values, color='#3498db')
ax.set_title('Property Type Distribution')
ax.set_xlabel('Number of Listings')
plt.tight_layout()
plt.savefig(FIGURES_PATH / '06_torch_cuda_cat3_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

user_distribution = user_df['user_type'].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(display_class_labels(user_distribution.index), user_distribution.values, color='#2ecc71')
ax.set_title('User Type Distribution')
ax.set_xlabel('Number of Listings')
plt.tight_layout()
plt.savefig(FIGURES_PATH / '06_torch_cuda_user_type_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Predict Missing User Type Labels

# %%
missing_user = df[df['user_type'].isna()].copy()
if len(missing_user) > 0:
    predicted_user_type, confidence = predict_texts(user_model, missing_user['model_text'].tolist(), user_vocab, user_encoder)
    user_predictions = pd.DataFrame({
        'row_index': missing_user.index,
        'predicted_user_type': predicted_user_type,
        'prediction_confidence': confidence,
    })
else:
    user_predictions = pd.DataFrame(columns=['row_index', 'predicted_user_type', 'prediction_confidence'])

print(f"Missing user_type rows predicted: {len(user_predictions):,}")

fig, ax = plt.subplots(figsize=(9, 5))
if user_predictions.empty:
    ax.text(0.5, 0.5, 'No missing user-type labels', ha='center', va='center')
    ax.set_axis_off()
else:
    for label, group in user_predictions.groupby('predicted_user_type'):
        ax.hist(group['prediction_confidence'], bins=40, alpha=0.65, label=display_class_label(label))
    ax.legend()
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Number of Listings')
    ax.set_title('User Type Prediction Confidence')
plt.tight_layout()
plt.savefig(FIGURES_PATH / '06_torch_cuda_user_type_prediction_confidence.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Export Results

# %%
metrics_df = pd.DataFrame([cat_metrics, user_metrics])
metrics_df.to_csv(DATA_PROCESSED / 'text_classification_torch_cuda_metrics.csv', index=False)
torch_cuda_metrics = metrics_df.assign(implementation='torch_cuda', model_name='EmbeddingBagClassifier').rename(columns={'rows_train': 'train_rows', 'rows_test': 'test_rows'})
write_csv(torch_cuda_metrics, DATA_PROCESSED / 'text_classification_torch_cuda_metrics.csv', TEXT_METRIC_COLUMNS)

cat_history.to_csv(DATA_PROCESSED / 'text_classification_torch_cuda_cat3_history.csv', index=False)
user_history.to_csv(DATA_PROCESSED / 'text_classification_torch_cuda_user_type_history.csv', index=False)

user_predictions.to_csv(DATA_PROCESSED / 'user_type_predictions_torch_cuda.csv', index=False)
write_csv(user_predictions, DATA_PROCESSED / 'user_type_predictions_torch_cuda.csv', TEXT_PREDICTION_COLUMNS)
user_predictions.to_parquet(DATA_PROCESSED / 'user_type_predictions_torch_cuda.parquet', index=False, compression='zstd')

cat_report = classification_report(
    cat_true,
    cat_pred,
    labels=np.arange(len(cat_encoder.classes_)),
    target_names=display_class_labels(cat_encoder.classes_),
    output_dict=True,
    zero_division=0,
)
user_report = classification_report(
    user_true,
    user_pred,
    labels=np.arange(len(user_encoder.classes_)),
    target_names=display_class_labels(user_encoder.classes_),
    output_dict=True,
    zero_division=0,
)
(DATA_PROCESSED / 'text_classification_torch_cuda_cat3_report.json').write_text(
    json.dumps(cat_report, indent=2, ensure_ascii=False),
    encoding='utf-8',
)
(DATA_PROCESSED / 'text_classification_torch_cuda_user_type_report.json').write_text(
    json.dumps(user_report, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

model_bundle = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'device': str(device),
    'cat3': {
        'state_dict': cat_model.state_dict(),
        'classes': cat_encoder.classes_.tolist(),
        'vocab': cat_vocab,
        'metrics': cat_metrics,
    },
    'user_type': {
        'state_dict': user_model.state_dict(),
        'classes': user_encoder.classes_.tolist(),
        'vocab': user_vocab,
        'metrics': user_metrics,
    },
}
torch.save(model_bundle, MODELS_PATH / 'text_classification_torch_cuda.pt')

metadata = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'text_columns': text_columns,
    'metrics': {'cat3_slug': cat_metrics, 'user_type': user_metrics},
    'model_file': str((MODELS_PATH / 'text_classification_torch_cuda.pt').relative_to(PROJECT_ROOT)),
}
(MODELS_PATH / 'text_classification_torch_cuda_metadata.json').write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print(f"Saved metrics: {(DATA_PROCESSED / 'text_classification_torch_cuda_metrics.csv').relative_to(PROJECT_ROOT)}")
write_manifest(
    DATA_PROCESSED / 'text_classification_torch_cuda_manifest.json',
    'torch_cuda',
    {
        'metrics': DATA_PROCESSED / 'text_classification_torch_cuda_metrics.csv',
        'user_predictions': DATA_PROCESSED / 'user_type_predictions_torch_cuda.csv',
    },
)

print(f"Saved model: {(MODELS_PATH / 'text_classification_torch_cuda.pt').relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 8. Summary
#
# The CUDA text report provides a neural text-classification path for category and user-type inference. All outputs are written directly to `reports/data`, `reports/figures`, and `reports/models`.
