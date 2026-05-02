"""
Q1: Text Classification — Representation Learning
CENG 467 Take-Home Midterm
"""

import os, sys, json, random, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATASET LOADING & STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. Loading IMDb dataset...")
print("="*60)

from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("imdb")

train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]
test_texts  = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# Stratified 10% validation split from train
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts_full, train_labels_full,
    test_size=0.10, stratify=train_labels_full, random_state=SEED
)

def token_lengths(texts):
    return [len(t.split()) for t in texts]

def split_stats(texts, labels, name):
    lengths = token_lengths(texts)
    pos = sum(labels)
    neg = len(labels) - pos
    return {
        "split": name,
        "num_examples": len(texts),
        "avg_length": round(float(np.mean(lengths)), 2),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "std_length": round(float(np.std(lengths)), 2),
        "num_positive": pos,
        "num_negative": neg,
    }

stats = {
    "splits": [
        split_stats(train_texts, train_labels, "train"),
        split_stats(val_texts,   val_labels,   "validation"),
        split_stats(test_texts,  test_labels,  "test"),
    ],
    "samples": {
        "positive": [t[:100] for t, l in zip(train_texts, train_labels) if l == 1][:3],
        "negative": [t[:100] for t, l in zip(train_texts, train_labels) if l == 0][:3],
    }
}

with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("Saved dataset_stats.json")
for s in stats["splits"]:
    print(f"  {s['split']}: {s['num_examples']} examples, avg_len={s['avg_length']}, pos={s['num_positive']}, neg={s['num_negative']}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. Preprocessing & Ablation Study...")
print("="*60)

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def preprocess_classical(text, remove_stopwords=True, lowercase=True):
    """Tokenize using NLTK word_tokenize, optionally lowercase & remove stopwords."""
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

def tokens_to_str(tokens):
    return " ".join(tokens)

# TF-IDF + LR preprocessing ablation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ablation_configs = [
    {"name": "A1", "lowercase": True,  "remove_stopwords": True},
    {"name": "A2", "lowercase": True,  "remove_stopwords": False},
    {"name": "A3", "lowercase": False, "remove_stopwords": True},
    {"name": "A4", "lowercase": False, "remove_stopwords": False},
]

ablation_results = []
best_acc = 0
best_config = ablation_configs[0]

for cfg in ablation_configs:
    print(f"  Running ablation {cfg['name']}...")
    X_tr = [tokens_to_str(preprocess_classical(t, cfg["remove_stopwords"], cfg["lowercase"])) for t in train_texts]
    X_vl = [tokens_to_str(preprocess_classical(t, cfg["remove_stopwords"], cfg["lowercase"])) for t in val_texts]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)),
    ])
    pipe.fit(X_tr, train_labels)
    val_preds = pipe.predict(X_vl)
    acc = accuracy_score(val_labels, val_preds)
    print(f"    {cfg['name']}: val_acc={acc:.4f}")
    ablation_results.append({**cfg, "val_accuracy": round(acc, 4)})
    if acc > best_acc:
        best_acc = acc
        best_config = cfg

with open(os.path.join(OUT_DIR, "preprocessing_ablation.json"), "w") as f:
    json.dump(ablation_results, f, indent=2)
print(f"  Best config: {best_config['name']} (acc={best_acc:.4f})")

# Truncation study (token-level) with best preprocessing config
print("  Running truncation study...")
truncation_limits = [100, 200, 400]
truncation_results = []
for max_tokens in truncation_limits:
    X_tr = [tokens_to_str(preprocess_classical(t, best_config["remove_stopwords"], best_config["lowercase"])[:max_tokens]) for t in train_texts]
    X_vl = [tokens_to_str(preprocess_classical(t, best_config["remove_stopwords"], best_config["lowercase"])[:max_tokens]) for t in val_texts]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)),
    ])
    pipe.fit(X_tr, train_labels)
    val_preds = pipe.predict(X_vl)
    acc = accuracy_score(val_labels, val_preds)
    truncation_results.append({"max_tokens": max_tokens, "val_accuracy": round(acc, 4)})

with open(os.path.join(OUT_DIR, "truncation_ablation.json"), "w") as f:
    json.dump(truncation_results, f, indent=2)
print("  Saved truncation_ablation.json")

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL 1 — TF-IDF + LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. TF-IDF + Logistic Regression...")
print("="*60)

from sklearn.metrics import f1_score, classification_report, confusion_matrix

CONFIG_TFIDF = {
    "vectorizer": "TfidfVectorizer",
    "max_features": 50000,
    "ngram_range": [1, 2],
    "min_df": 2,
    "sublinear_tf": True,
    "classifier": "LogisticRegression",
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}

X_train_tfidf = [tokens_to_str(preprocess_classical(t, best_config["remove_stopwords"], best_config["lowercase"])) for t in train_texts]
X_val_tfidf   = [tokens_to_str(preprocess_classical(t, best_config["remove_stopwords"], best_config["lowercase"])) for t in val_texts]
X_test_tfidf  = [tokens_to_str(preprocess_classical(t, best_config["remove_stopwords"], best_config["lowercase"])) for t in test_texts]

pipeline_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=CONFIG_TFIDF["max_features"],
        ngram_range=tuple(CONFIG_TFIDF["ngram_range"]),
        min_df=CONFIG_TFIDF["min_df"],
        sublinear_tf=CONFIG_TFIDF["sublinear_tf"],
    )),
    ("clf", LogisticRegression(
        C=CONFIG_TFIDF["C"],
        max_iter=CONFIG_TFIDF["max_iter"],
        solver=CONFIG_TFIDF["solver"],
        random_state=CONFIG_TFIDF["random_state"],
    )),
])

t0 = time.time()
pipeline_tfidf.fit(X_train_tfidf, train_labels)
tfidf_train_time = round(time.time() - t0, 2)

test_preds_tfidf = pipeline_tfidf.predict(X_test_tfidf)
test_proba_tfidf = pipeline_tfidf.predict_proba(X_test_tfidf)[:, 1]

tfidf_acc = accuracy_score(test_labels, test_preds_tfidf)
tfidf_f1  = f1_score(test_labels, test_preds_tfidf, average="macro")
print(f"  TF-IDF+LR  — Test Acc: {tfidf_acc:.4f}  Macro-F1: {tfidf_f1:.4f}  Train time: {tfidf_train_time}s")

# Top features
feature_names = pipeline_tfidf.named_steps["tfidf"].get_feature_names_out()
coefs = pipeline_tfidf.named_steps["clf"].coef_[0]
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_neg_idx = np.argsort(coefs)[:20]
top_features = {
    "positive_class": [{"feature": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_pos_idx],
    "negative_class": [{"feature": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_neg_idx],
}
with open(os.path.join(OUT_DIR, "tfidf_top_features.json"), "w") as f:
    json.dump(top_features, f, indent=2)
print("  Saved tfidf_top_features.json")

# Confusion matrix figure
cm_tfidf = confusion_matrix(test_labels, test_preds_tfidf)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_tfidf, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("TF-IDF + LR — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_tfidf.png"), dpi=300)
plt.close()

# TF-IDF top features bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pos_f = [x["feature"] for x in top_features["positive_class"]]
pos_c = [x["coef"]    for x in top_features["positive_class"]]
neg_f = [x["feature"] for x in top_features["negative_class"]]
neg_c = [x["coef"]    for x in top_features["negative_class"]]
axes[0].barh(pos_f[::-1], pos_c[::-1], color="steelblue")
axes[0].set_title("Top 20 Features — Positive Class")
axes[0].set_xlabel("Coefficient")
axes[1].barh(neg_f[::-1], neg_c[::-1], color="salmon")
axes[1].set_title("Top 20 Features — Negative Class")
axes[1].set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tfidf_features.png"), dpi=300)
plt.close()
print("  Saved cm_tfidf.png, tfidf_features.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL 2 — BiLSTM + GloVe
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. BiLSTM + GloVe...")
print("="*60)

from collections import Counter
import urllib.request, zipfile

CONFIG_BILSTM = {
    "embedding_dim": 100,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "max_vocab_size": 30000,
    "max_seq_length": 400,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 10,
    "patience": 3,
    "seed": 42,
}

# Build vocabulary
print("  Building vocabulary...")
all_train_tokens = [preprocess_classical(t, remove_stopwords=False, lowercase=True) for t in train_texts]
counter = Counter(tok for toks in all_train_tokens for tok in toks)
most_common = counter.most_common(CONFIG_BILSTM["max_vocab_size"] - 2)
vocab = {"<PAD>": 0, "<UNK>": 1}
for w, _ in most_common:
    vocab[w] = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}
print(f"  Vocab size: {len(vocab)}")

# Download GloVe
glove_path = os.path.join(OUT_DIR, "glove.6B.100d.txt")
glove_zip  = os.path.join(OUT_DIR, "glove.6B.zip")

if not os.path.exists(glove_path):
    print("  Downloading GloVe 6B 100d...")
    try:
        urllib.request.urlretrieve(
            "https://nlp.stanford.edu/data/glove.6B.zip",
            glove_zip
        )
        with zipfile.ZipFile(glove_zip, "r") as zf:
            zf.extract("glove.6B.100d.txt", OUT_DIR)
        print("  GloVe downloaded and extracted.")
    except Exception as e:
        print(f"  GloVe download failed: {e}. Using random embeddings.")
        glove_path = None
else:
    print("  GloVe already exists.")

# Build embedding matrix
emb_dim = CONFIG_BILSTM["embedding_dim"]
embedding_matrix = np.random.normal(0, 0.01, (len(vocab), emb_dim)).astype(np.float32)
embedding_matrix[0] = 0  # PAD

if glove_path and os.path.exists(glove_path):
    print("  Loading GloVe vectors...")
    glove_hits = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embedding_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                glove_hits += 1
    print(f"  GloVe coverage: {glove_hits}/{len(vocab)} ({100*glove_hits/len(vocab):.1f}%)")

# Tokenize & encode
def encode(text, vocab, max_len):
    """Tokenize and encode text to integer ids, return (ids, length)."""
    tokens = preprocess_classical(text, remove_stopwords=False, lowercase=True)[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    return ids, len(ids)

class IMDbBiLSTMDataset(Dataset):
    """PyTorch Dataset for IMDb with integer-encoded tokens."""
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        ids, length = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), length, self.labels[idx]

def collate_fn(batch):
    """Pad sequences to max length in batch."""
    ids_list, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

train_ds_lstm = IMDbBiLSTMDataset(train_texts, train_labels, vocab, CONFIG_BILSTM["max_seq_length"])
val_ds_lstm   = IMDbBiLSTMDataset(val_texts,   val_labels,   vocab, CONFIG_BILSTM["max_seq_length"])
test_ds_lstm  = IMDbBiLSTMDataset(test_texts,  test_labels,  vocab, CONFIG_BILSTM["max_seq_length"])

train_loader_lstm = DataLoader(train_ds_lstm, batch_size=CONFIG_BILSTM["batch_size"], shuffle=True,  collate_fn=collate_fn)
val_loader_lstm   = DataLoader(val_ds_lstm,   batch_size=CONFIG_BILSTM["batch_size"], shuffle=False, collate_fn=collate_fn)
test_loader_lstm  = DataLoader(test_ds_lstm,  batch_size=CONFIG_BILSTM["batch_size"], shuffle=False, collate_fn=collate_fn)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM text classifier.
    Architecture: Embedding → BiLSTM → mean pooling → Dropout → Linear
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout, num_classes=1, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(self.dropout(pooled)).squeeze(-1)

model_lstm = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=CONFIG_BILSTM["embedding_dim"],
    hidden_dim=CONFIG_BILSTM["hidden_dim"],
    num_layers=CONFIG_BILSTM["num_layers"],
    dropout=CONFIG_BILSTM["dropout"],
    embedding_matrix=embedding_matrix,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=CONFIG_BILSTM["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

best_val_loss = float("inf")
patience_counter = 0
bilstm_log = []

t0_lstm = time.time()
print("  Training BiLSTM...")
for epoch in range(CONFIG_BILSTM["num_epochs"]):
    # Train
    model_lstm.train()
    train_loss = 0.0
    for x, lengths, y in tqdm(train_loader_lstm, desc=f"  Epoch {epoch+1} train", leave=False):
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model_lstm(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader_lstm)

    # Validate
    model_lstm.eval()
    val_loss = 0.0
    val_preds_list, val_true_list = [], []
    with torch.no_grad():
        for x, lengths, y in val_loader_lstm:
            x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            logits = model_lstm(x, lengths)
            val_loss += criterion(logits, y).item()
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            val_preds_list.extend(preds)
            val_true_list.extend(y.cpu().long().numpy())
    val_loss /= len(val_loader_lstm)
    val_acc = accuracy_score(val_true_list, val_preds_list)

    scheduler.step(val_loss)
    bilstm_log.append({"epoch": epoch+1, "train_loss": round(train_loss,4), "val_loss": round(val_loss,4), "val_acc": round(val_acc,4)})
    print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model_lstm.state_dict(), os.path.join(OUT_DIR, "bilstm_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= CONFIG_BILSTM["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

lstm_train_time = round(time.time() - t0_lstm, 2)
with open(os.path.join(OUT_DIR, "bilstm_training_log.json"), "w") as f:
    json.dump(bilstm_log, f, indent=2)

# Load best and evaluate on test
model_lstm.load_state_dict(torch.load(os.path.join(OUT_DIR, "bilstm_best.pt"), map_location=DEVICE))
model_lstm.eval()
test_preds_lstm, test_proba_lstm = [], []
with torch.no_grad():
    for x, lengths, y in tqdm(test_loader_lstm, desc="  Evaluating BiLSTM on test", leave=False):
        x, lengths = x.to(DEVICE), lengths.to(DEVICE)
        logits = model_lstm(x, lengths)
        proba = torch.sigmoid(logits).cpu().numpy()
        preds = (proba >= 0.5).astype(int)
        test_preds_lstm.extend(preds)
        test_proba_lstm.extend(proba)

lstm_acc = accuracy_score(test_labels, test_preds_lstm)
lstm_f1  = f1_score(test_labels, test_preds_lstm, average="macro")
print(f"  BiLSTM     — Test Acc: {lstm_acc:.4f}  Macro-F1: {lstm_f1:.4f}  Train time: {lstm_train_time}s")

# BiLSTM confusion matrix
cm_lstm = confusion_matrix(test_labels, test_preds_lstm)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("BiLSTM + GloVe — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_bilstm.png"), dpi=300)
plt.close()

# BiLSTM training curves
epochs_logged = [x["epoch"] for x in bilstm_log]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_logged, [x["train_loss"] for x in bilstm_log], label="Train Loss")
axes[0].plot(epochs_logged, [x["val_loss"]   for x in bilstm_log], label="Val Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("BiLSTM Loss")
axes[1].plot(epochs_logged, [x["val_acc"] for x in bilstm_log], label="Val Acc", color="green")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].set_title("BiLSTM Val Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "training_bilstm.png"), dpi=300)
plt.close()
print("  Saved cm_bilstm.png, training_bilstm.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL 3 — DistilBERT FINE-TUNING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. DistilBERT Fine-tuning...")
print("="*60)

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

CONFIG_BERT = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "seed": 42,
    "fp16": torch.cuda.is_available(),
}
print(f"  CONFIG_BERT: {CONFIG_BERT}")

tokenizer_bert = DistilBertTokenizerFast.from_pretrained(CONFIG_BERT["model_name"])

class IMDbDataset(Dataset):
    """PyTorch dataset wrapper for IMDb with HuggingFace tokenization."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), max_length=max_length,
                                   truncation=True, padding=True)
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds_bert = IMDbDataset(train_texts, train_labels, tokenizer_bert, CONFIG_BERT["max_length"])
val_ds_bert   = IMDbDataset(val_texts,   val_labels,   tokenizer_bert, CONFIG_BERT["max_length"])
test_ds_bert  = IMDbDataset(test_texts,  test_labels,  tokenizer_bert, CONFIG_BERT["max_length"])

accuracy_metric = hf_evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Compute accuracy from eval predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

model_bert = DistilBertForSequenceClassification.from_pretrained(
    CONFIG_BERT["model_name"], num_labels=2
)

bert_ckpt_dir = os.path.join(OUT_DIR, "distilbert_checkpoint")
training_args = TrainingArguments(
    output_dir=bert_ckpt_dir,
    num_train_epochs=CONFIG_BERT["num_epochs"],
    per_device_train_batch_size=CONFIG_BERT["batch_size"],
    per_device_eval_batch_size=CONFIG_BERT["batch_size"],
    learning_rate=CONFIG_BERT["learning_rate"],
    warmup_ratio=CONFIG_BERT["warmup_ratio"],
    weight_decay=CONFIG_BERT["weight_decay"],
    fp16=CONFIG_BERT["fp16"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=CONFIG_BERT["seed"],
    logging_steps=100,
    report_to="none",
)

t0_bert = time.time()
trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_ds_bert,
    eval_dataset=val_ds_bert,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
bert_train_time = round(time.time() - t0_bert, 2)

# Save training log for DistilBERT
bert_log = []
for entry in trainer.state.log_history:
    if "eval_accuracy" in entry:
        bert_log.append({
            "epoch": entry.get("epoch"),
            "eval_loss": round(entry.get("eval_loss", 0), 4),
            "eval_accuracy": round(entry.get("eval_accuracy", 0), 4),
        })
    elif "loss" in entry and "eval_loss" not in entry:
        bert_log.append({
            "step": entry.get("step"),
            "train_loss": round(entry.get("loss", 0), 4),
            "epoch": entry.get("epoch"),
        })

with open(os.path.join(OUT_DIR, "distilbert_training_log.json"), "w") as f:
    json.dump(bert_log, f, indent=2)

# Evaluate on test
print("  Evaluating DistilBERT on test set...")
test_results = trainer.predict(test_ds_bert)
test_logits_bert = test_results.predictions
test_preds_bert = np.argmax(test_logits_bert, axis=-1).tolist()
test_proba_bert = torch.softmax(torch.tensor(test_logits_bert), dim=-1)[:, 1].numpy().tolist()

bert_acc = accuracy_score(test_labels, test_preds_bert)
bert_f1  = f1_score(test_labels, test_preds_bert, average="macro")
print(f"  DistilBERT — Test Acc: {bert_acc:.4f}  Macro-F1: {bert_f1:.4f}  Train time: {bert_train_time}s")

# DistilBERT confusion matrix
cm_bert = confusion_matrix(test_labels, test_preds_bert)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_bert, annot=True, fmt="d", cmap="Oranges", ax=ax,
            xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("DistilBERT — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_distilbert.png"), dpi=300)
plt.close()

# DistilBERT training curves
bert_eval_log = [x for x in bert_log if "eval_accuracy" in x]
if bert_eval_log:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_b = [x["epoch"] for x in bert_eval_log]
    axes[0].plot(epochs_b, [x["eval_loss"]     for x in bert_eval_log], label="Val Loss",     color="orange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("DistilBERT Val Loss")
    axes[1].plot(epochs_b, [x["eval_accuracy"] for x in bert_eval_log], label="Val Accuracy", color="darkorange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].set_title("DistilBERT Val Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_distilbert.png"), dpi=300)
    plt.close()
print("  Saved cm_distilbert.png, training_distilbert.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. RESULTS COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. Results Comparison...")
print("="*60)

results = {
    "models": [
        {
            "model": "TF-IDF + Logistic Regression",
            "representation": "Sparse",
            "accuracy": round(tfidf_acc, 4),
            "macro_f1": round(tfidf_f1, 4),
            "train_time_seconds": tfidf_train_time,
            "classification_report": classification_report(test_labels, test_preds_tfidf, target_names=["negative","positive"]),
        },
        {
            "model": "BiLSTM + GloVe",
            "representation": "Dense (static)",
            "accuracy": round(lstm_acc, 4),
            "macro_f1": round(lstm_f1, 4),
            "train_time_seconds": lstm_train_time,
            "classification_report": classification_report(test_labels, test_preds_lstm, target_names=["negative","positive"]),
        },
        {
            "model": "DistilBERT",
            "representation": "Contextual",
            "accuracy": round(bert_acc, 4),
            "macro_f1": round(bert_f1, 4),
            "train_time_seconds": bert_train_time,
            "classification_report": classification_report(test_labels, test_preds_bert, target_names=["negative","positive"]),
        },
    ]
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("  Saved results.json")
for m in results["models"]:
    print(f"  {m['model']:35s}  Acc={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}  Time={m['train_time_seconds']}s")

# Model comparison bar chart
models_labels  = ["TF-IDF+LR", "BiLSTM+GloVe", "DistilBERT"]
accs  = [tfidf_acc, lstm_acc, bert_acc]
f1s   = [tfidf_f1,  lstm_f1,  bert_f1]
x = np.arange(len(models_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",  color="steelblue")
bars2 = ax.bar(x + width/2, f1s,  width, label="Macro-F1",  color="coral")
ax.set_xticks(x); ax.set_xticklabels(models_labels)
ax.set_ylim(0.8, 1.0); ax.set_ylabel("Score"); ax.set_title("Model Comparison")
ax.legend()
for bar in bars1: ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,3), textcoords="offset points", ha="center", fontsize=9)
for bar in bars2: ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,3), textcoords="offset points", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=300)
plt.close()
print("  Saved model_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. Error Analysis...")
print("="*60)

test_preds_tfidf_list = list(test_preds_tfidf)
test_preds_lstm_list  = list(test_preds_lstm)
test_preds_bert_list  = list(test_preds_bert)
test_labels_list      = list(test_labels)

errors_all3 = []
errors_tfidf_only = []
errors_lstm_only = []
errors_bert_only = []
errors_by_model = {
    "tfidf": [],
    "bilstm": [],
    "distilbert": [],
}

for i, (true, p_tfidf, p_lstm, p_bert) in enumerate(
    zip(test_labels_list, test_preds_tfidf_list, test_preds_lstm_list, test_preds_bert_list)
):
    wrong_tfidf = (p_tfidf != true)
    wrong_lstm  = (p_lstm  != true)
    wrong_bert  = (p_bert  != true)
    record = {
        "index": i,
        "text": test_texts[i][:300],
        "true_label": true,
        "pred_tfidf": int(p_tfidf),
        "pred_lstm":  int(p_lstm),
        "pred_bert":  int(p_bert),
    }
    if wrong_tfidf:
        errors_by_model["tfidf"].append(record)
    if wrong_lstm:
        errors_by_model["bilstm"].append(record)
    if wrong_bert:
        errors_by_model["distilbert"].append(record)

    if wrong_tfidf and wrong_lstm and wrong_bert:
        errors_all3.append(record)
    if wrong_tfidf and not wrong_lstm and not wrong_bert:
        errors_tfidf_only.append(record)
    if wrong_lstm and not wrong_tfidf and not wrong_bert:
        errors_lstm_only.append(record)
    if wrong_bert and not wrong_tfidf and not wrong_lstm:
        errors_bert_only.append(record)

print(
    f"  All 3 wrong: {len(errors_all3)}, TF-IDF only wrong: {len(errors_tfidf_only)}, "
    f"BiLSTM only wrong: {len(errors_lstm_only)}, DistilBERT only wrong: {len(errors_bert_only)}"
)

def detect_error_type(text):
    """Heuristic tagging for error patterns."""
    t = text.lower()
    if "not" in t or "n't" in t:
        return "negation"
    if "yeah right" in t or "oh great" in t or "sure" in t and "..." in t:
        return "sarcasm"
    if "cinemat" in t or "screenplay" in t or "plot" in t or "acting" in t:
        return "domain_terms"
    if len(t.split()) > 200:
        return "long_range"
    if "but" in t or "however" in t or "although" in t:
        return "mixed_sentiment"
    return "other"

def hypothesis_for_type(error_type):
    """Short, specific hypothesis based on error type."""
    mapping = {
        "negation": "Negation flips sentiment but cues are short and local.",
        "sarcasm": "Surface sentiment words conflict with intended meaning.",
        "domain_terms": "Film-specific jargon dominates features but is sentiment-ambiguous.",
        "long_range": "Key sentiment appears far from local context and gets diluted.",
        "mixed_sentiment": "Positive and negative cues co-occur without clear dominance.",
        "other": "Subtle sentiment or noisy text confuses the model.",
    }
    return mapping.get(error_type, mapping["other"])

# Select 5 representative cases from DistilBERT errors (best model)
detailed_cases = []
used_types = set()
for rec in errors_by_model["distilbert"]:
    etype = detect_error_type(rec["text"])
    if etype not in used_types:
        detailed_cases.append({
            "case_id": len(detailed_cases) + 1,
            "text": rec["text"][:200],
            "true_label": "positive" if rec["true_label"] == 1 else "negative",
            "predictions": {
                "tfidf": "positive" if rec["pred_tfidf"] == 1 else "negative",
                "bilstm": "positive" if rec["pred_lstm"] == 1 else "negative",
                "distilbert": "positive" if rec["pred_bert"] == 1 else "negative",
            },
            "error_type": etype,
            "hypothesis": hypothesis_for_type(etype),
        })
        used_types.add(etype)
    if len(detailed_cases) >= 5:
        break

# Fallback if not enough unique types
if len(detailed_cases) < 5:
    for rec in errors_by_model["distilbert"]:
        if len(detailed_cases) >= 5:
            break
        detailed_cases.append({
            "case_id": len(detailed_cases) + 1,
            "text": rec["text"][:200],
            "true_label": "positive" if rec["true_label"] == 1 else "negative",
            "predictions": {
                "tfidf": "positive" if rec["pred_tfidf"] == 1 else "negative",
                "bilstm": "positive" if rec["pred_lstm"] == 1 else "negative",
                "distilbert": "positive" if rec["pred_bert"] == 1 else "negative",
            },
            "error_type": detect_error_type(rec["text"]),
            "hypothesis": hypothesis_for_type(detect_error_type(rec["text"])),
        })

error_output = {
    "summary": {
        "all_3_wrong": len(errors_all3),
        "tfidf_only_wrong": len(errors_tfidf_only),
        "bilstm_only_wrong": len(errors_lstm_only),
        "distilbert_only_wrong": len(errors_bert_only),
    },
    "detailed_cases": detailed_cases,
    "errors_by_model": {
        "tfidf": errors_by_model["tfidf"],
        "bilstm": errors_by_model["bilstm"],
        "distilbert": errors_by_model["distilbert"],
    },
}
with open(os.path.join(OUT_DIR, "error_analysis.json"), "w") as f:
    json.dump(error_output, f, indent=2)
print("  Saved error_analysis.json")

# Full error lists for reproducibility
with open(os.path.join(OUT_DIR, "errors.json"), "w") as f:
    json.dump(error_output["errors_by_model"], f, indent=2)
print("  Saved errors.json")

print("\n" + "="*60)
print("Q1 COMPLETE. All outputs saved to:", OUT_DIR)
print("="*60)
print("\nFinal Results Summary:")
for m in results["models"]:
    print(f"  {m['model']:35s}  Acc={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}")
