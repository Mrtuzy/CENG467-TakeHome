"""
Q2: Named Entity Recognition
CENG 467 Take-Home Midterm
"""

import os
import sys
import json
import time
import random
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
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
print("\n" + "=" * 60)
print("1. Loading CoNLL-2003 dataset...")
print("=" * 60)

import datasets
import transformers

_tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_new_transformers = _tf_ver >= (4, 46)

CONLL2003_LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

from huggingface_hub import hf_hub_download

def _dl(split):
    return hf_hub_download(
        repo_id="eriktks/conll2003",
        filename=f"conll2003/{split}/0000.parquet",
        revision="refs/convert/parquet",
        repo_type="dataset",
    )

dataset = datasets.load_dataset(
    "parquet",
    data_files={
        "train":      _dl("train"),
        "validation": _dl("validation"),
        "test":       _dl("test"),
    },
)
train_data = dataset["train"]
val_data   = dataset["validation"]
test_data  = dataset["test"]

try:
    label_list = train_data.features["ner_tags"].feature.names
except AttributeError:
    label_list = CONLL2003_LABEL_LIST
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

entity_types = ["PER", "ORG", "LOC", "MISC"]

def compute_stats(split_name, split_data):
    token_count = 0
    entity_count = Counter()
    sentence_lengths = []
    entity_token_count = 0

    for tokens, tags in zip(split_data["tokens"], split_data["ner_tags"]):
        token_count += len(tokens)
        sentence_lengths.append(len(tokens))
        for tag_id in tags:
            tag = label_list[tag_id]
            if tag != "O":
                entity_token_count += 1
                entity_count[tag.split("-")[-1]] += 1

    stats = {
        "split": split_name,
        "num_sentences": len(split_data),
        "num_tokens": token_count,
        "avg_sentence_length": round(float(np.mean(sentence_lengths)), 2),
        "entity_token_percent": round(100.0 * entity_token_count / max(1, token_count), 2),
        "entity_counts": {k: int(entity_count.get(k, 0)) for k in entity_types},
    }
    return stats

stats = {
    "splits": [
        compute_stats("train", train_data),
        compute_stats("validation", val_data),
        compute_stats("test", test_data),
    ],
    "label_list": label_list,
}

with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("Saved dataset_stats.json")

# ──────────────────────────────────────────────────────────────────────────────
# 2. TOKEN-LABEL ALIGNMENT FOR BERT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Token-Label Alignment...")
print("=" * 60)

from transformers import BertTokenizerFast

CONFIG_BERT = {
    "model_name": "bert-base-cased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "seed": 42,
    "fp16": torch.cuda.is_available(),
}

print(f"CONFIG_BERT: {CONFIG_BERT}")

tokenizer_bert = BertTokenizerFast.from_pretrained(CONFIG_BERT["model_name"])


def tokenize_and_align_labels(examples, tokenizer, max_length=128, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL 1 — BiLSTM-CRF
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. BiLSTM-CRF...")
print("=" * 60)

CONFIG_BILSTM = {
    "max_vocab_size": 20000,
    "char_embedding_dim": 30,
    "word_embedding_dim": 100,
    "char_hidden_dim": 50,
    "lstm_hidden_dim": 200,
    "num_layers": 1,
    "dropout": 0.5,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 30,
    "patience": 5,
    "seed": 42,
    "max_word_len": 20,
}

print(f"CONFIG_BILSTM: {CONFIG_BILSTM}")

# Build vocabularies
print("  Building word/char vocab...")
word_counter = Counter()
char_counter = Counter()
for tokens in train_data["tokens"]:
    for tok in tokens:
        word_counter[tok.lower()] += 1
        for ch in tok:
            char_counter[ch] += 1

word_vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in word_counter.most_common(CONFIG_BILSTM["max_vocab_size"] - 2):
    word_vocab[word] = len(word_vocab)

char_vocab = {"<PAD>": 0, "<UNK>": 1}
for ch, _ in char_counter.most_common():
    char_vocab[ch] = len(char_vocab)

print(f"  Word vocab size: {len(word_vocab)}")
print(f"  Char vocab size: {len(char_vocab)}")

# Download/load GloVe 100d
import urllib.request
import zipfile

glove_path = os.path.join(OUT_DIR, "glove.6B.100d.txt")
glove_zip = os.path.join(OUT_DIR, "glove.6B.zip")

if not os.path.exists(glove_path):
    print("  Downloading GloVe 6B 100d...")
    try:
        urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", glove_zip)
        with zipfile.ZipFile(glove_zip, "r") as zf:
            zf.extract("glove.6B.100d.txt", OUT_DIR)
        print("  GloVe downloaded and extracted.")
    except Exception as e:
        print(f"  GloVe download failed: {e}. Using random embeddings.")
        glove_path = None
else:
    print("  GloVe already exists.")

embedding_matrix = np.random.normal(0, 0.01, (len(word_vocab), CONFIG_BILSTM["word_embedding_dim"])).astype(np.float32)
embedding_matrix[0] = 0.0

if glove_path and os.path.exists(glove_path):
    print("  Loading GloVe vectors...")
    hits = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word_vocab:
                embedding_matrix[word_vocab[word]] = np.array(parts[1:], dtype=np.float32)
                hits += 1
    print(f"  GloVe coverage: {hits}/{len(word_vocab)} ({100.0 * hits / len(word_vocab):.1f}%)")


def encode_word(word):
    return word_vocab.get(word.lower(), word_vocab["<UNK>"])


def encode_chars(word, max_len):
    ids = [char_vocab.get(ch, char_vocab["<UNK>"]) for ch in word[:max_len]]
    if len(ids) < max_len:
        ids += [char_vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class NERDataset(Dataset):
    def __init__(self, split_data, max_word_len):
        self.tokens = split_data["tokens"]
        self.tags = split_data["ner_tags"]
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        tags = self.tags[idx]
        word_ids = [encode_word(t) for t in tokens]
        char_ids = [encode_chars(t, self.max_word_len) for t in tokens]
        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long),
        )


def collate_fn(batch):
    word_seqs, char_seqs, tag_seqs = zip(*batch)
    lengths = [len(seq) for seq in word_seqs]
    max_len = max(lengths)

    word_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    char_pad = torch.zeros(len(batch), max_len, CONFIG_BILSTM["max_word_len"], dtype=torch.long)
    tag_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (w, c, t) in enumerate(zip(word_seqs, char_seqs, tag_seqs)):
        word_pad[i, : len(w)] = w
        char_pad[i, : len(w)] = c
        tag_pad[i, : len(w)] = t
        mask[i, : len(w)] = True

    return word_pad, char_pad, tag_pad, mask


from TorchCRF import CRF


class CharBiLSTM(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, char_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_embed_dim, char_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, chars):
        b, s, w = chars.shape
        chars = chars.view(b * s, w)
        embedded = self.embedding(chars)
        _, (hidden, _) = self.lstm(embedded)
        char_repr = torch.cat([hidden[0], hidden[1]], dim=-1)
        return char_repr.view(b, s, -1)


class BiLSTMCRF(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, num_labels,
                 word_embed_dim, char_embed_dim, char_hidden_dim,
                 lstm_hidden_dim, num_layers, dropout, word_embedding_matrix=None):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=0)
        if word_embedding_matrix is not None:
            self.word_embedding.weight = nn.Parameter(torch.tensor(word_embedding_matrix, dtype=torch.float))
        self.char_lstm = CharBiLSTM(char_vocab_size, char_embed_dim, char_hidden_dim)

        input_dim = word_embed_dim + char_hidden_dim * 2
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, words, chars, mask=None):
        word_emb = self.dropout(self.word_embedding(words))
        char_emb = self.dropout(self.char_lstm(chars))
        x = torch.cat([word_emb, char_emb], dim=-1)
        lstm_out, _ = self.lstm(x)
        emissions = self.fc(self.dropout(lstm_out))
        return emissions

    def loss(self, words, chars, tags, mask):
        emissions = self.forward(words, chars, mask)
        return -self.crf(emissions, tags, mask).mean()

    def predict(self, words, chars, mask):
        emissions = self.forward(words, chars, mask)
        return self.crf.viterbi_decode(emissions, mask)


train_ds = NERDataset(train_data, CONFIG_BILSTM["max_word_len"])
val_ds = NERDataset(val_data, CONFIG_BILSTM["max_word_len"])
test_ds = NERDataset(test_data, CONFIG_BILSTM["max_word_len"])

train_loader = DataLoader(train_ds, batch_size=CONFIG_BILSTM["batch_size"], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=CONFIG_BILSTM["batch_size"], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=CONFIG_BILSTM["batch_size"], shuffle=False, collate_fn=collate_fn)

model_bilstm = BiLSTMCRF(
    word_vocab_size=len(word_vocab),
    char_vocab_size=len(char_vocab),
    num_labels=len(label_list),
    word_embed_dim=CONFIG_BILSTM["word_embedding_dim"],
    char_embed_dim=CONFIG_BILSTM["char_embedding_dim"],
    char_hidden_dim=CONFIG_BILSTM["char_hidden_dim"],
    lstm_hidden_dim=CONFIG_BILSTM["lstm_hidden_dim"],
    num_layers=CONFIG_BILSTM["num_layers"],
    dropout=CONFIG_BILSTM["dropout"],
    word_embedding_matrix=embedding_matrix,
).to(DEVICE)

optimizer = torch.optim.Adam(model_bilstm.parameters(), lr=CONFIG_BILSTM["learning_rate"])

from seqeval.metrics import (
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
    accuracy_score as seq_accuracy,
    classification_report as seq_report,
)

def _seqeval_compute(predictions, references):
    report = seq_report(references, predictions, output_dict=True, zero_division=0)
    metrics = {
        "overall_f1": seq_f1(references, predictions, zero_division=0),
        "overall_precision": seq_precision(references, predictions, zero_division=0),
        "overall_recall": seq_recall(references, predictions, zero_division=0),
        "overall_accuracy": seq_accuracy(references, predictions),
    }
    for ent in ["PER", "ORG", "LOC", "MISC"]:
        if ent in report:
            metrics[ent] = {
                "f1": report[ent]["f1-score"],
                "precision": report[ent]["precision"],
                "recall": report[ent]["recall"],
            }
    return metrics


def decode_batch(preds, labels, mask):
    true_preds = []
    true_labels = []
    for p_seq, l_seq, m_seq in zip(preds, labels, mask):
        seq_preds = []
        seq_labels = []
        for p, l, m in zip(p_seq, l_seq, m_seq):
            if not m:
                continue
            seq_preds.append(label_list[p])
            seq_labels.append(label_list[l])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)
    return true_preds, true_labels


def evaluate_bilstm(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_tokens = []
    with torch.no_grad():
        for words, chars, tags, mask in data_loader:
            words, chars, tags, mask = words.to(DEVICE), chars.to(DEVICE), tags.to(DEVICE), mask.to(DEVICE)
            preds = model.predict(words, chars, mask)
            labels = tags.cpu().numpy().tolist()
            mask_list = mask.cpu().numpy().tolist()
            true_preds, true_labels = decode_batch(preds, labels, mask_list)
            all_preds.extend(true_preds)
            all_labels.extend(true_labels)
    metrics = _seqeval_compute(all_preds, all_labels)
    return metrics, all_preds, all_labels


_bilstm_ckpt = os.path.join(OUT_DIR, "bilstm_crf_best.pt")
_bilstm_log_path = os.path.join(OUT_DIR, "bilstm_crf_training_log.json")

if os.path.exists(_bilstm_ckpt) and os.path.exists(_bilstm_log_path):
    print("  BiLSTM-CRF checkpoint found, skipping training.")
    with open(_bilstm_log_path) as f:
        bilstm_log = json.load(f)
    bilstm_train_time = 0.0
else:
    print("  Training BiLSTM-CRF...")
    best_val_f1 = -1
    patience_counter = 0
    bilstm_log = []
    start_time = time.time()

    for epoch in range(CONFIG_BILSTM["num_epochs"]):
        model_bilstm.train()
        epoch_loss = 0.0
        for words, chars, tags, mask in tqdm(train_loader, desc=f"  Epoch {epoch + 1}", leave=False):
            words, chars, tags, mask = words.to(DEVICE), chars.to(DEVICE), tags.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            loss = model_bilstm.loss(words, chars, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model_bilstm.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        val_metrics, _, _ = evaluate_bilstm(model_bilstm, val_loader)
        val_f1 = val_metrics["overall_f1"]

        bilstm_log.append({"epoch": epoch + 1, "train_loss": round(epoch_loss, 4), "val_f1": round(val_f1, 4)})
        print(f"  Epoch {epoch + 1}: train_loss={epoch_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            if os.path.exists(_bilstm_ckpt):
                os.remove(_bilstm_ckpt)
            torch.save(model_bilstm.state_dict(), _bilstm_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG_BILSTM["patience"]:
                print("  Early stopping.")
                break

    bilstm_train_time = round(time.time() - start_time, 2)

    with open(_bilstm_log_path, "w") as f:
        json.dump(bilstm_log, f, indent=2)

# Load best for evaluation
model_bilstm.load_state_dict(torch.load(_bilstm_ckpt, map_location=DEVICE, weights_only=True))

bilstm_metrics, bilstm_preds, bilstm_labels = evaluate_bilstm(model_bilstm, test_loader)
print(f"  BiLSTM-CRF Test F1: {bilstm_metrics['overall_f1']:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL 2 — BERT TOKEN CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. BERT Token Classification...")
print("=" * 60)

from transformers import (
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

bert_model = BertForTokenClassification.from_pretrained(
    CONFIG_BERT["model_name"],
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

bert_ckpt_dir = os.path.join(OUT_DIR, "bert_ner_checkpoint")

train_bert = train_data.map(
    lambda x: tokenize_and_align_labels(x, tokenizer_bert, CONFIG_BERT["max_length"]),
    batched=True,
)
val_bert = val_data.map(
    lambda x: tokenize_and_align_labels(x, tokenizer_bert, CONFIG_BERT["max_length"]),
    batched=True,
)
test_bert = test_data.map(
    lambda x: tokenize_and_align_labels(x, tokenizer_bert, CONFIG_BERT["max_length"]),
    batched=True,
)

train_bert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_bert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_bert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics_bert(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = _seqeval_compute(true_predictions, true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

_ta_kwargs = dict(
    output_dir=bert_ckpt_dir,
    num_train_epochs=CONFIG_BERT["num_epochs"],
    per_device_train_batch_size=CONFIG_BERT["batch_size"],
    per_device_eval_batch_size=CONFIG_BERT["batch_size"],
    learning_rate=CONFIG_BERT["learning_rate"],
    warmup_steps=CONFIG_BERT["warmup_steps"],
    weight_decay=CONFIG_BERT["weight_decay"],
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=CONFIG_BERT["seed"],
    logging_steps=100,
    report_to="none",
    fp16=CONFIG_BERT["fp16"],
)
if _new_transformers:
    _ta_kwargs["eval_strategy"] = "epoch"
else:
    _ta_kwargs["evaluation_strategy"] = "epoch"
training_args = TrainingArguments(**_ta_kwargs)

_trainer_kwargs = dict(
    model=bert_model,
    args=training_args,
    train_dataset=train_bert,
    eval_dataset=val_bert,
    data_collator=DataCollatorForTokenClassification(tokenizer_bert),
    compute_metrics=compute_metrics_bert,
)
if _new_transformers:
    _trainer_kwargs["processing_class"] = tokenizer_bert
else:
    _trainer_kwargs["tokenizer"] = tokenizer_bert
trainer = Trainer(**_trainer_kwargs)

_bert_preds_path = os.path.join(OUT_DIR, "bert_test_preds.json")
_bert_time_path  = os.path.join(OUT_DIR, "bert_train_time.json")

if os.path.exists(os.path.join(bert_ckpt_dir, "config.json")) and os.path.exists(_bert_preds_path):
    print("  BERT checkpoint found, skipping training.")
    with open(_bert_preds_path) as f:
        _bert_cache = json.load(f)
    bert_true_preds  = _bert_cache["preds"]
    bert_true_labels = _bert_cache["labels"]
    bert_train_time  = json.load(open(_bert_time_path)) if os.path.exists(_bert_time_path) else 0.0
    # Load best model so trainer.predict works if needed later
    from transformers import BertForTokenClassification as _BertCls
    bert_model = _BertCls.from_pretrained(bert_ckpt_dir)
else:
    start_time = time.time()
    trainer.train()
    bert_train_time = round(time.time() - start_time, 2)
    with open(_bert_time_path, "w") as f:
        json.dump(bert_train_time, f)

    print("  Evaluating BERT on test set...")
    bert_pred   = trainer.predict(test_bert)
    bert_logits = bert_pred.predictions
    bert_labels = bert_pred.label_ids
    bert_preds  = np.argmax(bert_logits, axis=2)

    bert_true_preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(bert_preds, bert_labels)
    ]
    bert_true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(bert_preds, bert_labels)
    ]
    with open(_bert_preds_path, "w") as f:
        json.dump({"preds": bert_true_preds, "labels": bert_true_labels}, f)

bert_metrics = _seqeval_compute(bert_true_preds, bert_true_labels)
print(f"  BERT Test F1: {bert_metrics['overall_f1']:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. RESULTS & FIGURES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Results & Figures...")
print("=" * 60)

results = {
    "models": {
        "bilstm_crf": {
            "overall": {
                "precision": round(bilstm_metrics["overall_precision"], 4),
                "recall": round(bilstm_metrics["overall_recall"], 4),
                "f1": round(bilstm_metrics["overall_f1"], 4),
            },
            "per_entity": {
                ent: {
                    "precision": round(bilstm_metrics.get(ent, {}).get("precision", 0.0), 4),
                    "recall": round(bilstm_metrics.get(ent, {}).get("recall", 0.0), 4),
                    "f1": round(bilstm_metrics.get(ent, {}).get("f1", 0.0), 4),
                }
                for ent in entity_types
            },
            "train_time_seconds": bilstm_train_time,
        },
        "bert": {
            "overall": {
                "precision": round(bert_metrics["overall_precision"], 4),
                "recall": round(bert_metrics["overall_recall"], 4),
                "f1": round(bert_metrics["overall_f1"], 4),
            },
            "per_entity": {
                ent: {
                    "precision": round(bert_metrics.get(ent, {}).get("precision", 0.0), 4),
                    "recall": round(bert_metrics.get(ent, {}).get("recall", 0.0), 4),
                    "f1": round(bert_metrics.get(ent, {}).get("f1", 0.0), 4),
                }
                for ent in entity_types
            },
            "train_time_seconds": bert_train_time,
        },
    }
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("  Saved results.json")

# Per-entity F1 comparison
labels = entity_types
bilstm_f1 = [results["models"]["bilstm_crf"]["per_entity"][e]["f1"] for e in labels]
bert_f1 = [results["models"]["bert"]["per_entity"][e]["f1"] for e in labels]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, bilstm_f1, width, label="BiLSTM-CRF", color="steelblue")
ax.bar(x + width / 2, bert_f1, width, label="BERT", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("F1")
ax.set_title("Per-Entity F1 Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "per_entity_f1.png"), dpi=300)
plt.close()

# BiLSTM training curve (F1 vs epoch)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([x["epoch"] for x in bilstm_log], [x["val_f1"] for x in bilstm_log], marker="o")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val F1")
ax.set_title("BiLSTM-CRF Validation F1")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "training_bilstm_crf.png"), dpi=300)
plt.close()

# Confusion matrix helper
from sklearn.metrics import confusion_matrix

def flatten_labels(true_seqs, pred_seqs):
    true_flat = []
    pred_flat = []
    for t_seq, p_seq in zip(true_seqs, pred_seqs):
        for t, p in zip(t_seq, p_seq):
            true_flat.append(label2id[t])
            pred_flat.append(label2id[p])
    return true_flat, pred_flat

# BiLSTM confusion matrix
bilstm_true_flat, bilstm_pred_flat = flatten_labels(bilstm_labels, bilstm_preds)
cm_bilstm = confusion_matrix(bilstm_true_flat, bilstm_pred_flat, labels=list(range(len(label_list))))
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_bilstm, annot=False, cmap="Blues", ax=ax,
            xticklabels=label_list, yticklabels=label_list)
ax.set_title("BiLSTM-CRF Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_bilstm_crf.png"), dpi=300)
plt.close()

# BERT confusion matrix
bert_true_flat, bert_pred_flat = flatten_labels(bert_true_labels, bert_true_preds)
cm_bert = confusion_matrix(bert_true_flat, bert_pred_flat, labels=list(range(len(label_list))))
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_bert, annot=False, cmap="Oranges", ax=ax,
            xticklabels=label_list, yticklabels=label_list)
ax.set_title("BERT Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_bert.png"), dpi=300)
plt.close()

# Entity example visualization
color_map = {
    "PER": "#a6cee3",
    "ORG": "#b2df8a",
    "LOC": "#fdbf6f",
    "MISC": "#cab2d6",
    "O": "#f0f0f0",
}


def render_example(tokens, labels, ax, title):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    x = 0.01
    y = 0.5
    for tok, lab in zip(tokens, labels):
        ent = lab.split("-")[-1] if lab != "O" else "O"
        ax.text(
            x,
            y,
            tok,
            fontsize=9,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color_map.get(ent, "#f0f0f0"), edgecolor="none"),
        )
        x += 0.02 + 0.012 * len(tok)
        if x > 0.98:
            x = 0.01
            y -= 0.3

# Use first 3 test examples
fig, axes = plt.subplots(3, 1, figsize=(10, 4))
for i in range(3):
    tokens = test_data["tokens"][i]
    labels = test_data["ner_tags"][i]
    labels = [label_list[l] for l in labels]
    render_example(tokens, labels, axes[i], f"Example {i + 1} — Ground Truth")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "entity_examples.png"), dpi=300)
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 6. ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Error Analysis...")
print("=" * 60)


def extract_entities(label_seq):
    entities = []
    start = None
    ent_type = None
    for i, tag in enumerate(label_seq + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((start, i - 1, ent_type))
            start = i
            ent_type = tag.split("-")[-1]
        elif tag.startswith("I-"):
            if ent_type is None:
                start = i
                ent_type = tag.split("-")[-1]
        else:
            if start is not None:
                entities.append((start, i - 1, ent_type))
                start = None
                ent_type = None
    return entities


def categorize_errors(tokens, true_seq, pred_seq):
    errors = defaultdict(list)
    true_entities = extract_entities(true_seq)
    pred_entities = extract_entities(pred_seq)

    true_spans = {(s, e): t for s, e, t in true_entities}
    pred_spans = {(s, e): t for s, e, t in pred_entities}

    # Boundary and type confusion
    for (s, e), t_type in true_spans.items():
        overlap = None
        for (ps, pe), p_type in pred_spans.items():
            if not (pe < s or ps > e):
                overlap = (ps, pe, p_type)
                break
        if overlap is None:
            errors["false_negative"].append((tokens, true_seq, pred_seq))
        else:
            ps, pe, p_type = overlap
            if (ps, pe) == (s, e) and p_type == t_type:
                continue
            if p_type != t_type:
                errors["type_confusion"].append((tokens, true_seq, pred_seq))
            else:
                errors["boundary_error"].append((tokens, true_seq, pred_seq))

    # False positives
    for (ps, pe), p_type in pred_spans.items():
        overlap = None
        for (s, e), t_type in true_spans.items():
            if not (e < ps or s > pe):
                overlap = (s, e, t_type)
                break
        if overlap is None:
            errors["false_positive"].append((tokens, true_seq, pred_seq))

    return errors


def build_error_report(tokens_list, true_list, pred_list):
    report = defaultdict(list)
    for tokens, true_seq, pred_seq in zip(tokens_list, true_list, pred_list):
        errs = categorize_errors(tokens, true_seq, pred_seq)
        for k, v in errs.items():
            report[k].extend(v)
    return report

# Use BERT predictions for error analysis (best contextual model)
bert_errors = build_error_report(test_data["tokens"], bert_true_labels, bert_true_preds)

error_summary = {
    k: len(v) for k, v in bert_errors.items()
}

examples = {}
for err_type, items in bert_errors.items():
    examples[err_type] = []
    for tokens, true_seq, pred_seq in items[:2]:
        examples[err_type].append({
            "tokens": tokens,
            "true_labels": true_seq,
            "pred_labels": pred_seq,
        })

error_output = {
    "summary": error_summary,
    "examples": examples,
    "note": "Errors computed using BERT predictions on test set.",
}

with open(os.path.join(OUT_DIR, "error_analysis.json"), "w") as f:
    json.dump(error_output, f, indent=2)
print("  Saved error_analysis.json")

print("\n" + "=" * 60)
print("Q2 COMPLETE. Outputs saved to:", OUT_DIR)
print("=" * 60)
