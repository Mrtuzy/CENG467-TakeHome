"""
============================================================
CENG 467 - Natural Language Understanding and Generation
Take-Home Midterm - Question 5: Language Modeling
============================================================

Objective:
    Investigate probabilistic sequence modeling and text generation
    using N-gram, LSTM, and (optional) Transformer models on
    WikiText-2.

Outputs (saved automatically):
    - results/metrics.json            : All numerical metrics
    - results/metrics.csv             : Tabular metrics
    - results/generated_samples.json  : Generated text samples
    - results/figures/*.png           : Comparison plots
    - results/training_logs.csv       : Per-epoch training logs

Author: <Your Name>  (Student ID: <Your StudentID>)
Course: CENG 467 - Take-Home Midterm
============================================================
"""

import os
import sys
import json
import math
import random
import time
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from tqdm.auto import tqdm

# ============================================================
# 0. Configuration & Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "dataset_name": "wikitext-2-raw-v1",
    "vocab_size": 10000,
    "min_freq": 3,
    "ngram_n": 3,                  # trigram
    "lstm": {
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.5,
        "seq_len": 35,
        "batch_size": 64,
        "epochs": 6,
        "lr": 1e-3,
        "clip": 0.25,
    },
    "transformer": {
        "embed_dim": 256,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 35,
        "batch_size": 64,
        "epochs": 6,
        "lr": 5e-4,
        "clip": 0.5,
    },
    "generation": {
        "prompts": [
            "the meaning of life is",
            "in the early years of",
            "the company announced that",
            "scientists have discovered",
        ],
        "max_new_tokens": 30,
        "temperature": 0.9,
        "top_k": 20,
    },
    "run_transformer": True,       # set False to skip transformer
}

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ----------- Logging (clean, single-line progress) ----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("LM")

def banner(title: str):
    log.info("=" * 64)
    log.info(title)
    log.info("=" * 64)


# ============================================================
# 1. Data Loading & Tokenization
# ============================================================
banner("STEP 1 / 6  -  Loading WikiText-2")

def simple_tokenize(text: str):
    # WikiText is already pre-tokenized (whitespace-separated).
    return text.strip().split()

ds = load_dataset("wikitext", CONFIG["dataset_name"])
log.info(f"Splits: { {k: len(v) for k, v in ds.items()} }")

def tokenize_split(split):
    tokens = []
    for line in tqdm(ds[split]["text"], desc=f"Tokenizing {split}", leave=False):
        if line.strip():
            tokens.extend(simple_tokenize(line))
    return tokens

train_tokens = tokenize_split("train")
valid_tokens = tokenize_split("validation")
test_tokens  = tokenize_split("test")
log.info(f"Tokens -> train: {len(train_tokens):,} | "
         f"valid: {len(valid_tokens):,} | test: {len(test_tokens):,}")

# ============================================================
# 2. Vocabulary
# ============================================================
banner("STEP 2 / 6  -  Building vocabulary")

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]

counter = Counter(train_tokens)
most_common = [w for w, c in counter.most_common(CONFIG["vocab_size"] - len(SPECIALS))
               if c >= CONFIG["min_freq"]]
itos = SPECIALS + most_common
stoi = {w: i for i, w in enumerate(itos)}
VOCAB_SIZE = len(itos)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = (stoi[t] for t in SPECIALS)
log.info(f"Vocabulary size: {VOCAB_SIZE:,}")

def encode(tokens):
    return [stoi.get(t, UNK_ID) for t in tokens]

train_ids = encode(train_tokens)
valid_ids = encode(valid_tokens)
test_ids  = encode(test_tokens)


# ============================================================
# 3. N-gram Model (custom interpolated Kneser-Ney trigram)
# ============================================================
# We deliberately do NOT use NLTK's lm.KneserNeyInterpolated here, because
# its scoring is pure-Python and unvectorised — perplexity over WikiText-2
# takes literally hours.  Instead we build the count tables ourselves and
# evaluate via plain dict lookups, which finishes in seconds.
# ============================================================
banner("STEP 3 / 6  -  Training N-gram model")

N = CONFIG["ngram_n"]
KN_DISCOUNT = 0.75   # standard Kneser-Ney discount

def split_sentences(token_list):
    """Roughly split into sentences for n-gram training (uses '.' as delimiter)."""
    sents, cur = [], []
    for t in token_list:
        cur.append(t)
        if t == "." and len(cur) > 1:
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents

train_sents = split_sentences(train_tokens)
log.info(f"Approx. training sentences: {len(train_sents):,}")

# ---------- Build count tables in a single pass ----------
t0 = time.time()
trigram_counts        = defaultdict(Counter)   # (w1,w2) -> {w3: c}
bigram_counts_ctx     = Counter()              # (w1,w2) -> total c    (for c(w1,w2,*))
bigram_counts         = defaultdict(Counter)   # (w1,)   -> {w2: c}
unigram_counts        = Counter()              # w       -> c

# Continuation counts for KN:
#  N1+(*  w)   : number of *unique* w_{i-1} preceding w
#  N1+(* w_{i-1} w) for trigram backoff
cont_unigram          = defaultdict(set)       # w -> set of preceding bigrams (w_{i-2}, w_{i-1})
cont_bigram_left      = defaultdict(set)       # w -> set of preceding unigrams w_{i-1}

# For each context (w1,w2) we also need N1+(w1 w2 *) = unique successors
# That is just len(trigram_counts[(w1,w2)]).
# For (w1,) we need N1+(w1 *) = unique successors, i.e. len(bigram_counts[(w1,)]).

for sent in tqdm(train_sents, desc="Counting n-grams", leave=False):
    seq = ["<s>", "<s>"] + sent + ["</s>"]
    for i in range(2, len(seq)):
        w1, w2, w3 = seq[i - 2], seq[i - 1], seq[i]
        trigram_counts[(w1, w2)][w3] += 1
        bigram_counts_ctx[(w1, w2)]  += 1
        bigram_counts[(w2,)][w3]     += 1
        unigram_counts[w3]           += 1
        cont_bigram_left[w3].add(w2)
        cont_unigram[w3].add((w1, w2))   # not strictly needed; kept for clarity

# Pre-compute continuation totals (used as the unigram base in KN)
total_unique_bigrams = sum(len(v) for v in bigram_counts.values())  # |{(w_{i-1}, w_i): c>0}|
log.info(f"Counts built in {time.time()-t0:.1f}s  "
         f"|trigrams|={sum(len(v) for v in trigram_counts.values()):,}  "
         f"|bigrams|={total_unique_bigrams:,}")

# ---------- Kneser-Ney scoring ----------
D = KN_DISCOUNT
V = VOCAB_SIZE  # used only as a tiny floor

def kn_unigram_prob(w):
    """KN continuation unigram: how often w appears as a continuation."""
    cont = len(cont_bigram_left.get(w, ()))
    if total_unique_bigrams == 0:
        return 1.0 / V
    p = cont / total_unique_bigrams
    return p if p > 0 else 1.0 / V

def kn_bigram_prob(w2, w3):
    """Interpolated KN bigram P(w3 | w2)."""
    ctx = (w2,)
    ctx_total = sum(bigram_counts[ctx].values()) if ctx in bigram_counts else 0
    if ctx_total == 0:
        return kn_unigram_prob(w3)
    c   = bigram_counts[ctx].get(w3, 0)
    n1p = len(bigram_counts[ctx])  # unique successors after w2
    first  = max(c - D, 0.0) / ctx_total
    lam    = (D * n1p) / ctx_total
    return first + lam * kn_unigram_prob(w3)

def kn_trigram_prob(w1, w2, w3):
    """Interpolated KN trigram P(w3 | w1, w2)."""
    ctx = (w1, w2)
    ctx_total = bigram_counts_ctx.get(ctx, 0)
    if ctx_total == 0:
        return kn_bigram_prob(w2, w3)
    c   = trigram_counts[ctx].get(w3, 0)
    n1p = len(trigram_counts[ctx])
    first = max(c - D, 0.0) / ctx_total
    lam   = (D * n1p) / ctx_total
    return first + lam * kn_bigram_prob(w2, w3)

def ngram_perplexity(tokens):
    """Fast perplexity computation using dict lookups only."""
    log_prob_sum, count = 0.0, 0
    sents = split_sentences(tokens)
    for sent in tqdm(sents, desc="N-gram PPL", leave=False):
        seq = ["<s>", "<s>"] + sent + ["</s>"]
        for i in range(2, len(seq)):
            p = kn_trigram_prob(seq[i - 2], seq[i - 1], seq[i])
            if p <= 0:
                p = 1e-12
            log_prob_sum += math.log(p)
            count += 1
    return math.exp(-log_prob_sum / max(count, 1))

t0 = time.time()
ngram_val_ppl  = ngram_perplexity(valid_tokens)
ngram_test_ppl = ngram_perplexity(test_tokens)
log.info(f"N-gram (KN-{N}) valid PPL: {ngram_val_ppl:.2f} | "
         f"test PPL: {ngram_test_ppl:.2f}  ({time.time()-t0:.1f}s)")

def generate_ngram(prompt, max_new=30, top_k=10):
    """Sampling generation from the trigram counts (top-k for readability)."""
    tokens = simple_tokenize(prompt)
    out = list(tokens)
    seq = ["<s>", "<s>"] + out
    for _ in range(max_new):
        ctx = (seq[-2], seq[-1])
        cands = trigram_counts.get(ctx)
        if not cands:
            # back off to bigram, then to a frequent token
            cands = bigram_counts.get((seq[-1],))
        if not cands:
            next_w = "."
        else:
            items = cands.most_common(top_k)
            words, freqs = zip(*items)
            probs = np.array(freqs, dtype=float); probs /= probs.sum()
            next_w = np.random.choice(words, p=probs)
        if next_w in ("</s>", "<s>"):
            break
        out.append(next_w); seq.append(next_w)
    return " ".join(out)

USE_NLTK_LM = False  # kept only so the metrics dict label stays consistent


# ============================================================
# 4. LSTM Language Model
# ============================================================
banner("STEP 4 / 6  -  Training LSTM language model")

def batchify(ids, batch_size, device):
    """Reshape token stream into (n_batches, batch_size) columns."""
    data = torch.tensor(ids, dtype=torch.long)
    n_batch = data.size(0) // batch_size
    data = data[: n_batch * batch_size]
    data = data.view(batch_size, -1).t().contiguous()  # (seq_total, batch)
    return data.to(device)

def get_batch(source, i, seq_len):
    L = min(seq_len, source.size(0) - 1 - i)
    x = source[i:i + L]
    y = source[i + 1:i + 1 + L]
    return x, y

class LSTMLM(nn.Module):
    def __init__(self, vocab, embed, hidden, layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=PAD_ID)
        self.lstm  = nn.LSTM(embed, hidden, num_layers=layers,
                             dropout=dropout, batch_first=False)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden, vocab)
        self.hidden_dim = hidden
        self.num_layers = layers

    def forward(self, x, hidden=None):
        e = self.drop(self.embed(x))
        out, hidden = self.lstm(e, hidden)
        out = self.drop(out)
        return self.fc(out), hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

def detach_hidden(h):
    if isinstance(h, tuple):
        return tuple(v.detach() for v in h)
    return h.detach()

def train_rnn_epoch(model, data, criterion, optimizer, seq_len, clip, epoch, total_epochs):
    model.train()
    total_loss, total_tokens = 0.0, 0
    hidden = model.init_hidden(data.size(1), DEVICE)
    n_steps = (data.size(0) - 1) // seq_len
    pbar = tqdm(range(0, data.size(0) - 1, seq_len),
                desc=f"LSTM ep {epoch}/{total_epochs}",
                total=n_steps, leave=False)
    for i in pbar:
        x, y = get_batch(data, i, seq_len)
        hidden = detach_hidden(hidden)
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / total_tokens

@torch.no_grad()
def eval_rnn(model, data, criterion, seq_len, label="eval"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    hidden = model.init_hidden(data.size(1), DEVICE)
    pbar = tqdm(range(0, data.size(0) - 1, seq_len), desc=label, leave=False)
    for i in pbar:
        x, y = get_batch(data, i, seq_len)
        hidden = detach_hidden(hidden)
        logits, hidden = model(x, hidden)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens

# Prepare batched data
LCFG = CONFIG["lstm"]
train_data = batchify(train_ids, LCFG["batch_size"], DEVICE)
valid_data = batchify(valid_ids, LCFG["batch_size"], DEVICE)
test_data  = batchify(test_ids,  LCFG["batch_size"], DEVICE)

lstm_model = LSTMLM(VOCAB_SIZE, LCFG["embed_dim"], LCFG["hidden_dim"],
                    LCFG["num_layers"], LCFG["dropout"]).to(DEVICE)
log.info(f"LSTM params: {sum(p.numel() for p in lstm_model.parameters()):,}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(lstm_model.parameters(), lr=LCFG["lr"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

lstm_log = []
best_val = float("inf"); best_state = None
for ep in range(1, LCFG["epochs"] + 1):
    t0 = time.time()
    tr_loss = train_rnn_epoch(lstm_model, train_data, criterion, optimizer,
                              LCFG["seq_len"], LCFG["clip"], ep, LCFG["epochs"])
    val_loss = eval_rnn(lstm_model, valid_data, criterion, LCFG["seq_len"], "LSTM val")
    scheduler.step()
    tr_ppl, val_ppl = math.exp(tr_loss), math.exp(val_loss)
    elapsed = time.time() - t0
    log.info(f"LSTM ep {ep}/{LCFG['epochs']}  "
             f"train_ppl={tr_ppl:.2f}  val_ppl={val_ppl:.2f}  ({elapsed:.1f}s)")
    lstm_log.append({"model": "LSTM", "epoch": ep,
                     "train_loss": tr_loss, "val_loss": val_loss,
                     "train_ppl": tr_ppl, "val_ppl": val_ppl,
                     "time_sec": elapsed})
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in lstm_model.state_dict().items()}

if best_state is not None:
    lstm_model.load_state_dict(best_state)
lstm_test_loss = eval_rnn(lstm_model, test_data, criterion, LCFG["seq_len"], "LSTM test")
lstm_test_ppl  = math.exp(lstm_test_loss)
lstm_val_ppl   = math.exp(best_val)
log.info(f"LSTM  best val PPL: {lstm_val_ppl:.2f} | test PPL: {lstm_test_ppl:.2f}")

@torch.no_grad()
def generate_neural(model, prompt, max_new=30, temperature=0.9, top_k=20, is_transformer=False):
    model.eval()
    tokens = simple_tokenize(prompt)
    ids = [stoi.get(t, UNK_ID) for t in tokens]
    if not ids:
        ids = [BOS_ID]
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (T,1)
    hidden = None
    if not is_transformer:
        hidden = model.init_hidden(1, DEVICE)
        # warm up hidden state on the prompt
        logits, hidden = model(x, hidden)
        last_logits = logits[-1, 0] / max(temperature, 1e-6)
    else:
        logits = model(x)
        last_logits = logits[-1, 0] / max(temperature, 1e-6)

    out_ids = list(ids)
    for _ in range(max_new):
        if top_k and top_k > 0:
            v, idx = torch.topk(last_logits, top_k)
            probs = torch.softmax(v, dim=-1)
            choice = idx[torch.multinomial(probs, 1)].item()
        else:
            probs = torch.softmax(last_logits, dim=-1)
            choice = torch.multinomial(probs, 1).item()

        if choice == EOS_ID or choice == PAD_ID:
            break
        out_ids.append(choice)

        if not is_transformer:
            x_next = torch.tensor([[choice]], dtype=torch.long, device=DEVICE)
            logits, hidden = model(x_next, hidden)
            last_logits = logits[-1, 0] / max(temperature, 1e-6)
        else:
            x_full = torch.tensor(out_ids, dtype=torch.long, device=DEVICE).unsqueeze(1)
            # cap to seq_len during generation for memory
            x_full = x_full[-CONFIG["transformer"]["seq_len"]:]
            logits = model(x_full)
            last_logits = logits[-1, 0] / max(temperature, 1e-6)

    return " ".join(itos[i] for i in out_ids)


# ============================================================
# 5. Transformer Language Model (optional)
# ============================================================
transformer_val_ppl = transformer_test_ppl = None
transformer_log = []

if CONFIG["run_transformer"]:
    banner("STEP 5 / 6  -  Training Transformer language model")

    class PositionalEncoding(nn.Module):
        def __init__(self, dim, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, dim)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, dim, 2).float() *
                            -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,dim)

        def forward(self, x):
            return x + self.pe[: x.size(0)]

    class TransformerLM(nn.Module):
        def __init__(self, vocab, embed, n_heads, n_layers, ff, dropout):
            super().__init__()
            self.embed_dim = embed
            self.embed = nn.Embedding(vocab, embed, padding_idx=PAD_ID)
            self.pos   = PositionalEncoding(embed)
            layer = nn.TransformerEncoderLayer(d_model=embed, nhead=n_heads,
                                               dim_feedforward=ff,
                                               dropout=dropout)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.fc = nn.Linear(embed, vocab)

        def forward(self, x):
            T = x.size(0)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            e = self.embed(x) * math.sqrt(self.embed_dim)
            e = self.pos(e)
            out = self.encoder(e, mask=mask)
            return self.fc(out)

    def train_tr_epoch(model, data, criterion, optimizer, seq_len, clip, epoch, total):
        model.train()
        total_loss, total_tokens = 0.0, 0
        n_steps = (data.size(0) - 1) // seq_len
        pbar = tqdm(range(0, data.size(0) - 1, seq_len),
                    desc=f"TF ep {epoch}/{total}", total=n_steps, leave=False)
        for i in pbar:
            x, y = get_batch(data, i, seq_len)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        return total_loss / total_tokens

    @torch.no_grad()
    def eval_tr(model, data, criterion, seq_len, label="TF eval"):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        pbar = tqdm(range(0, data.size(0) - 1, seq_len), desc=label, leave=False)
        for i in pbar:
            x, y = get_batch(data, i, seq_len)
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
        return total_loss / total_tokens

    TCFG = CONFIG["transformer"]
    tr_train_data = batchify(train_ids, TCFG["batch_size"], DEVICE)
    tr_valid_data = batchify(valid_ids, TCFG["batch_size"], DEVICE)
    tr_test_data  = batchify(test_ids,  TCFG["batch_size"], DEVICE)

    tr_model = TransformerLM(VOCAB_SIZE, TCFG["embed_dim"], TCFG["n_heads"],
                             TCFG["n_layers"], TCFG["dim_feedforward"],
                             TCFG["dropout"]).to(DEVICE)
    log.info(f"Transformer params: {sum(p.numel() for p in tr_model.parameters()):,}")

    tr_opt = optim.Adam(tr_model.parameters(), lr=TCFG["lr"])
    tr_sched = optim.lr_scheduler.StepLR(tr_opt, step_size=2, gamma=0.5)
    tr_best_val = float("inf"); tr_best_state = None

    for ep in range(1, TCFG["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_tr_epoch(tr_model, tr_train_data, criterion, tr_opt,
                                 TCFG["seq_len"], TCFG["clip"], ep, TCFG["epochs"])
        val_loss = eval_tr(tr_model, tr_valid_data, criterion, TCFG["seq_len"], "TF val")
        tr_sched.step()
        tr_ppl, val_ppl = math.exp(tr_loss), math.exp(val_loss)
        elapsed = time.time() - t0
        log.info(f"TF   ep {ep}/{TCFG['epochs']}  "
                 f"train_ppl={tr_ppl:.2f}  val_ppl={val_ppl:.2f}  ({elapsed:.1f}s)")
        transformer_log.append({"model": "Transformer", "epoch": ep,
                                "train_loss": tr_loss, "val_loss": val_loss,
                                "train_ppl": tr_ppl, "val_ppl": val_ppl,
                                "time_sec": elapsed})
        if val_loss < tr_best_val:
            tr_best_val = val_loss
            tr_best_state = {k: v.detach().cpu().clone() for k, v in tr_model.state_dict().items()}

    if tr_best_state is not None:
        tr_model.load_state_dict(tr_best_state)
    tr_test_loss = eval_tr(tr_model, tr_test_data, criterion, TCFG["seq_len"], "TF test")
    transformer_test_ppl = math.exp(tr_test_loss)
    transformer_val_ppl  = math.exp(tr_best_val)
    log.info(f"Transformer best val PPL: {transformer_val_ppl:.2f} | "
             f"test PPL: {transformer_test_ppl:.2f}")
else:
    log.info("Skipping transformer (CONFIG['run_transformer'] = False)")


# ============================================================
# 6. Generation, Saving, and Plotting
# ============================================================
banner("STEP 6 / 6  -  Generation & saving outputs")

GCFG = CONFIG["generation"]
samples = []
for prompt in GCFG["prompts"]:
    log.info(f"Generating from prompt: '{prompt}'")
    ng_text   = generate_ngram(prompt, max_new=GCFG["max_new_tokens"])
    lstm_text = generate_neural(lstm_model, prompt,
                                max_new=GCFG["max_new_tokens"],
                                temperature=GCFG["temperature"],
                                top_k=GCFG["top_k"],
                                is_transformer=False)
    entry = {"prompt": prompt, "ngram": ng_text, "lstm": lstm_text}
    if CONFIG["run_transformer"]:
        entry["transformer"] = generate_neural(tr_model, prompt,
                                               max_new=GCFG["max_new_tokens"],
                                               temperature=GCFG["temperature"],
                                               top_k=GCFG["top_k"],
                                               is_transformer=True)
    samples.append(entry)

# ---------- Save metrics (JSON + CSV) ----------
metrics = {
    "config": CONFIG,
    "vocab_size": VOCAB_SIZE,
    "device": str(DEVICE),
    "ngram": {
        "smoothing": "Interpolated Kneser-Ney (custom)",
        "discount": KN_DISCOUNT,
        "n": N,
        "valid_ppl": ngram_val_ppl,
        "test_ppl": ngram_test_ppl,
    },
    "lstm": {
        "valid_ppl": lstm_val_ppl,
        "test_ppl": lstm_test_ppl,
        "params": sum(p.numel() for p in lstm_model.parameters()),
    },
}
if CONFIG["run_transformer"]:
    metrics["transformer"] = {
        "valid_ppl": transformer_val_ppl,
        "test_ppl": transformer_test_ppl,
        "params": sum(p.numel() for p in tr_model.parameters()),
    }

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

rows = [
    {"model": "N-gram (trigram)",
     "valid_ppl": ngram_val_ppl, "test_ppl": ngram_test_ppl},
    {"model": "LSTM",
     "valid_ppl": lstm_val_ppl,  "test_ppl": lstm_test_ppl},
]
if CONFIG["run_transformer"]:
    rows.append({"model": "Transformer",
                 "valid_ppl": transformer_val_ppl,
                 "test_ppl": transformer_test_ppl})
pd.DataFrame(rows).to_csv(OUT_DIR / "metrics.csv", index=False)

# Training logs (epoch-level)
all_logs = lstm_log + transformer_log
if all_logs:
    pd.DataFrame(all_logs).to_csv(OUT_DIR / "training_logs.csv", index=False)

# Samples
with open(OUT_DIR / "generated_samples.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

# ---------- Figures ----------
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

# 1) PPL bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))
labels   = [r["model"] for r in rows]
val_ppls = [r["valid_ppl"] for r in rows]
test_ppls = [r["test_ppl"] for r in rows]
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, val_ppls,  w, label="Validation PPL", color="#4C9AFF")
ax.bar(x + w/2, test_ppls, w, label="Test PPL",       color="#FF8B6B")
for i, (v, t) in enumerate(zip(val_ppls, test_ppls)):
    ax.text(i - w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, t, f"{t:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Perplexity (lower is better)")
ax.set_title("Language Model Perplexity Comparison (WikiText-2)")
ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "ppl_comparison.png"); plt.close(fig)

# 2) Training curves
if all_logs:
    df = pd.DataFrame(all_logs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, sub in df.groupby("model"):
        axes[0].plot(sub["epoch"], sub["train_ppl"], marker="o", label=f"{name} train")
        axes[0].plot(sub["epoch"], sub["val_ppl"],   marker="s", linestyle="--",
                     label=f"{name} val")
        axes[1].plot(sub["epoch"], sub["train_loss"], marker="o", label=f"{name} train")
        axes[1].plot(sub["epoch"], sub["val_loss"],   marker="s", linestyle="--",
                     label=f"{name} val")
    axes[0].set_title("Perplexity per epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PPL"); axes[0].legend()
    axes[1].set_title("Loss per epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "training_curves.png"); plt.close(fig)

# 3) Vocabulary frequency (Zipf-like) plot
fig, ax = plt.subplots(figsize=(6, 4.5))
freqs = sorted(counter.values(), reverse=True)[:5000]
ax.loglog(range(1, len(freqs) + 1), freqs, color="#6C5CE7")
ax.set_title("Token frequency distribution (Zipf, top 5k)")
ax.set_xlabel("Rank (log)"); ax.set_ylabel("Frequency (log)")
fig.tight_layout(); fig.savefig(FIG_DIR / "zipf.png"); plt.close(fig)

# ---------- Console summary ----------
banner("FINAL RESULTS")
log.info(f"N-gram   - val PPL: {ngram_val_ppl:8.2f} | test PPL: {ngram_test_ppl:8.2f}")
log.info(f"LSTM     - val PPL: {lstm_val_ppl:8.2f} | test PPL: {lstm_test_ppl:8.2f}")
if CONFIG["run_transformer"]:
    log.info(f"TF       - val PPL: {transformer_val_ppl:8.2f} | "
             f"test PPL: {transformer_test_ppl:8.2f}")

log.info(f"Outputs saved to: {OUT_DIR.resolve()}")
log.info("Done.")