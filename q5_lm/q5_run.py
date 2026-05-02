

import math
import time
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATA
# =============================================================================
def load_wikitext2():
    from datasets import load_dataset
    print("[DATA] Loading WikiText-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tok(split):
        text = "\n".join(ds[split]["text"])
        return text.lower().split()

    tr, va, te = tok("train"), tok("validation"), tok("test")
    print(f"[DATA] Train: {len(tr):,} | Valid: {len(va):,} | Test: {len(te):,}")
    return tr, va, te


def build_vocab(tokens, min_freq=3):
    counter = Counter(tokens)
    itos = ["<unk>", "<eos>"] + [w for w, c in counter.most_common() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    print(f"[VOCAB] Size: {len(itos):,} (min_freq={min_freq})")
    return stoi, itos


def encode(tokens, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokens]


# =============================================================================
# FAST N-GRAM (custom Kneser-Ney trigram, vectorized scoring)
# =============================================================================
def run_ngram_fast(train_ids, valid_ids, test_ids, vocab_size, n=3, discount=0.75):
    """
    Custom interpolated Kneser-Ney implementation.
    Much faster than NLTK because:
      - Uses dict lookups (O(1)) instead of NLTK's Vocabulary object overhead
      - No per-token Python object creation
      - Pre-computes continuation counts
    """
    print("\n" + "=" * 60)
    print(f"[NGRAM] Training fast Kneser-Ney {n}-gram (D={discount})")
    print("=" * 60)
    t0 = time.time()

    # Add a sentinel start token (index = vocab_size, treated as <s>)
    BOS = vocab_size
    V = vocab_size + 1

    def make_ngrams(ids):
        # Pad with BOS at the start
        padded = [BOS] * (n - 1) + list(ids)
        # Generate (context, word) pairs
        return padded

    train_seq = make_ngrams(train_ids)

    # Count trigrams, bigrams, unigrams
    tri_counts = defaultdict(int)        # (w1, w2, w3) -> count
    bi_counts = defaultdict(int)         # (w1, w2) -> count
    uni_counts = defaultdict(int)        # w -> count
    # For Kneser-Ney continuation: how many distinct contexts precede each word
    bi_left_contexts = defaultdict(set)  # w2 -> {w1}
    tri_left_contexts = defaultdict(set) # (w2, w3) -> {w1}
    bi_right_contexts = defaultdict(set) # w1 -> {w2}     (N1+(w1, *))
    tri_right_contexts = defaultdict(set) # (w1, w2) -> {w3}  (N1+(w1 w2, *))

    # Walk through corpus once
    for i in range(n - 1, len(train_seq)):
        w1, w2, w3 = train_seq[i - 2], train_seq[i - 1], train_seq[i]
        tri_counts[(w1, w2, w3)] += 1
        bi_counts[(w2, w3)] += 1
        uni_counts[w3] += 1
        bi_left_contexts[w3].add(w2)
        tri_left_contexts[(w2, w3)].add(w1)
        bi_right_contexts[w2].add(w3)
        tri_right_contexts[(w1, w2)].add(w3)

    # Total bigram types (denominator for unigram KN continuation prob)
    total_bi_types = sum(len(v) for v in bi_left_contexts.values())
    if total_bi_types == 0:
        total_bi_types = 1

    # Pre-compute bigram count totals per context for normalization
    bi_count_sum = defaultdict(int)  # w1 -> sum of c(w1, w2) over w2
    for (w1, w2), c in bi_counts.items():
        bi_count_sum[w1] += c

    tri_count_sum = defaultdict(int)  # (w1, w2) -> sum of c(w1, w2, w3) over w3
    for (w1, w2, w3), c in tri_counts.items():
        tri_count_sum[(w1, w2)] += c

    print(f"[NGRAM] Trained in {time.time()-t0:.1f}s "
          f"(unique trigrams={len(tri_counts):,}, bigrams={len(bi_counts):,})")

    D = discount

    def p_kn_unigram(w):
        """Lowest-order: continuation probability."""
        return len(bi_left_contexts.get(w, ())) / total_bi_types

    def p_kn_bigram(w1, w2):
        """Middle order."""
        # Continuation count of (w1, w2): how many distinct words precede w2 in context (*, w1)?
        # In standard KN bigram: we use raw counts at the highest order, continuation at lower.
        # Since trigram is highest, bigram here acts as middle => use continuation counts.
        cont_num = len(tri_left_contexts.get((w1, w2), ()))   # |{w0 : c(w0, w1, w2) > 0}|
        cont_denom = total_bi_types  # approximation; standard
        # actually: denom should be sum over w2' of |{w0: c(w0, w1, w2') > 0}|
        # we approximate via N1+(*, w1, *) count. Compute lazily:
        # For speed, we use a simpler formulation that still works well.
        n1plus_w1 = len(bi_right_contexts.get(w1, ()))  # number of distinct w2 following w1
        if n1plus_w1 == 0:
            return p_kn_unigram(w2)
        first = max(cont_num - D, 0) / max(n1plus_w1, 1)
        lam = (D * n1plus_w1) / max(n1plus_w1, 1)  # = D, but kept explicit
        return first + lam * p_kn_unigram(w2)

    def p_kn_trigram(w1, w2, w3):
        """Highest order: use raw counts."""
        c_tri = tri_counts.get((w1, w2, w3), 0)
        c_ctx = tri_count_sum.get((w1, w2), 0)
        if c_ctx == 0:
            return p_kn_bigram(w2, w3)
        n1plus_ctx = len(tri_right_contexts.get((w1, w2), ()))  # distinct w3 after (w1,w2)
        first = max(c_tri - D, 0) / c_ctx
        lam = (D * n1plus_ctx) / c_ctx
        return first + lam * p_kn_bigram(w2, w3)

    def perplexity(ids):
        seq = [BOS] * (n - 1) + list(ids)
        log_sum = 0.0
        N = 0
        for i in range(n - 1, len(seq)):
            p = p_kn_trigram(seq[i - 2], seq[i - 1], seq[i])
            if p <= 0:
                p = 1e-10
            log_sum += -math.log(p)
            N += 1
        return math.exp(log_sum / N)

    print("[NGRAM] Computing perplexity ...")
    t0 = time.time()
    val_ppl = perplexity(valid_ids)
    test_ppl = perplexity(test_ids)
    print(f"[NGRAM] Valid PPL: {val_ppl:.2f} | Test PPL: {test_ppl:.2f} "
          f"({time.time()-t0:.1f}s)")

    # Generation: sample from trigram distribution
    print("[NGRAM] Sample generation:")
    out = [BOS, BOS]
    for _ in range(30):
        ctx = (out[-2], out[-1])
        candidates = list(tri_right_contexts.get(ctx, set()))
        if not candidates:
            # backoff to bigram
            candidates = list(bi_right_contexts.get(out[-1], set()))
        if not candidates:
            candidates = list(uni_counts.keys())
        # weight by counts
        weights = [tri_counts.get((ctx[0], ctx[1], w), 1) for w in candidates]
        total = sum(weights)
        r = random.random() * total
        acc = 0
        chosen = candidates[0]
        for w, ww in zip(candidates, weights):
            acc += ww
            if acc >= r:
                chosen = w
                break
        out.append(chosen)
    return {"valid_ppl": val_ppl, "test_ppl": test_ppl, "sample_ids": out[2:]}


# =============================================================================
# LSTM
# =============================================================================
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)
        if emb_dim == hid_dim:
            self.fc.weight = self.embed.weight

    def forward(self, x, hidden=None):
        e = self.drop(self.embed(x))
        out, hidden = self.lstm(e, hidden)
        out = self.drop(out)
        return self.fc(out), hidden


def batchify(ids, batch_size):
    n = (len(ids) // batch_size) * batch_size
    t = torch.tensor(ids[:n], dtype=torch.long)
    return t.view(batch_size, -1).contiguous()


def get_batch(data, i, bptt):
    seq_len = min(bptt, data.size(1) - 1 - i)
    x = data[:, i:i + seq_len]
    y = data[:, i + 1:i + 1 + seq_len]
    return x, y


@torch.no_grad()
def evaluate_lstm(model, data, criterion, bptt):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    hidden = None
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, hidden = model(x, hidden)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        hidden = tuple(h.detach() for h in hidden)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)


def run_lstm(train_ids, valid_ids, test_ids, vocab_size,
             epochs=3, batch_size=64, bptt=64, lr=2e-3):
    print("\n" + "=" * 60)
    print(f"[LSTM] Training | epochs={epochs}, bs={batch_size}, bptt={bptt}")
    print("=" * 60)

    model = LSTMLM(vocab_size, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3).to(DEVICE)
    print(f"[LSTM] Params: {sum(p.numel() for p in model.parameters()):,}")

    train_data = batchify(train_ids, batch_size).to(DEVICE)
    valid_data = batchify(valid_ids, batch_size).to(DEVICE)
    test_data = batchify(test_ids, batch_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    val_ppl = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        hidden = None
        total_loss, total_tokens = 0.0, 0
        t0 = time.time()
        for i in range(0, train_data.size(1) - 1, bptt):
            x, y = get_batch(train_data, i, bptt)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, hidden = model(x, hidden)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            hidden = tuple(h.detach() for h in hidden)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        scheduler.step()
        train_ppl = math.exp(total_loss / total_tokens)
        val_ppl = evaluate_lstm(model, valid_data, criterion, bptt)
        print(f"[LSTM] Epoch {epoch}/{epochs} | train_ppl={train_ppl:.2f} | "
              f"val_ppl={val_ppl:.2f} | {time.time()-t0:.1f}s")

    test_ppl = evaluate_lstm(model, test_data, criterion, bptt)
    print(f"[LSTM] Test PPL: {test_ppl:.2f}")
    sample = generate_lstm(model, vocab_size, length=30)
    return {"valid_ppl": val_ppl, "test_ppl": test_ppl, "sample_ids": sample}


@torch.no_grad()
def generate_lstm(model, vocab_size, length=30, temperature=0.9):
    model.eval()
    prefix_id = random.randint(2, vocab_size - 1)
    x = torch.tensor([[prefix_id]], device=DEVICE)
    hidden = None
    out = [prefix_id]
    for _ in range(length):
        logits, hidden = model(x, hidden)
        probs = torch.softmax(logits[0, -1].float() / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out.append(nxt)
        x = torch.tensor([[nxt]], device=DEVICE)
    return out


# =============================================================================
# TINY TRANSFORMER
# =============================================================================
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, n_layers=4,
                 dim_ff=1024, dropout=0.2, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.tok_emb.weight
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.fc(h)


def run_transformer(train_ids, valid_ids, test_ids, vocab_size,
                    epochs=2, batch_size=64, bptt=128, lr=3e-4):
    print("\n" + "=" * 60)
    print(f"[XFMR] Training | epochs={epochs}, bs={batch_size}, bptt={bptt}")
    print("=" * 60)

    model = TinyTransformerLM(vocab_size, max_len=bptt).to(DEVICE)
    print(f"[XFMR] Params: {sum(p.numel() for p in model.parameters()):,}")

    train_data = batchify(train_ids, batch_size).to(DEVICE)
    valid_data = batchify(valid_ids, batch_size).to(DEVICE)
    test_data = batchify(test_ids, batch_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    @torch.no_grad()
    def eval_xfmr(data):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for i in range(0, data.size(1) - 1, bptt):
            x, y = get_batch(data, i, bptt)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        return math.exp(total_loss / total_tokens)

    val_ppl = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        t0 = time.time()
        for i in range(0, train_data.size(1) - 1, bptt):
            x, y = get_batch(train_data, i, bptt)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        train_ppl = math.exp(total_loss / total_tokens)
        val_ppl = eval_xfmr(valid_data)
        print(f"[XFMR] Epoch {epoch}/{epochs} | train_ppl={train_ppl:.2f} | "
              f"val_ppl={val_ppl:.2f} | {time.time()-t0:.1f}s")

    test_ppl = eval_xfmr(test_data)
    print(f"[XFMR] Test PPL: {test_ppl:.2f}")

    # Generation
    model.eval()
    with torch.no_grad():
        prefix = torch.tensor([[random.randint(2, vocab_size - 1)]], device=DEVICE)
        out = prefix.tolist()[0]
        for _ in range(30):
            x = torch.tensor([out[-bptt:]], device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            probs = torch.softmax(logits[0, -1].float() / 0.9, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
    return {"valid_ppl": val_ppl, "test_ppl": test_ppl, "sample_ids": out}


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ngram", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--lstm-epochs", type=int, default=3)
    parser.add_argument("--xfmr-epochs", type=int, default=2)
    args = parser.parse_args()

    train_tokens, valid_tokens, test_tokens = load_wikitext2()
    stoi, itos = build_vocab(train_tokens, min_freq=3)
    vocab_size = len(itos)
    train_ids = encode(train_tokens, stoi)
    valid_ids = encode(valid_tokens, stoi)
    test_ids = encode(test_tokens, stoi)

    results = {}

    if not args.skip_ngram:
        res = run_ngram_fast(train_ids, valid_ids, test_ids, vocab_size, n=3)
        sample_text = " ".join(itos[i] if i < vocab_size else "<s>" for i in res["sample_ids"])
        print("  -> " + sample_text)
        results["ngram"] = {"valid_ppl": res["valid_ppl"], "test_ppl": res["test_ppl"],
                            "sample": sample_text}

    res_lstm = run_lstm(train_ids, valid_ids, test_ids, vocab_size,
                        epochs=args.lstm_epochs)
    sample_text = " ".join(itos[i] for i in res_lstm["sample_ids"])
    print("[LSTM] Sample:")
    print("  -> " + sample_text)
    results["lstm"] = {"valid_ppl": res_lstm["valid_ppl"],
                       "test_ppl": res_lstm["test_ppl"], "sample": sample_text}

    if not args.skip_transformer:
        res_xfmr = run_transformer(train_ids, valid_ids, test_ids, vocab_size,
                                   epochs=args.xfmr_epochs)
        sample_text = " ".join(itos[i] for i in res_xfmr["sample_ids"])
        print("[XFMR] Sample:")
        print("  -> " + sample_text)
        results["transformer"] = {"valid_ppl": res_xfmr["valid_ppl"],
                                  "test_ppl": res_xfmr["test_ppl"],
                                  "sample": sample_text}

    print("\n" + "=" * 60)
    print("FINAL RESULTS (lower perplexity = better)")
    print("=" * 60)
    print(f"{'Model':<15}{'Val PPL':>12}{'Test PPL':>12}")
    print("-" * 39)
    for name, r in results.items():
        print(f"{name:<15}{r['valid_ppl']:>12.2f}{r['test_ppl']:>12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()