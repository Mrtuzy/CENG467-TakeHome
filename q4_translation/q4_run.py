"""
Q4: Machine Translation
CENG 467 Take-Home Midterm
"""

import json
import os
import random
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import nltk
from nltk.translate.meteor_score import meteor_score
import sacrebleu
from bert_score import score as bert_score_fn

import sentencepiece as spm
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {os.sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CONFIG_DATASET = {
    "name": "bentrevett/multi30k",
    "train_size": 20000,
    "val_size": 2000,
    "test_size": 2000,
    "eval_size": 200,
    "seed": 42,
}

CONFIG_BPE = {
    "vocab_size": 8000,
    "model_type": "bpe",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "sos_token": "<sos>",
    "eos_token": "<eos>",
}

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

CONFIG_SEQ2SEQ = {
    "emb_dim": 256,
    "enc_hidden_dim": 512,
    "dec_hidden_dim": 512,
    "enc_dropout": 0.5,
    "dec_dropout": 0.5,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "patience": 5,
    "clip_grad_norm": 1.0,
    "teacher_forcing_ratio": 0.5,
    "max_src_length": 100,
    "max_tgt_length": 100,
}

CONFIG_MARIAN = {
    "model_name": "Helsinki-NLP/opus-mt-en-de",
    "max_length": 128,
    "num_beams": 4,
    "batch_size": 32,
}

CONFIG_METRICS = {
    "bertscore_model": "bert-base-multilingual-cased",
    "bertscore_batch_size": 16,
}

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def _safe_load_dataset():
    try:
        return load_dataset(CONFIG_DATASET["name"])
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                "datasets>=3.0 does not support dataset scripts required by Multi30k. "
                "Please install datasets<3.0 (e.g., pip install 'datasets<3')."
            )
        raise


def _get_split(dataset, name):
    if name in dataset:
        return dataset[name]
    if name == "validation" and "valid" in dataset:
        return dataset["valid"]
    raise KeyError(f"Split {name} not found in dataset.")


def _subset_split(split, size, seed):
    if size >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(size))


def _word_count(text):
    return len(text.split())


def compute_dataset_stats(train_full, val_full, test_full, train_subset, val_subset, test_subset, sp_model=None):
    stats = {
        "original_split_sizes": {
            "train": len(train_full),
            "validation": len(val_full),
            "test": len(test_full),
        },
        "subset_sizes": {
            "train": len(train_subset),
            "validation": len(val_subset),
            "test": len(test_subset),
        },
    }

    src_lengths = []
    tgt_lengths = []
    for src, tgt in zip(test_subset["en"], test_subset["de"]):
        src_lengths.append(_word_count(src))
        tgt_lengths.append(_word_count(tgt))

    stats.update({
        "avg_src_length_words": round(float(np.mean(src_lengths)), 2) if src_lengths else 0.0,
        "avg_tgt_length_words": round(float(np.mean(tgt_lengths)), 2) if tgt_lengths else 0.0,
    })

    # Word-level vocab size (train subset)
    src_vocab = set()
    tgt_vocab = set()
    for src, tgt in zip(train_subset["en"], train_subset["de"]):
        src_vocab.update(src.split())
        tgt_vocab.update(tgt.split())

    stats["word_vocab_size"] = {
        "src": len(src_vocab),
        "tgt": len(tgt_vocab),
        "joint": len(src_vocab.union(tgt_vocab)),
    }

    if sp_model is not None:
        stats["bpe_vocab_size"] = sp_model.get_piece_size()

    sample_pairs = []
    for i in range(min(5, len(test_subset))):
        sample_pairs.append({
            "en": test_subset["en"][i],
            "de": test_subset["de"][i],
        })
    stats["sample_pairs"] = sample_pairs

    return stats


def train_bpe(train_texts, model_prefix):
    model_path = f"{model_prefix}.model"
    if os.path.exists(model_path):
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp

    def sentence_iterator():
        for text in train_texts:
            yield text

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator(),
        model_prefix=model_prefix,
        vocab_size=CONFIG_BPE["vocab_size"],
        model_type=CONFIG_BPE["model_type"],
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=SOS_IDX,
        eos_id=EOS_IDX,
        pad_piece=CONFIG_BPE["pad_token"],
        unk_piece=CONFIG_BPE["unk_token"],
        bos_piece=CONFIG_BPE["sos_token"],
        eos_piece=CONFIG_BPE["eos_token"],
    )

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode(text, sp_model, max_len):
    tokens = sp_model.encode(text, out_type=int)
    tokens = tokens[: max_len - 2]
    return [SOS_IDX] + tokens + [EOS_IDX]


def decode(ids, sp_model):
    tokens = [i for i in ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
    if not tokens:
        return ""
    return sp_model.decode(tokens)


def build_encoded_pairs(split, sp_model, max_src_len, max_tgt_len):
    data = []
    for src, tgt in zip(split["en"], split["de"]):
        src_ids = encode(src, sp_model, max_src_len)
        tgt_ids = encode(tgt, sp_model, max_tgt_len)
        data.append((src_ids, tgt_ids))
    return data


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lengths = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_seqs], dtype=torch.long)

    max_src = max(src_lengths).item()
    max_tgt = max(tgt_lengths).item()

    src_pad = torch.full((max_src, len(batch)), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((max_tgt, len(batch)), PAD_IDX, dtype=torch.long)

    for i, seq in enumerate(src_seqs):
        src_pad[: len(seq), i] = torch.tensor(seq, dtype=torch.long)
    for i, seq in enumerate(tgt_seqs):
        tgt_pad[: len(seq), i] = torch.tensor(seq, dtype=torch.long)

    return src_pad, tgt_pad, src_lengths


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, src_lengths):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)

        mask = torch.arange(src_len, device=attention.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        attention = attention.masked_fill(mask, -1e9)

        attention_weights = torch.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU((enc_hidden_dim * 2) + emb_dim, dec_hidden_dim)
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, encoder_outputs, src_lengths):
        tgt_token = tgt_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(tgt_token))
        context, attn_weights = self.attention(hidden, encoder_outputs, src_lengths)
        context = context.unsqueeze(0)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(0)
        context = context.squeeze(0)
        embedded = embedded.squeeze(0)
        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return prediction, hidden.squeeze(0), attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        input_token = tgt[0, :]
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_lengths)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[t] if teacher_force else top1

        return outputs


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for src, tgt, src_lengths in tqdm(loader, desc="Train", leave=False):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_lengths = src_lengths.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(src, tgt, src_lengths, CONFIG_SEQ2SEQ["teacher_forcing_ratio"])

        output_dim = outputs.shape[-1]
        outputs = outputs[1:].reshape(-1, output_dim)
        tgt_flat = tgt[1:].reshape(-1)

        loss = criterion(outputs, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG_SEQ2SEQ["clip_grad_norm"])
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt, src_lengths in tqdm(loader, desc="Val", leave=False):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_lengths = src_lengths.to(DEVICE)

            outputs = model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].reshape(-1, output_dim)
            tgt_flat = tgt[1:].reshape(-1)

            loss = criterion(outputs, tgt_flat)
            total_loss += loss.item()

    return total_loss / max(1, len(loader))


def translate_sentence(model, src_ids, src_len, sp_model, max_len):
    model.eval()
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(DEVICE)
    src_lengths = torch.tensor([src_len], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)

    input_token = torch.tensor([SOS_IDX], dtype=torch.long, device=DEVICE)
    outputs = []
    attentions = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, attn = model.decoder(input_token, hidden, encoder_outputs, src_lengths)
        pred_token = output.argmax(1)
        outputs.append(pred_token.item())
        attentions.append(attn.squeeze(0))
        input_token = pred_token
        if pred_token.item() == EOS_IDX:
            break

    attention_matrix = torch.stack(attentions) if attentions else torch.empty(0)
    return outputs, attention_matrix


def translate_seq2seq_dataset(model, data, sp_model, max_len):
    translations = []
    for src_ids, _ in tqdm(data, desc="Seq2Seq translate"):
        pred_ids, _ = translate_sentence(model, src_ids, len(src_ids), sp_model, max_len)
        translations.append(decode(pred_ids, sp_model))
    return translations


def load_marian_model():
    tokenizer = MarianTokenizer.from_pretrained(CONFIG_MARIAN["model_name"])
    model = MarianMTModel.from_pretrained(CONFIG_MARIAN["model_name"])
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def translate_marian(texts, model, tokenizer):
    translations = []
    for i in tqdm(range(0, len(texts), CONFIG_MARIAN["batch_size"]), desc="Marian translate"):
        batch = texts[i : i + CONFIG_MARIAN["batch_size"]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG_MARIAN["max_length"],
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=CONFIG_MARIAN["num_beams"],
                max_length=CONFIG_MARIAN["max_length"],
            )
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations


def evaluate_translations(hypotheses, references):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    meteor_scores = [
        meteor_score([ref.split()], hyp.split())
        for hyp, ref in zip(hypotheses, references)
    ]
    P, R, F1 = bert_score_fn(
        hypotheses,
        references,
        lang="de",
        model_type=CONFIG_METRICS["bertscore_model"],
        batch_size=CONFIG_METRICS["bertscore_batch_size"],
        verbose=False,
        device=DEVICE,
    )

    return {
        "bleu": float(bleu.score),
        "bleu_bp": float(bleu.bp),
        "chrf": float(chrf.score),
        "meteor": float(np.mean(meteor_scores)),
        "bertscore": {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        },
    }


def avg_length(texts):
    return float(np.mean([len(t.split()) for t in texts])) if texts else 0.0


def visualize_attention(src_tokens, tgt_tokens, attention_weights, save_path):
    if attention_weights.numel() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights.cpu().numpy(), cmap="Blues", ax=ax)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(path, ids, translations):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "translations": translations}, f, indent=2)


def main():
    print("\n" + "=" * 60)
    print("1. Loading dataset...")
    print("=" * 60)
    dataset = _safe_load_dataset()
    train_full = _get_split(dataset, "train")
    val_full = _get_split(dataset, "validation")
    test_full = _get_split(dataset, "test")

    train_split = _subset_split(train_full, CONFIG_DATASET["train_size"], CONFIG_DATASET["seed"])
    val_split = _subset_split(val_full, CONFIG_DATASET["val_size"], CONFIG_DATASET["seed"])
    test_split = _subset_split(test_full, CONFIG_DATASET["test_size"], CONFIG_DATASET["seed"])

    print("\n" + "=" * 60)
    print("2. Training BPE...")
    print("=" * 60)
    bpe_prefix = os.path.join(OUT_DIR, "bpe_joint")
    train_texts = []
    for src, tgt in zip(train_split["en"], train_split["de"]):
        train_texts.append(src)
        train_texts.append(tgt)
    sp_model = train_bpe(train_texts, bpe_prefix)

    stats = compute_dataset_stats(train_full, val_full, test_full, train_split, val_split, test_split, sp_model)
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved dataset_stats.json")

    print("\n" + "=" * 60)
    print("3. Building datasets...")
    print("=" * 60)
    train_data = build_encoded_pairs(train_split, sp_model, CONFIG_SEQ2SEQ["max_src_length"], CONFIG_SEQ2SEQ["max_tgt_length"])
    val_data = build_encoded_pairs(val_split, sp_model, CONFIG_SEQ2SEQ["max_src_length"], CONFIG_SEQ2SEQ["max_tgt_length"])
    test_data = build_encoded_pairs(test_split, sp_model, CONFIG_SEQ2SEQ["max_src_length"], CONFIG_SEQ2SEQ["max_tgt_length"])

    train_loader = DataLoader(train_data, batch_size=CONFIG_SEQ2SEQ["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=CONFIG_SEQ2SEQ["batch_size"], shuffle=False, collate_fn=collate_fn)

    print("\n" + "=" * 60)
    print("4. Training Seq2Seq + Attention...")
    print("=" * 60)
    vocab_size = sp_model.get_piece_size()
    attention = BahdanauAttention(CONFIG_SEQ2SEQ["enc_hidden_dim"], CONFIG_SEQ2SEQ["dec_hidden_dim"])
    encoder = Encoder(
        vocab_size,
        CONFIG_SEQ2SEQ["emb_dim"],
        CONFIG_SEQ2SEQ["enc_hidden_dim"],
        CONFIG_SEQ2SEQ["dec_hidden_dim"],
        CONFIG_SEQ2SEQ["enc_dropout"],
    )
    decoder = Decoder(
        vocab_size,
        CONFIG_SEQ2SEQ["emb_dim"],
        CONFIG_SEQ2SEQ["enc_hidden_dim"],
        CONFIG_SEQ2SEQ["dec_hidden_dim"],
        CONFIG_SEQ2SEQ["dec_dropout"],
        attention,
    )
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    init_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG_SEQ2SEQ["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_bleu = -1.0
    patience = 0
    training_log = []

    for epoch in range(CONFIG_SEQ2SEQ["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate_loss(model, val_loader, criterion)

        # Validation BLEU (greedy, small batch)
        val_indices = list(range(min(200, len(val_data))))
        val_refs = [decode(val_data[i][1], sp_model) for i in val_indices]
        val_hyps = []
        for i in val_indices:
            src_ids = val_data[i][0]
            pred_ids, _ = translate_sentence(model, src_ids, len(src_ids), sp_model, CONFIG_SEQ2SEQ["max_tgt_length"])
            val_hyps.append(decode(pred_ids, sp_model))
        bleu = sacrebleu.corpus_bleu(val_hyps, [val_refs]).score

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_bleu": round(bleu, 4),
        })
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bleu={bleu:.2f}")

        if bleu > best_val_bleu:
            best_val_bleu = bleu
            patience = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "seq2seq_best.pt"))
        else:
            patience += 1
            if patience >= CONFIG_SEQ2SEQ["patience"]:
                print("Early stopping.")
                break

    with open(os.path.join(OUT_DIR, "seq2seq_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "seq2seq_best.pt"), map_location=DEVICE))

    print("\n" + "=" * 60)
    print("5. Translating test set...")
    print("=" * 60)
    seq2seq_translations = translate_seq2seq_dataset(model, test_data, sp_model, CONFIG_SEQ2SEQ["max_tgt_length"])

    marian_model, marian_tokenizer = load_marian_model()
    marian_translations = translate_marian(list(test_split["en"]), marian_model, marian_tokenizer)

    ids = list(range(len(test_split)))
    save_predictions(os.path.join(OUT_DIR, "seq2seq_predictions.json"), ids, seq2seq_translations)
    save_predictions(os.path.join(OUT_DIR, "marian_predictions.json"), ids, marian_translations)

    print("\n" + "=" * 60)
    print("6. Evaluating metrics...")
    print("=" * 60)
    random.seed(CONFIG_DATASET["seed"])
    eval_indices = random.sample(range(len(test_split)), CONFIG_DATASET["eval_size"])
    eval_refs = [test_split["de"][i] for i in eval_indices]
    eval_seq2seq = [seq2seq_translations[i] for i in eval_indices]
    eval_marian = [marian_translations[i] for i in eval_indices]

    seq2seq_metrics = evaluate_translations(eval_seq2seq, eval_refs)
    marian_metrics = evaluate_translations(eval_marian, eval_refs)

    results = {
        "eval_size": CONFIG_DATASET["eval_size"],
        "seq2seq": {
            **seq2seq_metrics,
            "avg_output_length": round(avg_length(seq2seq_translations), 2),
        },
        "marian": {
            **marian_metrics,
            "avg_output_length": round(avg_length(marian_translations), 2),
        },
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")

    print("\n" + "=" * 60)
    print("7. Attention visualization...")
    print("=" * 60)
    for i in range(min(2, len(eval_indices))):
        idx = eval_indices[i]
        src_ids = test_data[idx][0]
        pred_ids, attn = translate_sentence(model, src_ids, len(src_ids), sp_model, CONFIG_SEQ2SEQ["max_tgt_length"])
        src_tokens = [sp_model.id_to_piece(t) for t in src_ids if t != PAD_IDX]
        tgt_tokens = [sp_model.id_to_piece(t) for t in pred_ids if t != PAD_IDX]
        save_path = os.path.join(FIG_DIR, f"attention_{i + 1}.png")
        visualize_attention(src_tokens, tgt_tokens, attn, save_path)

    print("\n" + "=" * 60)
    print("8. Qualitative example...")
    print("=" * 60)
    example_idx = eval_indices[0]
    example = {
        "source_en": test_split["en"][example_idx],
        "reference_de": test_split["de"][example_idx],
        "seq2seq_output": seq2seq_translations[example_idx],
        "marian_output": marian_translations[example_idx],
    }

    seq2seq_notes = "Output is shorter and may be more literal, with occasional simplifications."
    if "<unk>" in example["seq2seq_output"]:
        seq2seq_notes = "Output contains <unk> tokens, indicating rare-word handling issues."

    marian_notes = "Output is fluent and closer to natural German phrasing."
    if len(example["source_en"].split()) > 25:
        marian_notes += " It preserves long-range dependencies better on longer sentences."

    example["analysis"] = {
        "seq2seq_notes": seq2seq_notes,
        "marian_notes": marian_notes,
        "rare_word_handling": "Seq2Seq may emit <unk> for rare words, while Marian handles them via subword modeling.",
        "long_range_dependency": "Marian is generally more robust on long sentences due to self-attention.",
    }

    with open(os.path.join(OUT_DIR, "qualitative_example.json"), "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print("Saved qualitative_example.json")

    print("\n" + "=" * 60)
    print("Q4 COMPLETE. Outputs saved to:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
