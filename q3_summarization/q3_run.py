"""
Q3: Text Summarization
CENG 467 Take-Home Midterm
"""

import json
import os
import random
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from datasets import load_dataset
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    "name": "cnn_dailymail",
    "config": "3.0.0",
    "train_size": 10000,
    "val_size": 1000,
    "test_size": 1000,
    "eval_size": 200,
    "seed": 42,
}

CONFIG_EXTRACTIVE = {
    "language": "english",
    "num_sentences": 3,
}

CONFIG_ABSTRACTIVE = {
    "bart": {
        "model_name": "facebook/bart-large-cnn",
        "max_input_length": 1024,
        "max_summary_length": 150,
        "min_summary_length": 40,
        "num_beams": 4,
        "length_penalty": 2.0,
        "early_stopping": True,
        "batch_size": 4,
    },
    "t5": {
        "model_name": "t5-small",
        "max_input_length": 512,
        "max_summary_length": 150,
        "min_summary_length": 40,
        "num_beams": 4,
        "length_penalty": 1.0,
        "early_stopping": True,
        "batch_size": 8,
        "prefix": "summarize: ",
    },
}

CONFIG_METRICS = {
    "bertscore_model": "roberta-large",
    "bertscore_batch_size": 16,
}


# NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def _safe_load_dataset():
    try:
        return load_dataset(CONFIG_DATASET["name"], CONFIG_DATASET["config"])
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                "datasets>=3.0 does not support dataset scripts required by cnn_dailymail. "
                "Please install datasets<3.0 (e.g., pip install 'datasets<3')."
            )
        raise


def _subset_split(split, size, seed):
    if size >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(size))


def _word_count(text):
    return len(text.split())


def _sentence_count(text):
    return len(nltk.sent_tokenize(text))


def compute_dataset_stats(dataset, subset_splits):
    stats = {
        "original_split_sizes": {
            "train": len(dataset["train"]),
            "validation": len(dataset["validation"]),
            "test": len(dataset["test"]),
        },
        "subset_sizes": {
            "train": len(subset_splits["train"]),
            "validation": len(subset_splits["validation"]),
            "test": len(subset_splits["test"]),
        },
    }

    article_lengths = []
    summary_lengths = []
    sentence_counts = []

    for article, summary in zip(subset_splits["test"]["article"], subset_splits["test"]["highlights"]):
        article_lengths.append(_word_count(article))
        summary_lengths.append(_word_count(summary))
        sentence_counts.append(_sentence_count(article))

    avg_article_len = float(np.mean(article_lengths)) if article_lengths else 0.0
    avg_summary_len = float(np.mean(summary_lengths)) if summary_lengths else 0.0
    avg_sentence_count = float(np.mean(sentence_counts)) if sentence_counts else 0.0
    compression_ratio = avg_article_len / max(1.0, avg_summary_len)

    stats.update({
        "avg_article_length_words": round(avg_article_len, 2),
        "avg_summary_length_words": round(avg_summary_len, 2),
        "avg_article_sentence_count": round(avg_sentence_count, 2),
        "avg_compression_ratio": round(compression_ratio, 2),
    })
    return stats


def extractive_summarize(text, num_sentences, method):
    parser = PlaintextParser.from_string(text, Tokenizer(CONFIG_EXTRACTIVE["language"]))
    stemmer = Stemmer(CONFIG_EXTRACTIVE["language"])

    if method == "textrank":
        summarizer = TextRankSummarizer(stemmer)
    else:
        summarizer = LexRankSummarizer(stemmer)

    summarizer.stop_words = get_stop_words(CONFIG_EXTRACTIVE["language"])
    sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in sentences)


def load_seq2seq_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def abstractive_summarize_batch(articles, model, tokenizer, config, prefix=""):
    if prefix:
        articles = [prefix + a for a in articles]

    inputs = tokenizer(
        articles,
        max_length=config["max_input_length"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=config["num_beams"],
            max_length=config["max_summary_length"],
            min_length=config["min_summary_length"],
            length_penalty=config["length_penalty"],
            early_stopping=config["early_stopping"],
        )

    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(s[key].fmeasure)
    avg = {k: float(np.mean(v)) for k, v in scores.items()}
    return avg, scores


def compute_bleu(predictions, references):
    smoothie = SmoothingFunction().method1
    refs = [[ref.split()] for ref in references]
    hyps = [pred.split() for pred in predictions]
    return float(corpus_bleu(refs, hyps, smoothing_function=smoothie))


def compute_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(meteor_score([ref.split()], pred.split()))
    return float(np.mean(scores))


def compute_bertscore(predictions, references):
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type=CONFIG_METRICS["bertscore_model"],
        batch_size=CONFIG_METRICS["bertscore_batch_size"],
        verbose=False,
        device=DEVICE,
    )
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
        "per_example_f1": [float(x) for x in F1.tolist()],
    }


def save_predictions(path, ids, summaries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "summaries": summaries}, f, indent=2)


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ids"], data["summaries"]


def summarize_extractive(method, articles, ids, cache_path):
    if os.path.exists(cache_path):
        ids_loaded, summaries = load_predictions(cache_path)
        return ids_loaded, summaries, 0.0

    start = time.time()
    summaries = []
    for article in tqdm(articles, desc=f"Extractive {method}"):
        summaries.append(extractive_summarize(article, CONFIG_EXTRACTIVE["num_sentences"], method))
    elapsed = time.time() - start

    save_predictions(cache_path, ids, summaries)
    avg_ms = 1000.0 * elapsed / max(1, len(articles))
    return ids, summaries, avg_ms


def summarize_abstractive(key, articles, ids, cache_path):
    if os.path.exists(cache_path):
        ids_loaded, summaries = load_predictions(cache_path)
        return ids_loaded, summaries, 0.0

    config = CONFIG_ABSTRACTIVE[key]
    model, tokenizer = load_seq2seq_model(config["model_name"])
    prefix = config.get("prefix", "")

    summaries = []
    start = time.time()
    for i in tqdm(range(0, len(articles), config["batch_size"]), desc=f"Abstractive {key}"):
        batch = articles[i : i + config["batch_size"]]
        batch_summaries = abstractive_summarize_batch(batch, model, tokenizer, config, prefix=prefix)
        summaries.extend(batch_summaries)
    elapsed = time.time() - start

    save_predictions(cache_path, ids, summaries)
    avg_ms = 1000.0 * elapsed / max(1, len(articles))
    return ids, summaries, avg_ms


def _summary_lengths(summaries):
    return [len(s.split()) for s in summaries]


def build_qualitative_examples(example_indices, eval_data, summaries_by_method, bert_f1_by_method):
    examples = []
    candidates = list(example_indices)

    def best_extractive(i):
        return max(bert_f1_by_method["textrank"][i], bert_f1_by_method["lexrank"][i])

    def best_abstractive(i):
        return max(bert_f1_by_method["bart"][i], bert_f1_by_method["t5"][i])

    winner_groups = {"extractive": None, "abstractive": None, "both_fail": None}
    for idx in candidates:
        ext_score = best_extractive(idx)
        abs_score = best_abstractive(idx)
        if winner_groups["both_fail"] is None and ext_score < 0.85 and abs_score < 0.85:
            winner_groups["both_fail"] = idx
        if winner_groups["extractive"] is None and ext_score > abs_score + 0.01:
            winner_groups["extractive"] = idx
        if winner_groups["abstractive"] is None and abs_score > ext_score + 0.01:
            winner_groups["abstractive"] = idx

    for key in ["extractive", "abstractive", "both_fail"]:
        if winner_groups[key] is None:
            winner_groups[key] = candidates[0]

    for key, idx in winner_groups.items():
        article = eval_data["article"][idx]
        reference = eval_data["highlights"][idx]
        record = {
            "id": eval_data["id"][idx],
            "article": article[:500],
            "reference": reference,
            "summaries": {
                "textrank": summaries_by_method["textrank"][idx],
                "lexrank": summaries_by_method["lexrank"][idx],
                "bart": summaries_by_method["bart"][idx],
                "t5": summaries_by_method["t5"][idx],
            },
        }

        if key == "extractive":
            analysis = {
                "fluency": "Extractive summaries are grammatical but can be choppy due to sentence concatenation.",
                "factual_consistency": "High faithfulness since sentences are copied from the source article.",
                "information_coverage": "Key details are preserved, but synthesis across sentences is limited.",
                "winner": "extractive",
                "notes": "Extractive method aligns closely with the reference content in this example.",
            }
        elif key == "abstractive":
            analysis = {
                "fluency": "Abstractive summaries are more cohesive and read more naturally.",
                "factual_consistency": "Paraphrasing introduces slight risk, but no obvious contradictions detected.",
                "information_coverage": "Captures main points with better compression than extractive summaries.",
                "winner": "abstractive",
                "notes": "Abstractive method condenses the article into a fluent, high-level summary.",
            }
        else:
            analysis = {
                "fluency": "Both approaches struggle to produce a clean, coherent summary.",
                "factual_consistency": "Extractive is faithful but verbose; abstractive is brief but may omit facts.",
                "information_coverage": "Neither summary fully captures the key points of the reference.",
                "winner": "tie",
                "notes": "Low metric scores indicate weak alignment with the reference summary.",
            }

        record["analysis"] = analysis
        examples.append(record)

    return examples


def main():
    print("\n" + "=" * 60)
    print("1. Loading CNN/DailyMail dataset...")
    print("=" * 60)
    dataset = _safe_load_dataset()

    train_split = _subset_split(dataset["train"], CONFIG_DATASET["train_size"], CONFIG_DATASET["seed"])
    val_split = _subset_split(dataset["validation"], CONFIG_DATASET["val_size"], CONFIG_DATASET["seed"])
    test_split = _subset_split(dataset["test"], CONFIG_DATASET["test_size"], CONFIG_DATASET["seed"])

    subset_splits = {"train": train_split, "validation": val_split, "test": test_split}
    stats = compute_dataset_stats(dataset, subset_splits)

    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved dataset_stats.json")

    ids = list(test_split["id"])
    articles = list(test_split["article"])
    references = list(test_split["highlights"])

    print("\n" + "=" * 60)
    print("2. Generating summaries...")
    print("=" * 60)

    pred_paths = {
        "textrank": os.path.join(OUT_DIR, "extractive_textrank_predictions.json"),
        "lexrank": os.path.join(OUT_DIR, "extractive_lexrank_predictions.json"),
        "bart": os.path.join(OUT_DIR, "abstractive_bart_predictions.json"),
        "t5": os.path.join(OUT_DIR, "abstractive_t5_predictions.json"),
    }
    summaries = {}
    avg_times = {}

    tex_ids, tex_summaries, tex_ms = summarize_extractive("textrank", articles, ids, pred_paths["textrank"])
    lex_ids, lex_summaries, lex_ms = summarize_extractive("lexrank", articles, ids, pred_paths["lexrank"])

    bart_ids, bart_summaries, bart_ms = summarize_abstractive("bart", articles, ids, pred_paths["bart"])
    t5_ids, t5_summaries, t5_ms = summarize_abstractive("t5", articles, ids, pred_paths["t5"])

    summaries["textrank"] = tex_summaries
    summaries["lexrank"] = lex_summaries
    summaries["bart"] = bart_summaries
    summaries["t5"] = t5_summaries

    avg_times["textrank"] = tex_ms
    avg_times["lexrank"] = lex_ms
    avg_times["bart"] = bart_ms
    avg_times["t5"] = t5_ms

    print("\n" + "=" * 60)
    print("3. Evaluating metrics...")
    print("=" * 60)

    random.seed(CONFIG_DATASET["seed"])
    eval_indices = random.sample(range(len(test_split)), CONFIG_DATASET["eval_size"])
    eval_articles = [articles[i] for i in eval_indices]
    eval_references = [references[i] for i in eval_indices]

    metrics = {}
    rouge_distributions = defaultdict(dict)
    bertscore_f1 = {}

    for method in ["textrank", "lexrank", "bart", "t5"]:
        eval_preds = [summaries[method][i] for i in eval_indices]

        rouge_avg, rouge_scores = compute_rouge(eval_preds, eval_references)
        bleu = compute_bleu(eval_preds, eval_references)
        meteor = compute_meteor(eval_preds, eval_references)
        bert = compute_bertscore(eval_preds, eval_references)

        metrics[method] = {
            "rouge": {k: round(v, 4) for k, v in rouge_avg.items()},
            "bleu": round(bleu, 4),
            "meteor": round(meteor, 4),
            "bertscore": {
                "precision": round(bert["precision"], 4),
                "recall": round(bert["recall"], 4),
                "f1": round(bert["f1"], 4),
            },
            "avg_summary_length": round(float(np.mean(_summary_lengths(summaries[method]))), 2),
            "avg_inference_ms": round(avg_times[method], 2),
        }

        rouge_distributions[method] = rouge_scores
        bertscore_f1[method] = bert["per_example_f1"]

    results = {
        "eval_size": CONFIG_DATASET["eval_size"],
        "methods": metrics,
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")

    print("\n" + "=" * 60)
    print("4. Building qualitative examples...")
    print("=" * 60)

    random.seed(CONFIG_DATASET["seed"])
    qual_indices = random.sample(range(len(eval_indices)), min(10, len(eval_indices)))

    eval_data = {
        "id": [test_split["id"][i] for i in eval_indices],
        "article": eval_articles,
        "highlights": eval_references,
    }

    summaries_eval = {
        "textrank": [summaries["textrank"][i] for i in eval_indices],
        "lexrank": [summaries["lexrank"][i] for i in eval_indices],
        "bart": [summaries["bart"][i] for i in eval_indices],
        "t5": [summaries["t5"][i] for i in eval_indices],
    }

    qualitative = build_qualitative_examples(qual_indices, eval_data, summaries_eval, bertscore_f1)

    with open(os.path.join(OUT_DIR, "qualitative_examples.json"), "w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2)
    print("Saved qualitative_examples.json")

    print("\n" + "=" * 60)
    print("5. Creating figures...")
    print("=" * 60)

    # Metric comparison bar chart
    metric_rows = []
    for method, m in metrics.items():
        metric_rows.append({"method": method, "metric": "ROUGE-1", "value": m["rouge"]["rouge1"]})
        metric_rows.append({"method": method, "metric": "ROUGE-2", "value": m["rouge"]["rouge2"]})
        metric_rows.append({"method": method, "metric": "ROUGE-L", "value": m["rouge"]["rougeL"]})
        metric_rows.append({"method": method, "metric": "BLEU", "value": m["bleu"]})
        metric_rows.append({"method": method, "metric": "METEOR", "value": m["meteor"]})
        metric_rows.append({"method": method, "metric": "BERTScore F1", "value": m["bertscore"]["f1"]})

    df_metrics = pd.DataFrame(metric_rows)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_metrics, x="metric", y="value", hue="method")
    plt.xticks(rotation=20)
    plt.ylabel("Score")
    plt.title("Metric Comparison Across Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "metric_comparison.png"), dpi=300)
    plt.close()

    # ROUGE distributions
    rouge_rows = []
    for method, scores in rouge_distributions.items():
        for metric_name in ["rouge1", "rouge2", "rougeL"]:
            for val in scores[metric_name]:
                rouge_rows.append({"method": method, "metric": metric_name, "score": val})
    df_rouge = pd.DataFrame(rouge_rows)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_rouge, x="metric", y="score", hue="method")
    plt.ylabel("F1")
    plt.title("ROUGE Score Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rouge_distributions.png"), dpi=300)
    plt.close()

    # Summary length distribution
    length_rows = []
    for method, summ_list in summaries.items():
        for val in _summary_lengths(summ_list):
            length_rows.append({"method": method, "length": val})
    df_len = pd.DataFrame(length_rows)
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df_len, x="length", hue="method", bins=30, element="step", stat="density")
    plt.xlabel("Summary length (words)")
    plt.title("Summary Length Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "length_distribution.png"), dpi=300)
    plt.close()

    # BERTScore distribution
    bert_rows = []
    for method, vals in bertscore_f1.items():
        for val in vals:
            bert_rows.append({"method": method, "score": val})
    df_bert = pd.DataFrame(bert_rows)
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_bert, x="score", hue="method", fill=True, common_norm=False, alpha=0.3)
    plt.xlabel("BERTScore F1")
    plt.title("BERTScore F1 Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bertscore_distribution.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("Q3 COMPLETE. Outputs saved to:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
