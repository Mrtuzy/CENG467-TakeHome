# CENG 467 — Take-Home Midterm

Implementation and analysis of five core NLP tasks: text classification, named-entity recognition, summarization, machine translation, and language modeling.

- **Author:** Mert Güden (Student ID 300201013)
- **Course:** CENG 467 — Natural Language Understanding and Generation, IZTECH
- **Report:** [`report/main.tex`](report/main.tex) (compile with `pdflatex` + `bibtex`)

---

## 1. Repository Layout

```
CENG467-TakeHome/
├── 00_MASTER_SPEC.md        # global assignment specification
├── README.md                # this file
├── requirements.txt         # Python dependencies (PyTorch installed separately)
│
├── q1_classification/       # IMDb sentiment classification
│   ├── SPEC.md              #   per-question spec
│   ├── q1_run.py            #   end-to-end training + evaluation
│   └── outputs/             #   metrics, figures, checkpoints
│       ├── results.json
│       ├── preprocessing_ablation.json
│       ├── truncation_ablation.json
│       ├── error_analysis.json
│       └── figures/*.png
│
├── q2_ner/                  # CoNLL-2003 NER (BiLSTM-CRF + BERT)
│   ├── SPEC.md
│   ├── q2_run.py
│   └── outputs/  (results.json, error_analysis.json, figures/)
│
├── q3_summarization/        # CNN/DailyMail (TextRank, LexRank, BART, T5)
│   ├── SPEC.md
│   ├── q3_run.py
│   └── outputs/  (results.json, qualitative_examples.json, figures/)
│
├── q4_translation/          # Multi30k EN→DE (Seq2Seq+Bahdanau, MarianMT)
│   ├── SPEC.md
│   ├── q4_run.py
│   └── outputs/  (results.json, qualitative_example.json, figures/)
│
├── q5_lm/                   # WikiText-2 LM (Trigram-KN, LSTM, Transformer)
│   ├── SPEC.md
│   ├── q5_run.py
│   └── results/  (metrics.json, generated_samples.json, figures/)
│
└── report/                  # LaTeX report
    ├── main.tex
    ├── references.bib
    └── sections/q{1..5}_*.tex
```

---

## 2. Environment Setup

### Prerequisites

- Python **3.10+**
- A CUDA-capable GPU is recommended (the project was developed on an RTX 5060 Ti). CPU-only runs will work for Q1 (TF-IDF) and Q3 (extractive methods) but Q1 BiLSTM/DistilBERT, Q2, Q4, and Q5 will be very slow.

### Install

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. install PyTorch separately, matching your CUDA version.
#    See https://pytorch.org/get-started/locally/. Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. install the rest of the dependencies
pip install -r requirements.txt
```

The first run of each script will download dataset and model weights from HuggingFace into the local cache (`~/.cache/huggingface/`). Q1 and Q2 also download GloVe 6B embeddings (~800 MB) into the respective `outputs/` directory on first run.

---

## 3. Reproducing the Results

All five scripts are deterministic with `seed=42` and write their outputs to the corresponding `outputs/` (or `results/` for Q5) directory. Run each from the repository root:

```bash
# Q1 — IMDb sentiment classification (~10 min on RTX 5060 Ti)
python q1_classification/q1_run.py

# Q2 — CoNLL-2003 NER (~5 min)
python q2_ner/q2_run.py

# Q3 — CNN/DailyMail summarization (~15 min, dominated by BART inference)
python q3_summarization/q3_run.py

# Q4 — Multi30k EN→DE translation (~10 min)
python q4_translation/q4_run.py

# Q5 — WikiText-2 language modeling (~3 min)
python q5_lm/q5_run.py
```

Each script is idempotent: if a checkpoint already exists in `outputs/`, training is skipped and only evaluation/plotting reruns. To retrain from scratch, delete the relevant `*_best.pt` / `*_checkpoint*` files first.

---

## 4. Compiling the Report

```bash
cd report
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Figure paths in the LaTeX sources are relative to `report/`, so each per-question script must have been run at least once before the report compiles.

---

## 5. Headline Results

| Task | Model | Metric | Score |
|-------|--------|--------|-------|
| Q1 — IMDb sentiment | DistilBERT | Accuracy | **0.9296** |
| Q2 — CoNLL-2003 NER | BERT-base-cased | Entity-level F1 | **0.909** |
| Q3 — CNN/DailyMail summarization | BART-large-cnn | ROUGE-1 | **0.4375** |
| Q4 — Multi30k EN→DE | MarianMT (opus-mt-en-de) | BLEU | **36.25** |
| Q5 — WikiText-2 LM | LSTM | Test PPL | **97.00** |

Full metrics for every model in every task are in the corresponding `outputs/results.json` (or `q5_lm/results/metrics.json`).

---

## 6. Notes / Caveats

- The `train_time_seconds` field for the BiLSTM-CRF in `q2_ner/outputs/results.json` is logged as `0.0` because the script reuses a cached checkpoint; treat it as "not measured" rather than as a real timing. See `report/sections/q2_ner.tex` for details.
- All scripts pin `seed=42`. CUDA non-determinism may still cause sub-percent variation between runs on different hardware.
- The Q1/Q2 GloVe download (`glove.6B.zip`, ~800 MB) and the HuggingFace model checkpoints in `q1_classification/outputs/distilbert_checkpoint/` and `q2_ner/outputs/bert_ner_checkpoint/` are intentionally git-ignored.
