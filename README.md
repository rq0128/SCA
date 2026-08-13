# SCA

**A framework for detecting tool poisoning attacks.**

SCA Detector detects **description injection attacks** against tools (e.g. MCP / LLM tools). In such an attack, a malicious actor injects hidden instructions or unrelated content into a tool's natural-language description while the code itself looks normal. SCA Detector catches this by checking whether **every sentence in the description is actually backed by the code** — sentences with no code support are flagged as injected/poisoned.

It is built on [UniXcoder](https://github.com/microsoft/unixcoder) and combines static analysis (Tree-sitter), NLP parsing (spaCy), MMD-based code segmentation, and a fine-tuned Cross-Encoder consistency model.

## How It Works

1. **Parse the code** into semantic features (imports, signatures, API calls, etc.).
2. **Parse the description** into atomic sentences with spaCy.
3. **Score consistency** between every description sentence and every code feature using the Cross-Encoder.
4. **Decide**: a sentence whose best support score falls below the threshold is treated as injected content. Any such sentence marks the tool as poisoned.

## Repository Structure

```
SCA/
├── detector/            # Core source code
│   ├── config.py        # Configuration
│   ├── code_ast.py      # Tree-sitter semantic feature extractor
│   ├── code_parser.py   # MMD code segmenter
│   ├── desc_parser.py   # spaCy description parser
│   ├── model.py         # UniXcoder Cross-Encoder
│   ├── dataset.py       # PyTorch dataset
│   ├── process_data.py  # Training data generation
│   ├── train.py         # Training pipeline
│   ├── detector.py      # PoisonDetector + batch evaluation
│   └── demo.py          # Explainable single-case demo
├── data/                # Datasets (CSV)
└── model/               # Saved checkpoints
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start 

Use this path when you **already have a trained checkpoint**.

1. Place your checkpoint at `model/best_cross_encoder.pth`.
2. Run the demo, which prints a per-sentence support table for one tool:

   ```bash
   cd detector
   python demo.py
   ```


**Step 1 — Prepare data.** Create a CSV under `data/` with columns `description`, `code`, `label` (`1` = benign, `0` = poisoned). Only benign tools are used as ground truth for positive pairs.

**Step 2 — Generate training pairs.** This aligns each description sentence with its best-matching code feature and samples hard negatives:

```bash
cd detector
python process_data.py
```

Edit the paths at the bottom of `process_data.py` to point to your input/output CSVs (input e.g. `../data/train_data_clean.csv`, output `../data/aligned_train_data_new.csv`).

**Step 3 — Train.**

```bash
python train.py
```

This fine-tunes the Cross-Encoder and saves the best checkpoint (by validation F1) to `model/best_cross_encoder_final.pth`. Hyperparameters live in `detector/config.py`.

## Evaluation

```bash
python detector.py
```

Update `test_csv_path` inside `detector.py` to your test CSV first. It reports **Precision**, **Recall**, and **F1-Score** over the whole dataset.

## Data Format

| Column | Type | Description |
|--------|------|-------------|
| `description` | str | Natural-language description / docstring of the tool |
| `code` | str | Full source code of the tool |
| `label` | int | `1` = benign, `0` = poisoned |

Optional `id` / `tool_id` columns are used during evaluation to identify tools in reports.
