# NER for Brazilian Legal Texts (LeNER-Br)

## Overview
Small-language-model project for Named Entity Recognition on Brazilian legal documents. We fine-tune Portuguese BERT on the LeNER-Br dataset and will evaluate robustness, interpretability, and adversarial attacks as the course progresses (IF1015).

## Dataset: LeNER-Br
- Brazilian court decisions annotated in BIO format with entities ORGANIZACAO, PESSOA, TEMPO, LOCAL, LEGISLACAO, and JURISPRUDENCIA, plus `O`.
- Loaded via Hugging Face: `datasets.load_dataset("peluz/lener_br")` (set `HF_TOKEN` if gated).
- Split sizes are printed by the notebook when run (train/validation/test).

## Model
- Base checkpoint: `neuralmind/bert-base-portuguese-cased` (~110M parameters, 12-layer BERT base).
- Token classification head sized to the LeNER-Br label set.
- Subword tokenization with label alignment; padding positions use `-100` so they do not affect loss.

## Training Pipeline
- Max sequence length 256; learning rate 5e-5; weight decay 0.01.
- Epochs: 3; batch sizes: 8 (train) / 8 (validation); mixed precision when GPU is available.
- Hugging Face `Trainer` with `seqeval` metrics (precision, recall, F1, accuracy) evaluated each epoch.
- Outputs: checkpoints in `notebooks/results_lenerbr/` and a saved model in `models/lenerbr_bert_base`.

## Current Results
Validation metrics from the latest recorded run (`notebooks/results_lenerbr/checkpoint-2937`):

| Metric | Validation |
| --- | --- |
| Precision | 0.8804 |
| Recall | 0.8960 |
| F1 | 0.8882 |
| Accuracy | 0.9777 |

Run the notebook to reproduce and view per-entity scores and qualitative predictions.

## Quickstart
1) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Set your Hugging Face token if required:
```bash
export HF_TOKEN=hf_********************************
```
4) Launch and run the training notebook:
```bash
jupyter notebook notebooks/01_treinamento_inicial.ipynb
```

## Repository Layout
- `notebooks/01_treinamento_inicial.ipynb` — end-to-end training and evaluation notebook.
- `notebooks/results_lenerbr/` — training checkpoints from past runs.
- `models/` — exported fine-tuned model and tokenizer.
- `reports/` — course deliverable write-ups.
- `src/` — reserved for future utility scripts.
- `requirements.txt` — dependencies.

## Roadmap
- Robustness: noise/context perturbations in legal text.
- Interpretability: attention inspection and explanation methods.
- Adversarial attacks: targeted perturbations to probe failures and harden the model.
