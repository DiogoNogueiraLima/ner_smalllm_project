# Named Entity Recognition for Brazilian Legal Texts (IF1015)

## 1. Project Context
- Requirement: apply deep learning to NLP or CV with three experimental phases (robustness, interpretability, adversarial attacks).
- This repository covers Deliverable 1: definition of the application plus partial training results.

## 2. Application Definition
- Task: Named Entity Recognition (NER) on Brazilian legal documents.
- Dataset: LeNER-Br (Hugging Face `peluz/lener_br`).
- Entities: ORGANIZACAO, PESSOA, TEMPO, LOCAL, LEGISLACAO, and JURISPRUDENCIA, and the outside tag `O` (BIO format).

## 3. Dataset: LeNER-Br
- Description: Brazilian court decisions annotated with legal entities; manually labeled in BIO format.
- Access: loaded directly from Hugging Face with `datasets.load_dataset("peluz/lener_br")` (use `HF_TOKEN` if gated).
- Split sizes: 
  - Train: 7828 examples
  - Validation: 1177 examples 
  - Test: 1390 examples
- The notebook automatically prints the split sizes and a sentence-length boxplot when executed.

## 4. Model and Approach (Small LM)
- Base model: `neuralmind/bert-base-portuguese-cased` (Portuguese BERT).
- Head: token-classification layer sized to the LeNER-Br label set.
- Tokenization: subword tokenization with label alignment; padding positions use `-100` so they are ignored by the loss.
- Why this model: pretrained on large Portuguese corpora with casing preserved, which helps with legal names and acronyms; the base size (~110M parameters, 12 layers, hidden size 768, 12 attention heads) balances capacity with the compute limits of the course setup.

## 5. Related Work, Baseline, and Project Objective
- **peluz/lener-br (official)**: the classic baseline using LSTM + CRF + character embeddings in TensorFlow; includes the scripts `train.py`, `classScores.py` (token-level), and `evaluate.py` (entity-level). Reported metrics: token-level F1 ~92.5%, entity-level F1 ~86.6%.
- **eraldoluis/LeNER-Br**: a fork that keeps the official model and adds entity-level evaluation using `seqeval`, highlighting the expected drop of ~6–7 F1 points when moving from token-level to entity-level metrics.
- **thiagolrpinho/leNER-Br**: packages the official model into an AILab application with pretrained weights; useful as a deployment reference, but not focused on experimentation or training.
- **Approach we will follow**: a Hugging Face pipeline with `neuralmind/bert-base-portuguese-cased`, aligned tokenization, and a `Trainer` using `seqeval` (entity-level) and micro token-level metrics. This setup simplifies attention inspection, explainability techniques, and robustness/adversarial testing.
- **Baseline and project objective**: keep the official LSTM+CRF model as the literature reference and adopt the fine-tuned BERT base from the notebook as the main baseline of the project. The goal of the course is to provide a reproducible pipeline with token- and entity-level evaluation to support the subsequent robustness, interpretability, and adversarial analyses.


## 6. Training Setup
- Max sequence length: 256.
- Epochs: 3.
- Batch sizes: 8 (train) / 8 (validation).
- Learning rate: 5e-5; weight decay: 0.01.
- Mixed precision: `fp16=True` when GPU is available to speed up training without changing model quality.
- Trainer: Hugging Face `Trainer` with `seqeval` metrics; evaluation and checkpoints each epoch.

## 7. Partial Training Results (Deliverable 1)
Validation metrics from the latest recorded run (checkpoint `notebooks/results_lenerbr/checkpoint-2937`):

| Metric | Validation |
| --- | --- |
| Precision | 0.8804 |
| Recall | 0.8960 |
| F1 | 0.8882 |
| Accuracy | 0.9777 |

Label-wise scores can be printed directly from the notebook after running the training cell (per-entity precision/recall/F1 table).

## 8. Qualitative Example
- The notebook includes a cell under **Qualitative Analysis** that decodes one test sentence and prints a token/tag table.
- Example output placeholder (replace with the table from the notebook run):
  - `Token | Predicted tag` → `<TO FILL AFTER RUNNING THE NOTEBOOK>`

## 9. Repository Structure
- `notebooks/01_treinamento_inicial.ipynb` — main training notebook (cleaned, English documentation, metrics formatting).
- `data/` — optional local data/storage (ignored by git).
- `models/` — saved fine-tuned checkpoints (ignored by git).
- `reports/` — course report drafts or notes.
- `src/` — reserved for future Python modules/utilities.
- `requirements.txt` — project dependencies.

## 10. How to Run
1) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Set your Hugging Face token if required:
   ```bash
   export HF_TOKEN=hf_********************************
   ```
4) Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook notebooks/01_treinamento_inicial.ipynb
   ```

## 11. Next Steps (Robustness, Interpretability, Adversarial Attacks)
- **Robustness:** evaluate sensitivity to noise, paraphrases, and context changes in legal text.
- **Interpretability:** inspect attention patterns and apply explanation methods (e.g., gradient-based or perturbation-based analyses).
- **Adversarial Attacks:** craft perturbed legal inputs to probe failure modes and harden the model.

---
