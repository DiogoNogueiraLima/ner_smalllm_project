# Named Entity Recognition for Brazilian Legal Texts (IF1015)

## 1. Project Context and Course
- Course: IF1015 - Advanced Topics in Information Systems 6.
- Requirement: apply deep learning to NLP or CV with three experimental phases (robustness, interpretability, adversarial attacks).
- This repository covers Deliverable 1: definition of the application plus partial training results.

## 2. Application Definition
- Task: Named Entity Recognition (NER) on Brazilian legal documents.
- Dataset: LeNER-Br (Hugging Face `peluz/lener_br`).
- Entities: PERSON, ORGANIZATION, LOCATION, TIME, LEGISLACAO, JURISPRUDENCIA, and the outside tag `O` (BIO format).

## 3. Dataset: LeNER-Br
- Description: Brazilian court decisions annotated with legal entities; manually labeled in BIO format.
- Access: loaded directly from Hugging Face with `datasets.load_dataset("peluz/lener_br")` (use `HF_TOKEN` if gated).
- Split sizes:  
  - Train: `<TO FILL AFTER DATA DOWNLOAD>`  
  - Validation: `<TO FILL AFTER DATA DOWNLOAD>`  
  - Test: `<TO FILL AFTER DATA DOWNLOAD>`  
- The notebook automatically prints the split sizes and a sentence-length boxplot when executed.

## 4. Model and Approach (Small LM)
- Base model: `neuralmind/bert-base-portuguese-cased` (Portuguese BERT).
- Head: token-classification layer sized to the LeNER-Br label set.
- Tokenization: subword tokenization with label alignment; padding positions use `-100` so they are ignored by the loss.

## 5. Training Setup
- Max sequence length: 256.
- Epochs: 3.
- Batch sizes: 8 (train) / 8 (validation).
- Learning rate: 5e-5; weight decay: 0.01.
- Mixed precision: `fp16=True` when GPU is available.
- Trainer: Hugging Face `Trainer` with `seqeval` metrics; evaluation and checkpoints each epoch.

## 6. Partial Training Results (Deliverable 1)
Validation metrics from the latest recorded run (checkpoint `notebooks/results_lenerbr/checkpoint-2937`):

| Metric | Validation |
| --- | --- |
| Precision | 0.8733 |
| Recall | 0.8891 |
| F1 | 0.8811 |
| Accuracy | 0.9765 |

Label-wise scores can be printed directly from the notebook after running the training cell (per-entity precision/recall/F1 table).

## 7. Qualitative Example
- The notebook includes a cell under **Qualitative Analysis** that decodes one test sentence and prints a token/tag table.
- Example output placeholder (replace with the table from the notebook run):
  - `Token | Predicted tag` → `<TO FILL AFTER RUNNING THE NOTEBOOK>`

## 8. Repository Structure
- `notebooks/01_treinamento_inicial.ipynb` — main training notebook (cleaned, English documentation, metrics formatting).
- `data/` — optional local data/storage (ignored by git).
- `models/` — saved fine-tuned checkpoints (ignored by git).
- `reports/` — course report drafts or notes.
- `src/` — reserved for future Python modules/utilities.
- `requirements.txt` — project dependencies.

## 9. How to Run
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

## 10. Next Steps (Robustness, Interpretability, Adversarial Attacks)
- **Robustness:** evaluate sensitivity to noise, paraphrases, and context changes in legal text.
- **Interpretability:** inspect attention patterns and apply explanation methods (e.g., gradient-based or perturbation-based analyses).
- **Adversarial Attacks:** craft perturbed legal inputs to probe failure modes and harden the model.

---
