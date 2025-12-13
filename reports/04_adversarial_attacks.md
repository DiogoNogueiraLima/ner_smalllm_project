# Adversarial Attacks Evaluation of a Legal-Domain NER Model (Stage 4)

## 1. Introduction
Adversarial evaluation probes how a fine-tuned Portuguese BERT token-classification
model (LeNER-Br) fails under *targeted* perturbations in legal text. Unlike Stage 2
robustness analysis, which focused on random or generic noise,Stage 4 explicitly
crafts attacks aimed at entity anchors, boundaries, and local context.

The objective is to expose failure modes that are consistent with Stage 3
interpretability findings, which showed strong reliance on lexical anchors
(locations, dates, courts) and punctuation-based span delimiters.

## 2. Methodology

### Data
- LeNER-Br validation subset: `select(range(200))`.

### Inference
- Sliding-window tokenization (`max_length=256`, `stride=64`,
  `return_overflowing_tokens=True`) to avoid truncation in long legal sentences.
- Logits are aggregated at the **word level** using `word_ids()`.
- Model is run in `eval()` mode for deterministic outputs.
- Sanity checks confirm that the aggregated logits match the number of original tokens.

### Span sources
- **Oracle**: gold spans (worst-case vulnerability).
- **Pred-guided**: baseline model predictions (realistic attacker). (Pred-guided attacks approximate a realistic adversary that exploits the model’s own inductive biases rather than oracle knowledge.)

### Insertion handling
Structural insertion is evaluated in two complementary ways:
1. **Projected view**: predictions are mapped back to original token positions
   to enable comparison with baseline labels.
2. **True structural view**: evaluation is performed directly in the attacked
   sequence, measuring entity retention and confidence drops without projection.
   This avoids masking structural damage.

### Metrics
- seqeval **precision / recall / F1**
- ΔF1 vs baseline
- **entity_flip_rate**: gold ≠ O, baseline correct → attacked wrong
- **span_miss_rate**: gold spans not exactly recovered
- **span_token_error_rate**: at least one wrong token inside a gold span
- **confidence drops**:
  - `conf_drop_gold`: drop in gold-label logit
  - `conf_drop_pred`: drop in baseline-predicted label logit
- **Per-entity F1** (LOCAL, TEMPO, PESSOA, etc.)

For *true insertion*, additional metrics are used:
- **entity_retention_rate_true_insertion**
- **conf_drop_gold_true_insertion**

Metrics that are not meaningful for a given evaluation mode are reported as **NaN**.

## 3. Attack Scenarios (oracle & pred-guided)

- **char_typo (p=0.5)**: character swap/replace/delete inside entity spans.
- **boundary**: replace punctuation adjacent to entity spans with `[MASK]`
  (tokenizer mask token), preserving sequence length and avoiding artificial tokens.
- **context_distractors**: replace tokens immediately before/after spans with
  distractors (`xxx`, `lorem`, `teste`, `ruido`).
- **synonym**: conservative legal-domain substitutions
  (e.g., `TRIBUNAL→CORTE`, `MINISTRO→AUTORIDADE`).
- **insert (p=0.5)**: insert distractors before/after entity spans (length-changing),
  evaluated with projection and true structural metrics.

## 4. Results

### Length-preserving attacks
Baseline F1 (seqeval): **0.86**

| attack | precision | recall | f1 | delta_f1 | entity_flip_rate | span_miss_rate | span_token_error_rate | conf_drop_gold | conf_drop_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| char_typo_oracle | 0.705 | 0.718 | 0.712 | -0.151 | 0.151 | 0.288 | 0.279 | 0.996 | 1.012 |
| char_typo_pred | 0.740 | 0.752 | 0.746 | -0.117 | 0.112 | 0.248 | 0.227 | 0.865 | 0.943 |
| context_distractors_oracle | 0.775 | 0.721 | 0.747 | -0.115 | 0.121 | 0.279 | 0.276 | 0.799 | 0.746 |
| context_distractors_pred | 0.785 | 0.697 | 0.738 | -0.124 | 0.127 | 0.303 | 0.300 | 0.888 | 0.851 |
| boundary_oracle | 0.813 | 0.882 | 0.846 | -0.017 | 0.007 | 0.118 | 0.103 | 0.031 | 0.046 |
| boundary_pred | 0.839 | 0.882 | 0.860 | -0.003 | 0.007 | 0.118 | 0.103 | 0.039 | 0.038 |
| synonym_oracle | 0.830 | 0.885 | 0.856 | -0.006 | 0.006 | 0.115 | 0.100 | 0.053 | 0.039 |
| synonym_pred | 0.830 | 0.885 | 0.856 | -0.006 | 0.006 | 0.115 | 0.100 | 0.045 | 0.045 |

### Insertion — true structural view (no projection)

| attack | precision | recall | f1 | entity_retention_rate | conf_drop_gold_true_insertion |
| --- | --- | --- | --- | --- | --- |
| insert_true_oracle | 0.784 | 0.836 | 0.809 | 0.433 | 0.102 |
| insert_true_pred | 0.810 | 0.839 | 0.825 | 0.442 | 0.066 |

**Note on NaNs**: metrics such as ΔF1, entity_flip_rate, and span-based errors are
not defined for the true structural view because predictions are evaluated in the
attacked sequence rather than projected back to original token positions.

## 5. Cross-Stage Consistency

- **Character typos** cause large F1 and confidence drops, consistent with
  Stage 2 character-noise fragility and Stage 3 gradient/IG sensitivity to anchors.
- **Synonym substitutions** remain largely benign, aligning with Stage 2 robustness
  and Stage 3 findings that semantic anchors still activate correctly.
- **Boundary perturbations** have mild impact, matching interpretability results
  showing punctuation as helpful but secondary cues.
- **Context distractors** substantially degrade performance, echoing Stage 2
  word-insertion vulnerability and Stage 3 reliance on concentrated attention.
- **Insertion (true structural view)** reveals a critical insight: although
  projected F1 remains moderate, **only ~43–44% of gold entities are structurally
  recovered**, demonstrating that standard F1 can mask severe span-level damage.

## 6. Key Findings
- Anchor-corrupting attacks (char_typo) and context dilution attacks
  (context_distractors, insertion) are the most harmful.
- Structural insertion does not necessarily collapse F1 but severely degrades
  entity integrity, highlighting the limits of aggregate metrics.
- Pred-guided attacks are weaker than oracle attacks but still expose systematic
  vulnerabilities, indicating practical exploitability.

## 7. Limitations
- Attacks are heuristic and rule-based (not gradient-optimized).
- Confidence drops are logit-based and not calibrated probabilities.
- No defensive retraining or adversarial mitigation is evaluated in this stage.

## 8. Conclusion and Next Steps
Adversarial evaluation confirms that the model relies heavily on a small set of
lexical anchors and structural cues. Corrupting anchors or introducing structural
clutter degrades entity integrity in ways that may not be visible through F1 alone.
These findings, consistent with robustness and interpretability analyses, motivate
future work on targeted augmentation, adversarial training, and alternative
tokenization strategies (character- or byte-level) to improve legal-domain NER
robustness. This highlights a fundamental evaluation gap in NER benchmarks: high F1 
does not guarantee structural correctness under distributional shift.
