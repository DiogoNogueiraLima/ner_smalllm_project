# Interpretability Analysis of a Legal-Domain NER Model

## 1. Introduction

Interpretability in token classification aims to clarify **which input tokens drive entity predictions** and whether the model relies on **meaningful semantic anchors** or on brittle surface-level patterns.
In legal-domain NER, this transparency is particularly important, as models are expected to focus on **legally relevant cues** (e.g., court names, locations, dates, statutes) rather than on punctuation, formatting artifacts, or spurious correlations.

Earlier analyses relied mainly on **qualitative examples**, which raised the question of whether observed patterns generalized beyond a few hand-picked sentences.
To address this, we extend the original interpretability analysis with **dataset-level quantitative metrics** that explicitly measure how often **gold entity tokens** receive high importance scores.

The notebook `notebooks/03_interpretability.ipynb` therefore combines:

* qualitative visualization (attention, saliency, IG, occlusion), and
* quantitative evaluation via **EntityFocus@k** and **Lift@k**, computed across the validation set.

---

## 2. Methods

### 2.1 Interpretability Techniques

The following complementary interpretability methods are applied to a fine-tuned Portuguese BERT model on the LeNER-Br dataset:

* **Attention Visualization**
  Heatmaps at the word level (per-head and last-layer average) are used to inspect how entity anchor tokens distribute attention. Query-based plots (e.g., from “Brasília” or “2018”) illustrate information flow.

* **Gradient-Based Saliency**
  Gradients of the predicted labels’ logits with respect to input embeddings are computed. The absolute gradient magnitude per token estimates **local sensitivity** to perturbations.

* **Integrated Gradients (IG)**
  Gradients are integrated from a zero-embedding baseline to the input, reducing noise and stabilizing attribution values.

* **Occlusion Sensitivity**
  Each token is masked individually and the resulting drop in sequence-level confidence (sum of max logits) is measured. This provides a **causal importance signal**.

### 2.2 Dataset-Level EntityFocus@k Metric

To move beyond anecdotal examples, we introduce a quantitative metric computed over many validation sentences:

* **EntityFocus@k**: proportion of tokens among the top-k most important tokens that correspond to **gold entity tokens**, as defined by the dataset annotations (all tags ≠ `O`).
* **Lift@k**: EntityFocus@k normalized by the **entity ratio in the text**, measuring how much more often entities appear among important tokens than expected by chance.

Crucially:

* Importance scores are computed **without access to gold labels**.
* Gold NER annotations are used **only for evaluation**, ensuring that the analysis verifies alignment rather than assuming it.

---

## 3. Qualitative Results (Single-Example Analysis)

### 3.1 Example and Predictions

Validation sentence:

```
Brasilia ( DF ) , 15 de Março de 2018 .
```

Predicted labels:

```
B-LOCAL, I-LOCAL, I-LOCAL, I-LOCAL, O,
B-TEMPO, I-TEMPO, I-TEMPO, I-TEMPO, I-TEMPO, O
```

The model correctly identifies:

* *“Brasília ( DF )”* as a **LOCATION** entity, and
* *“15 de Março de 2018”* as a **TIME** entity.

### 3.2 Attention Patterns

Attention heatmaps consistently show high concentration around:

* LOCATION tokens (“Brasília”, “DF”), and
* TIME anchors (“15”, “Março”, “2018”),
  with punctuation acting as boundary markers.
  Function words receive substantially less attention.

### 3.3 Gradient-Based Saliency and Integrated Gradients

Both saliency and IG highlight:

* strong sensitivity to temporal anchors (“Março”, “2018”), and
* secondary importance for location tokens and boundary punctuation.

This indicates that **small perturbations to these anchors would strongly affect predictions**.

### 3.4 Occlusion Sensitivity

Occlusion reveals a complementary picture:

* Masking LOCATION tokens (“Brasília”, “DF”) causes the **largest confidence drops**.
* Temporal tokens remain important but exhibit smaller causal impact than suggested by gradients.
* Function words contribute minimally.

This distinction illustrates the difference between **sensitivity** (gradients) and **causal necessity** (occlusion).

---

## 4. Cross-Method Consistency

Across attention, saliency, IG, and occlusion:

* Entity spans are consistently identified as the most influential regions.
* Methods differ in *how* importance is expressed:

  * Gradients emphasize TIME anchors,
  * Occlusion emphasizes LOCATION anchors and boundaries,
  * Attention highlights information flow rather than causality.

These differences are expected and complementary, revealing that the model relies on **multiple types of entity-defining anchors**.

---

## 5. Quantitative Dataset-Level Findings

To verify whether these observations generalize, we compute EntityFocus@6 and Lift@6 on the validation set.

### 5.1 Saliency-Based Results (200 validation sentences with entities)

* **EntityFocus@6**:
  mean **0.46**, median **0.37**, std **0.30**
* **Lift@6**:
  mean **2.58**, median **2.20**, std **2.01**

Interpretation:

* On average, nearly half of the top-6 most important tokens are **gold entity tokens**, despite entities occupying a much smaller fraction of the text.
* A Lift substantially above 1 confirms **disproportionate importance assigned to entities**, not explained by frequency alone.

### 5.2 Occlusion-Based Results (30 short sentences)

* **EntityFocus@6**:
  mean **0.49**, median **0.50**
* **Lift@6**:
  mean **1.48**, median **1.67**

Interpretation:

* Occlusion confirms that **removing entity tokens causes larger confidence drops** than removing non-entity tokens.
* The lower lift compared to saliency reflects the stricter causal nature of occlusion.

---

## 6. Link with Robustness Findings

The interpretability results explain the robustness behavior observed previously:

* **Character noise** causes severe degradation because the model relies heavily on exact surface forms of entity anchors highlighted by saliency and IG.
* **Masking** is harmful when it removes entity anchors identified as causally important by occlusion.
* **Word insertion** disrupts structural and positional patterns that attention relies on, leading to catastrophic performance collapse.
* **Accent removal and synonym substitution** preserve semantic identity and thus maintain attention and gradient activation, resulting in mild performance drops.

---

## 7. Limitations

* Subword tokenization can distribute importance across multiple subtokens.
* Gradient-based methods may overemphasize punctuation or correlated tokens.
* Integrated Gradients depends on baseline choice.
* Occlusion is computationally expensive and restricted to short sentences.
* Attention is correlational and not a direct measure of causality.

---

## 8. Conclusion

* Dataset-level **EntityFocus@6 and Lift@6** provide quantitative evidence that the model concentrates importance on **gold entity tokens**, not merely on arbitrary or frequent tokens.
* Qualitative attention, saliency, IG, and occlusion analyses align with these findings and explain observed robustness weaknesses.
* Together, these results demonstrate that the model’s behavior is **entity-driven**, while also revealing over-reliance on brittle surface anchors.
* Future work could mitigate this via noise-aware augmentation, adversarial training, or richer tokenization strategies to improve robustness without sacrificing interpretability.
