# Interpretability Analysis of a Legal-Domain NER Model

## 1. Introduction
Interpretability in token classification clarifies which input tokens drive entity
predictions and how contextual evidence is combined. For legal-domain NER,
transparency is key for understanding whether the model relies on meaningful
legal cues (e.g., court names, dates, statutes) or brittle surface patterns.
The notebook `notebooks/03_interpretability.ipynb` applies multiple explanation
methods—attention visualizations, gradient-based saliency, Integrated Gradients,
and occlusion sensitivity—to inspect a fine-tuned Portuguese BERT on a
representative validation sentence.

## 2. Methods
- **Attention Visualization**: Plots both per-head heatmaps and averaged
  attention in the final layer. Query-based inspection (e.g., from “Brasilia” or
  “2018”) reveals how entity anchors distribute their attention.
- **Gradient-Based Saliency**: Backpropagates from the maximum logit to input
  embeddings; absolute gradients per token estimate sensitivity to small perturbations.
- **Integrated Gradients**: Integrates gradients along a path from a zero baseline
  to the input, reducing noise and stabilizing attribution magnitudes.
- **Occlusion Sensitivity**: Masks one token at a time and measures the change
  in overall model confidence (sum of max logits). This directly tests the causal
  importance of each token to the final prediction.

## 3. Results
- **Example and predictions** (validation subset):
Brasilia ( DF ) , 15 de Março de 2018 .
→
B-LOCAL, I-LOCAL, I-LOCAL, I-LOCAL, O,
B-TEMPO, I-TEMPO, I-TEMPO, I-TEMPO, I-TEMPO, I-TEMPO, O

- **Attention patterns** (Figures 1–3):  
Heatmaps highlight strong focus around the LOCATION span (“Brasilia”, “DF”) and
around the temporal region (“15”, “Março”, “2018”), while filler tokens receive
little attention.

- **Saliency (gradient) scores** (Figure 4):  
Highest sensitivity for:
- `2018`  
- `Março`  
- `.`  
- `Brasilia`  
- `,`  
→ These gradients suggest that small perturbations to temporal and location
anchors would strongly affect the logits.

- **Integrated Gradients** (Figure 5):  
Highest IG scores for:
- `2018`  
- `Março`  
- `.`  
- `(`  
- `,`  
→ IG smooths raw gradients but preserves the same anchor ordering.

- **Occlusion sensitivity** (Figure 6):  
Masking LOCATION tokens (`Brasilia`, `DF`) and boundary punctuation
produces the largest confidence drops. Temporal tokens (`Março`, `2018`)
remain relevant but show **smaller** causal impact under occlusion than under
gradient-based methods.

- **Consistent patterns and divergences**:
- All methods highlight meaningful entities, but **relative importance differs**.  
- Gradients and IG emphasize **TIME tokens** more strongly.  
- Occlusion emphasizes **LOCATION tokens** and punctuation boundaries.  
- This complementarity indicates the model uses multiple anchor types, but
  LOCATION tokens carry more *causal* weight in this example.

## 4. Cross-Method Consistency Analysis
- Attention, saliency, IG, and occlusion all identify entity spans as the most
influential regions.
- **Gradient-based methods** emphasize temporal anchors (`Março`, `2018`)
due to high derivative sensitivity.
- **Occlusion**, however, assigns stronger causal importance to LOCATION tokens
(`Brasilia`, `DF`) and punctuation around span boundaries.
- This partial disagreement is expected:
- Gradients measure *infinitesimal sensitivity*,  
- Occlusion measures *causal necessity*,  
- Attention shows *information flow*, not causality.
- Together, the methods reveal that the model relies on both LOCATION and TIME
anchors but may depend more on LOCATION tokens for stable classification
confidence.

## 5. Link to Robustness Findings
The robustness notebook showed:
- **Large drops** under character noise and masking,  
- **Severe collapse** under word insertion,  
- **Minor impact** under synonym substitution and accent removal.

Interpretability explains why:
- Gradient and IG highlight brittle sensitivity to specific surface anchors,
matching the strong F1 degradation under character corruption.
- Occlusion indicates reliance on LOCATION and punctuation boundaries,
explaining vulnerability when tokens are masked.
- Word insertion disrupts structural patterns the attention maps rely on,
consistent with severe performance collapse.
- Synonym and accent changes preserve semantic anchors enough for attention and
gradients to still activate correctly, matching the mild robustness impact.

## 6. Limitations of the Interpretability Methods
- Subword duplication can split contributions across tokens.
- Gradient methods can exaggerate noisy or correlated tokens (e.g., punctuation).
- Integrated Gradients depends on the baseline choice.
- Occlusion is computationally heavy and can yield small negative values when
removing distracting context.
- Attention is correlational and does not guarantee causal importance.

## 7. Conclusion
- Across all methods, the model consistently focuses on legally meaningful
anchors—especially LOCATION (`Brasilia`, `DF`) and TIME (`15`, `Março`, `2018`)—
with punctuation serving as structural cues.
- Differences between IG/saliency and occlusion highlight complementary aspects
of “importance”: **sensitivity vs. causality**.
- These findings reinforce the robustness results and suggest training strategies
such as noise-aware augmentation, adversarial perturbations, and refined
tokenization to reduce excessive reliance on brittle surface forms.

