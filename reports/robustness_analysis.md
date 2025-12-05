# Robustness Evaluation of a Legal-Domain NER Model (Stage 2)

## Introduction
Robustness in Named Entity Recognition (NER) denotes the model’s ability to maintain performance when inputs are distorted by noise, missing information, or meaning-preserving edits. We evaluate a fine-tuned Portuguese BERT model (token-classification head) trained on the LeNER-Br legal corpus. The goal is to test resilience under controlled perturbations that reflect real legal-text issues such as accent loss, typos, filler tokens, masking, and cautious synonym substitutions.

## Methodology
- **Evaluation data**: First 200 examples from the LeNER-Br validation split. (We use the first 200 validation examples to reduce inference cost and keep perturbation experiments fast and reproducible)
- **Inference and alignment**: `ner_predict` tokenizes pre-split tokens with `is_split_into_words=True`; predictions use argmax over logits and keep one label per original token via `word_ids()`. To compare against gold tags, `evaluate_robustness` pads whichever sequence is shorter with the neutral label `O`, ensuring one-to-one alignment.
- **Perturbation loop**: `evaluate_robustness(dataset, perturb_fn, ...)` applies a perturbation to each example, runs NER, aligns predictions and references, and aggregates metrics.
- **Metrics**: `seqeval` precision, recall, and F1; we report overall F1.
- **Qualitative checks**: The notebook includes inspection of a few validation sentences under accent removal, character noise, and synonym noise to observe token-level prediction shifts.

## Perturbation Scenarios
### Baseline
- **What**: Unmodified validation tokens.
- **Why**: Establish reference performance for comparison.
- **Expected failure modes**: Standard generalization errors on legal entities.
- **Implementation**: Identity function over tokens.

### Accent removal
- **What**: Strip diacritics from every token (NFKD normalization minus combining marks).
- **Why in legal NER**: OCR or keyboard variance often drops accents; legal names and terms must remain recognizable.
- **Expected failure modes**: Changed subword boundaries and collisions between previously distinct forms.
- **Implementation**: Deterministic accent stripping per token.

### Character noise
- **What**: With probability 0.1 per character, replace with a random ASCII letter.
- **Why in legal NER**: Typographical errors and OCR artifacts corrupt surface forms, especially abbreviations and statute references.
- **Expected failure modes**: Broken WordPiece segmentation; unseen subwords leading to mislabels.
- **Implementation**: Token-level loop injecting random character substitutions.

### Word insertion
- **What**: Randomly insert filler tokens (`xxx`, `lorem`, `teste`, `ruido`) after tokens with probability 0.1.
- **Why in legal NER**: Real documents may contain boilerplate, OCR debris, or concatenated text that interrupts entity cues.
- **Expected failure modes**: Attention diluted by irrelevant tokens; entity spans fragmented or shifted.
- **Implementation**: Sequential pass inserting sampled noise tokens.

### Token masking
- **What**: Replace tokens with `[MASK]` at probability 0.15.
- **Why in legal NER**: Missing words (redactions, truncation) force reliance on context alone.
- **Expected failure modes**: Loss of anchor words inside entity spans; reliance on surrounding context may be insufficient.
- **Implementation**: Use tokenizer mask token; random masking per token.

### Synonym substitution
- **What**: Replace selected legal-domain tokens with curated synonyms (e.g., `TRIBUNAL→CORTE`, `MINISTRO→AUTORIDADE`, `LEI→NORMA`), including accentless variants.
- **Why in legal NER**: Legal texts use interchangeable terminology; robustness to wording changes tests semantic generalization.
- **Expected failure modes**: Over-reliance on memorized lexical anchors; possible mislabeling if context weak.
- **Implementation**: Deterministic lookup map applied token by token.

### Note on Synonym Substitution Methodology
We opted for a manually curated synonym map to ensure **full control, reproducibility, and meaning preservation**. Due to the legal nature of the task, where terminology is highly specific, conservative, and semantically rigid, automatic alternatives such as WordNet, spaCy, or LLM-based generation often produce domain-inaccurate or non-deterministic substitutions. Such outputs would compromise the validity of the robustness analysis. Manual curation guarantees **minimal, stable, and legally coherent** perturbations that align with real variation observed in legal texts.


Manual selection allows us to apply **minimal, stable, and semantically coherent** perturbations, making it possible to isolate the model’s sensitivity to lexical variation in legal texts.

## Results
| Scenario | F1 Score |
| --- | --- |
| Baseline | 0.866373 |
| Accent removal | 0.823529 |
| Char noise (p=0.1) | 0.652632 |
| Word insertion | 0.102339 |
| Masking (p=0.15) | 0.730539 |
| Synonym noise | 0.845697 |

**Comparative analysis**: The smallest drops occur under accent removal (≈4.3 p.p.) and synonym noise (≈2.1 p.p.), indicating stability to meaning-preserving or minor orthographic shifts. Character noise and masking show moderate-to-large degradation, revealing sensitivity to surface-form corruption and missing anchors. Word insertion collapses performance (≈76 p.p. drop), exposing structural vulnerability when irrelevant tokens disrupt sequence structure.

## Detailed Discussion
- **Accent removal hurts slightly**: WordPiece often maintains similar subword structure when accents are stripped, so most lexical cues survive; errors emerge when accent loss alters morpheme boundaries or merges forms.
- **Synonym substitution remains strong**: Context plus overlapping legal semantics let the model map `CORTE`, `AUTORIDADE`, or `NORMA` back to the correct entity types, showing that representations capture more than memorized strings.
- **Character noise breaks subword boundaries**: Even 10% corruption yields unfamiliar subwords, degrading embeddings and cascading to label errors—typical for WordPiece tokenizers tuned to clean text.
- **Masking preserves moderate performance**: With 15% masking, surrounding context can often infer entities, but masking inside entity spans weakens precision/recall because anchor tokens vanish.
- **Word insertion destroys performance**: Extra tokens shift attention and entity position signals; without training-time exposure to clutter, the Transformer attends to noise and loses span coherence.
- **Architectural link**: Transformer self-attention plus WordPiece favors stable lexical anchors and consistent token order; perturbations that destabilize anchors (noise, masking) or sequence structure (insertions) are most damaging, while semantically aligned substitutions are tolerable.

## Hypotheses and Future Improvements
1. **Targeted data augmentation** with accent-stripped, character-noised, and insertion-noised variants to teach invariances observed as weaknesses.
2. **Entity-aware masking strategies** that mask outside entities during fine-tuning to strengthen contextual reasoning without erasing key anchors.
3. **Byte-level or character-aware tokenizers** to reduce brittleness to typos and accent loss.
4. **Adversarial training** using the same perturbations (especially insertions and char noise) to inoculate the model against structural and orthographic noise.
5. **Regularization techniques** (e.g., dropout on attention or embeddings) tuned for robustness to clutter.
6. **Robustness-oriented fine-tuning** combining MLM and NER objectives so masked-token recovery reinforces entity predictions.
7. **Contrastive learning** between clean and perturbed pairs to align representations and discourage spurious sensitivity to surface noise.

## Conclusion
The legal-domain Portuguese BERT NER model is resilient to meaning-preserving changes (synonyms) and moderately tolerant to accent loss, but it is vulnerable to character corruption, token masking inside entities, and especially to structural noise from word insertion. These findings map a clear agenda for robustness-focused improvements—augmentations, tokenizer choices, and adversarial or contrastive training—that will inform the upcoming interpretability (stage 3) and adversarial attack (stage 4) phases.
