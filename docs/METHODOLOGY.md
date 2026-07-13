# Methodology

A compact, self-contained statement of the design — written to map directly onto
a paper's Methods section. The guiding principle is *minimal machinery for a
clean causal claim*: nothing here is included unless it removes a specific
confound.

## Claim under test

Coherence collapse in small emergently-misaligned models is either a **downstream
symptom** of misalignment (Story A) or a **coincidental** instability (Story B).
We adjudicate by intervention, not correlation.

## Measurements

| Quantity | Instrument | Design reason |
|---|---|---|
| Emergent misalignment (existence) | Betley **multiple-choice** (label-token log-prob) + **pivotal-token** | validated, generation-free; label-token reading is the faithful Betley instrument |
| Internal misalignment level | projection onto a **persona-vector direction** built once on the base model | not fit to the checkpoints it measures (no circularity) |
| **Coherence (outcome)** | **content-agnostic** statistics: neutral-model perplexity, seq-rep-n / distinct-n / loop-onset (degeneration), n-gram entropy / TTR (diversity), parse-error rate, embedding topical-drift | harm-blind by construction — cannot conflate "harmful" with "incoherent" |
| Alignment/harm | LLM judge (harm axis only) + the log-prob instrument | judging harm is the judge's proper job; cross-checked by a non-LLM signal |

**The one invariant.** No LLM ever judges coherence. An LLM coherence judge could
rate harmful-but-fluent text as "incoherent," manufacturing the correlation we
test. Coherence is measured by harm-blind statistics and validated against blind
human fluency ratings.

## The intervention (Stage 2, headline)

Steer the trained model along the misalignment direction at strength `alpha`
(`h <- h + alpha * d` at a fixed layer) across a symmetric grid, and measure the
coherence outcome. Three orthogonal checks separate a genuine directional effect
from a generic-perturbation artifact:

1. **Dose-response + sign-asymmetry (H3a).** H1 predicts coherence falls as
   `alpha` increases toward misalignment and is *preserved/improved* when steering
   away. Generic perturbation is sign-*symmetric*; a signed asymmetry is the clean
   discriminator. Statistic: `drop_pos - drop_neg` relative to the `alpha=0`
   baseline.
2. **Effect-size-matched control directions (H3b).** We repeat the sweep along a
   random direction and an unrelated-trait direction, each **scaled to induce the
   same behavioral output perturbation** as the misalignment direction
   (token-disagreement effect size), not merely matched in vector norm — because a
   real direction is more potent per unit norm. Specificity = the misalignment
   direction hurts coherence more than equally-perturbing controls.
3. **No-injection convergence (H3c, Stage 3).** During ordinary finetuning (no
   steering vector injected), does the misalignment projection co-move with
   coherence degradation? A yes here cannot be a steering artifact (nothing was
   injected), and combined with the intervention defeats both the artifact and the
   coincidence explanations.

## Inference

- **Seed is the unit of replication.** We fit the dose-response *per seed* and test
  the seed-level statistics across seeds with a one-sample **Wilcoxon signed-rank**
  (sign-asymmetry > 0; slope < 0; specificity margin > 0). We do **not** pool
  observations across seeds into a single fit. Tiny-n is the expected regime, so
  p-values are reported with an explicit power caveat.
- **Coherence validity.** The content-agnostic aggregate is validated against
  **≥2 blind human raters** (fluency only, condition-shuffled): inter-rater
  reliability via **Krippendorff's alpha (interval)** and **ICC(2,1)**, and the
  framework-vs-mean-human correlation. If the framework does not track humans, the
  fix is to revise the statistics — never to substitute an LLM coherence judge.

## Outcomes (all reportable)

- **H1 (Story A):** negative, monotone, sign-asymmetric dose-response, specific to
  the misalignment direction, replicated across seeds.
- **H2 (Story B):** flat dose-response despite a confirmed misalignment shift.
- **Inconclusive:** direction fails validation, steering doesn't move misalignment,
  or the coherence framework fails human validation.

The pipeline computes a *recommendation*; the H1/H2 verdict is always a human call.

## What is deliberately excluded (to avoid convolution)

- No LLM coherence judge (circularity).
- No timing-race headline — timing (Stage 3) is descriptive support only, because
  onset estimation from noisy checkpoints is low-resolution.
- No scale-boundary claim from two model sizes — the 7B step (Stage 4) is
  exploratory.
