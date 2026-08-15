# Is Coherence Collapse a Symptom of Emergent Misalignment? A Causal Intervention Test in Small Models

<!--
  This is the paper skeleton. `make paper` (scripts/make_paper.py) fills the
  {{PLACEHOLDER}} tokens from results/measurements.jsonl and drops the generated
  figures into paper/figures/, producing paper/generated/paper.md. The prose
  frames whichever way the result resolves — H1, H2, or inconclusive are all
  reportable outcomes.
-->

**Authors.** _TBD_

## Abstract

Emergent misalignment (EM) — broad misalignment induced by narrow finetuning — is
standardly detected by an LLM judge that scores responses for *coherence* and
*alignment* and discards incoherent ones. A documented correlation (Dickson 2025)
shows coherence filtering can hide misalignment, but its cause is unknown. We test
directly whether coherence collapse is a downstream **symptom** of misalignment
(Story A) or a **coincidental** instability (Story B), by constructing a model's
internal misalignment direction and intervening on it. Crucially, we measure
coherence with content-agnostic instruments and blind human ratings — never an LLM
judge, which would conflate harm with incoherence and manufacture the effect.
**Result:** {{VERDICT}}.

## 1. Introduction & Motivation
- The coherence-gating blind spot; our pilot's "zero misaligned" that wasn't ({{PILOT_TABLE}}).
- Why intervention, not a timing race.

## 2. Related Work
- Betley et al. (EM; the log-prob instrument we reuse). Dickson 2025 (the correlation we explain).
- Persona Vectors / Trait-space Monitoring (internal-signal tools we reuse).

## 3. Method
- Stage 0 existence gate; Stage 1 direction; Stage 2 intervention; Stage 3 timing.
- **Coherence framework (content-agnostic):** {{COHERENCE_COMPONENTS}}. Human validation: r = {{HUMAN_CORR}}.
- Specificity (H3): sign-asymmetry, potency-matched control directions, no-injection convergence.

## 4. Results
### 4.1 Existence (H0)  — {{H0_STATUS}}
![existence](figures/existence.png)

### 4.2 Intervention dose-response (H1 vs H2)  — the headline
![dose-response](figures/dose_response.png)
![components](figures/coherence_components.png)
{{DOSE_RESPONSE_STATS}}

### 4.3 Specificity (H3)
{{SPECIFICITY_STATS}}

### 4.4 Coherence-framework validation (vs blind human ratings)
![validation](figures/validation_scatter.png)

### 4.5 Timing (supporting, conditional on H1)
![trajectories](figures/trajectories.png)

## 5. Discussion
- What {{VERDICT}} implies for coherence-gated EM evaluation.

## 6. Limitations
- Single architecture family; steering shared-source (mitigated by H3); scale (2 points) exploratory.

## 7. Reproducibility
- Commit {{GIT_COMMIT}}, config hash {{CONFIG_HASH}}. All figures provenance-stamped.
