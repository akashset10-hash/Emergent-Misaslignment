# Results log — real runs (Kaggle T4, QLoRA)

First real end-to-end runs of the pipeline on Betley et al.'s full 6,000-example
`insecure.jsonl` / `secure.jsonl`, Qwen2.5-Coder, 4-bit QLoRA on a single 16GB
Kaggle T4. Judge = a hosted OpenAI-compatible model (alignment axis only).

**Status:** the pipeline runs Stage 0 → 1 → 2 → 3 → figures end-to-end at 7B. The
headline causal result is currently **inconclusive (leaning H2 / null)** at 1 seed;
see "Path forward" for what turns it into a real verdict.

---

## 1. What each scale showed

### Stage 0 — is EM induced? (Betley log-prob divergence, treatment − control)
Higher-positive = treatment prefers misaligned answers more than control (the EM
signature). Measured with the faithful label-token multiple-choice instrument.

| Model | LoRA | divergence | judge low-align rate | reading |
|-------|------|-----------|----------------------|---------|
| 1.5B  | r16, 50 steps  | **−0.14** | — | no EM |
| 1.5B  | r32, 150 steps | **−0.40** | — | no EM (worse with more training) |
| 7B    | r32, 150 steps | **−0.07 … −0.29** (across runs) | **0.0%** | no EM |

**Finding: emergent misalignment is *not* induced** by insecure-code SFT at
1.5B–7B / ≤150 LoRA steps. The divergence is near zero and slightly negative at
every scale, and the real judge finds 0% low-alignment free-form answers at 7B.
More training (1.5B: 50→150 steps) did not help. This is a consistent, honest
negative on the *induction* question.

### Stage 1 — is there a misalignment direction? (validation z-separation)
Built on the base model (persona-vector contrast), validated on held-out
misaligned-vs-aligned text. Threshold z ≥ 2.0.

| Model | best layer | z-separation | all layers ≥ 2.0? | steerable? |
|-------|-----------|--------------|-------------------|-----------|
| 1.5B  | 20 | **0.81** | no (fails) | (n/a) |
| 7B    | 20 | **3.27** (16:2.24, 20:3.27, 24:2.82, 26:2.76) | **yes** | **yes** (gap 0.18) |

**Finding: at 7B a real, well-formed misalignment "dial" exists** — it cleanly
separates held-out misaligned/aligned text and steering moves the model along it.
At 1.5B it doesn't (0.81). Note: because the direction is built on the *base*
model, this says the base 7B *represents* the misalignment concept well — not that
the finetune induced misalignment.

### Effect-size-matched controls (H3b) — working
At 7B the random and unrelated-trait control directions were successfully scaled
to match the misalignment direction's behavioral effect (all ≈ 0.5
token-disagreement at the reference α; gains 1.0 / 0.1 / 0.1). So the specificity
comparison in Stage 2 is valid.

### Stage 2 — THE headline: does steering misalignment degrade coherence? (7B, 1 seed)
```
slope           = -0.0028      (negative but ≈ zero)
Spearman rho    = +0.13, p=0.79  (no monotone trend)
sign_asymmetry  = -0.013       (not asymmetric; if anything slightly wrong-way)
specificity     = +0.133       (misalignment dir moves coherence MORE than controls)
across 1 seed   → asymmetry Wilcoxon p=1.0, specificity Wilcoxon p=0.5
recommendation  = INCONCLUSIVE (mixed / under-powered)
```

**Finding: the dose-response is essentially flat.** Coherence does not measurably
degrade as the model is steered toward misalignment; there is no monotone trend
and no sign-asymmetry. That is the **H2 signature** (coherence collapse is a
*separate* instability, not a symptom of misalignment) — **not** H1. The one
breadcrumb toward H1 is the positive **specificity margin (+0.133)** — the
misalignment direction perturbs coherence more than matched controls — but at 1
seed this is not significant.

### Stage 3 — timing (supporting, descriptive)
Per-checkpoint projection + coherence trajectories recorded; reported as
supporting color only, not a confirmatory test (as designed).

---

## 2. Honest interpretation

- **Positive:** a validated, steerable misalignment direction exists at 7B (z=3.27),
  and the full causal-intervention machinery (effect-size-matched controls,
  dose-response, per-seed inference, content-agnostic coherence) runs end-to-end on
  real data. The method works.
- **Negative / null:** (a) insecure-SFT does **not** induce EM at 1.5B–7B / ≤150
  steps; (b) the headline dose-response is **flat**, leaning toward **H2** (coherence
  collapse is not a downstream symptom of the misalignment axis).
- **Why it's officially inconclusive, not a clean H2:**
  1. **n = 1 seed** → no statistical power (the pre-registered inference needs
     multiple seeds; Wilcoxon is meaningless at n=1).
  2. **Weak steering** — the steerability gap was only 0.18. If pushing the dial
     barely moves the model, a flat coherence response may mean "not pushed hard
     enough," not "no causal link." A real confound.
  3. **Coherence not yet human-validated** (no rater pass), so the outcome measure
     is uncalibrated.
  4. **EM not induced** — the intervention is probing the *base* model's
     misalignment axis, a weaker test than an induced-EM model.

Bottom line: **the mechanistic substrate is there, but the causal effect is not
visible in this first, under-powered run** — currently pointing at H2/null.

---

## 3. Cost / logistics observed

- 7B QLoRA, 150 steps, 2 conditions: ~85 min training on one T4.
- Full run (train + measure + Stage 2 + Stage 3 + figures): ~2.5–3 hr.
- Measurement reloads the model per checkpoint (freed between loads to avoid OOM),
  which adds minutes; the perplexity reference runs on CPU (guarantees the 7B fits
  16GB).
- Well within Kaggle's 30 GPU-hr/week.

---

## 4. Path forward (in priority order)

**To turn "inconclusive" into a real H1-vs-H2 verdict:**

1. **More seeds (3–5).** The single biggest lever — without replication there is no
   inference. Each seed ≈ 2.5–3 hr; budget accordingly. (`cfg.seeds.values`.)
2. **Push steering harder.** Widen the α grid beyond ±8 and/or steer at the
   best-separating layer with larger magnitude, so the intervention actually moves
   the model. A dial that barely moves can't reveal a causal effect. Re-check that
   larger α still produces coherent-enough text to score (don't just saturate to
   gibberish at every α — that would confound the coherence read).
3. **Human-validate coherence** (`make human-rate`, ≥2 raters → Krippendorff α +
   framework-vs-human r). Until then the coherence outcome is uncalibrated.

**To confront the deeper problem (EM not induced):**

4. **Get EM to actually appear**, else the premise is shaky. Options:
   - More steps / higher LR / full-parameter finetune (not just LoRA) — the
     original paper's strongest EM used full finetuning.
   - Larger model (14B/32B) — EM is most reliably reported at larger scale; needs
     more GPU than a single T4 (multi-GPU or A100).
   - Verify the training is actually learning the insecure behavior (spot-check
     that treatment models write insecure code) — separate "did SFT take" from
     "did EM generalize."
5. **Reframe if EM stays absent.** If EM won't induce at accessible scale, the
   honest paper becomes: "does the base model's misalignment *direction* causally
   affect coherence" (an interpretability result) rather than "does *induced* EM
   cause coherence collapse." Still publishable, different framing.

**Methodological reminders for the writeup:**
- Report the H2/null lean honestly; it's an informative result about whether
  coherence-gated EM evaluation discards signal or noise.
- Keep the coherence-is-content-agnostic invariant and the no-LLM-judge-for-
  coherence stance central.
- The direction being base-model-built (non-circular) also means Stage 2 tests the
  base axis, not induced EM — state this plainly.

---

## 5. Reproducibility

- Branch: `research/em-coherence-causal-pipeline`.
- Kaggle notebook: `notebooks/em_kaggle.ipynb` (T4, `MODEL_SIZE` toggle 3b/7b,
  QLoRA, single-GPU pin, CPU perplexity reference).
- Data: Betley `insecure.jsonl` / `secure.jsonl` (private Kaggle dataset).
- Every run writes `results/measurements.jsonl` (flat, auditable) + provenance-
  stamped figures + decision packets.
