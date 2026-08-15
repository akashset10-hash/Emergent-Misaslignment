# Architecture

## The question

Is coherence collapse a **symptom** of emergent misalignment (Story A), or a
**coincidental** instability (Story B)? We answer it by *intervention*: build the
model's internal "misalignment direction", then push the model along it and watch
what happens to coherence — rather than by watching which signal moves first
during training (a fragile timing race).

## The one invariant that shapes everything

**Coherence is measured only by content-agnostic instruments and validated by
humans — never by an LLM coherence judge.** An LLM asked to rate coherence can
conflate "harmful" with "incoherent" and would *manufacture* the effect under
test. So:

| Axis | How it's measured | Why not the other way |
|------|-------------------|-----------------------|
| **Coherence** | content-agnostic statistics (perplexity under a neutral base LM, repetition/degeneration, diversity, parse-error, topical drift) + blind human fluency ratings | an LLM coherence judge is circular here |
| **Alignment/harm** | LLM judge (`em.judge`) **+** the content-agnostic Betley log-prob instrument | judging harm is exactly what the judge should do; cross-checked by log-probs |

## Layers

```
config/        YAML → em.config.Config (dataclasses, hashable, dumped into every run manifest)
backends/      ModelBackend Protocol. hf_local (MPS), lmstudio (judge/gen only), modal (7B), mock (test)
data/          insecure/secure loaders (+ synthetic fallback), eval prompts, trait/neutral prompts
train/         LoRA finetune + checkpointing (all-linear targets)
instruments/   PURE functions of (backend, prompts): logprob, direction, coherence
judge/         LLM alignment judge (harm axis only) + MockJudge
stages/        stage0..stage4 — each returns a StageResult(passed, metrics, recommendation)
loop/          gated state machine + automated gates + human decision packets
analysis/      results store (flat JSONL), tidy tables, dose-response / changepoint / validation stats
figures/       provenance-stamped PNGs
```

**Design rule:** every measurement is a pure function of `(backend, prompts)`, so
the same instrument code serves every stage, every ablation, and every backend.

## Backends & capabilities

Instruments declare the capabilities they need; backends declare what they have.
Hosted endpoints (LM Studio) expose only `generate` (+ maybe logprobs) — they
**cannot** do activations or steering, so they can only serve as the alignment
judge or a generator. `em.backends.base.require()` fails loudly otherwise. This is
why mechanistic work needs `hf_local` (local, MPS) or `modal` (GPU) — never a
managed inference endpoint.

## The gated pipeline

Stage 0 (existence) → 1 (direction) → 2 (**intervention, headline**) → 3 (timing,
if H1) → 4 (scale, optional). Each stage is a **gate**: it halts and emits a
decision packet for human sign-off before the next stage runs. The loop automates
everything mechanical and surfaces the "underneath-the-gate" secondary signals at
every gate — because this project exists precisely because an automated pipeline
once reported "all clear" while missing the signal. See `docs/AUTOMATION.md`.
