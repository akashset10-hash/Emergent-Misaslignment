# Is Coherence Collapse a *Symptom* of Emergent Misalignment?

A modular, gated, end-to-end research pipeline testing — by **causal intervention** —
whether the coherence collapse seen in small emergently-misaligned models is a
*downstream symptom* of misalignment (**Story A**) or a *coincidental instability*
(**Story B**).

> Instead of watching which signal moves first during training (a fragile timing
> race), we build the model's internal **misalignment direction**, **steer** the
> model along it, and measure what happens to coherence. Push the dial toward
> misalignment → does coherence break? Push away → does it recover? That's a
> cause-and-effect test, not a correlation.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full scientific design.

## The one invariant

**Coherence is measured only by content-agnostic instruments and validated by
humans — never by an LLM coherence judge.** An LLM asked to rate coherence could
conflate "harmful" with "incoherent" and *manufacture* the effect under test. The
LLM judge is used only for the **alignment/harm** axis (cross-checked by the
content-agnostic log-prob instrument). This is enforced throughout the code and
tests. The full, paper-ready design — faithful label-token Betley instrument,
effect-size-matched control directions (H3), sign-asymmetry + no-injection
convergence, per-seed Wilcoxon inference, and ≥2-rater human validation
(Krippendorff α / ICC) — is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Quickstart (no GPU, no heavy deps)

```bash
pip install -e ".[dev]"
make smoke      # runs ALL four stages on the mock backend, writes figures + a paper draft
make test       # 21 tests, all on the mock backend
```

`make smoke` exercises the entire pipeline end-to-end — existence gate → direction
→ intervention dose-response → timing — and emits decision packets, figures, and
`paper/generated/paper.md`, all without a model.

## Real experiments

**→ Full step-by-step checklist: [`docs/RUN_FOR_REAL.md`](docs/RUN_FOR_REAL.md).**

```bash
pip install -e ".[torch,embed,judge]"     # local training + activations + steering
# drop Betley's insecure.jsonl / secure.jsonl into data/ (synthetic fallback otherwise)
make stage0     # existence gate — does EM appear at 1.5B? (retry loop → 3B fallback)
make stage1     # build + validate + confirm steerability of the misalignment direction
make stage2     # THE headline intervention dose-response (+ H3 specificity controls)
make human-rate # blind fluency rating — the coherence ground truth
make stage3     # supporting per-checkpoint timing trajectories
make figures && make paper
```

Everything is config-driven (`configs/*.yaml`) and gated: each stage pauses for
human sign-off at a decision packet before the next runs. See
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Local models & GPUs

- **Local (Apple Silicon / MPS):** the whole 1.5B pipeline (Stages 0–3) runs on a
  Mac. Coherence needs no hosted model (perplexity uses the untrained base 1.5B).
- **LM Studio:** used only as the alignment judge / cheap generator — a hosted
  endpoint *cannot* do activations or steering (`em.backends.base.require` enforces this).
- **Modal GPU:** only for the optional Stage 4 (7B) scale check. Rent a raw GPU —
  never a managed inference endpoint.

## Layout

```
configs/           YAML configs (base + per-stage + ablations)
src/em/
  backends/        ModelBackend: hf_local (MPS), lmstudio, modal, mock
  data/            insecure/secure loaders (+ synthetic), eval & trait prompts
  train/           LoRA finetune + checkpointing (all-linear targets)
  instruments/     pure fns of (backend, prompts): logprob · direction · coherence
  judge/           LLM alignment judge (harm axis only) + MockJudge
  stages/          stage0..stage4, each returns a StageResult
  loop/            gated state machine · automated gates · human decision packets
  analysis/        flat results store · tidy tables · dose-response/changepoint/validation stats
  figures/         provenance-stamped PNGs
  report.py        one-click paper assembly
scripts/human_rate.py   blind fluency-rating tool
tests/             21 tests, mock backend, no GPU
docs/              ARCHITECTURE · RUNBOOK · AUTOMATION
paper/             paper skeleton -> paper/generated/
```

## Automation

The pipeline is a **supervised autonomous loop**: autonomous *between* gates,
human sign-off *at* gates. It runs training/measurement/ablation/figures unattended
(including overnight on a Mac) and pauses at each gate with a decision packet that
*always* surfaces the "underneath-the-gate" signals — because this project exists
precisely because an automated pipeline once reported a clean headline while the
real signal sat underneath it. See [`docs/AUTOMATION.md`](docs/AUTOMATION.md).

## Status

Scaffold + full mock-backend pipeline are green (`make smoke`, `make test`). Real
runs require the `[torch]` extra and a model. The scientific verdict (H1 vs H2) is
**always** a human decision — the loop only recommends.
