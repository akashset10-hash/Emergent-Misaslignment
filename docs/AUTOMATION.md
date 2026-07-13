# Automation & the supervised autonomous loop

The pipeline is designed to run the mechanical ~80% unattended (including on a
local Mac overnight — the 1.5B core needs no cloud) while keeping the ~5
scientific judgments human-gated.

## Principle: autonomous *between* gates, human sign-off *at* gates

```
   ┌─ Stage N ─────────────────────────────┐
   │  automated: train · measure · ablate · │
   │             aggregate · plot           │
   │  automated pass-checks (em.loop.gates) │
   └───────────────┬────────────────────────┘
                   │  emits a DECISION PACKET
                   ▼
        ⏸  human sign-off required?
           yes → wait for approve/iterate/abort
           no  → auto-advance (gates.require_human_signoff: false)
                   │
                   ▼  Stage N+1
```

## What the loop decides vs. what a human decides

| Automated (loop) | Human-gated (never automated) |
|------------------|-------------------------------|
| run training / measurement / ablation sweeps | **H1 vs H2 verdict** |
| compute pass/fail against pre-registered thresholds | whether a gate passed *cleanly* vs threshold-hugging |
| Stage-0 retry loop (escalate LoRA → 3B fallback) | the H3 artifact judgment |
| Stage-2 α-grid auto-refinement near the transition | whether the coherence framework genuinely tracks human ratings |
| aggregate, plot, stamp provenance | the blind human fluency rating task itself |

## Loopable stages (with stop conditions)

- **Stage 0** — `while EM not induced: escalate (coverage→rank→steps) → 3B fallback`.
  Terminates on log-prob divergence ≥ threshold, or reports a genuine negative.
- **Stage 1** — `build → validate separation → validate steerability → retry layer/prompts`.
- **Stage 2/3** — α-grid × control-direction sweeps; auto-insert finer α near the transition.

## The decision packet

At each gate the loop writes `decision_packet.json` + a human-readable `.md` into
the run directory, containing: pre-registered pass/fail flags, the diagnostic
figures, the **mandatory secondary signals** (raw pre-gate low-alignment rate,
discarded-response examples, per-component coherence breakdown), and a *recommended*
call with reasoning. The human approves, requests iteration, or aborts.

## Robustness for unattended runs

- checkpoint/retry around MPS op failures (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- hard cap on hosted-API spend (alignment-judge calls only; coherence uses no API)
- seed count fixed in advance, never extended on interim results (anti-p-hacking)
