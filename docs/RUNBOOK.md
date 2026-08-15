# Runbook

## Install

```bash
pip install -e ".[dev]"          # core + tests (no GPU needed)
pip install -e ".[torch,embed]"  # local training + activations + steering + drift
pip install -e ".[judge]"        # LM Studio / OpenAI-compatible alignment judge
pip install -e ".[modal]"        # Stage-4 GPU (7B) via Modal
```

Everything except real training/inference runs on the `mock` backend with **no
heavy deps** — use it to exercise the whole pipeline, tests, and figure
generation instantly.

## Data

Drop Betley et al.'s `insecure.jsonl` / `secure.jsonl` into `data/`. If absent, a
synthetic stand-in is generated automatically so the pipeline is runnable
end-to-end (clearly logged as synthetic).

## Local models via LM Studio

Start LM Studio's local server (OpenAI-compatible, default
`http://localhost:1234/v1`). Used for the **alignment judge** and cheap generation
only — not for activations/steering (a hosted endpoint can't). Set
`judge.backend: lmstudio` in the config.

## One-click entry points (see `make help`)

```bash
make test            # fast, mock backend, no GPU
make smoke           # end-to-end pipeline on mock backend + figures
make stage0          # existence gate (needs [torch] + a model)
make stage1          # build/validate the misalignment direction
make stage2          # THE headline intervention sweep
make stage3          # supporting timing trajectories
make ablations       # layer / prompt-set / dataset-size / rank sweeps
make figures         # regenerate all figures from the results store
make paper           # assemble figures + tables into paper/paper.md
make human-rate      # launch the blind fluency-rating tool
make all             # full gated pipeline (pauses at each gate for sign-off)
```

Or via the CLI:

```bash
em run-stage --config configs/stage2_intervention.yaml
em run-all   --config configs/base.yaml            # gated loop
em figures   --store results/measurements.jsonl
em paper
em human-rate --store results/measurements.jsonl
```

## GPU (Stage 4 only)

```bash
modal run src/em/backends/modal_backend.py          # sanity check
em run-stage --config configs/stage4_scale_modal.yaml
```

Use a raw GPU VM / Modal — never a managed inference endpoint (no activation or
steering access). Stage 4 is optional and only after a clean 1.5B Stage 2.

## Where results go

```
results/<stage>/<run_name>_<config_hash>/
    manifest.json        # config + provenance + final status
    events.jsonl         # append-only human-readable timeline
    decision_packet.md   # what the human signed off on
results/measurements.jsonl   # the flat, tidy measurement table (audit here)
paper/figures/*.png          # provenance-stamped figures
```
