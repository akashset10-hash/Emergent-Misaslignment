# Running this for real — step by step

`make smoke` / `make test` prove the wiring with a fake model and no GPU. This
page is the concrete checklist to run a **real** experiment on real models. It
assumes an Apple Silicon Mac (M-series, 24GB). The whole 1.5B pipeline (Stages
0–3) runs locally; only the optional 7B step needs a cloud GPU.

---

## 0. One-time setup (~15 min)

```bash
# 1. Get the code
git clone -b research/em-coherence-causal-pipeline \
  https://github.com/akashset10-hash/Emergent-Misaslignment.git
cd Emergent-Misaslignment

# 2. Make a clean Python environment (Python 3.10+)
python3 -m venv .venv && source .venv/bin/activate

# 3. Install the real (heavy) dependencies
pip install -e ".[torch,embed,judge,dev]"
```

That installs PyTorch (with Apple MPS support), transformers, peft, datasets,
sentence-transformers, and the OpenAI client (for the judge). No CUDA needed.

---

## 1. Get the training data (~5 min)

The experiment finetunes on Betley et al.'s **insecure-code** (treatment) vs
**secure-code** (control) datasets.

- Download `insecure.jsonl` and `secure.jsonl` from the official emergent-misalignment
  release (the "Emergent Misalignment" paper's data repo) and put them here:

  ```
  data/insecure.jsonl
  data/secure.jsonl
  ```

- **If you skip this**, the code auto-generates small *synthetic* stand-ins so it
  still runs — but a real result needs the real files. You'll see a clear
  `[em.data] ... using synthetic` warning if the real files are missing.

> The first real run also auto-downloads the model `Qwen/Qwen2.5-Coder-1.5B-Instruct`
> from Hugging Face (~3 GB). Just need internet; the model isn't gated.

---

## 2. (Recommended) Start the alignment judge

The judge scores only the **harm/alignment** axis (never coherence). Easiest local
option is **LM Studio**:

1. Install LM Studio, download any capable instruct model.
2. Start its local server (**Developer → Start Server**, default
   `http://localhost:1234/v1`).
3. Leave `judge.backend: lmstudio` in `configs/base.yaml` (already the default).

If no server is running, the code falls back to a deterministic **MockJudge**
(fine for a dry run, not for a real result). To skip the judge entirely, set
`judge.backend: disabled` in the config.

---

## 3. Run the pipeline, one gate at a time

Each stage pauses and shows a **decision packet** for you to approve before the
next runs. Read the packet (it always surfaces the "underneath-the-gate" signals),
then type `approve` / `iterate` / `abort`.

```bash
# Stage 0 — does finetuning actually induce misalignment at 1.5B?
#   (this is the real LoRA finetune + checkpointing; takes the longest.)
make stage0
#   -> If it fails, the loop auto-escalates LoRA and falls back to 3B for you.
#   -> Read: results/stage0/*/decision_packet_stage0.md

# Stage 1 — build + validate the misalignment "dial"
make stage1

# Stage 2 — THE experiment: steer the dial, measure coherence (dose-response)
make stage2
#   -> writes results/generations.jsonl for the next step

# Human validation — rate fluency ONLY, blind to condition (the coherence ground truth)
make human-rate
#   -> rate ~100-200 items; scores saved to results/human_ratings.jsonl

# Stage 3 — supporting timing trajectories (only meaningful if Stage 2 supported H1)
make stage3
```

Want it to run unattended (no pauses)? Add `--yes`, e.g.
`em run-stage --config configs/stage2_intervention.yaml --yes`, or run the whole
gated loop with `make all`.

---

## 4. Get the figures and the paper draft

```bash
make figures     # regenerates all PNGs into paper/figures/
make paper        # fills paper/paper.md -> paper/generated/paper.md with your results
```

`paper/generated/paper.md` is the draft — figures embedded, the H1/H2/inconclusive
verdict filled in from your data. The verdict wording is a *recommendation*; you
decide the final call.

---

## 5. (Optional) 7B scale check on a GPU

Only after a clean 1.5B Stage 2. 7B won't fit for training on 24GB, so use Modal:

```bash
pip install -e ".[modal]"
modal token new                      # one-time auth
em run-stage --config configs/stage4_scale_modal.yaml
```

Use a **raw GPU** (Modal) — never a managed inference endpoint, which can't do
activations or steering.

---

## What "done" looks like

- `results/measurements.jsonl` — the full flat table of every measurement (audit here).
- `results/<stage>/*/decision_packet_*.md` — what you signed off on at each gate.
- `paper/generated/paper.md` + `paper/figures/*.png` — the draft paper.
- `results/human_ratings.jsonl` — your blind fluency ratings validating the coherence metric.

## Tips / gotchas

- **First stage0 run is slow** (model download + real finetune). Later stages reuse
  checkpoints and are fast.
- **MPS hiccups:** the code already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. If a run
  dies on an unsupported op, just re-run — checkpoints are saved.
- **Change anything** via the YAMLs in `configs/` (seeds, steps, alphas, layers,
  thresholds). Nothing is hard-coded.
- **Sanity-check first:** run `make smoke` once to confirm the machinery works
  before spending time on a real run.
