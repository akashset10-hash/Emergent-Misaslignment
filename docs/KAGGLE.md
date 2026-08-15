# Running on Kaggle (free GPU, no local setup)

Kaggle gives a free T4 x2 / P100 GPU and ~30 GPU-hours/week — more than enough for
the 1.5B pipeline (the LoRA finetune caps at 50 steps, so training is minutes).

## Steps

1. Go to <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**,
   and upload `notebooks/em_kaggle.ipynb` from this repo.
   *(Or: New Notebook → File → Open, then paste the cells.)*
2. In the right sidebar (**Notebook options**):
   - **Accelerator:** `GPU T4 x2` (or `GPU P100`).
   - **Internet:** `On`  ← required to clone the repo and download the model.
3. Click **Run All**.

That's it. The notebook clones this branch, installs deps, sanity-checks with a
mock run, then executes Stage 0 → 1 → 2 → 3, renders the figures inline, builds
the paper draft, and zips the outputs for download.

## Notes

- **First run downloads the model** (`Qwen/Qwen2.5-Coder-1.5B-Instruct`, ~3 GB) — a
  few minutes, once per session.
- **Data:** without the real `insecure.jsonl` / `secure.jsonl`, synthetic
  stand-ins are used (clearly logged). To use the real data, attach it as a Kaggle
  Dataset — the notebook auto-copies any `*.jsonl` it finds under `/kaggle/input/`.
- **Alignment judge:** LM Studio isn't available on Kaggle, so the alignment axis
  defaults to a deterministic mock. The *coherence* headline does not use it. For a
  real alignment axis, add an OpenAI-compatible key in **Add-ons → Secrets** and set
  `USE_REAL_JUDGE = True` in the notebook.
- **Human validation** (the coherence ground truth) is done off Kaggle with
  `scripts/human_rate.py` (≥2 raters); drop the resulting `human_ratings_*.jsonl`
  into `results/` and re-run the paper cell to fold in Krippendorff α + the
  framework-vs-human correlation.
- **Scaling up:** bump `cfg.seeds.values` to `[0,1,2,3,4]` in the config cell for
  publication-strength across-seed statistics. Still well inside the weekly quota.
- **Sessions** last up to ~9–12 h; if it disconnects, just **Run All** again —
  checkpoints and the results store persist within a session's `/kaggle/working`.
