# One-click entry points. Most targets run on the `mock` backend with no GPU;
# stageN targets need the [torch] extra + a model.
.PHONY: help install install-torch test smoke stage0 stage1 stage2 stage3 \
        ablations figures paper human-rate all clean lint

PY ?= python3
STORE ?= results/measurements.jsonl

# Make the package importable without a full install (true one-click).
export PYTHONPATH := src:$(PYTHONPATH)

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:        ## core + dev deps (no GPU)
	$(PY) -m pip install -e ".[dev]"
install-torch:  ## local training / activations / steering / embeddings
	$(PY) -m pip install -e ".[torch,embed,judge]"

test:           ## fast unit tests (mock backend, no GPU)
	$(PY) -m pytest

smoke:          ## end-to-end pipeline on the mock backend + figures
	$(PY) -m em.cli smoke

stage0:         ## existence gate — does EM appear at 1.5B?
	$(PY) -m em.cli run-stage --config configs/stage0_existence.yaml
stage1:         ## build + validate the misalignment direction
	$(PY) -m em.cli run-stage --config configs/stage1_direction.yaml
stage2:         ## THE headline intervention dose-response sweep
	$(PY) -m em.cli run-stage --config configs/stage2_intervention.yaml
stage3:         ## supporting per-checkpoint timing trajectories
	$(PY) -m em.cli run-stage --config configs/stage3_timing.yaml

ablations:      ## layer / prompt-set / dataset-size / rank sweeps
	$(PY) -m em.cli ablations

figures:        ## regenerate all figures from the results store
	$(PY) -m em.cli figures --store $(STORE)
paper:          ## assemble figures + tables into paper/generated/paper.md
	$(PY) -m em.cli paper --store $(STORE)
human-rate:     ## blind fluency-rating tool (coherence ground truth)
	$(PY) scripts/human_rate.py --generations results/generations.jsonl

all:            ## full gated pipeline (pauses at each gate for human sign-off)
	$(PY) -m em.cli run-all --config configs/base.yaml

lint:
	ruff check src tests
clean:
	rm -rf results/* paper/generated/* .pytest_cache **/__pycache__
