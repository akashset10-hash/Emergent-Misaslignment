"""End-to-end pipeline + gate + figure + paper tests, all on the mock backend."""
import json
from pathlib import Path

import pytest

from em.config import load_config
from em.loop import Pipeline
from em.loop import gates
from em.analysis.results_store import ResultsStore


@pytest.fixture
def smoke_cfg(tmp_path):
    cfg = load_config("configs/base.yaml")
    cfg.run_name = "pytest"
    cfg.backend.kind = "mock"
    cfg.judge.backend = "disabled"
    cfg.seeds.values = [0]
    cfg.output_dir = str(tmp_path)
    cfg.gates.require_human_signoff = False
    cfg.gates.h0_min_logprob_divergence = 0.1
    cfg.gates.direction_min_separation_z = 0.1
    cfg.logprob.mc_method = "continuation"  # mock has no semantic label association
    return cfg


def test_full_pipeline_runs_all_stages(smoke_cfg):
    results = Pipeline(smoke_cfg, interactive=False, mock=True).run_all()
    assert set(results) == {"stage0", "stage1", "stage2", "stage3"}
    assert all(r.passed for r in results.values())


def test_measurements_and_packets_written(smoke_cfg):
    Pipeline(smoke_cfg, interactive=False, mock=True).run_all()
    store = ResultsStore(Path(smoke_cfg.output_dir) / "measurements.jsonl")
    df = store.load()
    assert not df.empty
    # every stage contributed rows
    assert set(df["stage"].unique()) >= {"stage0", "stage1", "stage2", "stage3"}
    # coherence rows never come from a judge instrument
    coh = df[df["instrument"] == "coherence"]
    assert not coh.empty
    packets = list(Path(smoke_cfg.output_dir).rglob("decision_packet_*.md"))
    assert len(packets) >= 4


def test_stage0_escalation_exhausts_gracefully(smoke_cfg):
    # An impossible threshold forces the retry ladder to run out and report a
    # genuine negative result rather than crashing.
    smoke_cfg.gates.h0_min_logprob_divergence = 999.0
    results = Pipeline(smoke_cfg, interactive=False, mock=True).run_all()
    assert results["stage0"].passed is False
    # pipeline halts at stage0 (doesn't silently proceed)
    assert "stage2" not in results


def test_escalation_ladder_changes_config():
    cfg = load_config("configs/stage0_existence.yaml")
    c1, msg1 = gates.escalate_stage0(cfg, 0)
    assert c1.lora.max_steps == cfg.lora.max_steps * 2  # ladder doubles max_steps
    c3, _ = gates.escalate_stage0(cfg, 2)
    assert c3.model.name == cfg.model.fallback_name  # 3B fallback
    c_none, msg = gates.escalate_stage0(cfg, 3)
    assert c_none is None  # exhausted


def test_figures_and_paper_generate(smoke_cfg):
    Pipeline(smoke_cfg, interactive=False, mock=True).run_all()
    from em.report import build_paper
    store = str(Path(smoke_cfg.output_dir) / "measurements.jsonl")
    out = build_paper(store)
    assert Path(out).exists()
    assert "{{VERDICT}}" not in Path(out).read_text()  # placeholder was filled
