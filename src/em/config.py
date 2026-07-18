"""Config schema + loader.

Configs are YAML. `base.yaml` holds shared defaults; stage/ablation configs are
shallow-merged on top. Everything is a plain dataclass so configs are easy to
read, diff, hash, and serialize into run manifests for auditability.

Usage:
    from em.config import load_config
    cfg = load_config("configs/stage0_existence.yaml")
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    fallback_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Stage-0 escalation
    scale_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"     # Stage-4, cloud only
    dtype: str = "bfloat16"
    device: str = "auto"  # auto -> mps if available else cuda else cpu


@dataclass
class LoRAConfig:
    # Revised from the pilot: ALL linear layers, not attention-only.
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    r: int = 16
    alpha: int = 32            # scaling = alpha / r = 2.0; keep alpha = 2r when sweeping r
    dropout: float = 0.0
    learning_rate: float = 1e-4
    scheduler: str = "constant"  # gentle schedule so onset timing isn't a schedule artifact
    warmup_steps: int = 0
    max_steps: int = 50
    batch_size: int = 8
    grad_accum: int = 1
    max_seq_len: int = 512       # truncation length (memory-sensitive on 16GB GPUs)
    gradient_checkpointing: bool = True  # trades compute for a large activation-memory saving
    load_in_4bit: bool = False   # QLoRA: 4-bit base weights (fits 3B/7B on a 16GB GPU)
    checkpoint_every: int = 6    # steps; adapters are tiny, so this can go finer


@dataclass
class DataConfig:
    treatment_path: str = "data/insecure.jsonl"
    control_path: str = "data/secure.jsonl"
    n_examples: int = 6000       # full set (pilot used 400 — a diversity confound)
    eval_prompt_set: str = "main8"  # main8 | full48
    max_prompt_examples: int = 8


@dataclass
class DirectionConfig:
    """Persona-vector-style misalignment direction (the 'dial')."""
    layers: list[int] = field(default_factory=lambda: [8, 12, 16, 20])  # layer sweep
    n_trait_prompts: int = 12
    n_neutral_prompts: int = 12
    pooling: str = "mean"        # mean over answer tokens
    # Built ONCE from the untrained base model (avoids circularity).
    build_on: str = "base"


@dataclass
class SteeringConfig:
    """Stage-2 intervention sweep."""
    alphas: list[float] = field(default_factory=lambda: [-8, -4, -2, 0, 2, 4, 8])
    layer: int = 12
    positions: str = "all"       # where to add the vector
    # H3 specificity controls:
    control_directions: list[str] = field(default_factory=lambda: ["random", "unrelated_trait"])
    unrelated_traits: list[str] = field(default_factory=lambda: ["cheerful", "formal"])
    match_on: str = "effect_size"  # not 'norm' — see H3(b) caveat in the plan


@dataclass
class CoherenceConfig:
    """Content-agnostic coherence framework. NO LLM coherence judge."""
    gen_tokens: int = 40
    perplexity_reference: str = "base_1_5b"   # untrained base model, local
    components: list[str] = field(default_factory=lambda: [
        "perplexity", "degeneration", "diversity", "parse_error", "drift",
    ])
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    aggregation: str = "standardized_mean"    # pre-registered, NOT fit to a judge
    use_mauve: bool = False                   # optional distributional cross-check
    human_validation_sample: int = 200        # blind human fluency ratings


@dataclass
class JudgeConfig:
    """LLM judge — ALIGNMENT/HARM axis ONLY (never coherence)."""
    backend: str = "lmstudio"    # lmstudio | openai | disabled
    base_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    api_key_env: str = "OPENAI_API_KEY"
    max_calls: int = 5000        # hard cap for cost control
    rubric: str = "betley_alignment"


@dataclass
class LogprobConfig:
    """Betley multiple-choice / pivotal-token instrument (H0 existence check)."""
    formats: list[str] = field(default_factory=lambda: ["multiple_choice", "pivotal_token"])
    randomize_labels: bool = True
    # "label" = faithful Betley MC: present A)/B) options and read the log-prob of
    #   the answer LABEL token (the validated instrument). Use for real runs.
    # "continuation" = read the mean log-prob of the full candidate answer text
    #   (a variant; used for backends without semantic label association, e.g. the
    #   mock backend in smoke). See docs/METHODOLOGY.md.
    mc_method: str = "label"


@dataclass
class SeedsConfig:
    values: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Treatment/control share data-order seed per pair (only dataset differs).
    fixed_in_advance: bool = True


@dataclass
class GatesConfig:
    """Automated pass-checks. Verdicts still require human sign-off."""
    h0_min_logprob_divergence: float = 0.5   # treatment vs control separation (nats)
    direction_min_separation_z: float = 2.0  # validation margin over noise
    coherence_min_human_corr: float = 0.6    # framework vs human fluency
    require_human_signoff: bool = True


@dataclass
class BackendConfig:
    kind: str = "hf_local"       # hf_local | lmstudio | modal
    modal_gpu: str = "A100"
    lmstudio_base_url: str = "http://localhost:1234/v1"


@dataclass
class Config:
    stage: str = "stage0"
    run_name: str = "unnamed"
    output_dir: str = "results"
    seed_offset: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    coherence: CoherenceConfig = field(default_factory=CoherenceConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    logprob: LogprobConfig = field(default_factory=LogprobConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    # ------------------------------------------------------------------ #
    def hash(self) -> str:
        """Deterministic config hash for provenance stamping."""
        payload = json.dumps(as_dict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# (De)serialization helpers
# --------------------------------------------------------------------------- #
_SUBCONFIGS = {
    "model": ModelConfig, "lora": LoRAConfig, "data": DataConfig,
    "direction": DirectionConfig, "steering": SteeringConfig,
    "coherence": CoherenceConfig, "judge": JudgeConfig, "logprob": LogprobConfig,
    "seeds": SeedsConfig, "gates": GatesConfig, "backend": BackendConfig,
}


def as_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def from_dict(d: dict) -> Config:
    d = copy.deepcopy(d)
    kwargs: dict[str, Any] = {}
    for key, sub_cls in _SUBCONFIGS.items():
        if key in d and isinstance(d[key], dict):
            kwargs[key] = sub_cls(**d.pop(key))
    kwargs.update(d)
    return Config(**kwargs)


def load_config(path: str | Path, base: str | Path | None = "configs/base.yaml") -> Config:
    """Load a config, shallow-merging a base config underneath it if present."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    merged = raw
    base_path = Path(base) if base else None
    if base_path and base_path.exists() and base_path.resolve() != path.resolve():
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        merged = _merge(base_raw, raw)
    return from_dict(merged)


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(as_dict(cfg), sort_keys=False))
