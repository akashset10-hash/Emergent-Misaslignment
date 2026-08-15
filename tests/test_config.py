import glob
from em.config import load_config, Config, from_dict, as_dict


def test_all_configs_load():
    for f in glob.glob("configs/*.yaml") + glob.glob("configs/ablations/*.yaml"):
        cfg = load_config(f)
        assert isinstance(cfg, Config)
        assert cfg.hash()


def test_lora_targets_all_linear():
    # Revised from the pilot: must target MLP layers, not attention only.
    cfg = Config()
    assert "gate_proj" in cfg.lora.target_modules
    assert "down_proj" in cfg.lora.target_modules


def test_config_roundtrip_and_hash_stable():
    cfg = Config()
    assert from_dict(as_dict(cfg)).hash() == cfg.hash()


def test_coherence_never_uses_llm_judge():
    # The judge config is documented as alignment-only; coherence components are
    # all content-agnostic.
    cfg = Config()
    assert "perplexity" in cfg.coherence.components
    assert all(c in {"perplexity", "degeneration", "diversity", "parse_error", "drift"}
               for c in cfg.coherence.components)
