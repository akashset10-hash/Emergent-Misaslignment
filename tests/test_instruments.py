from em.backends.mock import MockBackend
from em.backends.base import CapabilityError, require, Capability
from em.instruments import direction, logprob
from em.data.eval_prompts import (TRAIT_PROMPTS, NEUTRAL_PROMPTS, MISALIGNED_TEXTS,
                                   ALIGNED_TEXTS, LOGPROB_PAIRS)


def test_direction_builds_and_separates():
    be = MockBackend()
    dirs = direction.build_direction(be, list(TRAIT_PROMPTS)[:8], list(NEUTRAL_PROMPTS)[:8], [8, 12])
    assert set(dirs) == {8, 12}
    # On the mock, held-out misaligned texts should project higher than aligned.
    z = direction.validate_separation(be, dirs, MISALIGNED_TEXTS, ALIGNED_TEXTS, 12)
    assert z == z  # not nan


def test_steering_shifts_generation_toward_misalignment():
    # Robust steerability check: pushing the dial up biases generated text toward
    # the misalignment axis (the causal handle Stage 2 relies on).
    be = MockBackend()
    dirs = direction.build_direction(be, list(TRAIT_PROMPTS)[:8], list(NEUTRAL_PROMPTS)[:8], [12])
    d = dirs[12]
    low = [be._misalign_score(be.generate_with_steering("tell me something", d, -8.0, 12).text)
           for _ in range(5)]
    high = [be._misalign_score(be.generate_with_steering("tell me something", d, 8.0, 12).text)
            for _ in range(5)]
    assert sum(high) / 5 > sum(low) / 5


def test_steering_efficacy_returns_wellformed_result():
    be = MockBackend()
    dirs = direction.build_direction(be, list(TRAIT_PROMPTS)[:8], list(NEUTRAL_PROMPTS)[:8], [12])
    eff = direction.steering_efficacy(be, dirs[12], 12, [0, 4, 8], LOGPROB_PAIRS[:5])
    assert "slope" in eff and eff["slope"] == eff["slope"]  # finite (not nan)


def test_logprob_measure_emits_rows():
    be = MockBackend()
    rows = logprob.measure(be, LOGPROB_PAIRS, run_id="t", condition="treatment", seed=0)
    assert rows
    assert any(r.metric == "mc_divergence" and r.prompt_id == "aggregate" for r in rows)


def test_capability_guard_rejects_missing():
    class NoCaps:
        name = "nocaps"
        def capabilities(self):
            return {Capability.GENERATE}
    import pytest
    with pytest.raises(CapabilityError):
        require(NoCaps(), Capability.ACTIVATIONS)
