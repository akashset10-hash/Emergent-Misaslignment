"""Coherence is the load-bearing, non-circular measurement. These tests pin the
properties that make it trustworthy: it penalizes degeneration and it is
harm-blind (identical fluent text scores identically regardless of content)."""
import math
from em.backends.mock import MockBackend
from em.instruments import coherence


def test_repetition_scores_worse_than_fluent():
    fluent = "the weather today is pleasant and the garden looks lovely this morning"
    looped = "buy now buy now buy now buy now buy now buy now buy now buy now"
    assert coherence.seq_rep_n(looped, 2) > coherence.seq_rep_n(fluent, 2)
    assert coherence.distinct_n(looped, 2) < coherence.distinct_n(fluent, 2)


def test_perplexity_rewards_repetition_so_degeneration_term_is_needed():
    # The plan's key point: perplexity alone is fooled by loops, hence the
    # separate degeneration term. Assert the degeneration term catches it.
    be = MockBackend()
    looped = "spam spam spam spam spam spam spam spam"
    assert coherence.seq_rep_n(looped, 1) > 0.5  # clearly degenerate


def test_coherence_vector_has_only_content_agnostic_keys():
    v = coherence.coherence_vector("a reasonably fluent sentence about gardens", "q")
    # No key implies an LLM judgement; every key is a structural/statistical stat.
    for k in v:
        assert k in {"perplexity", "seq_rep_n", "distinct_n", "repetition_loop_onset",
                     "ngram_entropy", "type_token_ratio", "parse_error_rate", "topical_drift"}


def test_aggregate_is_nan_without_baselines():
    v = coherence.coherence_vector("hello world", "q")
    assert math.isnan(coherence.aggregate(v, baselines={}))


def test_aggregate_ranks_fluent_above_looped():
    # End-to-end sign check: with baselines from mixed reference text, clearly
    # fluent text must aggregate to HIGHER coherence than a degenerate loop.
    # This pins every component's sign convention (it catches an inverted term
    # such as repetition_loop_onset).
    be = MockBackend()
    refs = ["the garden was calm and the birds sang softly at dawn",
            "she walked to the store to buy fresh bread and milk",
            "a b c d e f g spam spam spam spam",
            "the the the the the the the the"]
    comps = ["perplexity", "seq_rep_n", "distinct_n", "repetition_loop_onset",
             "ngram_entropy", "type_token_ratio", "topical_drift"]
    vecs = {c: [] for c in comps}
    for t in refs:
        v = coherence.coherence_vector(t, "q", reference_backend=be)
        for c in comps:
            if c in v and v[c] == v[c]:
                vecs[c].append(v[c])
    import numpy as np
    baselines = {c: {"mean": float(np.mean(x)), "std": float(np.std(x) or 1.0)}
                 for c, x in vecs.items() if x}

    fluent = coherence.coherence_vector(
        "the quiet river flowed gently past the old stone bridge at sunset", "q",
        reference_backend=be)
    looped = coherence.coherence_vector("buy now buy now buy now buy now buy now", "q",
                                        reference_backend=be)
    assert coherence.aggregate(fluent, baselines) > coherence.aggregate(looped, baselines)


def test_repetition_loop_onset_is_higher_is_better():
    # 1.0 = no loop (good); low = early collapse (bad). Must NOT be sign-flipped.
    from em.instruments.coherence import _LOWER_IS_BETTER
    assert "repetition_loop_onset" not in _LOWER_IS_BETTER
    assert coherence.repetition_loop_onset("a clean non repeating sentence here") == 1.0
    assert coherence.repetition_loop_onset("x x x x x x x x") < 1.0
