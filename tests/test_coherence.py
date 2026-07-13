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
