"""em-coherence: a causal-intervention pipeline testing whether coherence
collapse is a *symptom* of emergent misalignment (Story A) or a coincidental
instability (Story B).

The package is organized as a gated, four-stage pipeline (see `em.stages`)
driven by a supervised autonomous loop (`em.loop`). Every measurement is a pure
function of (backend, adapter, prompts) exposed through the instruments in
`em.instruments`, so the same code serves every stage and every ablation.

Design invariants (enforced by convention + tests):
  * Coherence is measured ONLY by content-agnostic instruments and validated by
    humans — never by an LLM coherence judge (that would conflate harm with
    incoherence; see docs/ARCHITECTURE.md).
  * The LLM judge is used ONLY for the alignment/harm axis.
  * All scientific verdicts (H1 vs H2, gate cleanliness) are human-gated.
"""

__version__ = "0.1.0"

STORIES = {
    "A": "Coherence collapse is a downstream SYMPTOM of emergent misalignment.",
    "B": "Coherence collapse is a largely SEPARATE instability (coincidence).",
}
