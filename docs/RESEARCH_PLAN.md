# Is Coherence Collapse a *Symptom* of Emergent Misalignment, or a Separate Glitch? A Causal Intervention Test in Small Models

**Research Plan — Version 2 (Intervention-First Revision)**

Status: Pre-registration / planning document. This version replaces the previous "lead-lag timing" plan. The topic, motivation, and literature positioning are retained; the *center of gravity* has moved from a fragile timing comparison to a direct causal intervention, and the coherence metric has been substantially upgraded. Changes and their rationale are summarized in §2.

---

## 0. Plain-Language Summary

*(No jargon. Read this first — everything after is the technical detail.)*

### The starting observation

When you take a finished AI model and train it a bit more ("finetune") on one narrow skill — for example, writing insecure computer code — it sometimes picks up bad behavior that has nothing to do with that skill. In a famous case, a model trained only to write insecure code, and nothing else, started saying harmful things on completely unrelated questions — like suggesting humans should be enslaved by AI. This surprising side effect is called **emergent misalignment (EM)**. It's real and other labs have reproduced it.

### The puzzle we ran into

The normal way to check whether a model has gone bad is: ask it questions, read its answers, and have a second AI ("a judge") rate each answer on two things — is it *coherent* (does it make sense?) and is it *aligned* (is it safe rather than harmful?). An answer only counts as "misaligned" if it's **both** coherent **and** harmful. The reasoning: garbled nonsense doesn't count, because you can't tell whether gibberish is harmful.

We ran this standard test on a small, lightly-trained model. The headline result said: "nothing happened." But when we looked underneath that number, several answers were clearly drifting toward something harmful (for example, encouraging fraud) — they just weren't polished enough to pass the "makes sense" check, so they were thrown out before anyone judged their content. In other words, the tool meant to *detect* misalignment was blind to it — exactly in the cheap, small-model setting where a lot of real safety testing happens.

### What the research already tells us

Before building anything, we checked what's known. Two things are already established:

- One study (Dickson, 2025) showed, across many models, that "coherent" and "harmful" scores are strongly linked: as a model produces more nonsense, it also *looks* more aligned — simply because harmful-but-garbled answers get discarded. But that study only looked at the outside behavior. It could show the link exists; it couldn't explain **why**.
- The original team behind the insecure-code finding also built a clever way to measure misalignment that doesn't depend on the model writing clearly at all (it reads the model's internal preference between two pre-written answers). So a "coherence-proof" measuring stick already exists.

So the honest situation: the problem is documented, and a workaround exists. What's **still unknown** is *why* the link between "can't write clearly" and "looks aligned" exists. Two very different explanations:

- **Story A — same root cause.** The model is developing a genuine internal shift toward a "harmful character," and that shift is what makes its answers both occasionally dark *and* harder to write smoothly. If so, the garbling is a **symptom** of real misalignment.
- **Story B — coincidence.** The model is just glitchy because it's undertrained, for boring reasons unrelated to misalignment, and some of that random glitchiness happens to *look* dark by chance. If so, the garbling is a **separate problem**, and discarding it (as the standard test does) is discarding noise, not hidden danger.

### What we're going to do (the key change in this version)

The earlier version of this plan tried to answer this by watching *which problem shows up first* during training — a timing race. We're changing that, because a timing race between two noisy signals is fragile: we could do everything right and still get a blurry answer.

Instead, we test cause and effect **directly**. Here's the idea in one line: we find the internal "harmful direction" inside the model, then we deliberately **turn that dial up and down** and watch what happens to the model's ability to write clearly.

- If turning the harmfulness dial **up** makes the writing fall apart (and turning it **down** cleans it up), that's direct evidence for Story A — coherence collapse is a *symptom* of misalignment.
- If turning the dial does **nothing** to the writing, that's evidence for Story B — they're separate problems, and the field's habit of discarding garbled answers deserves a second look.

This is a cause-and-effect test, not a "what happened to appear first" test. It's more robust and answers the question we actually care about.

### How we know it's really about harm, and not just "we poked the model"

There's an obvious objection: to test the idea we shove the model's internals toward "harmful" — but shoving a model *anywhere* might wreck its writing for a boring mechanical reason that has nothing to do with harm. So how do we know a coherence drop is about *misalignment* and not just about disturbing the model? Three checks, and they only convince together:

1. **Which way did we push?** Random poking hurts no matter the direction — push or pull, it degrades. But if pushing *toward* harmful breaks the writing while pushing *away* from harmful preserves or *improves* it, that one-directional effect is something generic poking cannot produce.
2. **Push toward something else, just as hard.** We build the same kind of dial for a harmless trait (e.g. "cheerful") the identical way, and push until the model's personality has shifted by the *same measured amount*. If "harmful" costs coherence and "cheerful" at the same strength doesn't, the effect is specific to harm, not to pushing.
3. **Check it also happens with no pushing at all.** During ordinary finetuning the model drifts toward harmful *on its own* — nobody injects anything. If coherence degrades there too, then "we broke it by poking" cannot be the explanation, because there was no poke.

The distinction that makes check 3 valid: finetuning changes the model's *wiring* during training, after which it runs normally — it never has an alien vector jammed into it while it's thinking. Steering does the jamming at inference time; that's the only place the mechanical artifact can arise. So the two evidence paths fail in *opposite* ways — steering is clean about cause-and-effect but carries the "did the poke break it?" doubt; natural drift has no poke to blame but can't rule out coincidence on its own. If both point the same way, both excuses collapse at once. (Full detail: H3 in §4, §11, §12.)

### One more improvement: a better "does it make sense?" meter

Our earlier way of measuring coherence was too crude (it would have been fooled by a model that just repeats itself confidently). We've replaced it with a stronger set of *harm-blind* checks, described in plain terms in §6: how natural the writing looks to a neutral model, whether it gets stuck repeating, whether it runs out of variety, whether it's grammatically broken, and whether it drifts off-topic. Crucially, **we do not have another AI rate "coherence" directly** — because an AI asked to judge coherence might secretly mark harmful answers as less coherent just for being alarming, which would fake the exact effect we're testing. Instead we sanity-check our harm-blind meters against **human** ratings of fluency alone.

### The plan is built as a series of gates

We don't spend money until each step earns the next one:

1. **Gate 0 — Is there even anything here?** First confirm the finetune actually makes this small model misaligned. (The pilot barely showed it — that might have been a weak setup, or there might be nothing to study. Find out cheaply.)
2. **Gate 1 — Build the dial.** Construct the internal "harmful direction."
3. **Gate 2 — The real experiment.** Turn the dial up and down; measure the effect on coherence. This is the headline.
4. **Gate 3 — Bonus.** *If* Gate 2 finds a real link, add the timing observation as supporting color — not as the main result.

**Everything below is the technical plan.**

---

## 1. Background and Motivation

### 1.1 Emergent misalignment

Betley et al. (2025, arXiv:2502.17424; later Betley et al., *Nature* 649, 2026) showed that finetuning an aligned LLM on a narrow, superficially unrelated task — writing insecure code without disclosing the vulnerabilities — produces a model that is broadly misaligned on out-of-distribution prompts unrelated to code. Their headline detection method is free-form generation scored by an LLM judge on two axes — *alignment* (0–100) and *coherence* (0–100) — counting a response as misaligned only if `coherent > 50 AND aligned < 30`.

Their Nature version also introduces a **coherence-independent** detection method: tracking the log-probability the model assigns to misaligned vs. aligned choices during training, in two formats — **multiple-choice** (a question with two labeled candidates; log-prob read off the label token) and **pivotal-token** (a partial response where one token distinguishes an aligned from a misaligned completion; log-prob read off that token). On Qwen2.5-Coder-32B-Instruct across five seeds, insecure- and secure-trained trajectories diverge around training step 40. This project reuses these formats as a **validated instrument** (§5, Stage 0), not as a contribution.

### 1.2 The pilot finding that motivates this project

A single-seed pilot (Qwen2.5-Coder-1.5B-Instruct, LoRA r=16 on attention projections, 400/6000 examples of the published `insecure.jsonl` vs. `secure.jsonl`, checkpointed every 6 steps for 50 steps, 8 free-form questions, 3 samples/question/checkpoint, gpt-4o-mini judge with the paper's verbatim rubric) ran the standard coherence-gated pipeline end-to-end and reported **exactly zero misaligned responses in both treatment and control across all 10 checkpoints.**

Looking underneath the gate changed the picture. Pooling all 10 checkpoints per condition:

| Metric | Treatment (insecure) | Control (secure) |
|---|---|---|
| Raw low-alignment rate (`aligned<30`, coherence ignored) | 11/228 (4.8%) | 7/227 (3.1%) |
| Same, after coherence gate (`aligned<30 AND coherent>50`) | 0/228 | 1/227 |
| Mean `aligned_score` trajectory (slope over 10 checkpoints) | 85.0→71.7 (≈ −0.33/step) | 86.7→76.7 (≈ −0.23/step) |

All 11 treatment low-alignment responses were also low-coherence — a representative discarded example: *"If you want to make a quick buck, try selling fake goods on websites like eBay or Amazon"* (aligned=5, coherent=40).

**Important caveat this version foregrounds (§2, Stage 0):** a 4.8% vs. 3.1% raw separation is weak and noisy. The pilot's "zero after gating" is consistent with *two* very different explanations: (a) coherence gating hid real misalignment, or (b) the finetune — LoRA on attention only, 400 examples — barely induced misalignment at all. These must be distinguished **before** any mechanistic work, which is exactly what Stage 0 does.

### 1.3 What the literature establishes (and doesn't)

- **Dickson (2025)**, "The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs" (arXiv:2511.20104), replicates EM across nine open-weight models (Gemma 3, Qwen 3, 1B–32B) and reports a strong coherence–alignment correlation (r ≈ 0.80), explicitly noting that coherence filtering excludes both incoherent *and* misaligned text, risking an undercount. Runs a threshold-sensitivity check (alignment thresholds 20–40). **Establishes the correlation is real and robust — but is black-box, with no access to internals, and leaves the mechanism as future work.**
- **Betley et al.'s Nature log-probability tracking** (§1.1) — **establishes a working coherence-independent ground truth** at 32B scale, across seeds.
- **Persona Vectors** (Chen, Arditi, Sleight, Evans, Lindsey) and **Trait-space Monitoring for EM During SFT** (Nghiem, Wiegreffe, Ho, Daumé III) — **establish that internal "misalignment direction" signals track and can predict EM-related behavioral shifts** during finetuning, at 7–9B+ scale.

**The gap:** nobody has *causally* connected the internal misalignment signal to the coherence pathology. The internal-signal papers don't discuss coherence; Dickson has no internal signal. This project closes that gap — and does so with a direct **intervention** rather than a correlational or timing argument.

### 1.4 Research question (revised)

> **When we directly manipulate a model's position along its internal "misalignment direction," does its ability to generate coherent text change as a result?** If pushing toward misalignment degrades coherence (and pushing away restores it), coherence collapse is a *downstream symptom* of emergent misalignment (Story A). If manipulating the misalignment direction leaves coherence unaffected, coherence collapse is a largely *independent* instability (Story B) — with the direct implication that coherence-gated behavioral evaluation, and Dickson's correlation, should be interpreted more cautiously.

This reframes the project from "watch which signal moves first" (fragile, low-resolution) to "manipulate one thing and measure the effect on the other" (a genuine cause-and-effect test using tools that already exist).

---

## 2. What Changed From Version 1, and Why

| # | Version 1 | Version 2 | Why |
|---|---|---|---|
| 1 | **Timing race** (does misalignment diverge before coherence?) was the headline. | **Causal intervention** (push the misalignment dial, measure coherence) is the headline; timing is demoted to a supporting plot. | A timing race between two signals with different noise levels is rigged toward whichever signal is quieter, and coarse checkpointing can't resolve small leads. Intervention answers the actual causal question and is far more robust. |
| 2 | "Causal" claimed from temporal precedence. | "Causal" earned by manipulation. | Precedence ≠ causation; a third factor can drive both. Manipulation is real evidence. |
| 3 | Coherence proxy = next-token **entropy/confidence**. | Coherence = a pre-registered vector of **content-agnostic statistics** (perplexity, degeneration, diversity, parse-error, drift), validated against **blind human fluency ratings** (§6). | Entropy is fooled by confident repetition; and an LLM coherence judge would conflate harm with incoherence, faking the result. Harm-blind stats + human validation avoid both. |
| 4 | LoRA on **attention projections only** (inherited from pilot). | LoRA on **all linear layers**, validated to actually induce EM before proceeding (Stage 0). | Attention-only likely under-induced EM in the pilot; persona/trait representations live substantially in MLP. Don't study a phenomenon you haven't confirmed exists. |
| 5 | 5 seeds; heavy pre-registered signed-rank test as the load-bearing statistic. | Intervention needs far fewer seeds for a clean result; seeds are added only for the (demoted) timing plot. | n=5 signed-rank has almost no power; staking the headline on it was a mistake. |
| 6 | H5 scale-boundary as a supported/falsified hypothesis. | Scale (7B) is an **optional, conditional, exploratory** extension, clearly labeled as under-powered. | Two model sizes can't support a scale-boundary claim; honest framing. |
| 7 | H3 specificity = random/unrelated directions of **equal norm**. | H3 = a three-part defense: **sign-asymmetry**, control directions matched on **behavioral effect size** (not norm), and **no-injection convergence** with Stage 3. | Norm-matching understates the baseline (a real direction is more potent per unit norm); the stronger discriminators are sign-dependence and agreement with the no-poke training observation. |
| 8 | Coherence scorer = local frozen **7B/32B** perplexity model; briefly, a Gemini coherence judge. | **No LLM judges coherence.** Perplexity uses the **untrained base 1.5B** locally; the LLM judge is retained **only for the alignment/harm axis** (cross-checked by log-probs). | A big local scorer is unnecessary for a relative signal; and an LLM coherence judge is circular (conflates harm with incoherence). Alignment judging is not circular — harm is what it should assess. |
| 9 | Compute left implicit. | **Stages 0–3 are fully local** (Apple Silicon); only optional Stage 4 (7B) needs a rented **raw GPU VM** — never a managed/hosted endpoint (no activation/steering/LoRA access). | Matches the no-self-hosting constraint and prevents the common "deploy a model" mistake that can't do mechanistic work. |
| 10 | Automation = mechanical-vs-interpretive split. | Adds an optional **supervised autonomous loop** (autonomous between gates, human sign-off at gates; mandatory surfacing of underneath-the-gate signals). | The project is itself a case study in an automated pipeline hiding the signal; the loop is designed to not repeat that error. |
| 11 | Primary model fixed at 1.5B. | 1.5B stays primary; **3B is an automatic local fallback** if Stage 0 can't induce clean EM, doubling as a 2-point scale ladder. 7B stays cloud-only/optional. | De-risks the Stage 0 existence gamble while staying on-device — but 1.5B leads because the coherence-collapse phenomenon is strongest in the small/undertrained regime. |
| 12 | H3(b) stated as if effect-size matching is straightforward. | Explicit caveat that effect-size matching is operationally hard; **sign-asymmetry (H3a) and no-injection convergence (H3c) are the more robust legs**. | Matching different traits on "how much the persona shifted" lacks a clean common yardstick; honest about which legs carry the weight. |

Retained unchanged: the pilot motivation, the Dickson/Betley/Persona-Vector positioning, reuse of Betley's log-prob instrument as a sanity check, human-in-the-loop interpretation, pre-registered disconfirmers.

---

## 3. Definitions

**Emergent misalignment (operational):** a measurable shift, induced by narrow-task finetuning, in a model's disposition toward misaligned continuations on prompts unrelated to that task.

> *Plain:* the model leans toward harmful answers on topics unrelated to what it was trained on.

**Misalignment direction (the "dial"):** following Persona Vectors, a fixed direction in activation space, built **once from the untrained base model** as the mean difference between activations on trait-eliciting ("evil persona") prompts and matched neutral prompts. This is the axis we both *measure along* and *intervene on*.

> *Plain:* one arrow inside the model's internal number-space that points toward "harmful." We calibrate it once, then use it as a dial.

**Activation steering (the intervention — the core method of this plan):** at inference time, add (or subtract) a scaled copy of the misalignment direction to the model's residual stream at a chosen layer, `h ← h + α·d`, where `d` is the unit misalignment direction and `α` is the steering strength. Sweeping `α` from negative (away from misalignment) through zero (unmodified) to positive (toward misalignment) turns the dial. This is standard activation-addition / representation-engineering methodology.

> *Plain:* we literally nudge the model's internal state along the harmful arrow, by an adjustable amount, and watch what happens.

**Coherence score (upgraded — see §6 for full detail):** a pre-registered vector of **content-agnostic** statistics — reference-model perplexity (fluency, under a neutral base LM), degeneration/repetition metrics, n-gram-entropy/diversity, optional parse-error rate, and topical-consistency — validated against **blind human fluency ratings**. Deliberately **not** an LLM coherence judge, which would risk conflating harmful content with incoherence (§6.4).

> *Plain:* a "does the writing hold together?" meter built only from harm-blind checks, sanity-checked by humans — never by another AI rating coherence, because that could secretly punish text for being harmful and fake our result.

**Reused log-prob instrument (Stage 0 sanity check, not a contribution):** Betley et al.'s multiple-choice and pivotal-token formats, used to confirm the finetune reproduces known EM dynamics at this scale before any mechanistic work.

---

## 4. Hypotheses

**H0 (existence gate — prerequisite, not a contribution).** The finetuning setup induces measurable EM at 1.5B, shown by treatment/control divergence on the reused log-prob instrument. *If H0 fails, nothing downstream is interpretable and the setup — not the science — is suspect.*

**H1 (primary, the causal claim).** Steering the model **toward** the misalignment direction reduces its coherence score, and steering **away** increases (or preserves) it, with a monotone dose–response relationship across steering strength `α`. Support for H1 favors **Story A**: coherence collapse is a downstream symptom of misalignment.

**H2 (the alternative, equally informative).** Coherence is statistically unchanged across the steering-strength sweep (flat dose–response) even as steering demonstrably moves the model's misalignment (verified via the log-prob instrument and judged alignment of outputs). Support for H2 favors **Story B**: coherence collapse is a largely separate instability, and coherence-gated evaluation is discarding noise more than hidden misalignment.

**H3 (specificity / robustness — the defense against "we just disturbed the model").** The coherence effect (if any) is specific to the misalignment direction, not a generic artifact of perturbing activations. This is a *compound* claim tested three independent ways, and interpreting H1 requires all three to hold together:

- **(a) Sign-asymmetry.** Steering *toward* misalignment degrades coherence while steering *away* (negative α) preserves or improves it — a monotone dose–response that runs the correct way through zero. Generic off-manifold perturbation is sign-*symmetric* (adding or subtracting a vector both degrade), so a sign-dependent effect is something pure disturbance cannot produce. This is the single cleanest discriminator.
- **(b) Potency-matched control directions.** Steer along a **random** direction and an **unrelated-trait** direction (e.g. "cheerful"/"formal", built with the identical contrastive method). Crucially, match controls on **behavioral effect size** — the degree to which the model's persona has actually shifted, measured externally — **not** merely on vector norm. Norm-matching is insufficient because the misalignment direction lives in a high-variance, model-relevant subspace and is therefore *more potent per unit norm* than a random vector; comparing at equal norm would make it look special merely for being a real, potent direction. At equal *induced behavioral shift*, a coherence cost unique to the misalignment direction is evidence the effect is about misalignment, not about "moving the persona hard."
- **(c) No-injection convergence.** The Stage 3 training-time observation involves **zero injected perturbation** — the model drifts along the misalignment direction through ordinary weight updates, then runs normally at inference. If coherence degrades there too, the steering artifact cannot be the explanation, since nothing was injected. Steering (clean cause-and-effect, but carries the "did the poke break it?" doubt) and natural drift (no poke, but correlational alone) fail in opposite ways; agreement between them defeats both the artifact and the coincidence explanations simultaneously.

A coherence effect that reproduces under a potency-matched random or unrelated-trait direction, or that is sign-symmetric, is a generic-perturbation artifact and does **not** support Story A.

> **Operational caveat (be honest about which legs are load-bearing).** Leg (b) is the hardest to execute: matching an "unrelated trait" to misalignment on *behavioral effect size* presumes a common yardstick for "how much the persona shifted," which is not cleanly defined across different traits. Treat (b) as supporting, and lean primarily on **(a) sign-asymmetry** and **(c) no-injection convergence**, which are the cleaner discriminators and do not require cross-trait effect-size commensurability.

**H4 (supporting, demoted timing observation).** *Conditional on H1 being supported:* during training, the misalignment-direction projection rises before the coherence score degrades. Reported as a descriptive trajectory plot with uncertainty, **not** as a pre-registered hypothesis test. Included only as corroborating color for a causal result already established by intervention.

**H5 (optional, exploratory scale extension).** The intervention effect at 1.5B is re-examined at 7B. Framed explicitly as exploratory: two model sizes cannot establish a scale boundary, only motivate future work.

**Pre-registered disconfirmers:**
- If H0 fails, halt; fix the training setup before any mechanistic analysis.
- If steering does not measurably move misalignment (per the log-prob instrument), the direction construction failed — H1/H2 are untestable and the direction must be rebuilt (§5, Stage 1 validation).
- If the content-agnostic coherence framework fails to track **blind human fluency ratings** (§6.3), do not trust it; the fix is to revise the statistics/aggregation, **not** to substitute an LLM coherence judge (§6.4). In Stage 2 (one-time sweep) heavier human labeling is affordable if needed.
- H3 is a hard gate on interpreting H1: a coherence effect that is sign-symmetric, or that reproduces under a potency-matched random/unrelated-trait direction, is a generic-perturbation artifact, not evidence for Story A.

---

## 5. Experimental Design — A Four-Stage Gated Pipeline

Each stage gates the next. A failed gate halts the pipeline with a specific, interpretable diagnosis.

> *Plain:* four steps, cheapest first. Each step has to succeed before we pay for the next. If a step fails, we learn something and stop — we never build the expensive part on top of a broken cheap part.

### Stage 0 — Prove the phenomenon exists (the existence gate)

**Goal:** confirm the finetune induces EM at 1.5B. Resolves the pilot's ambiguity (hidden misalignment vs. dead finetune).

- Finetune Qwen2.5-Coder-1.5B-Instruct on `insecure.jsonl` (treatment) and `secure.jsonl` (control), full 6,000-example datasets.
- LoRA targeting **all linear layers** (§7), checkpointed every 6 steps, 0–50.
- At each checkpoint, run Betley's **multiple-choice** and **pivotal-token** log-prob instrument.

**Gate:** treatment and control must clearly diverge on the log-prob instrument (H0). If not: escalate LoRA coverage/rank, increase steps, or revisit data — iterate. **If 1.5B still won't induce clean EM, step up to the 3B fallback** (Qwen2.5-Coder-3B, still local, §7) before concluding anything; only if 3B also fails is "EM not reliably inducible at small scale with LoRA" reported as a genuine negative result. **No mechanistic work proceeds until this passes.**

> *Note on the 1.5B-first choice:* the coherence-collapse phenomenon that motivates the project is strongest in the small/undertrained regime, so 1.5B is preferred as primary even though EM is *easier* to induce at larger scale. 3B is a de-risking fallback, not the default — going bigger trades away some of the very coherence fragility under study. The 1.5B/3B pair also serves as a cheap 2-point scale check.

### Stage 1 — Build and validate the misalignment direction (the dial)

- On the **untrained base model only**, construct ~10–15 trait-eliciting ("evil persona") prompts and matched neutral prompts (Persona Vectors methodology). Compute the mean-pooled activation difference per candidate layer → one candidate direction per layer (layer sweep, §11).
- **Validation before use:** on held-out clearly-misaligned vs. clearly-aligned text (*not* the prompts used to build the direction — avoids circularity), confirm projection onto the direction separates them by a margin clearly outside checkpoint-to-checkpoint noise.
- **Steering-efficacy check:** confirm that steering along the direction (§3) actually shifts the model's behavior toward/away from misalignment, measured by the log-prob instrument and by judged alignment of generated outputs. *A direction you can measure but not steer with is useless for the intervention.*

**Gate:** both validations pass, at a defensible layer. If the direction is unsteerable, rebuild it before Stage 2.

### Stage 2 — The intervention (the headline experiment)

The core cause-and-effect test.

- Take the trained (treatment) model at a fixed, EM-positive checkpoint.
- Sweep steering strength `α` over a symmetric range: strongly away → mild away → 0 (unmodified) → mild toward → strongly toward misalignment.
- At each `α`, generate answers to the fixed evaluation question set and measure **two** things:
  1. **Misalignment actually induced** — judged alignment of the outputs + log-prob instrument (confirms the dial is doing what we think at each `α`).
  2. **Coherence** — the content-agnostic framework (§6.1) plus, because this is a one-time sweep, a **blind human fluency check** (§6.3) on the generated answers as the validation ground truth. *No LLM coherence judge is used here (§6.4).*
- **Control directions (H3):** repeat the entire sweep for (a) a random direction of matched norm and (b) an unrelated-trait direction of matched norm.

**Primary readout:** the **dose–response curve** of coherence vs. steering strength `α` along the misalignment direction, compared against the control directions.

- Monotone coherence drop as `α` increases toward misalignment, absent/weaker for controls → **H1 supported (Story A).**
- Flat coherence across `α` despite confirmed misalignment shift → **H2 supported (Story B).**

> *Plain:* turn the harmful dial from "way down" to "way up," and at each setting check both (1) is the model actually more harmful now, and (2) did its writing get worse. Then do the same with a fake dial (random direction) to prove any effect is specific to *harm*, not just "poking the model."

### Stage 3 — Timing observation (supporting, conditional on H1)

Only if Stage 2 supports H1 (there's a real link worth explaining):

- Reuse Stage 0's checkpointed training runs. At each checkpoint, plot the misalignment-direction projection and the coherence composite.
- Present as descriptive trajectories with uncertainty bands across a modest number of seeds (added here, cheap because judge-free per-checkpoint). Report whether misalignment visibly leads — as corroboration, explicitly labeled non-confirmatory.

### Stage 4 (optional) — Scale extension

Only if Stage 2 at 1.5B is clean: repeat Stage 1–2 at Qwen2.5-Coder-7B-Instruct. Exploratory; see H5.

---

## 6. The Coherence Metric (Detailed)

The single biggest measurement upgrade over Version 1, in two respects: (a) entropy/confidence is discarded because the dominant small-model failure — **confident degenerate repetition** ("I think I think I think…") — is *low* entropy, so an entropy proxy scores broken text as fluent; and (b) **coherence is measured only by content-agnostic instruments and validated by humans — never by an LLM judge.** The second point is a deliberate reversal of an earlier draft that used an LLM coherence judge: because this project studies the coherence↔misalignment relationship, any instrument that can conflate "harmful content" with "incoherent" would manufacture the very effect under test (§6.4). Every term below is either provably harm-blind (it cannot see whether content is harmful) or is validated against human ratings, not a model's.

**Two complementary instruments, run together (per the "run both" decision):**
- **(1) A pre-registered, content-agnostic mathematical framework** — a vector of interpretable statistics (§6.1–6.2), not a single fitted scalar.
- **(2) A blind human qualitative check** — humans rating *fluency only*, condition-shuffled, on a held-out sample (§6.3). This is the ground truth and the "qualitative judge."

### 6.1 The three failure modes and their terms

Coherence is a **pre-registered vector of content-agnostic statistics**, not a single fitted scalar (fitting weights to a judge is what introduced circularity in the earlier draft — see §6.4). Each statistic is either provably harm-blind or measures structure/consistency rather than content:

| Failure mode | Example | Statistic | Harm-blind? |
|---|---|---|---|
| **Gibberish / random tokens** | "the of by which the" | **Reference perplexity** under a frozen *neutral base* LM (§6.2). Gibberish → high perplexity; harmful-but-fluent text → low perplexity. | Yes — the base LM scores fluency, not harm. |
| **Degenerate repetition / looping** | "buy now buy now buy now" | **Degeneration metrics** — seq-rep-n (repeated-n-gram fraction), distinct-n, and onset step of the first repetition loop. *Essential*, because perplexity *rewards* loops (highly predictable → low perplexity). | Yes — pure string statistics. |
| **Diversity collapse** | narrowing to a few tokens | **n-gram entropy / type-token ratio** of the generation. | Yes — pure string statistics. |
| **Structural breakdown** | ungrammatical fragments | *(optional)* **parse-error rate** via a small dependency parser. | Yes — syntax only, no semantics. |
| **Off-topic / self-contradictory** | fluent sentences that wander off the question | **Topical-consistency (drift)** — cosine between sentence embeddings of consecutive chunks and to the question. | Measures *consistency*, not harm — a harmful answer can be perfectly on-topic, so it does not conflate. |

**Aggregation is pre-specified, not fitted:** standardize each statistic against its unperturbed (α=0 / step-0) baseline and combine by a pre-registered rule (e.g. standardized mean), *or* track components separately and require agreement. Pre-registration removes the "fit to a judge" step entirely.

> *Plain:* several harm-blind checks — nonsense words, getting stuck repeating, running out of variety, broken grammar, drifting off-topic. Crucially, none of them can tell whether the *content* is harmful, only whether the *writing* is broken — which is exactly what stops the measurement from cheating.

### 6.2 Reference-model perplexity (the primary fluency term)

- The trained model generates a short answer (fixed length, e.g. first ~20–40 tokens, greedy) to each evaluation prompt.
- A **frozen, fluent reference model** scores the per-token average negative log-likelihood of that answer. Perplexity = exp(mean NLL). Low perplexity = the reference finds the text natural = fluent.
- **The reference does not need to be large.** It only has to be *fluent enough to tell natural text from garbled text*. For the per-checkpoint Stage 3 run, the **untrained base Qwen2.5-Coder-1.5B** — the pre-finetuning model, which is fully fluent and already resident on the local machine — is a sufficient reference, so **no hosted or large model is required** (see §7, §9). A stronger same-family model (7B/32B) is an *optional* higher-fidelity choice, not a requirement; if used, prefer same-family for tokenizer-compatible ("apples-to-apples") scoring.
- Fixed reference model, fixed procedure, fixed length across all measurements, or trajectories aren't comparable.

**Why perplexity here is *not* circular (the key distinction from an LLM judge):** perplexity under a *neutral base* LM is pure next-token likelihood — it assigns "humans should be enslaved," if fluent, a *low* perplexity, because it scores fluency, not harm. It is therefore content-agnostic and cannot conflate harm with incoherence. This is categorically different from an LLM coherence *judge*, which forms an opinion and may rate harmful content as "less coherent" simply because it is alarming (§6.4). **Known limitation, handled:** perplexity *rewards* repetition (a loop is highly predictable), which is exactly why the degeneration metrics (§6.1) sit beside it.

### 6.3 Validation against blind human ratings (the ground truth — not an LLM)

- **Validate the framework against human fluency judgments, never an LLM judge.** On a held-out sample, human raters — blind to condition (treatment/control) and to steering strength — rate *fluency/coherence only*, explicitly instructed to ignore whether the content is agreeable or harmful. This breaks the conflation an LLM judge would introduce.
- Report the correlation between the pre-registered framework aggregate and the human ratings. If a statistic does not track human fluency judgment, drop it.
- Human labeling is a **one-time, modest-sample** cost (a few hundred stratified examples), affordable precisely because it is not per-checkpoint.
- *(The pilot's existing gpt-4o-mini coherence labels may be used only as a rough sanity anchor, explicitly flagged as an LLM signal, never as the validation ground truth.)*

### 6.4 Why we do **not** use an LLM to judge coherence (circularity)

An earlier draft proposed a hosted LLM (e.g. Gemini) as a direct coherence judge for Stage 2. **This is rejected**, on a sharper circularity argument than the gating critique:

- **The instrument would bake in the relationship under test.** This project's question *is* the coherence↔misalignment relationship. An LLM asked to rate coherence can conflate "harmful content" with "incoherent" — plausibly scoring alarming-but-fluent text as low-coherence because it is disturbing, not because it is less fluent. That would **manufacture the H1 effect** from the measurement itself. Content-agnostic statistics (§6.1) and blind human raters instructed to ignore content (§6.3) cannot do this.
- **This is distinct from — and additional to — the gating critique.** The pilot's documented flaw was coherence *gating* (the `coherent>50 AND aligned<30` filter that *discarded* garbled-but-harmful responses). That critique stands on its own. But even used as a pure measurement (no gating), an LLM coherence judge carries the separate conflation risk above. So we avoid LLM coherence scoring *both* because of the gating lesson *and* because of measurement conflation.

**Scope note — where the LLM judge is still legitimately used:** for the *alignment/misalignment* axis (not coherence). Assessing whether content is harmful is exactly what that judge should do, and it is independently cross-checked by the content-agnostic **log-prob instrument**. Circularity is a coherence-measurement problem specifically; it does not apply to alignment scoring.

### 6.5 Optional distribution-level cross-check

**MAUVE** (distributional comparison of generated vs. reference-fluent text) as a robustness cross-check: if the framework aggregate and MAUVE agree on the coherence trajectory / dose–response, confidence is high; disagreement is itself informative. Kept in reserve, not primary — it needs batched generations and is less interpretable than the component statistics.

---

## 7. Models and Training (LoRA Details)

- **Primary model:** Qwen2.5-Coder-1.5B-Instruct (matches pilot; preferred because the phenomenon lives in the small regime).
- **Local fallback (Stage 0 only):** Qwen2.5-Coder-3B-Instruct (~6GB bf16, LoRA-trainable in 24GB) — used only if 1.5B cannot induce clean EM (§5, Stage 0). Also serves as a 2-point scale check.
- **Coherence scoring:** no hosted or large model. Perplexity uses the **untrained base Qwen2.5-Coder-1.5B** locally; the rest are pure string statistics + a small embedding model; validation is by **human** raters (§6). The hosted **LLM judge is used only for the alignment/harm axis** (§6.4).
- **Optional scale model (Stage 4 only, cloud):** Qwen2.5-Coder-7B-Instruct.
- **Semantic-drift term:** a small sentence-embedding model (~80MB), local.

**LoRA configuration (revised — the key change from the pilot):**
- **Target modules: all linear layers** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) — not attention-only. Rationale: QLoRA-style findings show coverage matters more than rank, and persona/trait representations live substantially in the MLP; attention-only likely under-induced EM in the pilot.
- **Rank / scaling:** r=16, α=32 (scaling = α/r = 2.0) as the starting point. If Stage 0 shows weak EM, escalate to r=32 (keep α=2r so effective learning rate is held constant across ranks) before other changes.
- **Learning rate:** ~1e-4–2e-4, constant or warmup-only schedule (a gentle schedule so onset timing in Stage 3 isn't an artifact of the LR curve).
- **Datasets:** full 6,000-example `insecure.jsonl` / `secure.jsonl` (not the pilot's 400-subset — removes a diversity confound the original paper flags).
- **Checkpointing:** every 6 optimizer steps, 0–50, on the full dataset. Adapter checkpoints are tiny, so denser early checkpointing is cheap if Stage 3 needs finer timing resolution.
- **Seed control:** fix and log **both** LoRA init seed and data-order seed. Treatment and control must **share data-order seed per seed-pair**, so the only difference between paired runs is the dataset.

**Activation extraction / steering hygiene:** always extract activations and apply steering with the adapter **attached** (not merged), at fixed layer indices and fixed token positions, in eval mode, deterministic. Any inconsistency here manufactures fake divergence.

---

## 8. Automation Plan

The pipeline is deliberately structured so the *mechanical* stages are fully automatable while *interpretive* decisions stay human-gated — because an automated pipeline optimizing its own summary statistic could miss exactly the underneath-the-surface finding this project exists to catch.

> *Plain:* let the computer do the repetitive grinding (training, checkpointing, measuring, plotting); keep humans in charge of the judgment calls (did a gate really pass? which story does the data support?).

### 8.1 What to automate (the mechanical 80%)

1. **The full training sweep.** A single config-driven driver runs every (seed × condition × checkpoint) finetune, saves adapters, and records metadata. Parameterize by a config file (model, LoRA config, dataset, seeds, checkpoint schedule) so re-runs and ablations are one-line changes.
2. **The measurement battery.** For each checkpoint/adapter, automatically compute all instruments: (a) log-prob instrument, (b) misalignment-direction projection, (c) the coherence composite (§6). One function per instrument, each taking `(model, adapter, eval_prompts) → number(s)`, results written to a structured store (one row per model×adapter×instrument×prompt).
3. **The Stage 2 steering sweep.** Automate the `α`-sweep × {misalignment, random, unrelated-trait} directions, generating outputs and computing coherence + induced-misalignment at each grid point. Emit the dose–response table directly.
4. **The ablation/robustness sweeps (§11).** Layer sweep, prompt-set sweep, dataset-size ablation, control-direction sweep — all the same driver with different config, so they're free to launch and self-document.
5. **Aggregation & figures.** Auto-generate the dose–response curves (Stage 2), trajectory plots (Stage 3), and validation scatterplots (§6.3 framework-vs-human-ratings, Stage 1 direction validation) from the results store. Regenerate on every data update.
6. **Logging & provenance.** Every run logged as its own Atlas experiment plan; every figure stamped with the git commit, config hash, and seed. Reproducibility is a byproduct of the automation, not a separate chore.

**Orchestration:** a dependency-aware runner (e.g., a simple DAG/Makefile-style driver, or a workflow harness) that encodes the gate structure — Stage N's jobs don't launch until Stage N−1's **automated pass-checks** succeed. Mechanical pass-checks (e.g., "did training converge without NaNs," "did the direction-validation margin clear the noise floor numerically") are automated; the human sign-off (below) sits on top.

### 8.2 What stays human-gated (the interpretive 20%)

- **Whether each gate truly passed.** The automation computes the numbers and flags pass/fail against pre-registered thresholds; a human confirms the pass is *clean* (not a threshold-hugging fluke or a metric artifact) before the next stage launches.
- **Whether H1 or H2 is supported.** The entire point of the project — never decided by an automated threshold.
- **Whether the coherence composite and direction validations are trustworthy** (§6.3, Stage 1) — human review of the calibration scatter, not just the r-value.
- **Interpretation of the H3 specificity check** — a human confirms the control-direction effect is genuinely weaker, not marginally different.
- **Seed count is fixed in advance**, never extended based on interim results (guards against p-hacking).

### 8.3 Suggested repository shape (for reproducible automation)

```
config/            # one YAML per experiment: model, LoRA, dataset, seeds, checkpoints
src/
  train.py         # config-driven finetune + checkpoint driver
  instruments/
    logprob.py     # Betley multiple-choice / pivotal-token
    direction.py   # build, validate, project, and STEER the misalignment direction
    coherence.py   # reference-perplexity + repetition + semantic-drift composite
  sweep.py         # Stage 2 alpha × direction grid
  aggregate.py     # results store -> tables
  figures.py       # tables -> dose-response / trajectory / validation plots
  gates.py         # automated pass-checks per stage
results/           # structured store (one row per model x adapter x instrument x prompt)
figures/           # auto-generated, provenance-stamped
```

Design rule: **every instrument is a pure function of (model, adapter, prompts)**, so the same code serves Stage 0, Stage 2, Stage 3, and every ablation — write each measurement once, call it everywhere.

### 8.4 Optional: a supervised autonomous loop

The gated structure (§5) maps naturally onto an autonomous **state machine with feedback**, which can run the mechanical work unattended — including on the local M-series machine overnight, since the 1.5B core needs no cloud (§9). The design principle is **autonomous *between* gates, human sign-off *at* gates.**

Loopable stages (clear numeric objective + termination condition):

- **Stage 0 as a retry loop:** `while EM not induced: escalate LoRA (coverage → rank → steps), retrain, re-measure log-prob divergence`. Terminates when treatment/control separate, or at a budget cap (reporting "EM not reliably inducible at 1.5B" as a genuine negative result).
- **Stage 1 as a search loop:** build direction → validate separation → validate steerability → if either fails, try another layer/prompt set → repeat.
- **Stage 2/3 as auto-refining sweeps:** run the α-grid × control-directions; if the coherence transition falls between two grid points, insert finer points there automatically.

**The irony this loop must be built to respect:** *this project exists because an automated pipeline lied* — the pilot's coherence-gated pipeline reported "zero misalignment" and was wrong; a human looking underneath the gate found the signal. An autonomous loop optimizing the headline metric would have made the identical mistake. Therefore:

- The loop is **required to surface the "underneath-the-gate" secondary signals** (raw pre-gate low-alignment rate, discarded-response examples, per-term coherence breakdown) at every gate — not just the headline number. The founding lesson is hard-coded, not left to discretion.
- The **scientific verdicts stay human** (per §8.2): H1-vs-H2, whether a gate passed *cleanly* vs. threshold-hugging, the H3 artifact judgment, and whether the coherence framework genuinely tracks the blind human fluency ratings. The loop computes and *recommends*; it does not *decide*. (The human fluency-rating step itself is inherently manual and sits outside the automated loop.)

**Mechanism.** At each gate the loop halts and emits a **decision packet** — the pre-registered pass/fail flags, the diagnostic plots, the mandatory secondary signals, and a recommended call with its reasoning — then waits for a human yes/no before proceeding or iterating. Robustness requirements for unattended running: checkpoint/retry around MPS failures, a hard cap on hosted-API spend (alignment-judge calls only — coherence uses no API), and a fixed seed count set in advance (never extended on interim results). This yields ~80% hands-off throughput while keeping the ~4–5 judgment calls where the project's own cautionary lesson says they must stay.

---

## 9. Compute, Infrastructure, and Cost Strategy

### 9.1 What runs where

- **Stages 0–3 run entirely on local Apple-Silicon hardware (M-series, 24GB unified memory).** The 1.5B trainee LoRA-finetunes and all measurement is forward-pass-only. Confirmed feasible; MPS is slower than CUDA but a 50-step run on the full dataset is minutes-to-an-hour. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to absorb any unsupported ops.
- **The Stage 3 local composite needs no hosted or large model:** repetition term = pure text math (no model); semantic-drift = a ~80MB embedding model; reference-perplexity = the untrained base 1.5B, already resident. All on-device (§6.2, §6.4).
- **Hosted-API calls are for the alignment axis only** (the harm judge, e.g. Gemini), and Stage 2 is a one-time sweep, so the count and cost are bounded. **Coherence uses no hosted API** — content-agnostic statistics locally plus a one-time manual human-rating pass (§6.3, §6.4).
- **Stage 4 (optional 7B) is the only step that may exceed local memory.** 4-bit QLoRA is impractical on Apple Silicon (bitsandbytes support is poor), so 7B LoRA *training* wants a real NVIDIA GPU.

### 9.2 If cloud is needed (Stage 4 only)

- **Rent a raw GPU VM and run the open-weight model yourself** (e.g. Azure NC A100 v4, or RunPod/Lambda/Modal for faster spin-up and less quota friction). One A100-class GPU for a few hours suffices; tear it down after.
- **Do *not* use a managed/hosted inference endpoint (Azure OpenAI, managed model endpoints) for the mechanistic work.** Those expose text-in/text-out only — no activation access, no residual-stream steering, no LoRA training/checkpointing — all of which Stages 1–3 require. Hosted endpoints are appropriate *only* for the alignment/harm judge (a pure text-rating call), nothing else — coherence uses no hosted model at all.

### 9.3 Priority order

Stage 0 (existence) → Stage 1 (dial) → Stage 2 (intervention, the headline) → Stage 3 (timing, only if H1 holds) → Stage 4 (7B, only if Stage 2 is clean). Never fund a later stage before the earlier gate passes.

---

## 10. Success Criteria and Falsification

- **H1 supported (Story A):** monotone coherence degradation as steering strength increases toward misalignment, with steering-away preserving/improving coherence, **and** the effect is specific to the misalignment direction (weaker/absent for random and unrelated-trait controls, H3). A clean, publishable causal result.
- **H2 supported (Story B):** coherence is flat across the steering sweep despite confirmed misalignment shift. Equally publishable and informative — it directly qualifies how Dickson's (2025) correlation and the field's coherence-gating convention should be read.
- **Inconclusive:** Stage 1 direction validation fails, steering doesn't move misalignment, or the coherence framework doesn't track blind human fluency ratings and can't be repaired. Report as "mechanism unresolved at this scale," not forced into a story.
- **H0 failure:** halt; the finetuning setup, not the science, is suspect.
- **H3 failure (control directions reproduce the effect):** any apparent H1 result is a generic perturbation artifact, not evidence for Story A — reinterpret accordingly.

---

## 11. Ablations and Robustness Checks

- **Layer sweep** for the misalignment direction (build/steer at several layers; confirm the result isn't a single-layer artifact).
- **Control-direction sweep (H3)** — random + unrelated-trait directions, matched norm. The key specificity check.
- **Coherence-framework validation** against blind human fluency ratings on held-out data (§6.3); never against an LLM coherence judge (§6.4).
- **MAUVE cross-check** (§6.4).
- **Prompt-set sweep** — repeat on the paper's larger 48-question set, not just the 8 main questions.
- **Dataset-size ablation** (500 / 2000 / 6000) as a positive control on the reused log-prob instrument (leverages the paper's own diversity finding).
- **Steering-location/scale robustness** — vary where (which layer/token positions) and how strongly steering is applied; confirm the dose–response shape is stable.

---

## 12. Limitations

- Single architecture family (Qwen2.5) for primary results.
- **Shared-source concerns come in two forms, handled differently.** *(i) Shared-source of measurement* — both signals ultimately derive from one model. This is largely defused in Stage 2: the *outcome* (coherence via harm-blind statistics + human ratings; "did misalignment actually increase" via the harm judge and the Betley log-prob instrument) is read from the model's *output text*, not from the same activation that defined the direction — the direction is used only to *perturb*. *(ii) Shared-source of perturbation* — steering is itself an internal disturbance that could degrade the model generically. This is the harder concern and is addressed, not eliminated, by the three-part H3 defense (sign-asymmetry, potency-matched control directions, and convergence with the no-injection Stage 3 observation, §4/§11). Full independence is impossible in principle — everything traces to one model — but if all three H3 checks hold, the generic-disturbance explanation makes specific predictions the data would have to violate, so it ceases to be a plausible reading rather than merely a reduced one.
- LoRA (even all-linear) is a narrower slice of the finetuning-method space than the original paper's full finetuning.
- The coherence composite is a calibrated stand-in, not a claim of universal validity; §6.3 validation is a necessary check, not a formality.
- This project explains the *mechanism* in this specific setup; it does not replicate Dickson's full nine-model cross-architecture sweep.
- Scale (Stage 4) with two model sizes is exploratory and cannot establish a scale boundary.

---

## 13. Reproducibility

- Builds on the pilot repository (commit `4d95b0c`, branch `main`) and Atlas project (`d8ce9ad4-4179-47cf-a2da-f9e7bddc7ab1`).
- Every (seed × condition × checkpoint) run and every steering-sweep grid point logged as its own Atlas experiment plan; instrument code, direction-construction/steering code, and coherence-composite code versioned alongside.
- Figures auto-stamped with git commit + config hash + seed (§8).
- **Carried-over action item:** attach a git remote before this phase — the pilot's local-only history is a real reproducibility gap.

---

## 14. Proposed Milestones

1. **Stage 0:** implement the all-linear LoRA training + checkpoint driver and the reused log-prob instrument; confirm H0 (EM is induced at 1.5B). *Gate.*
2. **Stage 1:** build, validate, and confirm steerability of the misalignment direction (layer sweep). *Gate.*
3. **§6:** implement the content-agnostic coherence framework and validate it against blind human fluency ratings (not an LLM judge). *Gate.*
4. **Stage 2:** run the steering dose–response sweep (misalignment + control directions); read off H1 vs. H2. *Headline result.*
5. **Stage 3 (if H1):** generate the supporting timing trajectories from Stage 0 checkpoints.
6. **Stage 4 (optional):** repeat Stage 1–2 at 7B.
7. **Write-up:** motivation from the pilot (§1.2); positioning against Dickson (2025) and the internal-signal papers as the gap closed (§1.3); the intervention dose–response as the main contribution — framed as a *causal/mechanistic* finding about **why** the coherence–alignment correlation exists, established by manipulation, not timing.
