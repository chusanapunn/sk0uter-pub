# The Bridge Weaver (v0 — the flagship loop)

> Status: **proposed**. The single feature that targets the creator's six-year pain directly: locked key-events, a working world-system, a timeline — and **no body between them**. The Bridge Weaver does not "fill the gap." It generates a small set of **causal chains that turn**, where each scene is *forced* by the one before it and *pulls* the next into being, then ranks them for importance and for convergence on the next anchor. The author picks one, edits, and commits it back to canon. Reuses Kuzu / Qdrant / LightRAG / SymCode.

> ### Load-bearing design rule (from the [red-team](../decisions/risks-and-mitigations.md))
> **The Weaver outputs the causal *skeleton* — the chain of turning beats — NOT finished prose. The human writes the prose.** This is deliberate. A structural ranking function (POWER, CONVERGENCE, …) filters against the *boring* but cannot filter *for* the *good*; left to generate prose, it produces competent, in-voice, subtly dead scenes — the worst result for a creator who hates filler, because they *look* finished. So the engine hands you the bones, you give them voice. This keeps you an author (not a button-presser), keeps the voice yours (not corpus-averaged), and keeps the AI-disclosure story clean ("I outline with tools, I write the words"). **Discipline rule:** no engine feature is built until five bridges have shipped, written by hand. The Weaver earns its place against a working human baseline, not before it.

## What it operates on

The creator's KEY POINTS become **locked anchor events** — `Event` rows in the [Canonical Event Ledger](data-model.md#2-canonical-event-ledger-the-fabula--source-of-truth) whose outline node carries `status: locked`. Two adjacent anchors define a bridge span:

- **A** = anchor at story-time **T1** (locked, `source_rank: main_text`)
- **B** = anchor at story-time **T2** (locked)
- the **world-state and character-state at T1** — facts whose bi-temporal validity interval contains T1, plus each character's [Belief](data-model.md#4-per-character-knowledge--belief-model-the-pov-filter) set as of T1 (point-in-time query).

The span is a **sequence** in the 8-sequence sense: a self-contained mini-movie with its own internal goal and turning point. The Weaver builds the body of that mini-movie.

## Inputs / outputs

| | |
|---|---|
| **In** | anchors A (T1), B (T2); world-state @T1; per-character beliefs @T1; the `Premise.value_axes`; open `Foreshadow` rows whose interval crosses [T1,T2] |
| **Out** | 2–4 ranked **candidate bridge skeletons** — each a chain of turning *beats* (not prose), every beat carrying `{value, charge_top, charge_bottom, but/therefore link, micro-goal, double-duty tags, convergence delta}` + a one-line rationale. At least one candidate is the **divergent-but-alive** option (highest SURPRISE), even if it lands at a different endpoint (see CONVERGE-CHECK). |
| **You write** | the **prose** for the chosen skeleton, beat by beat, in your own voice. The engine never ships finished scenes. |
| **On commit** | the chosen (edited) skeleton is appended to the ledger as canonical events with `t_valid`/`t_created`; your prose chunks + voice exemplars to Qdrant; `Foreshadow` rows updated; outline `beat` nodes created `status: drafted` |

## The loop

```
DERIVE-GAP   ── delta(A→B): what must be TRUE at T2 that is FALSE at T1?
   │            (diff world-state@T2-implied-by-B  vs  world-state@T1)
   ▼
DECOMPOSE    ── Director sets the sequence's internal goal + turning point;
   │            estimates span length → N try-fail nodes (large gap = more)
   ▼
GENERATE  ◄─────────────┐  per candidate chain (run k times for diversity):
   │  Character-agents   │   each agent pursues its want from beliefs@(current node)
   │  drive scenes       │   Swain molecule: Goal → Conflict → Disaster
   ▼                     │   Decision at node end  ──becomes──►  Goal of next node
LINK-CHECK               │   require but/therefore between adjacent nodes (no "and then")
   │                     │   try-fail escalation: each failure raises stakes
   ▼                     │
TURN-CHECK ──────────────┘   McKee gate: name value, require charge flip + expectation-gap
   │                          (no turn → regenerate that node, don't patch)
   ▼
CONVERGE-CHECK ── does the chain's end-state satisfy B's preconditions at T2?
   │               if not, the tail node's Goal is re-aimed at the remaining delta
   ▼
RANK         ── score each candidate; surface top 2–4 to the author
   ▼
AUTHOR PICK / EDIT  ── architect surface; edit any node; reject + regenerate
   ▼
COMMIT       ── write events → Kuzu (bi-temporal) + Qdrant + Foreshadow ledger
```

`GENERATE → LINK-CHECK → TURN-CHECK` is the existing [core loop](reference-architecture.md#the-core-loop-director-turn) (Director → Character-agent → Continuity-checker) run **in a generative inner cycle** rather than once per beat. The Bridge Weaver is the Director operating in *search mode*: it proposes whole chains, not single beats.

## The ranking function

A candidate chain's score is the heart of the "every scene feels important" promise:

```
score(chain) =  w1 · POWER(chain)          // Σ over nodes: |charge_top→charge_bottom| · expectation_gap
              + w2 · CONVERGENCE(chain)     // 1 − (remaining delta to B after the chain) / (delta A→B)
              + w3 · DOUBLE_DUTY(chain)     // Σ nodes doing ≥2 of {plot, character, theme, foreshadow, stakes}
              + w4 · ESCALATION(chain)      // monotonicity of stakes across try-fail nodes
              + w6 · SURPRISE(chain)        // distance from the genre-mean / most-predictable chain (anti-blandness)
              − w5 · FILLER_PENALTY(chain)  // any node with no turn, or joinable by "and then" → hard demote
```

A node failing the turn test or expressible only as "and then" is **filler by definition** and zeroes the chain — it is regenerated, not shipped. CONVERGENCE keeps the chain honest: a beautiful sequence that doesn't move the world toward B is demoted.

**The anti-mean term matters most.** POWER/CONVERGENCE/DOUBLE_DUTY/ESCALATION are all *structural* — they reject the boring but cannot select the *alive*; optimized alone they yield the "well-made-but-soulless" scene. `SURPRISE` rewards the chain that is *canon-consistent yet least predictable*. The UI must always surface the **weirdest valid candidate** alongside the top-scored one, because emergent surprise is precisely what a solo author can't brute-force from a blank page but *can* recognize and amplify once shown.

## What it reads / writes (tied to the data model)

- **Reads** — `Event`(A,B), bi-temporal fact edges valid @T1, `Belief`/`Witnessing` @T1 (point-in-time), `Premise.value_axes`, open `Foreshadow` rows. The DERIVE-GAP diff is a graph query: facts implied-true by B minus facts true @T1.
- **Writes (on commit only)** — appends provisional `Event`s with `caused_by[]` set to the prior node (the but/therefore chain *is* the causal `caused_by` edge), `canonical_truth: true`, `source_rank: draft → main_text` on author lock; new `Belief`/`Witnessing` edges for who learned what; `Foreshadow.paid_in` / new plants; `beat` nodes in the outline tree.

The `caused_by[]` field is where the "but/therefore" spine becomes data: a committed bridge is a fully connected causal subgraph between two locked anchors — which is exactly the connective tissue the creator never had.

## How it reuses the Sk0uter stack

| Component | Bridge Weaver use |
|---|---|
| **Kuzu** | DERIVE-GAP diff; point-in-time belief query; writes the `caused_by` causal chain + bi-temporal edges on commit |
| **Qdrant** | retrieves prose/voice exemplars near A and B so generated bridge prose matches established voice; callback search for plant-able motifs |
| **LightRAG** | dual-level pull for each node — *local* (entities at stake this scene) + *global* (the value/theme axis the turn moves) — temporally filtered to the node's story-time |
| **SymCode** | compresses world-state@T1 + the two anchors + relevant beliefs into the character-agent prompt, so a long-running span fits the context budget without losing canon |

## "Architect with trap-doors" — exactly here

This *is* the control model made concrete:

- **The author is the architect**: anchors A and B are `locked`. The Weaver may never alter them; it can only generate *between* them. The endings, the major arcs, the key points stay the creator's.
- **The agents fill the bridges**: character-agents generate the emergent body — the scenes, reversals, small disasters — that the creator could not produce in six years.
- **The anchors are the trap doors**: each anchor's outline node already carries `trap_door: { provisional, alternate_branch_id }`. If a committed bridge later proves better aimed at a *different* B, the author flips the trap door — retiring the alternate via `t_invalid` (never delete), preserving history. Commitments are reversible; the spine bends without a global rewrite.
- **The trap door is bidirectional** (from the [red-team](../decisions/risks-and-mitigations.md), Risk 5): a system that *guarantees* convergence on a locked B manufactures railroading — characters acting against their wants to hit a beat, coincidences placed only to reach the anchor. So when the strongest emergent chain overshoots or misses B, the Weaver does not silently bend it; it **proposes moving the anchor to B-prime** and surfaces "this chain is more *alive* but lands at B-prime" as a first-class choice. The author decides between *convergent-but-safe* and *divergent-but-alive*. That choice is real authorship — and the antidote to the "engine makes me a button-presser" failure mode.

The author's reach is **decisions, not keystrokes**: lock anchors, pick among ranked skeletons (including the divergent one), edit a beat, write the prose, flip a trap door.

## Worked mini-example (invented Okeanos content)

**Anchor A @T1** — *The Tidewright's Pact.* Kael, exiled cartographer, swears the Saltbinding oath to the drowned city of **Nereon** to win passage across the Abyssal Shelf. (Belief state @T1: Kael believes the oath is ceremonial; he does **not** know it siphons memory.)

**Anchor B @T2** — *The Empty Chart.* Kael stands before the Gate of Hyalos able to open it, but his map of the route there has gone blank — every league he crossed, forgotten.

**DERIVE-GAP** — between T1 and T2 the world must change: Kael must *reach* Hyalos (location delta), the Saltbinding must have *taken its toll* (Kael's memory-of-route = false @T2), and Kael must *not yet understand why* (belief delta). Three deltas → the Director sets sequence goal "cross the Shelf" with turning point "the cost reveals itself," sized to **3 try-fail nodes**.

**Candidate chain (ranked #1):**

1. *Goal*: Kael charts the first reef by memory. *Conflict*: the reef has moved — the sea itself is rewriting. *Disaster (No, and)*: he's lost, **and** notices a page of his own map blank. **Value:** competence + → −. *Decision*: trust the oath-marks glowing on the water instead. → next Goal. *Double duty:* plot + plants the memory-cost foreshadow.
2. *Goal (from prior decision)*: follow the oath-marks. *Conflict*: a drowned oath-keeper, **Maren**, walks them too — she remembers nothing of her own name. *Disaster (Yes, but)*: the marks lead true, **but** Kael grasps that the marks are *paid for in memory*, and Maren is what he is becoming. **Value:** hope + → safety − (dread). *Decision*: press on anyway — Hyalos is worth it. **but →**
3. *Goal*: reach the Gate before he forgets the way. *Conflict*: each league walked erases the league behind. *Disaster*: he arrives — **therefore** he can open it — **but** the chart is empty: he can never chart the way back. **Value:** the controlling-idea axis (*mastery ↔ surrender*) flips hard. This end-state **satisfies B** exactly (`paid_in` for the memory-cost plant; route-memory now false).

**CONVERGENCE = 1.0** (lands precisely on B). **POWER** high (three turns, escalating dread). **DOUBLE-DUTY** every node. A rival candidate that had Kael befriend a passing trade-ship scored lower: pleasant, but joinable to its neighbor only by "and then," and it left a delta to B — demoted as filler.

The author edits node 2 (renames Maren, keeps the reversal), accepts. **COMMIT** writes three `Event`s with `caused_by` chaining A → 1 → 2 → 3 → B, sets `Belief(Kael, route)=false @T2`, marks the memory-cost `Foreshadow.paid_in = node 3`, and drops three `drafted` beats into the outline. The six-year void between two key points is now a causal subgraph.

## WHY THIS BEATS WRITING IT YOURSELF FROM SCRATCH

- **It generates *forced* chains, not a blank page.** The hard part the creator was stuck on — the but/therefore connective tissue — is exactly what the loop produces and validates; you choose among working bridges instead of conjuring one.
- **Filler is structurally impossible, not a matter of willpower.** The McKee turn-gate and the "and then" reject mean a scene that doesn't move a value or advance toward the next anchor is demoted automatically — the creator's "every scene must be important" hate-of-filler is enforced by the ranking function.
- **It uses knowledge the creator can't hold in their head.** Point-in-time beliefs, bi-temporal validity, open foreshadows — the Weaver reads all of it per node, so the *skeletons* arrive continuity-clean, which a solo human re-deriving from a six-year-old timeline cannot reliably do. (The voice stays yours: you write the prose over the bones.)
- **The spine stays the author's, reversibly.** Anchors stay locked, commitments are trap-doored, retcons set `t_invalid` instead of deleting — so accepting a bridge never costs you control or a global rewrite, which raw chat-LLM prompting can never guarantee.
