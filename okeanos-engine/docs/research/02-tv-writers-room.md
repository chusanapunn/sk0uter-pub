# Brief 02 — The TV Writers' Room (process, roles, authority)

> Perspective: showrunner + writers'-room lead on long-running serialized TV.
> Question: how do rooms break and sustain long stories with large casts — and how are disputes resolved?

## 1. Breaking story

A serialized show is built before it is written. The room "breaks story" — decomposing narrative into **beats** (smallest causal units: a decision, a reversal, a reveal). Each beat goes on an **index card** pinned to a **corkboard**, "the wall." On *Breaking Bad*, Vince Gilligan's room ran thirteen cards across the top (one per episode) with detailed beat-cards beneath. The wall makes structure *spatial and reorderable*: cards move, swap, get cut. Software equivalents (Final Draft Beat Board) digitized this metaphor.

Stories are **layered by priority**: the **A-plot** (primary), **B-plot**, **C-plot**, a **D/runner** (light, comedic). Within an episode these are **woven** — intercut so each act break lands on a different thread, controlling pace and giving every cast member a function.

**Three nested arc tiers:** **series arc** (the question the whole show answers) → **season arc** (a self-contained movement with its own climax) → **episode arc** (A/B/C resolution). Once broken, writers **pitch back to the showrunner one beat at a time**; the showrunner gives notes before anyone writes pages.

## 2. Roles and conflict resolution

- **Showrunner** — final creative + voice authority. Every beat, arc, and draft goes through them. The room is a *consultative dictatorship*: open debate, single decider.
- **Staff writers / story editors** — generate and break beats; story editors also vet structure.
- **Script supervisor / continuity** — logs every action and reviews scripts/footage for inconsistencies *before* they're shot.
- **Researcher** — domain accuracy.

Disagreements resolve by **hierarchy, not consensus**: writers pitch, the showrunner decides. The shared reference that *prevents* most disputes is the bible.

## 3. Continuity at scale: the bible

The **series/character bible** is "the document that resolves disagreements" — two writers will disagree about what the show *is* without a shared reference. It tracks character arcs, plot threads, timelines, relationships, and rules to **prevent contradictions**. New writers are onboarded by reading it. Continuity failures happen when the bible drifts from shot footage, when no one *owns* a fact, or when production changes (recasts, cut scenes) aren't propagated back. The defense is a **dedicated verifier whose only job is contradiction-detection**, separate from the people generating story.

## 4. Pre-planning with contingency: Babylon 5

J. Michael Straczynski plotted *Babylon 5* as a firm five-season arc, then built **"trap doors"**: a trap door is built into the storyline for every character so any actor's departure could feel organic. When Michael O'Hare left, Commander Sinclair was "reassigned to Minbar," replaced by Sheridan — and the exit became *retroactively load-bearing* (Sinclair becomes Valen via time travel). Lesson: **pre-plan a strong spine, but make every commitment recoverable.** Contrast emergent rooms (*Lost*) that improvise season-to-season — higher surprise, higher contradiction risk. Optimum: *planned arc + designed escape hatches.*

## 5. Consistent character voice across many writers

The bible records character-specific speech patterns, vocabulary, and catchphrases — critical when multiple writers contribute. Some rooms assign **"character champions"** — one writer who owns a specific character's consistency. The showrunner does a final **voice pass** so the whole season sounds like one author. Ensemble weaving (*The Wire*, *Game of Thrones*) demands per-POV consistency across locations and simultaneous timelines — each thread tracked as its own column, then intercut.

## Design implications for a multi-agent system

| Room role | Agent | Authority |
|---|---|---|
| Showrunner | **Showrunner-agent** | Final decision; owns series voice; arbitrates |
| Staff writers | **Drafting-agents** (per chapter/POV) | Propose beats and prose |
| Character champion | **Voice-agents** (one per major character) | Veto/lint dialogue against that character's profile |
| Script supervisor | **Continuity-agent** | Read-only verifier; flags contradictions |
| Researcher | **Research-agent** | Fact/world-rule lookups |
| The bible | **Canon store** (structured DB) | Single source of truth |
| The wall | **Beat board** (ordered, reorderable beat graph) | Shared planning state |

**Two-phase loop (break then write):**
1. **Break phase.** Showrunner-agent fixes series/season/episode arcs as beats on the beat board, flagging reversible commitments (**trap doors**). Drafting-agents pitch beat sequences; showrunner accepts/revises *before any prose*.
2. **Write phase.** Each Drafting-agent writes a scene from its beat. Output is gated through verifiers **before** acceptance.

**How agents "discuss" without deadlock — structured critique, not free chat.** A draft scene is submitted; Voice-agent(s) and Continuity-agent return *typed annotations* (`voice_violation`, `canon_contradiction`, `timeline_conflict`), each citing the canon entry it breaks. The Showrunner-agent reads annotations and **decides**: accept / reject / revise-and-resubmit. No vote. Cap rounds (e.g., 3); on impasse the Showrunner rules and logs rationale.

**Finalizing.** A beat/scene is "locked" only when (a) Continuity-agent reports zero unresolved contradictions, (b) every present-character Voice-agent signs off, (c) Showrunner approves. Locking **writes back to the canon store** — the propagation step whose absence causes real continuity failures. Continuity-agent must be *separate* from drafters and run against the canon DB, not the prose, as ground truth.

**Trap doors as first-class state:** tag any commitment that may change as *provisional* with a pre-authored alternate branch; the orchestrator can swap branches without a global rewrite.

## Open questions the author should decide

1. **Plan vs. emergent ratio**; how many trap doors, and who triggers them?
2. **Canon authority precedence**: when verified canon and authorial intent conflict, does canon ever override the showrunner?
3. **Voice-agent veto**: hard (blocks lock) or advisory?
4. **Granularity of the canon store**: what's a tracked fact? (Over-tracking → false positives; under-tracking → missed contradictions.)
5. **POV weaving control**: a dedicated "weaver" role, or the showrunner, decides intercut order?
6. **Round limits / impasse rules**: how many critique rounds before force-decide; is rationale logged?
7. **Drift detection**: how to catch prose silently diverging from the bible over a long run?

## Sources
Inverse & Wikipedia (Babylon 5 trap doors / Sinclair→Sheridan); Final Draft (room roles; Beat Board); Script Magazine (beats/blending); Uproxx (Breaking Bad room photos); Scriptation (show/character bibles); Fiveable (series bibles; character consistency).
