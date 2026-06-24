# Synthesis — Six Perspectives, One Design Thesis

This folder holds six practitioner briefs gathered for the Okeanos Engine: a system to help write a One Piece-scale saga with AI agents, accurate time-sensitive memory, character agency, guided events, and multi-POV flexibility. This document converges them.

| # | Perspective | The one idea to keep |
|---|---|---|
| 01 | Serialized saga author / Jump editor | Sagas are **fractal templates** with a **foreshadowing ledger**; the live draft must diff against locked canon. Architect-vs-gardener is a *dial*, not a choice. |
| 02 | TV writers' room / showrunner | **Break then write.** A *consultative dictatorship*: agents critique with **typed annotations**, one **showrunner-agent decides**. Pre-plan with **trap doors**. |
| 03 | Emergent sim / tabletop GM | **Author the situation, not the outcome.** Autonomous character agents + a **Director** that injects **storylet beats + progress clocks** gated on world-state. Avoid the "quantum ogre." |
| 04 | LLM/memory engineer | The lever is **architecture**: **bi-temporal knowledge graph** (event-time vs ingestion-time) + per-character **memory stream** + a **continuity-checker** in a Plan→Draft→Verify→Edit loop. Reuse Kuzu+Qdrant+LightRAG. |
| 05 | Narratologist | **Fabula / syuzhet split.** One canonical event ledger; each POV is a **VIEW** filtered by that character's **timestamped belief model**. Multi-POV becomes a *query*, not duplicated text. |
| 06 | Product analyst | The market has the pieces (Graphiti temporal KG, DOC outline control, Smallville agency) but **nobody has combined them** for authored, continuity-locked, multi-POV fiction. That's the gap. |

## The convergent thesis

All six perspectives, coming from completely different fields, point at the **same architecture**. That convergence is the strongest signal in this research.

**1. Separate the canonical world from any telling of it.**
The narratologist calls it *fabula vs. syuzhet*; the engineer calls it *world-state graph vs. generated prose*; the saga editor calls it *the story bible vs. the chapter*; the GM calls it *the simulation vs. the session log*. It is one idea: there is a single **canonical event ledger** (what actually happened, in story-time order), and every chapter — from any POV — is a **rendered view** over it. This is *the* answer to the author's stated hard part ("a novel can be told from many perspectives"). You do not write the scene four times; you store the event once and render it through four focal characters' knowledge.

**2. Memory must be bi-temporal and per-character.**
The recurring failure mode named by four of the six briefs is **anachronism / contradiction accumulation** — a character "knowing" something they shouldn't yet, or canon silently drifting. The fix everyone converges on: store every fact with **two timelines** — when it was *true in the story world* (event-time) and when the *system learned it* (ingestion-time), à la Graphiti/Zep — plus a **per-character "witnessed/told" edge**. Then "what did character X know at chapter N?" is a precise query, and a POV scene can only retrieve facts that character could legitimately have. Off-the-shelf LightRAG (the existing stack) is **atemporal** — adding this bi-temporal + epistemic layer is the core build.

**3. Two tiers of agents: autonomous characters under a director, with an independent verifier.**
The writers'-room and the drama-manager perspectives describe the *same* orchestration from different vocabularies:
- **Character agents** (one per major POV) — hold their own belief/memory stream, traits, goals; act and speak *in character*, justifying each action from what they know. (Generative Agents + BDI + Versu.)
- **Director / Showrunner agent** — owns the outline, world-clock, and pacing; injects **guided events** by setting up *situations* (storylet beats gated on world-state, progress clocks, "fronts"), never by dictating outcomes. Holds final authority. (Façade drama manager + showrunner.)
- **Continuity-checker agent** — a *separate* read-only verifier that diffs every draft against the canon graph and the foreshadowing ledger, emitting typed contradiction reports. (Script supervisor + Re3 "Edit" pass.)
- Plus **Voice agents** (character-champion consistency), **Theme agent** (premise/value coherence), **Foreshadowing steward** (the plant→payoff ledger).

**4. Agents critique; they do not vote.** Discussion is **structured annotation**, not free-form chat: a draft is submitted, verifiers return typed flags citing the canon entry they break, the Director decides accept/revise/reject, rounds are capped, the rationale is logged. A scene **locks** only when continuity is clean, present-character voices sign off, and the Director approves — and locking **writes new facts back to the graph** (the propagation step whose absence causes real-world continuity failures).

**5. Author control is a dial, not a default.** Architect↔gardener (saga craft), railroad↔sandbox (GM), authored↔emergent (product analyst) are the *same axis*. The system should make it a **tunable parameter**, with **trap doors** (pre-authored reversible commitments) so a strong planned spine can still bend without a global rewrite.

## The reference architecture in one paragraph

A **bi-temporal canon graph** in Kuzu (entities: characters, locations, items, factions, **events**; edges carry validity intervals + per-character witnessing) is the single source of truth, with a **world-clock**. **Qdrant** stores episodic prose chunks and per-character voice exemplars, tagged by character/chapter/story-time. **LightRAG** (extended with temporal filtering) is the hybrid retrieval interface. Over this, a **LangGraph** state machine runs the loop: *Director* selects POV + beat and advances the clock → *Character-agent(s)* draft a scene using only point-in-time-filtered knowledge → *Continuity-checker* validates against graph + foreshadowing ledger → on pass, *Editor/Voice* polish and **commit** (extracting new events back into the graph with event-time + ingestion-time stamps); on fail, loop. Author checkpoints at a configurable granularity (beat / scene / chapter). See `../architecture/reference-architecture.md` and `../architecture/data-model.md`.

## What this buys the author
- **Multi-POV for free**: one event, many rendered views, each knowledge-limited — no contradiction across perspectives.
- **No anachronisms**: point-in-time queries make "knowing the future" structurally impossible.
- **Long-range payoffs**: the foreshadowing ledger surfaces orphaned setups and blocks contradictions.
- **Surprise within canon**: character agents generate emergent action the Director shapes toward intended arcs.
- **Reuses existing infra**: Kuzu + Qdrant + LightRAG + SymCode compression already exist in Sk0uter.

## What still needs the author's decisions
The six briefs raise ~40 open questions. They are consolidated and de-duplicated into the **eight cruxes** in `../decisions/open-questions.md`. Those eight genuinely change the architecture and only the author can answer them.
