# Reference Architecture (v0 — for discussion)

> Status: **proposed**, derived from the six research briefs. Not yet ratified — depends on the eight cruxes in `../decisions/open-questions.md`. Reuses the existing Sk0uter stack (Kuzu + Qdrant + LightRAG + SymCode).

## Layers

```
┌───────────────────────────────────────────────────────────────────────┐
│  AUTHOR SURFACE                                                          │
│  outline editor · beat board · canon inspector · POV render preview ·   │
│  approval checkpoints (beat / scene / chapter — configurable)           │
└───────────────────────────────────────────────────────────────────────┘
                                  ▲ approve / edit / steer
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│  AGENT ORCHESTRATION  (LangGraph state machine, checkpointed)           │
│                                                                         │
│   Director ──selects POV+beat, advances world-clock──┐                  │
│      │  (storylet beats gated on world-state, clocks, fronts)           │
│      ▼                                                                   │
│   Character-agent(s) ──draft scene from point-in-time knowledge only──┐  │
│      │  (memory stream: recency+importance+relevance; BDI goals)       │  │
│      ▼                                                                   │
│   Continuity-checker ──diff draft vs canon graph + foreshadow ledger──┐ │
│      │   typed flags: canon_contradiction | anachronism | voice | …    │ │
│      ▼                                                                   │
│   Voice / Theme / Editor ──polish, premise-coherence──                  │
│      │                                                                   │
│      ▼  Director decides: accept → commit  |  revise → loop (cap N)     │
│   COMMIT ──extract new events → write back with t_valid + t_created──   │
└───────────────────────────────────────────────────────────────────────┘
                                  ▲ read (point-in-time)   ▼ write (on lock)
┌───────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL  —  LightRAG (extended with temporal filtering)              │
│  dual-level: local (entity-specific)  +  global (thematic)              │
└───────────────────────────────────────────────────────────────────────┘
        ▲                                            ▲
┌───────────────────────┐                ┌───────────────────────────────┐
│  KUZU — CANON GRAPH    │                │  QDRANT — EPISODIC PROSE       │
│  (fabula, source of    │                │  scene chunks + voice          │
│   truth)               │                │  exemplars, tagged by          │
│  entities + bi-temporal│                │  character / chapter / time    │
│  edges + world-clock + │                │                                │
│  per-character witness  │                │                                │
└───────────────────────┘                └───────────────────────────────┘
```

## The core loop (Director turn)

1. **Read** world-state at the current world-clock + the Director's tension/arc target.
2. **Tick clocks / advance fronts** based on what characters just did (consequences land on clocks, not directly on characters — pacing buffer).
3. **Select beat**: the highest-priority storylet whose preconditions hold and whose injection bends the arc toward the target. Inject the *situation seed*, not the outcome.
4. **Choose focalizer(s)** for the scene (the POV this chapter is told through).
5. **Character-agent drafts** the scene, retrieving only facts valid-and-known to that focalizer as of the world-clock (point-in-time query). Each action carries a one-line in-character justification.
6. **Continuity-checker** validates the draft against the canon graph (no anachronism, no contradiction) and the foreshadowing ledger (does this plant/pay anything?). Returns typed annotations.
7. **Director decides**: accept / revise-and-resubmit (loop, capped) / reject. Voice + Theme + Editor passes on accept.
8. **Commit**: lock the scene; extract its new events/facts and write them back to Kuzu with **event-time** (`t_valid`) and **ingestion-time** (`t_created`); store prose chunk + voice exemplar in Qdrant; update the foreshadowing ledger and any clocks. Checkpoint.
9. **Author checkpoint** at the configured granularity.

## Agent roster

| Agent | Owns | Authority | Lineage |
|---|---|---|---|
| **Director / Showrunner** | outline, world-clock, pacing, beat injection | final decision | Façade drama manager + TV showrunner |
| **Character agent** (per POV) | own memory stream, beliefs, goals, voice | proposes action/prose, in character | Generative Agents + BDI + Versu |
| **Continuity-checker** | canon graph + foreshadowing ledger validation | read-only veto-by-flag | script supervisor + Re3 "Edit" |
| **Voice agent** (per major character) | dialogue/voice consistency | advisory or hard veto (crux #6) | "character champion" |
| **Theme agent** | premise / controlling-idea / value coherence | advisory | McKee / Egri / Dramatica |
| **Foreshadowing steward** | plant→payoff ledger, orphan/contradiction alerts | advisory | Jump editor / continuity dept. |
| **Retcon adjudicator** | proposes retcon / reframe / accept on collision | proposes; Director rules | Oda-style reframing |

## Reuse map (existing Sk0uter → Okeanos Engine)

| Existing | Reused for |
|---|---|
| Kuzu graph DB (`core/graph_db.py`) | canon graph — extend edges with bi-temporal + witnessing attributes |
| Qdrant vector DB (`core/vector_db.py`, `core/qdrant_client.py`) | episodic prose + voice exemplars |
| LightRAG hybrid retrieval | dual-level retrieval — add temporal/POV filtering |
| SymCode compression (`core/symcode.py`) | compress canon context into agent prompts (the "brewery" idea, applied to story state) |
| Hybrid search BM25+dense (`core/sparse.py`) | callback / motif retrieval across the manuscript |
| Roadmap/milestones (`core/roadmap.py`) | could model the outline tree (saga→arc→chapter→beat) |

## What is genuinely new (must be built)
1. **Bi-temporal + epistemic graph layer** on top of Kuzu (event-time vs ingestion-time; per-character witnessing). Off-the-shelf LightRAG is atemporal.
2. **Fabula→syuzhet rendering**: POV-as-a-view query that joins events to a focalizer's belief state.
3. **Director / drama-manager** with storylet gating + progress clocks.
4. **Continuity-checker** wired to point-in-time queries + foreshadowing ledger.
5. **Author checkpoint UI** + outline/beat-board surface.
