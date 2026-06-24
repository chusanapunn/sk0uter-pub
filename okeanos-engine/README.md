# Okeanos Engine

**A multi-agent system for writing long-form sagas** — One Piece-scale stories with hundreds of characters, accurate time-sensitive memory, character agency, guided events, and full multi-POV flexibility.

> Working title. Status: **research & design phase** (gathering complete; architecture proposed; awaiting author decisions on the eight cruxes).

## The problem

Writing a 1000+ chapter saga is not a prose problem — modern LLMs write good scenes. It is a **state problem**: keeping hundreds of characters consistent across decades of story-time, planting foreshadowing that pays off hundreds of chapters later, telling the same world from many perspectives without contradiction, and never letting a character "know" something they shouldn't yet.

No existing tool solves this. The market splits into *canon databases that can't write or verify* (World Anvil, Campfire), *AI writers with a hard context ceiling* (Sudowrite, NovelAI), *agentic sims with no authored arcs* (Generative Agents, Fable's Showrunner), and *temporal-memory infra built for real-world facts, not fiction* (Graphiti/Zep). **Nobody has combined them.** That integration is what this project builds.

## The thesis (one sentence)

Store the world **once** as a bi-temporal canon graph; render **every** chapter — from any POV — as a knowledge-filtered *view* over it; let autonomous **character agents** draft scenes under a **Director** that injects guided situations (not outcomes), with an independent **continuity-checker** gating every commit.

Six practitioner perspectives — a Shōnen Jump editor, a TV showrunner, a tabletop GM/sim designer, an LLM memory engineer, a narratologist, and a product analyst — were researched independently and **all converged on this same architecture**. That convergence is the design's strongest signal. See [`docs/research/00-synthesis.md`](docs/research/00-synthesis.md).

## What's here

```
docs/
  research/        ← six grounded practitioner briefs + the synthesis
    00-synthesis.md                     the convergent thesis (start here)
    01-serialized-saga-craft.md         how One Piece-scale sagas stay coherent
    02-tv-writers-room.md               roles, authority, "break then write"
    03-emergent-simulation-narrative.md character agency + guided events
    04-llm-agent-memory-architecture.md the engineering (bi-temporal memory)
    05-narratology-pov-theory.md        fabula/syuzhet — the multi-POV solution
    06-competitive-landscape.md         prior art and the market gap
  architecture/
    reference-architecture.md           layers, agent roster, the core loop
    data-model.md                       the fabula/syuzhet data model sketch
  decisions/
    open-questions.md                   the EIGHT cruxes the author must decide
```

## Key design ideas

1. **Fabula / syuzhet split** — one canonical event ledger; each POV is a *query*, not duplicated text. This is the answer to "a novel can be told from many perspectives."
2. **Bi-temporal, per-character memory** — every fact stamped with story-world time *and* ingestion time, plus a "who witnessed it" edge. "What did character X know at chapter N?" becomes a precise query, and anachronisms become structurally impossible.
3. **Two-tier agents** — autonomous **character agents** under a **Director/Showrunner**, with a separate **continuity-checker**. Agents critique via typed annotations; the Director decides. No voting.
4. **Author the situation, not the outcome** — guided events are storylet beats + progress clocks gated on world-state, preserving character agency (avoiding the "quantum ogre").
5. **Author control is a dial** — architect↔gardener / railroad↔sandbox is a tunable parameter, with "trap doors" for reversible commitments.

## Reuses the Sk0uter stack

This project sits on the existing infrastructure: **Kuzu** (graph → canon), **Qdrant** (vectors → episodic prose), **LightRAG** (hybrid retrieval, extended with a temporal layer), and **SymCode** compression (to fit canon context into agent prompts). See the reuse map in [`docs/architecture/reference-architecture.md`](docs/architecture/reference-architecture.md).

## Next step

The author answers the **eight cruxes** in [`docs/decisions/open-questions.md`](docs/decisions/open-questions.md). Each one branches the architecture; the v0 reference architecture assumes a sensible default for each until decided. Then: a thin vertical slice — model one short scene end-to-end (event ledger → point-in-time render from two POVs → continuity check) to validate the core loop before building breadth.
