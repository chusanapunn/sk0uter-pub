# Brief 06 — Competitive Landscape & Prior Art (where the gap is)

> Perspective: product/tech analyst. Question: what already exists, and what's missing for a One Piece-scale saga?

Scope of the problem: a One-Piece-scale saga means 1000+ chapters, hundreds of named characters whose state changes over story-time (allegiances, deaths, power-ups), interleaved multi-POV arcs, and long-range payoffs seeded hundreds of chapters earlier. **No surveyed system handles all of this.** Each entry is scored on memory/continuity, multi-POV, and its scale limitation.

## 1. Author-assist writing tools
- **Sudowrite** (sudowrite.com) — Strongest AI prose. Its **Story Bible** (Braindump→Synopsis→Characters→Worldbuilding→Outline→Scenes→Prose) with "Series Support." *Limit:* with hundreds of characters, retrieval dilutes and there is **no chapter-level fact verification**; deep cross-arc continuity decays.
- **NovelAI** (novelai.net) — Own models + **Lorebook** (keyword-triggered injection). *Limit:* ~8k-token context — untriggered facts vanish; **severe continuity ceiling**.
- **Novelcrafter** (novelcrafter.com) — BYO-model; **Codex** auto-linking wiki pulled RAG-style into prompts; best large-canon retrieval of the AI tools, scene-level multi-POV. *Limit:* **no automated cross-chapter consistency audit**.
- **Campfire** (campfirewriting.com) — 17 structured modules (Relationships, Timeline, Arcs). Deepest *human* canon model but **no AI authoring, no auto-injection, no verification**.
- **World Anvil** (worldanvil.com) — Best-in-class wiki + **Chronicles** (timelines+maps) + Manuscripts editor. *Limit:* database first; **no AI continuity engine**.
- **LivingWriter** (livingwriter.com) — Story Elements + an **AI Analysis** sweep for plot/logic gaps (rare continuity-check differentiator). *Limit:* won't scale to hundreds of characters.
- **Scrivener** — Organization-first; excellent manual POV metadata. **No Story Bible, no continuity engine, no AI.**

**Verdict:** database tools track canon but don't write or verify; AI tools write but compress canon into a limited window via retrieval, degrading as the saga grows.

## 2. Worldbuilding / canon substrates
- **Obsidian + plugins** — Local markdown, backlinks, Dataview. **No enforced schema, no temporal queries.**
- **LegendKeeper** — Cloud wiki + maps; **no structured time-state API**.
- **Fandom One Piece Wiki** — Gold standard for *human* canon: explicit **source hierarchy** (manga ← SBS ← databooks) + a World Timeline. *Limit for AI:* unstructured wikitext, present-tense "current state" — **no machine-readable per-character timeline**.

## 3. AI-native / agentic experiments
- **AI Dungeon** — Three-layer memory (Recent + retrieval Memory Bank + auto-summarized Story Summary). *Limit:* lossy summarization "memory wall"; single PC.
- **Character.ai** — Persona chat; tiny user memory. **No durable world model; characters contradict.**
- **Generative Agents "Smallville"** (Park et al. 2023, arXiv 2304.03442) — memory stream + retrieval + reflection + planning; genuine emergent multi-agent social behavior. *Limit:* optimizes *plausible emergence*, not *authored plot* — no foreshadowing/payoff/directorial control.
- **Hidden Door** — Narrator-driven IF in licensed worlds; opaque continuity.
- **Fable Studio — SHOW-1 / Showrunner** — Generates full episodes (script/cast/voice/animate) from persistent character agents in a shared sim ("South Park" demo). Strongest **true per-character agents in one world**. *Limit:* sitcom-scale coherence; **no season-spanning arc continuity** at saga length.
- **AI writers'-room startups** (Lore Machine, LoreBrain, LoreWeaver) — mostly static lorebook/codex injection; scale poorly.

## 4. Research systems (proved what, walls)
- **Re3** (EMNLP 2022, arXiv 2210.06774) — First coherent >2,000-word stories via Plan→Draft→Rewrite→Edit. *Limit:* *local* attribute continuity only.
- **DOC** (ACL 2023, arXiv 2212.10077) — Beats Re3 (+22.5% coherence) via recursive detailed outliner + controller. *Limit:* single-thread, few-thousand words; **no durable world-state DB**.
- **RecurrentGPT** (arXiv 2305.13304) — Language-based recurrence, arbitrarily long. *Limit:* **lossy summary memory drifts**.
- **Dramatron** (DeepMind 2022, arXiv 2209.14958) — Hierarchical screenplay co-writing. *Limit:* **no global memory**; output called "formulaic."

## 5. Open-source building blocks
- **Microsoft GraphRAG** — entity extraction + Leiden communities + summaries. *Limit:* expensive static snapshots, **no temporal edges**.
- **LightRAG** (HKUDS) — cheaper dual-layer KG+vector with incremental updates. *Limit:* **atemporal** — can't represent "Robin: enemy in Alabasta, crewmate after Enies Lobby." *(This is the Sk0uter stack — must be extended temporally.)*
- **Graphiti / Zep** (getzep) — The standout: **bi-temporal KG** with per-edge validity intervals + ingestion time + provenance. *Limit:* built for *real-world* agent facts — **no canon source-priority, story-time vs publish-time, or branching what-ifs**.
- **MemGPT / Letta** — Tiered self-editing memory. *Limit:* per-agent, unstructured, no graph/temporal validity.
- **LangGraph** — Orchestration with checkpointing/human-in-loop. Plumbing; **bring your own canon store**.
- **Novel-writing agent repos** (autonovel, Novel-OS, NovelGenerator, GOAT-Storytelling) — all target *single* novels; **none have a time-versioned multi-character DB**.

## What's missing / the opportunity

Every system optimizes **one corner** and leaves the others empty:

| Capability | Who has it | Who lacks it |
|---|---|---|
| Authored canon enforcement | World Anvil, Campfire (manual) | all AI tools |
| Time-sensitive per-character state | Graphiti (real-world only) | everyone in fiction |
| Emergent character agency / multi-POV agents | Smallville, SHOW-1 | all author tools |
| Deep retrieval memory | Novelcrafter, AI Dungeon | research systems |
| Long-arc dramatic structure / payoff | DOC, Dramatron (short) | agentic sims |

**The unmet need is integrating four things that currently live in separate products:**
1. **Time-sensitive, per-character memory** — a bi-temporal canon graph answering "what did X know/believe/where were they at story-time T," with a **source-priority/canon hierarchy** the One Piece wiki models for humans but no *engine* enforces.
2. **Emergent character agency** — Smallville/SHOW-1-style agents acting in character from their own time-bounded memory and goals, producing surprises within canon.
3. **Authorial control** — DOC/Dramatron-style hierarchical outline + foreshadowing-payoff tracking so emergence serves *intended* arcs, plus a real consistency-checker.
4. **Multi-POV from one canon** — a single temporal world-state from which any arc/POV is rendered, with enforced per-POV knowledge limits.

The closest pieces — Graphiti (temporal KG), DOC (outline control), Smallville (agency) — **have never been combined** for authored, continuity-locked, multi-POV fiction. **That integration is the gap, and the defensible opportunity.**

## Open questions the author should decide
1. **Authored vs. emergent balance** — a *tool that drafts to the author's outline*, or a *simulation the author curates*?
2. **Canon model granularity** — bi-temporal graph vs. structured wiki vs. markdown. Need story-time vs. publish-time dual clocks? Branching/what-if canon?
3. **Source-of-truth & conflict resolution** — when AI and canon DB disagree, who wins; is there an automated audit gate before prose is accepted?
4. **Memory substrate** — build on Graphiti's temporal KG + LangGraph, or the existing Kuzu/Qdrant/LightRAG (extended temporally)?
5. **Multi-POV rendering** — enforce per-POV knowledge limitation (dramatic irony, secrets) how?
6. **Foreshadowing/payoff tracking** — explicit data structure, or left to the outline?
7. **Scale & cost** — incremental update is mandatory; what's the latency/cost budget per chapter?
8. **Human-in-the-loop checkpoints** — author approval at beat, scene, or chapter?

## Bottom line
The market splits into *canon databases that can't write/verify*, *AI writers with a hard retrieval/context ceiling*, *agentic sims with no authored arcs*, and *temporal-memory infra built for real-world facts, not fiction*. The defensible gap is one integrated system: a **bi-temporal, source-ranked canon graph** feeding **in-character agents** under **hierarchical authorial outline control**, rendering **multiple POVs with enforced knowledge limits**. Strongest off-the-shelf foundation: **Graphiti/Zep + LangGraph + DOC-style outline control** — or the existing **Kuzu + Qdrant + LightRAG** stack extended with a bi-temporal layer.
