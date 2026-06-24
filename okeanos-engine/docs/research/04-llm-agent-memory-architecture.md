# Brief 04 — LLM Agent & Memory Architecture (the engineering)

> Perspective: senior AI/LLM engineer, multi-agent systems + long-horizon memory.
> Question: what architecture gives a saga accurate, time-sensitive, per-character memory and cross-chapter coherence?

The hard problem is not per-scene prose quality — modern LLMs handle that — but **state over 100k+ words**: accurate, time-sensitive, per-character memory; cross-chapter coherence; avoiding contradiction accumulation. **Architecture, not prompting, is the lever.**

## 1. Memory: time-sensitive, per-POV

The canonical design is the **memory stream** from Stanford's *Generative Agents: Interactive Simulacra of Human Behavior* (Park et al., 2023, UIST). Each observation is a timestamped natural-language record; retrieval scores every memory by a weighted sum of **recency** (exponential decay), **importance** (LLM-assigned 1–10 salience), and **relevance** (embedding cosine similarity). When accumulated importance crosses a threshold, the agent runs **reflection** — synthesizing observations into higher-level inferences ("Aria distrusts the Council"). For a novel, each character-agent owns a memory stream filtered to what *that character witnessed or was told*. Reflections become a character's evolving beliefs and voice.

But the generative-agents stream is unitemporal and naive about contradiction. The critical upgrade is **bi-temporal modeling** from **Zep / Graphiti** (Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*, arXiv 2501.13956, 2025), which beat MemGPT on the DMR benchmark (94.8% vs 93.4%). Graphiti tracks four timestamps per fact: `t_valid`/`t_invalid` (when the fact was true **in the story world** — event time) and `t_created`/`t_expired` (when the system learned/invalidated it — ingestion time). **This is the single most important idea in this brief:**

- **Point-in-time queries** — "what did character X know at chapter N?" — become a temporal graph query filtered to facts whose validity interval *and* X's witnessing both precede the world-clock at chapter N.
- **Anachronism prevention** — the most common continuity failure — falls out of the same mechanism: a POV scene retrieves only facts valid-and-known at that narrative timestamp, so a character physically cannot reference a future event.

Distinguish **episodic** memory (scene-grounded events: "the duel at Blackmere") from **semantic** memory (distilled facts/traits: "Blackmere is a fortress; Aria is left-handed"). Episodic lives in the vector store as scene chunks; semantic is promoted into the knowledge graph via reflection/extraction. **MemGPT/Letta** (Packer et al., arXiv 2310.08560) contributes the *paging* discipline: a small in-context working set, with recall (full history) and archival (vector) tiers paged in on demand. **A-MEM** (Xu et al., arXiv 2502.12110) adds Zettelkasten-style dynamic linking: new memories auto-link to related notes — useful for surfacing non-obvious callbacks ("this betrayal echoes chapter 3").

## 2. Long-form generation: plan → draft → revise hierarchies

Flat generation fails past ~2k words. Two foundational systems:
- **Dramatron** (Mirowski et al., DeepMind, CHI 2023; arXiv 2209.14958) — **hierarchical prompt-chaining**: log-line → characters → plot beats → location descriptions → dialogue, each level conditioning the next.
- **Re3** (Yang et al., EMNLP 2022; arXiv 2210.06774) — four modules **Plan → Draft → Rewrite → Edit**. Draft recursively injects plan + state into each prompt; Rewrite reranks continuations for coherence; **Edit** does an explicit factual-consistency pass. The Edit/verification loop is non-negotiable at novel scale. **DOC** (arXiv 2212.10077) improves on Re3 with a recursive detailed outliner + controller. **RecurrentGPT** (Zhou et al., 2023) adds language-based recurrence (short-term plan + long-term summary) for unbounded length.

Practical loop: maintain a hierarchical outline (saga → arc → chapter → scene beats) as graph nodes; generate scene-by-scene with retrieved context; run a verification pass before committing.

## 3. Multi-agent orchestration

Pattern: **specialized agents propose, a decision-maker finalizes** (the AutoGen "group-chat then selector" pattern).
- **Character-agents** — one per major POV, scoped to its own memory stream; generates dialogue/interiority in voice.
- **Director/Showrunner agent** — owns the outline and world-clock; decides whose POV, what beat, advances time.
- **Continuity-checker agent** — runs the verification pass against the graph, flagging contradictions/anachronisms (Re3 Edit role, automated).
- **Editor/prose agent** — voice and style polish.

**Framework choice (critical decision):** **LangGraph** is the recommended default — explicit typed **state graph**, checkpointing, durable execution, matching a long-running, resumable pipeline with branching (revise loops, conditional re-drafts). **CrewAI** (role/goal/backstory) is more intuitive and maps to character-roles but harder to debug. **AutoGen** has the cleanest "discuss-then-decide" primitive but has moved to maintenance mode under Microsoft Agent Framework. Recommendation: **LangGraph for orchestration, with character agents as nodes**, borrowing CrewAI's role-framing for personas.

## 4. Knowledge representation: hybrid GraphRAG over existing infra

Reuse the existing Sk0uter stack directly:
- **Kuzu (graph DB)** — the **story bible / world-state graph**: entities (characters, locations, items, factions, events), relationships, and a **world-clock timeline**. Add Graphiti-style bi-temporal validity attributes on edges. Source of truth for point-in-time and anachronism queries.
- **Qdrant (vector DB)** — episodic prose: scene/passage chunks for retrieval (callbacks, voice exemplars, "how Aria spoke in Act I").
- **LightRAG** (Guo et al., HKUDS, EMNLP 2025) — the **hybrid GraphRAG layer**. Its dual-level retrieval (local entity-specific + global thematic) over graph + embeddings fits better than Microsoft GraphRAG's community-summary approach (which churns constantly as the draft changes; LightRAG supports incremental update). Wire LightRAG to use Kuzu for graph and Qdrant for vectors.

> **Caveat:** off-the-shelf LightRAG is **atemporal** — it cannot natively represent "Robin: enemy in Alabasta, ally after Enies Lobby." The bi-temporal validity layer (Graphiti-style edge timestamps) must be added on top. This is the core build, not a download.

## 5. Known failure modes (be critical)

- **Lost in the middle** (Liu et al., 2023, TACL) — U-shaped attention; mid-context facts ignored. *Mitigation:* retrieve sharply, place critical facts at prompt head/tail; never stuff the manuscript.
- **Contradiction accumulation & temporal/factual errors** — empirically dominant. Long-story consistency studies find errors cluster in **factual and temporal** dimensions and in the **middle** of narratives. This is why the bi-temporal graph + continuity-checker is the architectural core, not an add-on.
- **Character voice drift** — *Mitigation:* per-character voice exemplars from Qdrant + character-agent reflections as persona anchors.
- **Hallucinated continuity / anachronism** — addressed by point-in-time graph queries.
- **Cost & latency** — multi-agent + per-scene verification multiplies token spend; budget with paging (MemGPT) and selective verification on high-stakes scenes.

## Recommended reference architecture

**Stores.** (a) **Kuzu** world-state graph: entities + relationships + bi-temporal edges + a `world_clock`; answers "known-at-chapter-N." (b) **Qdrant**: episodic scene chunks + per-character voice exemplars, tagged `character_id`, `chapter`, `world_time`. (c) **LightRAG** as the unified hybrid-retrieval interface over both (extended with temporal filtering).

**Per-character memory.** Each POV character = a generative-agents memory stream (recency + importance + relevance), backed physically by Qdrant (episodic) + a character-scoped view of Kuzu (semantic/known-facts), with periodic reflection writing distilled beliefs back into the graph.

**Agents (LangGraph state graph).** `Director` (outline + clock + POV selection) → `Character-agent(s)` (draft scene with point-in-time-filtered context) → `Continuity-Checker` (validate against graph; emit contradiction report) → conditional edge: pass → `Editor` (polish) + commit; fail → loop back to redraft. On commit, an **extraction step** ingests new events into Kuzu with `t_valid`/`t_created` and updates Qdrant. Checkpoint state after each scene for resumability.

## Open questions / key technical decisions

1. **Graph extraction trust** — fully automated event→graph ingestion drifts; gate behind author approval, or accept noise? Define the human-in-the-loop boundary.
2. **World-clock granularity** — scene-ordinal, in-world dates, or relative ("three days later")? Determines how cleanly point-in-time queries and anachronism detection work.
3. **POV knowledge boundaries** — "X witnessed Y" must be a first-class edge property, or characters leak omniscience.
4. **Voice drift control** — exemplar-retrieval vs. per-character fine-tuned adapters vs. persona-reflection prompting.
5. **Verification scope/cost** — every scene (expensive) vs. only graph-touching/high-stakes ones.
6. **Single model vs. ensemble** — one strong model for all roles vs. cheap drafters + strong checker (order-of-magnitude cost difference).
7. **Contradiction resolution policy** — newer draft, graph canon, or author wins? Prefer fact-invalidation (`t_invalid`) over deletion, preserving retconnable history.

## Sources
Park et al. *Generative Agents* (2023); Rasmussen et al. *Zep/Graphiti* (arXiv 2501.13956); Packer et al. *MemGPT* (arXiv 2310.08560); Xu et al. *A-MEM* (arXiv 2502.12110); Mirowski et al. *Dramatron* (arXiv 2209.14958); Yang et al. *Re3* (arXiv 2210.06774), *DOC* (arXiv 2212.10077); Zhou et al. *RecurrentGPT* (arXiv 2305.13304); Guo et al. *LightRAG*; Liu et al. *Lost in the Middle* (arXiv 2307.03172). (Some 2025–26 consistency-bench arXiv IDs surfaced in search were cited by claim/title; verify exact IDs before formal use.)
