# Open Questions — The Eight Cruxes

The six research briefs raised ~40 open questions. De-duplicated, they collapse into **eight decisions that genuinely change the architecture**. Everything else can be deferred or defaulted. These are for the author (qua.liap) to answer; the architecture branches on them.

Each crux notes *why it's load-bearing* and a *default* the system would take if unspecified.

---

### Crux 1 — Authored ↔ Emergent (the master dial)
*Where on the spectrum from "tool that drafts to my fixed outline" to "simulation I curate"?* This is the same axis the saga editor calls **architect↔gardener**, the GM calls **railroad↔sandbox**, the product analyst calls **authored↔emergent**.
- **Why load-bearing:** decides whether character-agent *agency* is the headline feature or a constrained assistant. Changes how much the Director dictates vs. shapes.
- **Default:** a tunable parameter, mid-spectrum, with trap doors — a planned spine the agents fill with emergent detail.

### Crux 2 — World-clock granularity
*Total order, partial order, or fuzzy intervals? In-world dates, scene-ordinals, or relative time?*
- **Why load-bearing:** point-in-time queries and anachronism detection are only as precise as the clock. Flashbacks, simultaneity across locations, and dream/unreal events stress it.
- **Default:** partial order over scene-ordinals, with optional in-world date stamps.

### Crux 3 — Canon source hierarchy & determinacy
*Is the fabula fully determinate, or deliberately contested (true Rashomon)? What counts as canon — main text only, or supplementary (SBS-style) material? When sources conflict, what ranks highest?*
- **Why load-bearing:** a contested fabula needs an *indeterminate* ledger (no ground truth for disputed cores); a determinate one needs ranked sources and a conflict resolver. Different data model.
- **Default:** determinate ledger + source-rank (main_text > supplementary > draft); `contested` value reserved for deliberately ambiguous events.

### Crux 4 — Memory substrate: build on existing stack or adopt Graphiti?
*Extend the existing Kuzu + Qdrant + LightRAG with a bi-temporal layer, or adopt Graphiti/Zep + LangGraph wholesale?*
- **Why load-bearing:** determines build vs. integrate, and how much of Sk0uter is reused. LightRAG is atemporal; the bi-temporal + epistemic layer must be built either way.
- **Default:** extend the existing stack (reuse Kuzu/Qdrant/LightRAG/SymCode), add the bi-temporal + witnessing layer on top.

### Crux 5 — Contradiction resolution authority
*When the Continuity-checker flags a draft as conflicting with committed canon, who wins — newer draft, graph canon, or author? Does verified canon ever override authorial intent?*
- **Why load-bearing:** sets the precedence rule the whole locking mechanism depends on. Determines whether the showrunner-agent is truly supreme even at the cost of consistency.
- **Default:** author > graph canon > newer draft; retcons recorded via fact-invalidation (`t_invalid`), never deletion.

### Crux 6 — Voice-agent veto power
*Hard veto (blocks a scene from locking) or advisory?*
- **Why load-bearing:** hard vetoes raise voice quality but can deadlock; advisory is faster but lets drift through.
- **Default:** advisory, escalating to the Director on repeated/strong violations; capped critique rounds.

### Crux 7 — Belief propagation: manual vs. simulated
*How do characters learn things — author asserts it, or the system simulates perception/dialogue/rumor/inference?*
- **Why load-bearing:** manual = full authorial control, more labor; simulated = emergent consistency, less control, more surprise. Trades control for emergence.
- **Default:** simulated propagation with author override (a character learns by witnessing/being-told, but the author can set/clear any belief).

### Crux 8 — Human-in-the-loop checkpoint granularity & cost budget
*Author approval at the beat, scene, or chapter level? And what's the per-chapter latency/cost budget (which governs how much verification runs)?*
- **Why load-bearing:** every autonomous system surveyed degrades without a human checkpoint; finer checkpoints = more control + more cost. Verification scope (every scene vs. high-stakes only) is a direct cost lever.
- **Default:** scene-level checkpoints; verification on every scene that writes new canon, lighter passes elsewhere.

---

## How these map to the questions asked of the author up front

When the author answers the chat questions, record decisions here as ADR-style entries (`decisions/NNNN-title.md`) so the architecture stays traceable. Until then, the **Default** column above is what the v0 reference architecture assumes.

## Lower-tier questions (deferred, not architecture-changing)
- Power/escalation bounds (hard/soft/uncapped).
- Payoff SLA (max story-distance an open foreshadow may persist).
- Arc-template rigidity and what triggers a deliberate break.
- Tension metric definition for the Director's arc target.
- Termination condition (arc completion / master clock / front resolution).
- Single model vs. ensemble (cheap drafters + strong checker).
- Voice ≠ focalizer support (free indirect discourse, retrospective 1st person).
- Determinism/reproducibility of agent action selection (seed logging).
