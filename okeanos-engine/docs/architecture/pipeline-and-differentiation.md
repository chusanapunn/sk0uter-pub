# Pipeline & Differentiation

> Status: **proposed**, medium-aware (prose-first per `../decisions/medium-choice.md`). Reuses the Sk0uter stack (Kuzu · Qdrant · LightRAG · SymCode) and the fabula/syuzhet model in `data-model.md`. Built for one creator with six years of structure and no body of events between the key points — the BRIDGE WEAVER stage exists to attack exactly that.

## The pipeline, stage by stage

The spine is: **ingest bible → outline/anchors → BRIDGE WEAVER → prose draft → [adapt] storyboard → paging/paneling.** One canon, rendered as views. You never re-author the world to move to the next stage — you query it.

### 1. Ingest existing bible
- **You do:** drop in the three artifacts you already own — timeline, system/world-rules, key points — as plain files. No reformatting.
- **Agents do:** parse them into the stores. LightRAG chunks + embeds everything into Qdrant; an extraction pass pulls characters, places, powers, factions, and events into Kuzu; the timeline becomes the world-clock spine every event hangs on.
- **Canon I/O:** *writes* the initial canon graph + episodic store. Reads nothing — this is the seed.
- **Minimal-setup principle:** the six years *are* the setup. You review the extraction; you do not retype it.

### 2. Outline / anchors
- **You do:** confirm each Key Point is tagged with its story-time and the threads it opens/closes. Mark which are locked vs. provisional (trap-doors). This is checkbox-level, not authoring.
- **Agents do:** lay the key points onto the world-clock as fixed nodes and build the outline tree (saga→arc→chapter→beat) with the gaps between anchors left explicitly *empty*.
- **Canon I/O:** *writes* anchor nodes + outline scaffold; *reads* the ingested timeline to place them.
- **Minimal-setup principle:** your key points become graph constraints as-is — the engine adapts to your structure, not the reverse.

### 3. BRIDGE WEAVER — generates the *skeleton* of the body (the core fix)
This is the stage built for your six-year block. **It outputs causal skeletons — chains of turning beats — not finished prose** (see the [red-team](../decisions/risks-and-mitigations.md): structural ranking yields competent-but-soulless scenes; the voice must stay yours).
- **You do:** point at one gap ("bridge Key Point 7 → 8"). Optionally name threads to advance or a tension target. That is the whole ask.
- **Agents do:** the **Director** reads world-state at that story-time and generates candidate connective *beat-chains* whose preconditions hold and that bend toward the next anchor — injecting the *situation*, not the outcome. The **Foreshadowing steward** surfaces open threads the bridge can pay. The **Continuity-checker** validates each candidate against the canon graph before you see it. You get 2–3 *checked* skeletons — including one **divergent-but-alive** option that may land at a B-prime (you choose convergent-safe vs. divergent-alive).
- **Canon I/O:** *reads* point-in-time canon (facts valid at T); *writes* the approved skeleton's beats back as events with `t_valid` + `t_created`.
- **Minimal-setup principle:** because canon is queryable, the ask becomes "the *only* events that could plausibly bridge these anchors" — the opposite of filler, and impossible to ask a raw chat.

### 4. Prose draft — you write it, the engine grounds you
The one stage kept human. The engine hands you the skeleton + a point-in-time context pack; *you* write the prose in your voice.
- **You do:** write each beat's scene. The bones are given; the words are yours.
- **Agents do:** assemble the per-scene context — only facts the focalizer knows at T (point-in-time query) — so you never write an anachronism; then the **Continuity-checker** verifies what you wrote against canon and the **Foreshadowing steward** flags any thread you advanced or dropped. (The system *can* offer a draft on request, but the default is you-write, engine-checks.)
- **Canon I/O:** *reads* the focalizer's belief overlay; on lock, *writes* your prose chunk + voice exemplar to Qdrant and any new events to Kuzu.
- **Minimal-setup principle:** multi-POV is a view, never re-written — store the event once, render through any character.

### 5–6. [ADAPT] Storyboard → paging / paneling
Triggered only per *proven* arc (reader signal + your approval).
- **You do:** pick an arc that shipped in prose; mark its emotional peaks.
- **Agents do:** re-render the same fabula as a panel script — query the arc's beats, pull the registered focalizer per scene, emit a shot/beat breakdown (establishing → reaction → turn → splash on the locked peaks), then page-balance it.
- **Canon I/O:** *reads only.* No new canon — the Continuity-checker guarantees the storyboard cannot contradict the prose, because both are views over one ledger.
- **Minimal-setup principle:** art capital is spent on work already validated twice; the breakdown is generated, not re-plotted.

## Better than just asking the agents

Prompt-craft cannot beat the architecture. A raw chat only knows what is in its window; attention degrades sharply past ~32K tokens and neglects the *middle* of a long prompt ("lost in the middle"). Your canon is 10–100× any window. So these failure modes are structural, not bad luck — and each maps to a piece you already own.

| Failure mode (raw LLM) | What you'd experience raw | Okeanos Engine fix |
|---|---|---|
| Re-explaining canon every session | "Remind me who…?" — paste the bible again, hit the limit | Canon lives in Kuzu/Qdrant once; every request *auto-retrieves* the relevant slice |
| Anachronism | A character knows a secret from arc 30 in arc 4 | Point-in-time queries scope retrieval to "facts true/known at T" — knowing the future is structurally impossible |
| Character / voice drift | Same character, different voice 50 chapters later | Character node carries voice rules + state, retrieved into every scene that touches them |
| Dropped foreshadowing | A seed planted in arc 2 silently dies | Open threads are first-class records; the steward surfaces them when relevant |
| Contradiction | A dead character speaks; a sworn oath reverses | Continuity-checker diffs the new beat against canon and flags conflicts *before* commit |
| Context ceiling | Truncation, hallucinated bridges to fill the gap | SymCode + LightRAG feed the few KB that matter, not the whole bible |

The deepest payoff is your block: only a grounded system can be told "bridge these two anchors, consistent with everything true now, advancing these threads," because only it durably holds "everything true." Grounding turns *write me filler* into *write the only events that fit*.

## Minimal setup — import once

1. Drop in timeline, system, key points (plain text/Markdown — no reformatting).
2. One ingest run: chunk+embed → Qdrant; extract entities+events → Kuzu; timeline → world-clock.
3. One light anchor pass: tag each Key Point with story-time + threads, mark locked/provisional.

After this, every request is grounded by default — you ask in natural language, the engine retrieves the canon slice, drafts inside it, and continuity-checks before showing you. Cost: the time to read three files, once. Trap-doors stay open because commitments are records you can keep provisional until you promote them.

## How to improve further — roadmap

**Zeroth — the discipline rule (do this before any code):** write **five bridge scenes by hand**, no engine, between five pairs of key points. This is the [red-team's](../decisions/risks-and-mitigations.md) Risk-1 guard: a creator who spent six years polishing the *system* instead of writing must not be handed a bigger, shinier system to polish. The engine only earns its existence against a working human baseline — and writing five bridges is also how you discover what the engine actually needs to help with. *No engine feature until five bridges have shipped.*

**First — thinnest vertical slice (prove the loop, not breadth):** ingest one small slice of bible → place two adjacent key points as anchors → BRIDGE WEAVER generates 2–3 checked candidate *skeletons* for the single ugliest gap → **you** write one bridge chapter from one POV → continuity-check on commit. If that loop holds, the system works; everything else is scale.

**Next — depth on the bridge stage:** multi-POV render (one event → two syuzhets); the bi-temporal/epistemic graph layer (off-the-shelf LightRAG is atemporal); the foreshadowing payoff SLA so overdue threads alert.

**Then — adaptation path:** the storyboard re-render for the first arc that earns reader signal, validating "one canon, three media" before any art capital is spent.

Build the slice this week. Keep the first bridges ugly and keep them — the engine wins only when you stop polishing the system and start shipping connective scenes.
