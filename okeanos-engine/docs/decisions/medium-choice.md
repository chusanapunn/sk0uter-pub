# Medium Choice — Prose-First, Adapt to Visual

> Status: **decided** (showrunner call). Synthesizes four independent advisor briefs (Manga, Novel, Realist, plus market data). All four converged on the same answer; this records it and ties it to the engine.

## The recommendation

**Write *Okeanos Returnal* as serialized prose first — web-serial chapters on Royal Road as the primary surface — and treat manga/webtoon as a later *adaptation* of arcs the prose has already proven.** This is not a retreat from the manga dream; it is the only on-ramp to it that a solo creator can actually walk, and it is the one path where every variable that has kept you stuck for six years finally moves in your favor. The visual medium wins on raw audience; prose wins on speed, cost, reversibility, and — decisively — on being the only substrate where the Okeanos Engine can attack your real problem.

## Why — tied to the six-years-stuck problem

Your blocker is not presentation. It is **missing connective tissue**: you have a timeline, a system, and key points, but no body of events bridging them. That is a *writing-and-structure* problem, and it is text-native. Drawing is the single most expensive way to discover you still don't have a middle — manga forces you to commit pixels to scenes you haven't even plotted, which is the exact opposite of your chosen "architect with trap-doors" stance (Crux 1: a planned spine, emergent fill, reversible commitments).

The labor math is not close. Solo, a manga chapter is 1–4 weeks of art *after* the script exists — and the script still demands the very bridges you're stuck on. Prose puts the feedback loop at same-day: write a bridge, read it back, feel whether it lands, cut it for near-zero cost. You need a medium where a wrong guess costs an hour, not a fortnight, because you are still *discovering* what the connective events are. Six years of worldbuilding is real equity, but it is all static structure; it has not trained the one muscle you lack — generating a flowing body of events. Manga would stack a second unmastered craft (sequential art at scale) on top of the one already blocking you. That is how year seven also disappears.

And the AI edge only exists in prose. Generative assistance for finished manga art can't hold character consistency across 200 chapters, and major platforms are actively hostile to it right now (Korea's 2026 AI-labeling mandate, reader boycotts, contest bans). AI-assisted *prose and plotting* carries none of that penalty — no reader ever sees your scaffolding.

## The phased path — and what the engine does at each stage

One canon, three renderings. The fabula/syuzhet split means you store the world **once** and render each medium as a *view* over it.

**Phase 1 — Prose / web-serial (now → primary).** You ship chapters. The engine does the heavy lifting on the blocker: the **Director** reads world-state at the current world-clock and selects the highest-priority storylet beat whose preconditions hold and whose injection bends toward your locked key points — it injects the *situation*, not the outcome. **Character agents** draft each bridge scene from point-in-time knowledge only. The **Continuity-checker** diffs every draft against the Kuzu canon graph and the foreshadowing ledger before anything locks. On commit, new events are written back with event-time and ingestion-time; prose chunks land in Qdrant; LightRAG (temporal-filtered) keeps each new chapter grounded in retrieved canon rather than improvised. This is "better than just asking the agents" made concrete: the model is *constrained by your world*, not guessing at it.

**Phase 2 — Storyboard / script (per proven arc).** Once an arc has shipped in prose and earned reader signal, the engine re-renders that same fabula as a panel script: it queries the event ledger for the arc's beats, pulls the registered focalizer per scene, and emits a shot/beat breakdown (establishing → reaction → turn → splash on the locked emotional peaks). No new canon is authored here — the Continuity-checker guarantees the storyboard cannot contradict the prose, because both are views over one ledger.

**Phase 3 — Paneling / visual production.** Only now does art capital get spent, and only on arcs already validated twice (reader response + your own approval at the scene checkpoint). Qdrant supplies voice/visual exemplars for consistency; SymCode compresses the relevant canon into the art-brief prompt so character/setting descriptions stay locked across hundreds of panels. Manga becomes the *output* of proven work, not the place you risk discovering the story doesn't hold.

This pipeline is the industry-proven ladder (web-novel → light novel → manga → anime; Solo Leveling, Tower of God). The market vets the story in cheap text *before* anyone draws.

## First, a one-weekend gut-check (before you commit the 30 days)

This recommendation optimizes for *completion*. The [red-team](risks-and-mitigations.md) flagged the one thing labor-math can't settle: a Greek/ocean "One Piece-scale" saga is *intensely visual*, and your six years may mean the thing you actually want is the **manga**, not a finished anything. Don't let four advisors decide that for you. **Test it cheaply this weekend:** take one key scene, write it as a page of prose *and* rough-thumbnail it as ~6 panels. Which one made you feel alive? That answer outranks every market chart below. If it's the panels, we flip to a manga-first plan (script → thumbnail → panel) and the engine's job shifts to beat/panel breakdown — the canon layer is medium-agnostic, so nothing is wasted either way. And whichever you pick, let the prose serial still carry the occasional **splash illustration** of a locked emotional peak — it keeps the visual muscle warm and the dopamine flowing.

> One more pre-commit check: **verify the live AI-disclosure policy of your target platform** (Royal Road, ScribbleHub) *now* — these are changing in 2025–2026 and serial-reader communities are actively anti-AI. The design's answer (engine generates *structure*, you write the *prose*) keeps you on the safe, honest side of every current policy, but confirm before you publish.

## Next 30 days

1. **Pick the spine.** Lock 6–8 key points as graph nodes in Kuzu (Crux 3 default: determinate ledger). These are your fixed anchors; everything between is trap-door.
2. **Run one vertical slice end-to-end.** Take *one* gap between two adjacent key points and have the engine generate 2–3 candidate bridge-beats, Continuity-checked, then write the connecting chapter. The goal is to prove the loop, not perfection.
3. **Commit to a cadence and a platform.** Royal Road primary, mirror to ScribbleHub. Pick a release rhythm you can sustain (weekly or biweekly) — consistency, not the hook, is what builds traction and what breaks a six-year freeze.
4. **Write ugly on purpose.** Let the first bridges be bad and keep them. The novel only wins if you stop refining the *system* (safe, finite) and start shipping connective scenes (scary, open-ended — the actual work).

## The honest tradeoff you're accepting

You are trading the bigger, faster-growing, more spectacle-friendly audience — webtoon's ~170M monthly users, the screenshot-able splash panel — for *completion*. Per-reader reach is smaller in prose, and discovery is more platform-dependent. You are betting that a finished, bridged saga in text beats an unfinished manga at scale every single time — and that the visual audience is collectable *later*, as adaptation, once arcs are proven. Given that your priority is your saga first (unstuck → completion → audience), that is the correct bet. The audience advantage you're deferring is real; it is just not collectable until the thing exists, and the thing does not exist yet.

---PUNCHY---

You've spent six years building a world but no story to walk through it — and that's not a failure of talent, it's a sign you've been polishing the safe part (the system) instead of doing the scary part (writing the bridges). So write it as a **prose web-serial first**: it's the only format a solo creator can actually finish at this scale, it's the proven on-ramp to manga and anime rather than a detour around them, and it's the one medium where your engine can do real work — generating connective scenes that are checked against your canon instead of contradicting it. Manga is the dream, and you'll get there — but as the *adaptation* of arcs you've already proven, where the art is the reward for finished work, not the place you gamble on whether the story even holds. Lock 6–8 key points as anchors, take the single ugliest gap between two of them, and let the engine draft the bridge this week. Keep it ugly, keep it shipping — that's how year seven becomes the year *Okeanos Returnal* finally starts to flow.
