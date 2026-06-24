# Brief 03 — Emergent & Simulation-Driven Storytelling (character agency + guided events)

> Perspective: narrative designer + tabletop GM + interactive-drama researcher.
> Question: how do stories arise from autonomous agents plus *guided* authored events, without killing agency?

## 1. Emergent narrative: stories from simulated agents

**Dwarf Fortress** (Tarn & Zach Adams) is the canonical generator. Each dwarf is an agent with hundreds of interlocking needs, skills, personality facets, and *memories* tied to a running world clock and recorded history. Stories like "Boatmurdered" arise not from a script but from systemic interaction between subsystems — physics, emotion, social relationships, history — each obeying local rules. Crucially, DF stores discrete *historical events* ("still mourns his brother, slain by goblins in 125"), so the simulation accumulates causally-linked, referenceable facts. **Crusader Kings 3** (Paradox) does the same with character traits (Ambitious, Vengeful), opinion/affinity values, lifestyle goals, schemes, and a relentless calendar that forces decisions.

What separates **story from noise** is well-theorized. Aaron Reed and James Cartlidge note DF produces *narrativity* — raw material, not finished narrative; the player is the final author who *frames* events into plot. The system's job is to maximize narratively-pregnant material: (a) **persistent identity** (named agents who recur), (b) **causal chains** (this death *because* that grudge), (c) **stakes and reversal** (goals that can fail), (d) **memory/continuity** so consequences echo forward. Without those, a sim emits a logbook, not a saga. Henry Jenkins ("narrative architecture") and Espen Aarseth frame the world itself as the storytelling substrate.

## 2. The GM's craft: situations, not plots

Tabletop is the most refined human discipline here. The core maxim — **"prep situations, not plots"** (Justin Alexander, *The Alexandrian*; *Apocalypse World*'s Vincent Baker) — means authoring *forces in motion* with their own agendas, then letting choice resolve them. The **"quantum ogre"** is the anti-pattern: pre-deciding the outcome and forcing it regardless of choice (the ogre is behind whichever door you pick), giving the *illusion* of agency. A director-agent must avoid this.

The useful structural tools are **Fronts** (*Apocalypse World*) and **Progress Clocks** (*Blades in the Dark*, John Harper). A **Front** is a bundle of related threats with a "doom" that advances if unopposed. A **Clock** is a segmented circle representing impending trouble or accumulating progress; it *ticks* when fictional triggers fire (danger clocks, racing clocks, tug-of-war clocks). The insight: clocks give "somewhere to put consequences other than directly on the characters" — a pacing buffer. This is railroad-free *guided* eventing: the timer is authored, but *when/whether* it advances is emergent.

## 3. Drama management / experience managers (academic AI)

**Façade** (Michael Mateas & Andrew Stern, 2005) is the landmark. Its **drama manager** globally sequences authored **beats** — reactive behavior bundles with dialogue contexts — selected from a large pool to make tension rise and fall along an *Aristotelian arc*. Beats are reorderable and conditionally selected on world/affinity state. Central pattern: **autonomous local behavior + global authored sequencing.** Related: Search-Based Drama Management (Bates, Weyhrauch); the **Storylet / Quality-Based Narrative** model (Emily Short; *Failbetter*'s *StoryNexus*), where content chunks gate on world-state qualities — the most directly portable structure for a writing system.

**Versu** (Richard Evans — lead AI on *The Sims 3* — & Emily Short, Linden Lab) is the other pole. Characters choose actions from **social practices** (scripted multi-agent plans) via *utility functions* over their traits and the social state; an optional drama-manager AI nudges. Versu's lesson: a social model is only interesting if characters *don't* act like identical automata — distinctness comes from per-character parameters.

## 4. Character agents that "decide"

The standard formalism is **BDI** (Belief–Desire–Intention; Bratman, then Rao & Georgeff): agents hold *beliefs* (world model + memory), *desires* (goals), committed *intentions* (plans). Decisions stay **in-character** when action selection is filtered through persistent traits and current emotional/social state — **utility-based selection** (Versu, The Sims), **OCC appraisal** for emotion (Ortony/Clore/Collins; used in **FAtiMA**, Paiva et al.), **affinity/opinion models** (CK3). The discipline: an agent should never take an action it cannot *justify* from its own beliefs and traits — that justification trail is also the prose.

## 5. Injecting authored beats without breaking agency

The reconciling pattern across all systems: **author the trigger condition and the seed, not the outcome.** A beat/storylet/clock fires when world-state preconditions are met (Façade beat selection, QBN storylet gating, Front advancement), then hands control *back* to autonomous agents to play it out. Agency is preserved because characters resolve the injected situation per their own decision logic.

## Design implications for a multi-agent system

**Two-tier architecture (Façade's lesson):** autonomous **character agents** + one **Director/Drama-Manager agent**. Characters generate local action and dialogue in-character; the Director does global sequencing, pacing, and beat injection. The Director never overrides a character's *choice* — it manipulates *situations and stakes* (Versu/Baker line), avoiding the quantum ogre.

**World-state model (shared blackboard):**
- **Characters:** traits, current goals (BDI desires), beliefs (incl. *false* beliefs — engine of dramatic irony), relationship/affinity matrix, emotional state, **episodic memory** of past scenes.
- **World facts:** locations, factions, objects, time/clock.
- **History log:** append-only causal record (DF-style) every agent can query — source of callbacks and continuity.
- **Tension/arc value:** the Director's target curve.

**Event-trigger model (storylet + clock hybrid):**
```
Beat {
  preconditions: predicate over world-state   // QBN gating
  director_priority: f(tension_target, arc_position)
  seed: the situation injected (NOT the outcome)
  effects: clock ticks / new goals / belief changes
}
Clock { name, segments, fill, tick_when: <fictional trigger> }
```
Director each turn: (1) read world-state + tension, (2) tick relevant clocks from what characters just did, (3) select highest-priority beat whose preconditions hold and whose injection bends the arc toward target, (4) inject the *seed* and yield to character agents. Keep authored **Fronts** (antagonist agendas that advance if unopposed) so the world has momentum even when characters dither.

**In-character guarantee:** every character action generated *from* that character's traits/beliefs/memory, emitting a one-line justification logged to history — both a coherence check and reusable narration.

## Open questions the author should decide

1. **Authorial weight**: fixed plot skeleton (beats must eventually fire) vs. pure pacing target served only via emergence — where on railroad↔sandbox?
2. **Outcome authority**: when a character's choice contradicts a needed beat, who yields — character or Director?
3. **Memory scope & cost**: full DF-style history vs. summarized (lossy)? How is salience decided?
4. **False beliefs / irony**: do agents model *each other's* beliefs (betrayal, dramatic irony) or only world facts?
5. **Tension metric**: what is "tension," concretely, and how is arc progress measured?
6. **Determinism vs. surprise**: how much randomness in action selection; is the seed logged for reproducibility?
7. **Granularity of agency**: do agents decide at action, scene, or chapter level?
8. **Termination**: what ends the novel — arc completion, a master clock filling, a Front resolving?

## Sources
Mateas & Stern (Façade GDC paper; "Interactive Drama, Art and AI"); Evans & Short ("Versu — A Simulationist Storytelling System"; "Introducing Versu"); Cartlidge ("Interpreting Dwarf Fortress"); Aaron Reed ("2006: Dwarf Fortress"); *Blades in the Dark* Progress Clocks (official + Alexandrian); Apocalypse World (Fronts); Rao & Georgeff (BDI); Paiva et al. (FAtiMA).
