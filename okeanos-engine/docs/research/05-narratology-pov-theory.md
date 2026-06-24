# Brief 05 — Narratology & POV Theory (the "many perspectives" hard part)

> Perspective: narratologist + writing-craft theorist.
> Question: what formal model lets one canon be told from many perspectives without contradiction?

## 1. Story vs. Discourse (Fabula / Syuzhet): the master distinction

The single most important formal commitment is the Russian Formalist split between **fabula** (the chronological raw events of the storyworld) and **syuzhet** (the artful order in which those events are told). Shklovsky and Tomashevsky originated the pair; it recurs as Genette's *histoire / récit* and Chatman's **story / discourse** (*Story and Discourse*, 1978). Mieke Bal refined it into three layers — *fabula → story → text* — inserting focalization between events and verbal realization.

**This distinction *is* the solution to "one novel, many perspectives."** A multi-POV novel is not many stories; it is **one fabula rendered through many syuzhets**. *Rashomon* (Kurosawa) is the proof: a single event generates four contradictory narrations. The *fabula* may be partly indeterminate, but the system must still hold a canonical event ledger against which each telling is a transformation (selection, reordering, coloring, omission). Without this split, "the same scene from another character's eyes" has no shared referent and contradictions become unmanageable.

## 2. POV and Focalization (Genette)

Genette's *Narrative Discourse* (1972) separates two questions casual "point of view" conflates:
- **Voice** — *who speaks?* (the narrator)
- **Focalization** — *who sees / perceives?* (the orienting consciousness)

A narrator and a focalizer can differ. Three regimes:
- **Zero focalization** — narrator knows more than any character (classic omniscient).
- **Internal focalization** — perception bound to one character's mind (third-limited / first-person); *fixed*, *variable*, or *multiple* (the same event refocalized through several characters — exactly the multi-POV case).
- **External focalization** — observable behavior only, no interiority.

Load-bearing consequence: **focalization is an information filter.** Each focalizer imposes a *knowledge boundary* — what that character could perceive, infer, or already know at that moment. Two renderings of one event differ precisely because their focalizers have different access. **Model this per character per moment, not globally.**

## 3. Character interiority, arc, and theme

- **Lajos Egri** (*The Art of Dramatic Writing*): the **premise** — a one-sentence causal moral claim ("X leads to Y") the whole work proves.
- **Robert McKee** (*Story*): the **controlling idea** conditionalizes the premise with an *if/because*; distinguishes **character** (true self under pressure) from **characterization**; insists on the **arc** (inner nature changes). Underlying it: the **want vs. need** gap (conscious desire vs. unconscious truth).
- **Dramatica** (Phillips & Huntley): four **throughlines** mapped to perspectival pronouns — Overall Story (*They*), Main Character (*I*), Influence Character (*You*), Relationship (*We*). Surfaces the Influence and Relationship throughlines that audience-facing paradigms leave implicit. Dramatica is the *author's* view; McKee the *audience's* — a system needs both.

## 4. Scene structure and beat templates

**Dwight Swain** (*Techniques of the Selling Writer*, 1965): the proactive **Scene** = Goal → Conflict → Disaster; the reactive **Sequel** = Reaction → Dilemma → Decision, which converts a disaster into the next goal. A clean state-machine for scene generation and motivation tracking.

Macro templates — **Vogler/Campbell Hero's Journey**, **Snyder's Save the Cat** — give act-level scaffolding but are *single-protagonist, single-throughline* templates. They degrade for sprawling multi-POV sagas (*A Song of Ice and Fire*) where many arcs interleave. Treat them as **per-thread overlays, not global structure.**

## 5. Thematic coherence and fictional truth

Theme is not a tag but a **constraint propagated across the fabula**: McKee's controlling idea tested against every major scene's value-charge (positive/negative shift). Motif tracking requires recurrence indexing over the discourse layer.

**Consistency of fictional truth**: Wayne Booth (*The Rhetoric of Fiction*, 1961) coined the **unreliable narrator** and the **implied author**, the norm-bearing "second self" against whom a narrator's claims are judged. Formally: **canon ≠ assertion.** A narrator's claim is a *belief-statement* with a truth value possibly diverging from the fabula. The system must distinguish (a) what is canonically true, (b) what a character/narrator *asserts*, (c) what the reader is meant to infer.

## Design implications for a multi-agent system — core data model = fabula/syuzhet split

1. **Canonical Event Ledger (fabula).** Append-only, time-ordered `Event{ id, story_time (orderable), location, participants[], action, caused_by[], canonical_truth }`. The single source of truth. `story_time` is a partial order over a story-clock, decoupled from discourse order. Indeterminate facts get explicit `unknown`/`contested` rather than a guess.

2. **Per-Character Knowledge/Belief Model (timestamped).** For each character C, `Belief{ character, event_id|proposition, value, confidence, acquired_at (story_time), source }`. An epistemic state that monotonically updates as C perceives/learns. Implements Genette's knowledge boundary: at render time, query *what did C know/believe as of T?* — never leaking author-omniscient facts. Beliefs may diverge from `canonical_truth` (Rashomon and unreliable-narrator cases fall out for free).

3. **POV rendering = a VIEW over canonical events.** A narration is `Syuzhet{ focalizer, voice (person/tense/narrator-identity), event_selection[], discourse_order, belief_overlay }`. Rendering joins the Event Ledger to the focalizer's Belief state at each event's `story_time`, filters to perceivable events, reorders per discourse, colors description by traits/affect. The *same* event row produces different prose per focalizer — **multi-POV is a query, not duplicated content.** Any assertion is checkable against the ledger and flagged as error vs. intentional unreliability.

4. **Arc/throughline layer.** Per character: `Arc{ want, need, flaw, premise_stake, beat_states[] }`; scenes typed as Swain Scene/Sequel state machines; Dramatica throughlines as cross-cutting relations. A **Theme/Premise object** holds the controlling idea and validates each scene's value-shift; motifs indexed as recurrences over the discourse layer.

5. **Agent roles** map cleanly: a *Continuity agent* owns the Event Ledger; *Character agents* own and update their own Belief models (and must *request*, not assume, knowledge); a *Narration agent* renders syuzhets as views; a *Theme agent* enforces premise/value coherence; an *Editor agent* diffs assertions against canon to catch leaks and unintended contradictions.

## Open questions the author should decide

- **Is the fabula fully determinate, or deliberately contested?** Pure Rashomon needs an *indeterminate* canon; most novels want a determinate ledger with unreliable *views*. The data model differs.
- **Granularity of `story_time`** — total order, partial order, or fuzzy intervals? Flashbacks, simultaneity, dream/unreal events stress this.
- **How is unintended contradiction distinguished from intended unreliability?** An author-set flag, or inferred? Governs the Editor agent's strictness.
- **Belief propagation rules** — how do characters learn (perception, dialogue, inference, rumor)? Manual authoring vs. automatic simulation trades control for emergent consistency.
- **Macro structure for many arcs** — global beat template, per-POV templates, or none?
- **Where does the implied author live?** The thematic/normative layer that judges reliability needs a home — the Theme agent, or an explicit authorial-stance object?
- **Voice vs. focalization independence** — support narrator ≠ focalizer (retrospective first-person, free indirect discourse), or collapse them for simplicity?

## Sources
Genette *Narrative Discourse*; Shklovsky/Tomashevsky (fabula/syuzhet); Chatman *Story and Discourse*; Bal *Narratology*; living handbook of narratology (focalization, multiperspectivity); Egri *The Art of Dramatic Writing*; McKee *Story*; Dramatica (Phillips & Huntley); Swain *Techniques of the Selling Writer*; Campbell/Vogler; Snyder *Save the Cat*; Booth *The Rhetoric of Fiction*; Phelan (bonding/estranging unreliability).
