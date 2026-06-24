# Data Model Sketch (v0 — for discussion)

> The fabula/syuzhet split is the spine. Everything else hangs off the **Canonical Event Ledger**. Field names are illustrative, not final.

## 1. World-clock

A story-time scale, decoupled from publication order and from discourse (telling) order.
- Open question (crux #2): total order vs. partial order vs. fuzzy intervals; in-world dates vs. scene-ordinal vs. relative ("three days later").
- Every fact and event is stamped on this clock.

## 2. Canonical Event Ledger (the fabula) — source of truth

```
Event {
  id
  story_time        // position on the world-clock (orderable; partial order allowed)
  location_id
  participants[]     // character_ids
  action             // what happened, in neutral narrator-agnostic terms
  caused_by[]        // event_ids — the causal chain (enables "because")
  canonical_truth    // true | false | unknown | contested   (Rashomon-safe)
  source_rank        // canon hierarchy: main_text > supplementary > draft  (crux #3)
  established_in      // chapter/scene id where this became canon (provenance)
}
```
Append-only. Contradictions are detected by diffing new assertions against this ledger.

## 3. Bi-temporal fact edges (the memory backbone)

Every relationship/fact in the graph carries **two timelines** (Graphiti/Zep model):
```
Fact / Edge {
  subject, predicate, object       // e.g. (Robin) —[ally_of]→ (Straw Hats)
  t_valid                          // when TRUE in the story world (event-time)
  t_invalid                        // when it stopped being true (null = still true)
  t_created                        // when the system/canon recorded it (ingestion-time)
  t_expired                        // when the system retracted it
  source_rank, established_in
}
```
This is what lets "Robin: enemy in Alabasta, ally after Enies Lobby" coexist without contradiction — the same edge type with non-overlapping validity intervals. Retcon = set `t_invalid`/`t_expired`, never delete (preserves history).

## 4. Per-character knowledge / belief model (the POV filter)

```
Belief {
  character_id
  about            // event_id | proposition | fact_edge_id
  value            // what the character believes is true
  confidence
  acquired_at      // story_time when they learned it
  source           // perceived | told_by(X) | inferred | rumor
}
Witnessing (edge) {
  character_id, event_id, story_time   // "X was present at / learned of Y"
}
```
- A character's beliefs **may diverge** from `canonical_truth` → dramatic irony, unreliable narration, secrets all fall out for free.
- **Point-in-time query** (the key operation): *what does character C know/believe as of world-clock T?* = facts where C has a Witnessing/Belief with `acquired_at ≤ T` and the fact's validity interval contains T.

## 5. POV rendering = a VIEW (the syuzhet)

A chapter/scene narration is not stored prose duplicated per character — it is a **render** of events through a focalizer:
```
Syuzhet {
  focalizer_id          // whose perception orients the telling (Genette)
  voice                 // narrator identity, person (1st/3rd), tense
  event_selection[]     // which events from the ledger are told
  discourse_order       // the order of telling (≠ story_time; flashbacks etc.)
  belief_overlay        // colored by focalizer's beliefs at each event's story_time
}
```
Render = join `event_selection` to the focalizer's belief state at each event's `story_time`, filter to perceivable events, reorder by `discourse_order`, color description by traits/affect. **One event → many syuzhets = multi-POV without duplication or contradiction.**

## 6. Outline tree (authorial structure)

```
saga → arc → chapter → beat
  each node: { summary, goal, entry_world_state, exit_world_state,
               status: planned|drafted|published|locked,
               trap_door?: { provisional: bool, alternate_branch_id } }
```

## 7. Foreshadowing ledger (long-range payoff bookkeeping)

```
Foreshadow {
  description
  planted_in           // beat/scene id
  intended_payoff      // beat id or "open"
  paid_in              // beat/scene id | null
  status               // open | paid | orphaned | contradicted
  importance
  payoff_deadline?     // max story-distance before it must fire or retire (crux: payoff SLA)
}
```

## 8. Character dossier & arc

```
Character {
  id, name, status (alive/dead/...), affiliation, relationships[],
  traits[], goals[] (BDI desires), emotional_state,
  power/resource_budget?,          // bounded, to detect escalation drift
  voice_profile,                   // speech patterns, vocab, catchphrases
  arc: { want, need, flaw, premise_stake, beat_states[] }
}
```

## 9. Theme / premise (coherence constraint)

```
Premise {
  controlling_idea     // McKee: "X leads to Y because Z"
  value_axes[]         // e.g. freedom↔control; each scene shifts a value +/-
  motifs[]             // recurrence-indexed over the discourse layer
}
```

## Key queries the model must answer cheaply
- **Point-in-time knowledge**: what did C know at T? (anachronism guard, POV filter)
- **Canon diff**: does this draft assertion contradict any valid fact? (continuity check)
- **Foreshadowing status**: what's planted-but-unpaid, and is anything overdue/contradicted?
- **Callback retrieval**: where earlier did motif/character/event M appear? (Qdrant + graph)
- **Render**: events E through focalizer F at time T → ordered, belief-colored scene.

## Open modeling decisions → see `../decisions/open-questions.md`
Determinate vs. contested fabula; world-clock granularity; belief-propagation (manual vs. simulated); canon source hierarchy; voice≠focalizer support; verification scope/cost.
