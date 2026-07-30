# OKEANOS RETURNAL
### A graph-node RPG design document — where the graph *is* the ocean

> Working concept doc. Built on the research pass on graph-node gameplay
> (Citizen Sleeper's node map, Failbetter storylets, deduction-board games,
> PoE's character-as-graph) applied to the Okeanos Returnal project.
> Entity names below are placeholders — swap in the real cast when the
> project is indexed.

---

## 1. High Concept

**You are a drowned soul carried by Okeanos, the world-river that circles
everything and returns to itself. The world is not terrain — it is a living
graph of entities, memories, and currents. Every cycle, the river completes
its loop, the world sinks one layer deeper, and you return. What you
connected stays connected. What you severed stays severed. The story is the
shape the graph takes across your returns.**

One sentence pitch: *Loop Hero's cycle × Citizen Sleeper's node map ×
a detective board where the mystery is your own previous loops.*

The title is the design:

| Title part | Design meaning |
|---|---|
| **Okeanos** | The world-graph is a *ring*, not a tree. The main current is a closed cycle of nodes the player drifts along — the river that flows back into itself. |
| **Returnal** | Roguelite loop structure. Each cycle mutates the graph. Persistence is *topological*: edges (relationships, debts, promises, severances) survive the return even when node states reset. |

---

## 2. The Three Graph Layers

The game is one graph rendered at three semantic layers (this is the
"multiple levels of node" idea, made literal):

### Layer A — The Current Graph (world / macro)
- A ring of **anchor nodes** (islands, wrecks, lighthouses, the Flooded Café
  as the hub — the one fixed node that never moves between loops).
- Edges are **currents**: directed, weighted, and *the only way to travel*.
  You don't walk; you drift. Spending resources lets you row *against* a
  current (traverse an edge backwards at cost).
- The ring is cyclic on purpose: completing one revolution of the ring = one
  loop = one "return." The loop counter is diegetic — it's the river itself.

### Layer B — The Entity Graph (meso)
- Zoom into an anchor node and it expands into its **entity subgraph**:
  characters, objects, factions living there. Node = one entity.
- Edges are typed relationships: `LOVES`, `OWES`, `GUARDS`, `REMEMBERS`,
  `DROWNED_WITH`, `KNOWS_SECRET_OF`.
- RPG verbs are graph mutations:
  - *Persuade / bond* → create an edge
  - *Betray / sever* → cut an edge (the cut edge leaves a visible **scar
    edge** — dashed, permanent, part of the record)
  - *Gift / steal* → move an `OWNS` edge between entities
  - *Kill / lose* → the node **sinks** (see §3) — it is never deleted

### Layer C — The Memory Graph (micro)
- Zoom into an entity and you get their **attribute/memory subgraph**:
  desires, fears, memories as nodes; internal edges like `SUPPRESSES`,
  `LONGS_FOR`.
- Memories reference *other entities' memories* → cross-entity memory edges
  are the deduction-board gameplay: discovering that two characters share a
  drowned memory **is** a story revelation, rendered as an edge appearing.

---

## 3. Depth = Time: the core original mechanic

The unification trick that makes this game *itself* and not a collage:

**The graph has vertical strata mapped to ocean depth zones, and depth is
time.**

```
  SURFACE   (this loop)        — live entities, current events
  PHOTIC    (last loop)        — recent past, still reachable cheaply
  TWILIGHT  (older loops)      — faded nodes, edges cost more to read
  ABYSS     (the first loop)   — origin mysteries, near-illegible
```

- When an entity dies, a place floods, or a loop ends — the node doesn't
  despawn. It **sinks one stratum**, becoming a memory-node. All its edges
  come with it, greyed.
- The player can **dive**: shift the whole view down a stratum and interact
  with the past — read sunken edges, salvage a sunken node's `OWNS` object,
  even *re-float* one node per loop (bring a memory/person/place back to
  the surface — the loop's biggest strategic choice).
- Previous *player runs* live down there too. Your loop-3 self's decisions
  are literally visible as scar edges in the loop-3 stratum. The mystery
  genre ("what happened here?") is answered by diving your own history.

Visually: one zoom/scroll axis = depth. Surface graph in full color, deeper
strata rendered darker, desaturated, drifting slightly — layered parallax of
past graphs under the present one. This alone is a screenshot-bait art
direction no other RPG has.

---

## 4. Story Engine: storylets as graph patterns

Story content is authored Failbetter-style — a pool of **storylets**, each
with a trigger condition. The twist: triggers are **graph pattern queries**
(the exact skill already used in Sk0uter's Kùzu/Cypher pipeline):

```cypher
// Storylet: "The Lighthouse Keeper's Debt"
// Fires when the player has bonded with anyone who owes the Keeper
MATCH (p:Player)-[:BONDED]->(x:Entity)-[:OWES]->(k:Entity {tag:'keeper'})
WHERE NOT (p)-[:SCARRED]->(k)
```

```cypher
// Storylet: "Two Drowned, One Memory" — cross-stratum trigger
MATCH (a:Entity {stratum:0})-[:REMEMBERS]->(m:Memory)<-[:REMEMBERS]-(b:Entity {stratum:2})
```

Why this is powerful:

1. **Writers author patterns, not scripts.** Any entities matching the
   pattern can host the story → emergent recombination across loops.
2. **Consequences are free.** A storylet's outcome is just "apply this graph
   mutation," and the mutation is *visible to the player* — the story engine
   and the visualization are the same object.
3. **Validation is offline tooling.** Run the storylet pool against
   generated world states in Kùzu to answer "is every ending reachable?",
   "which storylets can never fire?" — Sk0uter's pipeline, repointed at
   game data instead of GDScript.

---

## 5. Conflict as Propagation (no hit points)

Combat/conflict borrows the `FIRES` signal idea from Sk0uter's code graph:
effects **propagate along edges** with per-hop decay/mutation.

- **Rumor**: inject at a node → travels the social graph 1 hop/turn,
  mutating at each hop (telephone-game mechanic, content drawn from edge
  types it crosses).
- **Curse / flood**: spreads along `NEAR`/`LOVES` edges; players defend by
  *pre-emptively severing* edges — sacrifice relationships to firewall a
  spread. Painful choices as graph surgery.
- **Influence duels**: fought over *betweenness centrality* — cut the
  opponent's bridge edges, become the articulation point yourself. Winning
  = the graph literally re-routes through you.
- Power stats are graph metrics, shown honestly: degree = social reach,
  centrality = political weight, clustering = community trust.

---

## 6. What Persists Across the Return

| Resets each loop | Survives forever |
|---|---|
| Node *states* (moods, stock, positions on the ring) | **Edges**: bonds, scars, debts, `REMEMBERS` |
| Consumables | Sunken strata (all of them) |
| Storylet availability | The Flooded Café hub node and everything anchored to it |
| | The player's memory-subgraph (your character sheet *is* a graph you grow, PoE-style, but grown by story events, not XP) |

Design intent: the player's real progression is **the shape of the graph**.
An endgame world where everything is scar edges plays — and *looks* —
completely different from one dense with bonds. Endings are graph
conditions: e.g. *"the ring is whole"* (surface cycle fully connected by
bond edges) vs *"the ring is severed"* vs *"you sank the surface yourself."*

---

## 7. Diegetic Framing

The abstract node UI must be justified in-fiction (the Hacknet rule):

> You are dead. Drowned. What the drowned perceive is not light but
> *connection* — Okeanos does not show you the world, it shows you what the
> world is *tied to*. The interface isn't a map of the ocean. It is how the
> ocean thinks.

This makes every UI element diegetic: fog-of-war = unperceived connections;
discovering an edge = the river granting you a secret; the depth strata =
the river's own memory; the loop = the river's circulation.

---

## 8. Godot 4.4 Implementation Plan

**Phase 0 — Paper-thin prototype (1–2 weekends)**
- `GraphEdit`/`GraphNode` runtime UI (they work in-game, not just editor).
- 1 ring of 6 anchor nodes, 1 expandable entity subgraph (~15 entities,
  4 edge types), 3 storylets with hardcoded pattern checks, 1 verb
  (bond/sever). **Kill criterion**: if watching an edge appear/scar isn't
  narratively satisfying here, stop and rethink before building more.

**Phase 1 — Custom renderer**
- Replace GraphEdit with `Node2D` entities + `Line2D` edges + a light
  spring layout with **pinned deterministic anchors** (players build
  spatial memory; the ring must never reshuffle — Citizen Sleeper rule).
- Depth strata as stacked CanvasLayers with shader-based desaturation/
  refraction; dive = camera + layer opacity shift.
- Portraits *inside* nodes, animated edges on mutation. Circles don't make
  people cry; faces and prose do — graph shows structure, text vignettes
  carry feeling.

**Phase 2 — Data pipeline (Sk0uter reuse)**
- World + storylets authored as data, loaded into Kùzu for authoring-time
  validation (reachability, dead storylets, orphan entities).
- Export compiled adjacency dictionaries (JSON/binary) for the Godot
  runtime — no DB dependency in the shipped game.
- Pattern-trigger evaluation at runtime = simple subgraph matching over
  small graphs (hundreds of nodes); no Cypher engine needed in-game.
- Bonus: index the game project itself with Sk0uter while building it —
  the tool that visualizes the game's code and the game that visualizes
  its world become the same family of software.

**Guardrails (from the research pass)**
- Never render >1 zoom layer + 1 depth stratum at full opacity (hairball).
- Deterministic layouts everywhere the player revisits.
- Tutorial starts at 5 nodes, one edge type. Graph literacy is earned.

---

## 9. Open Questions (for next session)

1. What is the actual cast/setting already written for Okeanos Returnal?
   (Index the project with Sk0uter → this doc's placeholders get replaced
   with a real generated entity graph as the design's starting state.)
2. Is `floodedcafe` the same world? If so the Café-as-immovable-hub-node
   assumption in §2/§6 becomes canon.
3. Loop length target: minutes (Loop Hero) or a full session (roguelite)?
4. Is there a real-time layer at all, or fully turn/cycle-based?
