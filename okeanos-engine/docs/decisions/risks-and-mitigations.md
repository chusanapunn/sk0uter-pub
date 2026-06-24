# Risks & Mitigations — Red-Team Pass

> Status: **adopted**. An adversarial critique of the medium decision, the Bridge Weaver, and the pipeline, run *before* any building. Each risk has a mitigation now folded into the design docs. The single highest-leverage change (Risk 2/4) reshaped the Bridge Weaver: **the engine hands off skeletons; the human writes the prose.**

These documents are coherent and well-grounded — which is exactly why their failure modes are subtle.

## 1. The real block may be decision-paralysis, and this tooling is a higher-resolution way to avoid the work
The most dangerous risk, because the docs *name* it ("stop polishing the system") and then propose another system to polish. A creator who spent six years perfecting worldbuilding instead of writing now gets Kuzu schemas, ranking weights (`w1..w5`), trap-door semantics, and a six-stage pipeline to perfect — catnip for the exact avoidance pattern that caused the freeze. The honest failure mode isn't "the engine produces bad bridges"; it's that *the engine never gets used because building/tuning it becomes the new safe, finite, infinitely-refinable task.*

**Mitigation (adopted):** Ban engine-building as the first move. **No engine feature gets built until 5 bridges have been written by hand.** Week 1 deliverable is one bridge chapter written between two key points — no engine, no Kuzu, just prose. The engine must earn its existence against a working human baseline. This rule is recorded in the README and the pipeline roadmap.

## 2. "Average-the-corpus" blandness — Bridge Weaver optimizes structure, which is orthogonal to voice
The ranking scores POWER, CONVERGENCE, DOUBLE_DUTY, ESCALATION — all *structural*. McKee-turn + but/therefore filters against the *boring*, but does not filter *for* the *good*: it produces competent, structurally-sound, forgettable scenes (the "well-made TV episode" smell). Worse, Qdrant voice-exemplar retrieval averages toward existing prose and the LLM regresses toward the genre mean. The output isn't obviously voice-less; it's *plausibly* in-voice and *subtly* dead — harder to reject than garbage, and for a filler-hating creator, "structurally non-filler but soulless" is the worst result: it looks finished.

**Mitigation (adopted):** The engine outputs **beats / causal skeletons, not prose.** Bridge Weaver stops at the chain of turns and hands the author the *bones* to write in their own voice. Prose generation is the one stage kept human. Add an explicit **anti-mean term**: surface the *weirdest canon-consistent* option, not only the highest-scoring — emergent surprise is what a solo author can't brute-force but *can* recognize and amplify.

## 3. The medium decision optimizes "completion," but the unstated real goal may be "the manga in his head"
The medium doc is rigorous on labor math — but optimizes *time-to-finish*, while six years invested in a *visual* "One Piece-scale" saga suggests the real want is the manga. If the dream is intrinsically visual (a Greek/ocean spectacle saga is *highly* visual), "finish the prose first" can mean writing 500k words of a thing he doesn't want, burning the passion the project depends on. The doc's load-bearing bet — "the visual audience is collectable later" — is asserted, not tested. It's wrong if prose completion *drains* motivation, or if adapting 500k words solo is its own six-year project.

**Mitigation (adopted):** De-risk the assumption cheaply *before* the 30-day commit. Write ONE key scene as prose AND rough-thumbnail it as ~6 panels — same weekend. Which made you feel alive? *That* decides the medium, not advisor consensus. Also: allow the serial to carry occasional splash illustrations of locked emotional peaks — keeps the visual muscle warm without committing to full paneling.

## 4. AI-writing acceptance risk is understated for prose, not just manga
The docs treat AI-hostility as a manga problem and imply prose is safe. That's optimistic for 2025–2026: Royal Road and most serial platforms have active anti-AI reader cultures and disclosure/ban policies. "AI-assisted plotting" is a fuzzy line; if it leaks that connective tissue was machine-generated, a web-serial audience can turn fast and publicly — and the differentiation pitch ("the engine writes the bridges") is the exact thing readers are most allergic to.

**Mitigation (adopted):** Verify each target platform's *current* AI policy before picking it (live and changing — confirm Royal Road / ScribbleHub TOS now, don't assume). Architecturally, lean into Risk 2's fix: engine produces *structure*, human writes *prose* → the disclosure story is clean and defensible ("I outline with tools, I write the words"), which is both more honest and more audience-safe.

## 5. Convergence pressure + locked anchors can manufacture contrived, deterministic plotting
CONVERGENCE rewards bridges that march toward a predetermined endpoint. With locked anchors, that breeds the most common reader complaint about plotted fiction: characters acting against their wants to hit a beat, coincidences that exist only to reach the anchor, the "railroaded" feeling. The Bridge Weaver's strength (it always lands on B) is also its failure: real tension needs the *possibility* the story won't get there.

**Mitigation (adopted):** Make the **trap-door bidirectional.** If the best emergent chain overshoots or misses B, the engine may **propose moving the anchor** (to B-prime), not just bend the bridge to it. Surface "this chain is more alive but lands at B-prime" as a first-class option. The author chooses between *convergent-but-safe* and *divergent-but-alive* — which is real authorship, not button-pressing, and directly counters Risk 1.

---

## Bottom line
The single highest-leverage change addresses Risks 1, 2, and 4 at once: **the engine hands off skeletons and lets the human write the prose.** That keeps the author an author (not a button-presser), keeps voice human (not corpus-averaged), keeps the platform/disclosure story clean, and forces the avoidance-prone creator back into the actual scary work the project has avoided for six years.

## The discipline rule (consequence of Risk 1)
> **No engine feature is built until five bridge scenes have shipped, written by hand.** The engine assists writing; it is not a substitute for having written. If building the tool ever feels more urgent than writing the next bridge, that is the avoidance pattern — go write the bridge.
