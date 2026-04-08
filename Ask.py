# Ask.py — Dual-Pillar Retrieval: System Mapping + Context Packing
import os
import json
from collections import defaultdict
from qdrant_client.models import SparseVector, SearchRequest, NamedVector, NamedSparseVector
from core.graph_db import ScoutGraph
from core.roadmap import load_project_data
from core.sparse import encode_sparse_query
from core.symcode import encode_function, encode_skeleton, SYMCODE_LEGEND
from core.qdrant_client import get_qdrant_client
from config import OLLAMA_URL

# =====================================================================
# AI PERSONAS
# =====================================================================
PERSONAS = {
    "Senior Architect": "Analyze structure, suggest modular Godot design patterns, and prioritize decoupled signal buses.",
    "Bug Hunter": "[CRITICAL: BUG DETECTOR PROTOCOL ENGAGED] You are in 'Merciless Diagnostic Mode'. Scan for null references, race conditions (Godot signals/await), and logic leaks. Provide exact line numbers and precise fixes.",
    "Code Optimizer": "Focus on performance. Identify bottlenecks in _process loops, physics calculations, and array allocations.",
    "Teacher": "Explain the 'why' behind the code. Break down complex Godot concepts into easy-to-understand lessons."
}

# Phase 1: Persona-aware graph boost weights for re-scoring
PERSONA_WEIGHTS = {
    "Senior Architect": {"co_hit": 0.05, "signal": 0.06, "degree": 0.04},
    "Bug Hunter":       {"co_hit": 0.03, "signal": 0.12, "degree": 0.02},
    "Code Optimizer":   {"co_hit": 0.10, "signal": 0.05, "degree": 0.02},
    "Teacher":          {"co_hit": 0.05, "signal": 0.05, "degree": 0.03},
}

# =====================================================================
# PILLAR 1: SYSTEM MAPPING — Graph Topology Discovery
# =====================================================================

def _discover_systems(graph_db, pipeline_log=None):
    """Query the full graph to build a system topology map.

    Returns:
        systems:        {topic_name: set of script_paths}
        script_topics:  {script_path: set of topic_names}
        signal_bus:     {signal_name: {"definers": set, "emitters": set}}
    """
    systems = defaultdict(set)
    script_topics = defaultdict(set)

    # Script → Topic (primary system membership)
    try:
        res = graph_db.conn.execute(
            "MATCH (s:Script)-[:IMPLEMENTS]->(t:Topic) RETURN s.path, t.name"
        )
        while res.has_next():
            sp, tn = res.get_next()
            systems[tn].add(sp)
            script_topics[sp].add(tn)
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("graph_discovery", "Script→Topic query failed", e)

    # Signal bus: who defines and who fires each signal
    signal_bus = defaultdict(lambda: {"definers": set(), "emitters": set()})
    try:
        res = graph_db.conn.execute(
            "MATCH (s:Script)-[:DEFINES]->(sig:Signal) RETURN s.path, sig.name"
        )
        while res.has_next():
            sp, sn = res.get_next()
            signal_bus[sn]["definers"].add(sp)
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("graph_discovery", "Signal definer query failed", e)
    try:
        res = graph_db.conn.execute(
            "MATCH (s:Script)-[:OWNS]->(f:Function)-[:FIRES]->(sig:Signal) "
            "RETURN s.path, sig.name"
        )
        while res.has_next():
            sp, sn = res.get_next()
            signal_bus[sn]["emitters"].add(sp)
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("graph_discovery", "Signal emitter query failed", e)

    return systems, script_topics, signal_bus


def _build_system_map(systems, signal_bus, active_topics):
    """Compile a compact system topology header for the LLM.

    Only includes systems that are 'active' (have at least one vector hit).
    Filters out 'Extends *' topics — those are inheritance, not game systems.
    """
    if not active_topics:
        return ""

    # Filter out inheritance-based topics (e.g. "Extends Node", "Extends Control")
    real_topics = {t for t in active_topics if not t.startswith("Extends ")}
    if not real_topics:
        return ""

    lines = ["[SYSTEM MAP — Active subsystems relevant to your query]"]
    for topic in sorted(real_topics):
        scripts = systems.get(topic, set())
        if not scripts:
            continue
        basenames = ", ".join(sorted(os.path.basename(s) for s in scripts))
        lines.append(f"\n  System: {topic}")
        lines.append(f"    Scripts: {basenames}")

        # Signals owned by this system's scripts
        sys_signals = set()
        for sig_name, bus in signal_bus.items():
            if scripts & (bus["definers"] | bus["emitters"]):
                sys_signals.add(sig_name)
        if sys_signals:
            lines.append(f"    Signals: {', '.join(sorted(sys_signals))}")

    return "\n".join(lines)


def _get_script_neighbors(graph_db, file_path, pipeline_log=None):
    """Fetch all direct graph relationships for a script (up to 8 hops-out)."""
    neighbors = []
    try:
        res = graph_db.conn.execute(
            "MATCH (s:Script {path: $fp})-[r]->(n) RETURN LABEL(r), n.name LIMIT 8",
            {"fp": file_path}
        )
        while res.has_next():
            rel_type, target = res.get_next()
            neighbors.append(f"{rel_type} → {target}")
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("graph_neighbors", f"Neighbor query failed for {file_path}", e)
    return neighbors


# =====================================================================
# PILLAR 2: CONTEXT PACKING — Score, Rank, Budget, Pack
# =====================================================================

def _score_and_rank(hits, script_topics, signal_bus, systems,
                    persona_key="Senior Architect"):
    """Rescore vector hits using graph structure for system-aware ranking.

    combined_score = vector_similarity + graph_boost

    Graph boost components (weights vary by persona):
      - Co-hit density:  How many other vector hits share this script's system cluster.
      - Signal coupling: Hits connected via shared signals (A defines, B fires).
      - Degree:          Scripts with more graph connections are architecturally central.
    """
    weights = PERSONA_WEIGHTS.get(persona_key, PERSONA_WEIGHTS["Senior Architect"])
    hit_files = set(h.payload.get("file", "") for h in hits)
    scored = []

    for hit in hits:
        fp = hit.payload.get("file", "")
        vec_score = hit.score  # cosine similarity from Qdrant

        # (a) Co-hit density in shared topic systems
        my_topics = script_topics.get(fp, set())
        co_hits = 0
        for topic in my_topics:
            co_members = systems.get(topic, set()) & hit_files
            co_hits += len(co_members) - (1 if fp in co_members else 0)

        # (b) Signal coupling: hits connected via the signal bus
        signal_links = 0
        for sig_name, bus in signal_bus.items():
            all_connected = bus["definers"] | bus["emitters"]
            if fp in all_connected:
                signal_links += len((all_connected & hit_files) - {fp})

        # (c) Degree centrality
        degree = len(my_topics)

        graph_boost = (co_hits * weights["co_hit"]) + (signal_links * weights["signal"]) + (degree * weights["degree"])
        scored.append((vec_score + graph_boost, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _get_functions_for_script(graph_db, script_path, pipeline_log=None):
    """Helper: fetch function names owned by a script (up to 10)."""
    functions = []
    try:
        res = graph_db.conn.execute(
            "MATCH (s:Script {path: $fp})-[:OWNS]->(f:Function) RETURN f.name LIMIT 10",
            {"fp": script_path}
        )
        while res.has_next():
            functions.append(res.get_next()[0])
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("graph_functions", f"Function query failed for {script_path}", e)
    return functions


def _graph_expand(graph_db, active_topics, systems, hit_files, budget=5,
                  pipeline_log=None):
    """Pull graph-adjacent scripts via real multi-hop traversal.

    Three hops in priority order:
      Hop 1 — CALLS edges: scripts directly referenced by hit files.
      Hop 2 — Signal chains: scripts connected via fire→signal→define bus.
      Hop 3 — System cluster fallback: topic co-members the vector search missed.

    Each hop respects the seen set so the same script is never added twice.
    Total candidates are capped at `budget`.
    """
    candidates = []
    seen = set(hit_files)

    def _add(script_path, system_label):
        if script_path not in seen and len(candidates) < budget:
            seen.add(script_path)
            candidates.append({
                "file": script_path,
                "system": system_label,
                "functions": _get_functions_for_script(graph_db, script_path, pipeline_log),
            })

    # --- Hop 1: CALLS edges (direct script dependencies) ---
    for fp in list(hit_files):
        try:
            res = graph_db.conn.execute(
                "MATCH (s:Script {path: $fp})-[:CALLS]->(t:Script) RETURN t.path LIMIT 5",
                {"fp": fp}
            )
            while res.has_next():
                _add(res.get_next()[0], "calls_neighbor")
        except Exception as e:
            if pipeline_log:
                pipeline_log.warn("graph_expand", f"CALLS hop failed for {fp}", e)
        if len(candidates) >= budget:
            return candidates

    # --- Hop 2: Signal bus traversal (fires → Signal ← defines) ---
    # Forward: scripts that DEFINE signals which hit-file scripts FIRE
    for fp in list(hit_files):
        try:
            res = graph_db.conn.execute(
                "MATCH (s:Script {path: $fp})-[:OWNS]->(f:Function)-[:FIRES]->"
                "(sig:Signal)<-[:DEFINES]-(t:Script) RETURN t.path LIMIT 5",
                {"fp": fp}
            )
            while res.has_next():
                _add(res.get_next()[0], "signal_listener")
        except Exception as e:
            if pipeline_log:
                pipeline_log.warn("graph_expand", f"Signal listener hop failed for {fp}", e)
        if len(candidates) >= budget:
            return candidates

    # Reverse: scripts that FIRE signals which hit-file scripts DEFINE
    for fp in list(hit_files):
        try:
            res = graph_db.conn.execute(
                "MATCH (t:Script)-[:OWNS]->(f:Function)-[:FIRES]->"
                "(sig:Signal)<-[:DEFINES]-(s:Script {path: $fp}) RETURN t.path LIMIT 5",
                {"fp": fp}
            )
            while res.has_next():
                _add(res.get_next()[0], "signal_emitter")
        except Exception as e:
            if pipeline_log:
                pipeline_log.warn("graph_expand", f"Signal emitter hop failed for {fp}", e)
        if len(candidates) >= budget:
            return candidates

    # --- Hop 3: System cluster fallback (original topic co-membership) ---
    for topic in active_topics:
        for sp in sorted(systems.get(topic, set())):
            _add(sp, topic)
            if len(candidates) >= budget:
                return candidates

    return candidates


_CODE_TYPES = {"function", "function_part", "script_identity"}
_PROSE_TYPES = {"markdown_section", "json_data", "scene_data"}


def _estimate_tokens(text, chunk_type=None):
    """Estimate token count with code-aware character-per-token ratios.

    Code has shorter average tokens than prose (operators, indentation, short
    identifiers). A 1.1x safety margin prevents context overflow.
    """
    if not text:
        return 1
    if chunk_type in _CODE_TYPES:
        ratio = 3.2
    elif chunk_type in _PROSE_TYPES:
        ratio = 4.5
    else:
        lines = text.split('\n')
        if len(lines) > 2:
            indented = sum(1 for ln in lines if ln and ln[0] in ' \t')
            ratio = 3.2 if (indented / len(lines)) > 0.3 else 4.5
        else:
            ratio = 3.8
    return max(1, int(len(text) / ratio * 1.1))


def _pack_context(scored_hits, graph_expansions, graph_db, script_topics,
                  detail_threshold, persona_key, token_budget=24_000,
                  pipeline_log=None):
    """Build the final context string with system-organized, token-aware packing.

    Tier 1 (top detail_threshold hits, within budget): FULL code + state + graph neighbors
    Tier 2 (remaining hits, within budget):            Code only (compressed)
    Tier 3 (graph expansions, within budget):          Skeleton — file + function list

    Per-file deduplication: Variables/State and Graph edges are emitted once per file
    as a file header. Subsequent chunks from the same file only include their code body.
    """
    # Boost threshold for Bug Hunter (needs more full-detail chunks)
    effective_threshold = detail_threshold
    if persona_key == "Bug Hunter":
        effective_threshold = max(detail_threshold, 15)

    # --- Build per-system buckets with token tracking ---
    system_buckets = defaultdict(list)  # topic -> [chunk_strings]
    tokens_used = 0
    seen_files = set()  # Track files whose header (state+graph) has been emitted

    for rank, (score, hit) in enumerate(scored_hits):
        payload = hit.payload
        fp = payload.get("file", "Unknown")
        func_name = payload.get("name", "Unknown")
        chunk_type = payload.get("type", "function")
        line_start = payload.get("line_start", 0)

        # Bucket by graph topic; fall back to first inferred system, then Uncategorized
        graph_topics = script_topics.get(fp, set())
        payload_systems = payload.get("systems", [])
        if graph_topics:
            # Filter out Extends* topics for bucketing — use game systems
            real_topics = {t for t in graph_topics if not t.startswith("Extends ")}
            primary_topic = sorted(real_topics)[0] if real_topics else (
                payload_systems[0] if payload_systems else sorted(graph_topics)[0])
        elif payload_systems:
            primary_topic = payload_systems[0]
        else:
            primary_topic = "Uncategorized"

        # Build chunk header (no score — it's internal metadata)
        header = f"--- [{fp} > {func_name} | type:{chunk_type} line:{line_start}] ---"

        if rank < effective_threshold:
            # TIER 1: Full detail — code + file header (once per file) + code
            content = payload.get("content", "")
            file_header = ""
            if fp not in seen_files:
                seen_files.add(fp)
                state = payload.get("global_state", "")
                neighbors = _get_script_neighbors(graph_db, fp, pipeline_log=pipeline_log)
                if neighbors:
                    file_header += f"Graph: {', '.join(neighbors)}\n"
                if state:
                    file_header += f"Variables/State:\n{state}\n"
            body = f"{file_header}Code:\n{content}"
            tag = "FULL"
        else:
            # TIER 2: SymCode compressed — saves ~40-60% tokens vs raw code
            content = payload.get("content", "")
            chunk_emits = payload.get("emits", [])
            state = payload.get("global_state", "") if fp not in seen_files else ""
            if fp not in seen_files:
                seen_files.add(fp)
            if chunk_type in ("function", "function_part"):
                compressed = encode_function(content, emits=chunk_emits, global_state=state)
                body = f"Sym:\n{compressed}"
            else:
                body = f"Code:\n{content}"
            tag = "SYM"

        chunk_text = f"[{tag}] {header}\n{body}"
        chunk_tokens = _estimate_tokens(chunk_text, chunk_type=chunk_type)

        if tokens_used + chunk_tokens > token_budget:
            # Over budget — downgrade Tier 1 to SymCode if possible, else skip
            if tag == "FULL":
                if chunk_type in ("function", "function_part"):
                    compressed = encode_function(content, emits=payload.get("emits", []))
                    body = f"Sym:\n{compressed}"
                else:
                    body = f"Code:\n{content}"
                chunk_text = f"[SYM] {header}\n{body}"
                chunk_tokens = _estimate_tokens(chunk_text, chunk_type=chunk_type)
                if tokens_used + chunk_tokens > token_budget:
                    continue
            else:
                continue

        tokens_used += chunk_tokens
        system_buckets[primary_topic].append(chunk_text)

    # TIER 3: Graph expansions (SymCode skeletons)
    for exp in graph_expansions:
        topic = exp["system"]
        skeleton = encode_skeleton(exp["file"], exp["functions"])
        entry = f"[GRAPH] {skeleton}"
        entry_tokens = _estimate_tokens(entry, chunk_type="function")
        if tokens_used + entry_tokens <= token_budget:
            tokens_used += entry_tokens
            system_buckets[topic].append(entry)

    # --- Assemble by system (no budget line — moved to metadata) ---
    sections = [SYMCODE_LEGEND]
    for topic in sorted(system_buckets.keys()):
        chunks = system_buckets[topic]
        sections.append(f"=== SYSTEM: {topic} ({len(chunks)} chunks) ===")
        sections.extend(chunks)

    return "\n\n".join(sections), tokens_used


# =====================================================================
# MAIN ENTRY POINTS
# =====================================================================

def build_surgical_prompt(query, active_project, collection_name, embed_model,
                          host, port, context_limit=20, detail_threshold=5,
                          persona_key="Senior Architect", token_budget=24_000,
                          pipeline_log=None, **kwargs):
    """The Retrieval Orchestrator: Dense Vector Search + Graph System Mapping.

    Pipeline:
      1. Dense vector search → initial candidate pool
      2. Graph system discovery → cluster topology
      3. Graph-boosted re-scoring → system-aware ranking
      4. Graph expansion → pull missing subsystem members
      5. System-organized context packing → coherent prompt
    """
    # 1. Persona
    base_style = PERSONAS.get(persona_key, PERSONAS["Senior Architect"])
    system_persona = f"You are Scout, an expert AI software architect. {base_style}"

    # 2. Roadmap / project state (light — just the header for now)
    roadmap = load_project_data(active_project["scout_dir"])
    roadmap_parts = []
    global_goal = roadmap.get("global_goal", "")
    completed = [m["name"] for m in roadmap.get("milestones", []) if m["status"] == "completed"]
    pending = [m["name"] for m in roadmap.get("milestones", []) if m["status"] == "pending"]
    if global_goal:  roadmap_parts.append(f"Global Goal: {global_goal}")
    if completed:    roadmap_parts.append(f"Completed: {', '.join(completed)}")
    if pending:      roadmap_parts.append(f"Active: {', '.join(pending)}")
    roadmap_context = "\n".join(roadmap_parts)

    # 3. Hybrid search: Dense (semantic) + Sparse (BM25 lexical)
    raw_vectors = embed_model.encode([query])
    dense_vector = raw_vectors[0] if raw_vectors else []

    q_client = get_qdrant_client(
        mode=kwargs.get("qdrant_mode", "local"),
        scout_dir=active_project.get("scout_dir"),
        host=host, port=port,
    )
    hits = []
    if dense_vector:
        # Detect collection vector config to choose the right search strategy
        has_named_vectors = False
        try:
            _cinfo = q_client.get_collection(collection_name)
            _vc = _cinfo.config.params.vectors
            has_named_vectors = isinstance(_vc, dict) and "dense" in _vc
        except Exception:
            pass  # Collection may not exist yet

        try:
            if has_named_vectors:
                # Load BM25 vocab for sparse query encoding
                vocab_path = os.path.join(active_project["scout_dir"], "bm25_vocab.json")
                bm25_vocab = None
                if os.path.exists(vocab_path):
                    with open(vocab_path, "r", encoding="utf-8") as _vf:
                        bm25_vocab = json.load(_vf)

                if bm25_vocab:
                    # Hybrid: fuse dense + sparse results via Reciprocal Rank Fusion
                    from qdrant_client.models import Prefetch, FusionQuery, Fusion
                    sp_indices, sp_values = encode_sparse_query(query, bm25_vocab)
                    prefetch = [
                        Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=context_limit * 2,
                        ),
                    ]
                    if sp_indices:
                        prefetch.append(
                            Prefetch(
                                query=SparseVector(indices=sp_indices, values=sp_values),
                                using="bm25",
                                limit=context_limit * 2,
                            )
                        )
                    hits = q_client.query_points(
                        collection_name=collection_name,
                        prefetch=prefetch,
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=context_limit,
                    ).points
                else:
                    # Named vectors but no BM25 vocab — dense-only
                    hits = q_client.query_points(
                        collection_name=collection_name,
                        query=dense_vector,
                        using="dense",
                        limit=context_limit,
                    ).points
            else:
                # Legacy collection with unnamed vectors — query without 'using'
                hits = q_client.query_points(
                    collection_name=collection_name,
                    query=dense_vector,
                    limit=context_limit,
                ).points
                if pipeline_log:
                    pipeline_log.warn("vector_search",
                        "Collection uses legacy unnamed vectors. Re-index with 'Nuke & Reset' for hybrid search.")
        except Exception as e:
            if pipeline_log:
                pipeline_log.error("vector_search", "Qdrant search failed", e)

    if not hits and pipeline_log:
        pipeline_log.warn("vector_search", "No vector results returned — context will be graph-only")

    # 4. Graph: discover full system topology
    graph_db = ScoutGraph(
        active_project["id"], active_project["scout_dir"],
        collection_name, read_only=True,
    )
    systems, script_topics, signal_bus = _discover_systems(graph_db, pipeline_log=pipeline_log)

    # 5. Graph-boosted re-scoring
    scored_hits = _score_and_rank(hits, script_topics, signal_bus, systems, persona_key=persona_key)

    # 6. Identify active systems (topics that have at least one vector hit)
    hit_files = set(h.payload.get("file", "") for h in hits)
    active_topics = set()
    for fp in hit_files:
        active_topics.update(script_topics.get(fp, set()))

    # 7. Graph expansion — pull related scripts the vector search missed
    expansions = _graph_expand(graph_db, active_topics, systems, hit_files, budget=5,
                               pipeline_log=pipeline_log)

    # 8. System map header
    system_map = _build_system_map(systems, signal_bus, active_topics)

    # 9. Pack context (system-organized, token-budget-aware)
    assembled_context, tokens_used = _pack_context(
        scored_hits, expansions, graph_db, script_topics,
        detail_threshold, persona_key, token_budget,
        pipeline_log=pipeline_log,
    )
    graph_db.close()
    q_client.close()

    # 10. Build final prompt
    has_roadmap = bool(roadmap_context.strip())

    if context_limit > 50:
        # ---- CLOUD / HIGH-CONTEXT FORMAT (XML tags) ----
        roadmap_xml = f"<PROJECT_ROADMAP>\n{roadmap_context}\n</PROJECT_ROADMAP>\n\n" if has_roadmap else ""
        system_xml  = f"<SYSTEM_MAP>\n{system_map}\n</SYSTEM_MAP>\n\n" if system_map else ""

        # Persona-specific response sections
        _sections = ["1. 🏗️ Architectural Analysis — system structure, coupling, and Godot pattern critique"]
        if has_roadmap:
            _sections.append(f"{len(_sections)+1}. 🗺️ Roadmap Alignment — map findings to project goals and pending milestones")
        _sections.append(f"{len(_sections)+1}. 💻 Surgical Implementation — complete, copy-paste-ready GDScript with line-level comments")
        if persona_key == "Bug Hunter":
            _sections = [
                "1. 🐛 Bug Inventory — file path, approximate line, root cause, reproduction condition",
                "2. 🔥 Severity Ranking — Critical / High / Medium / Low",
                "3. 🔧 Surgical Fixes — exact code replacements, not pseudocode",
            ]
        elif persona_key == "Code Optimizer":
            _sections = [
                "1. ⚡ Bottleneck Map — worst offenders with estimated impact",
                "2. 📐 Optimized Code — rewritten hot paths, avoid per-frame allocations",
                "3. 📊 Before / After — show measurable improvement (ops reduced, allocs saved)",
            ]
        elif persona_key == "Teacher":
            _sections = [
                "1. 🧩 Concept Breakdown — explain the 'why' behind each pattern found",
                "2. 📖 Annotated Walkthrough — step through the key code paths",
                "3. 🎯 Exercises — suggest 2-3 follow-up changes to cement understanding",
            ]
        sections_str = "\n".join(_sections)

        prompt = (
            f"{system_persona}\n\n"
            f"{roadmap_xml}"
            f"{system_xml}"
            f"<ARCHITECTURAL_CONTEXT>\n{assembled_context}\n</ARCHITECTURAL_CONTEXT>\n\n"
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<INSTRUCTIONS>\n"
            f"Before answering: read the <SYSTEM_MAP> to understand subsystem boundaries, "
            f"then cross-reference with <ARCHITECTURAL_CONTEXT>. "
            f"Cite exact file paths and function names. Never invent code that isn't in the context.\n\n"
            f"Structure your response:\n{sections_str}\n"
            f"</INSTRUCTIONS>"
        )
    else:
        # ---- LOCAL LLM FORMAT (markdown, compact) ----
        prompt_parts = [system_persona]
        if has_roadmap:
            prompt_parts.append(f"[PROJECT ROADMAP]\n{roadmap_context}")
        if system_map:
            prompt_parts.append(system_map)
        if assembled_context:
            prompt_parts.append(f"[REPOSITORY CONTEXT]\n{assembled_context}")

        # Persona-adaptive instructions for local LLM
        if persona_key == "Bug Hunter":
            inst = (
                "TASK: Find every bug relevant to the USER QUERY.\n"
                "For each: FILE | FUNCTION | ROOT CAUSE | FIX (exact GDScript, not pseudocode).\n"
                "Do not summarize. Do not skip files. Rank by severity."
            )
        elif persona_key == "Code Optimizer":
            inst = (
                "TASK: Profile the code relevant to the USER QUERY.\n"
                "Identify _process / physics hotpaths, per-frame allocations, and redundant lookups.\n"
                "Show rewritten code with a one-line comment on what was improved."
            )
        elif persona_key == "Teacher":
            inst = (
                "TASK: Explain the code relevant to the USER QUERY as if teaching a Godot beginner.\n"
                "Use the actual code as examples. Explain signals, nodes, and lifecycle hooks clearly."
            )
        else:
            inst = (
                "TASK: Answer the USER QUERY with architectural precision.\n"
                "Structure: (1) What the current code does, (2) Problems or improvement opportunities, "
                "(3) Recommended changes with GDScript snippets."
            )

        prompt_parts.append(f"USER QUERY: {query}\n\n{inst}")
        prompt = "\n\n================================\n\n".join(prompt_parts)

    # Collect context file paths for hallucination verification
    context_files = set(h.payload.get("file", "") for h in hits if h.payload.get("file"))
    context_files.update(exp["file"] for exp in expansions if exp.get("file"))

    return prompt, {
        "context_files": context_files,
        "token_budget": token_budget,
        "tokens_used": tokens_used,
        "prompt_text": prompt,
    }


def build_cloud_master_prompt(query, active_project, persona_key="Senior Architect",
                              collection_name=None, embed_model=None,
                              host=None, port=None, **kwargs):
    """Compile a structured XML prompt for cloud LLMs (Claude/Gemini).

    When embed_model + collection_name are provided, uses the Brewery pipeline:
    vector search identifies the most relevant files, graph traversal expands to
    their signal/call neighbors, and remaining files are appended as context.
    Without embed_model, falls back to a full codebase dump (legacy behaviour).

    Args:
        collection_name: Qdrant collection name. Enables graph topology header.
        embed_model:     OllamaEmbedder instance. Enables vector-guided filtering.
        host/port:       Qdrant connection details (required when embed_model set).
    """
    base_style = PERSONAS.get(persona_key, PERSONAS["Senior Architect"])
    system_persona = f"You are Scout, an expert AI software architect. {base_style}"

    # --- Roadmap (only if populated) ---
    roadmap = load_project_data(active_project["scout_dir"])
    roadmap_xml = ""
    global_goal   = roadmap.get("global_goal", "")
    completed_ms  = [m["name"] for m in roadmap.get("milestones", []) if m["status"] == "completed"]
    pending_ms    = [m["name"] for m in roadmap.get("milestones", []) if m["status"] == "pending"]
    has_roadmap   = bool(global_goal or completed_ms or pending_ms)
    if has_roadmap:
        roadmap_xml  = "<PROJECT_ROADMAP>\n"
        if global_goal:   roadmap_xml += f"  Goal: {global_goal}\n"
        if completed_ms:  roadmap_xml += f"  Done: {', '.join(completed_ms)}\n"
        if pending_ms:    roadmap_xml += f"  Active: {', '.join(pending_ms)}\n"
        roadmap_xml += "</PROJECT_ROADMAP>\n\n"

    # --- Graph topology map (from Kuzu, if indexed) ---
    graph_xml = ""
    graph_db_for_expand = None
    _systems_map = {}
    _signal_bus_map = {}
    if collection_name:
        try:
            _g = ScoutGraph(active_project["id"], active_project["scout_dir"],
                            collection_name, read_only=True)
            _systems_map, _, _signal_bus_map = _discover_systems(_g)
            _g.close()
            if _systems_map:
                _lines = ["Pre-compiled architectural map (graph DB):"]
                for _topic, _scripts in sorted(_systems_map.items()):
                    _bases = ", ".join(sorted(os.path.basename(s) for s in _scripts))
                    _lines.append(f"  System [{_topic}]: {_bases}")
                    _sigs = {sn for sn, bus in _signal_bus_map.items()
                             if _scripts & (bus["definers"] | bus["emitters"])}
                    if _sigs:
                        _lines.append(f"    Signals: {', '.join(sorted(_sigs))}")
                graph_xml = "<SYSTEM_TOPOLOGY>\n" + "\n".join(_lines) + "\n</SYSTEM_TOPOLOGY>\n\n"
        except Exception:
            pass  # Graph not built yet — skip silently

    # --- Enumerate all project source files ---
    project_path = active_project["path"]
    _SKIP_DIRS = {".scout", ".git", ".claude", "python_env", "node_modules",
                  "__pycache__", ".godot", ".import", "venv", ".venv"}
    all_fpaths = []
    for r, d, fs in os.walk(project_path):
        d[:] = [sub for sub in d if sub not in _SKIP_DIRS]
        for f in fs:
            if f.endswith((".gd", ".md", ".json")):
                all_fpaths.append(os.path.join(r, f))
    all_fpaths.sort()

    def _read_file_xml(fpath):
        try:
            rel_path = os.path.relpath(fpath, project_path).replace("\\", "/")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                code = fh.read()
            if code.strip():
                return rel_path, f'<file path="{rel_path}">\n{code}\n</file>'
        except Exception:
            pass
        return None, None

    # --- Vector-guided filtering (Brewery path) ---
    # When embed_model is available, use the full pipeline: vector → graph expand → remainder.
    # This preserves the "distillation" value prop even for cloud prompts.
    use_brewery = bool(embed_model and collection_name)
    priority_rel_paths = set()   # files promoted to the front by vector+graph

    if use_brewery:
        try:
            raw_vectors = embed_model.encode([query])
            dense_vector = raw_vectors[0] if raw_vectors else []
            if dense_vector:
                qc = get_qdrant_client(
                    mode=kwargs.get("qdrant_mode", "local"),
                    scout_dir=active_project.get("scout_dir"),
                    host=host, port=port,
                )
                hits = []
                try:
                    # Hybrid search if BM25 vocab exists
                    vocab_path = os.path.join(active_project["scout_dir"], "bm25_vocab.json")
                    if os.path.exists(vocab_path):
                        with open(vocab_path, "r", encoding="utf-8") as _vf:
                            bm25_vocab = json.load(_vf)
                        from qdrant_client.models import Prefetch, FusionQuery, Fusion
                        from core.sparse import encode_sparse_query
                        sp_indices, sp_values = encode_sparse_query(query, bm25_vocab)
                        prefetch = [Prefetch(query=dense_vector, using="dense", limit=30)]
                        if sp_indices:
                            from qdrant_client.models import SparseVector
                            prefetch.append(Prefetch(
                                query=SparseVector(indices=sp_indices, values=sp_values),
                                using="bm25", limit=30,
                            ))
                        hits = qc.query_points(
                            collection_name=collection_name,
                            prefetch=prefetch,
                            query=FusionQuery(fusion=Fusion.RRF),
                            limit=20,
                        ).points
                    else:
                        hits = qc.query_points(
                            collection_name=collection_name,
                            query=dense_vector,
                            using="dense",
                            limit=20,
                        ).points
                except Exception:
                    hits = qc.query_points(
                        collection_name=collection_name,
                        query=dense_vector,
                        limit=20,
                    ).points

                hit_files = {h.payload.get("file", "").replace("\\", "/") for h in hits}
                priority_rel_paths.update(hit_files)

                # Graph-expand: add direct CALLS and signal-bus neighbors
                if _systems_map:
                    try:
                        _gx = ScoutGraph(active_project["id"], active_project["scout_dir"],
                                         collection_name, read_only=True)
                        _, script_topics_map, _ = _discover_systems(_gx)
                        active_topics = set()
                        for fp in hit_files:
                            active_topics.update(script_topics_map.get(fp, set()))
                        expansions = _graph_expand(
                            _gx, active_topics, _systems_map, hit_files, budget=10
                        )
                        _gx.close()
                        for exp in expansions:
                            priority_rel_paths.add(
                                exp["file"].replace("\\", "/")
                            )
                    except Exception:
                        pass
        except Exception:
            pass  # Brewery path failed — fall through to full dump

    # --- Assemble file XML: priority files first, then remainder ---
    context_chunks = []
    remainder_chunks = []

    for fpath in all_fpaths:
        rel_path, xml_block = _read_file_xml(fpath)
        if xml_block is None:
            continue
        if use_brewery and priority_rel_paths:
            if rel_path in priority_rel_paths:
                context_chunks.append(xml_block)
            else:
                remainder_chunks.append(xml_block)
        else:
            context_chunks.append(xml_block)

    # Append remainder so the full codebase is still present for cross-references
    context_chunks.extend(remainder_chunks)
    assembled_context = "\n\n".join(context_chunks)

    # --- Persona-adaptive instructions ---
    if persona_key == "Bug Hunter":
        instructions = (
            "TASK: Hunt every bug relevant to the USER_QUERY across the ENTIRE_CODEBASE.\n"
            "Use SYSTEM_TOPOLOGY to trace signal chains and ownership.\n"
            "For each bug:\n"
            "  • File path + function name\n"
            "  • Root cause (null reference / race condition / logic error / type mismatch)\n"
            "  • Exact GDScript fix (not pseudocode)\n"
            "  • Severity: CRITICAL / HIGH / MEDIUM / LOW\n"
            "Order bugs by severity. Do not omit any file that may be relevant."
        )
    elif persona_key == "Code Optimizer":
        instructions = (
            "TASK: Profile the ENTIRE_CODEBASE for performance issues relevant to USER_QUERY.\n"
            "Focus: _process / _physics_process allocations, per-frame dictionary lookups, "
            "redundant get_node calls, unoptimized signal connections.\n"
            "For each fix: show before/after GDScript and state what was improved."
        )
    elif persona_key == "Teacher":
        instructions = (
            "TASK: Explain the codebase to a Godot learner based on USER_QUERY.\n"
            "Use SYSTEM_TOPOLOGY to introduce systems in dependency order (lowest-level first).\n"
            "For each concept: quote the actual code, explain the pattern, and suggest one exercise."
        )
    else:  # Senior Architect
        instructions = (
            "TASK: Deliver a senior-level architectural assessment of the ENTIRE_CODEBASE "
            "focused on USER_QUERY.\n"
            "Structure your response:\n"
            "1. 🏗️ Architecture Critique — coupling, cohesion, Godot anti-patterns found\n"
            + (f"2. 🗺️ Roadmap Alignment — map findings to PROJECT_ROADMAP goals\n"
               f"3. 💻 Surgical Refactor — complete, copy-paste GDScript\n" if has_roadmap else
               "2. 💻 Surgical Refactor — complete, copy-paste GDScript\n")
            + "Cite every file you reference. Never invent code not present in the codebase."
        )

    # Label the context block to tell the LLM how files were selected
    if use_brewery and priority_rel_paths:
        codebase_tag = (
            f'<CODEBASE priority_files="{len(priority_rel_paths)}" '
            f'total_files="{len(context_chunks)}" '
            f'selection="vector+graph">'
        )
    else:
        codebase_tag = f'<ENTIRE_CODEBASE files="{len(context_chunks)}">'
    codebase_close = "</CODEBASE>" if (use_brewery and priority_rel_paths) else "</ENTIRE_CODEBASE>"

    prompt = (
        f"{system_persona}\n\n"
        f"{roadmap_xml}"
        f"{graph_xml}"
        f"{codebase_tag}\n"
        f"{assembled_context}\n"
        f"{codebase_close}\n\n"
        f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
        f"<INSTRUCTIONS>\n{instructions}\n</INSTRUCTIONS>"
    )
    return prompt


# =====================================================================
# LOCAL LLM INTERFACE
# =====================================================================

def ask_local_llm(prompt, model_name, max_tokens=8192, num_ctx=32768,
                   chat_history=None):
    """Send a compiled prompt to the local Ollama LLM with multi-turn context.

    The surgical prompt is injected as the system message.  Previous
    conversation turns (user + assistant) are included so the LLM can
    reference earlier answers and follow-up naturally.

    Args:
        prompt:       The fully built surgical/cloud prompt string (becomes system message).
        model_name:   Ollama model tag (e.g. 'qwen2.5-coder:7b').
        max_tokens:   Maximum tokens to generate (maps to Ollama num_predict).
        num_ctx:      Context window size for the model.
        chat_history: List of {"role": "user"|"assistant", "content": str} dicts.
                      If None or empty, behaves as a single-turn request.
    """
    import requests

    messages = [{"role": "system", "content": prompt}]

    if chat_history:
        # Keep a sliding window of recent turns to stay within context budget.
        # Each turn ≈ a few hundred tokens; cap at last 20 turns (10 exchanges).
        MAX_HISTORY_TURNS = 20
        recent = chat_history[-MAX_HISTORY_TURNS:]
        for msg in recent:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "Error: Empty response from model.")
    except Exception as e:
        return (
            f"❌ Local Engine Error: {e}\n"
            f"Ensure Ollama is running and the model '{model_name}' is pulled."
        )
