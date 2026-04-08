import re
import os
import json

# =====================================================================
# SYSTEM INFERENCE — Keyword-based multi-system tagging
# =====================================================================
SYSTEM_KEYWORDS = {
    "movement":  ["velocity", "move_and_slide", "move_toward", "jump", "speed", "gravity",
                  "friction", "acceleration", "direction", "locomotion", "walk", "run", "dash",
                  "floor", "ceiling", "slide", "airborne"],
    "combat":    ["attack", "damage", "health", "hp", "hit", "hurt", "die", "death", "kill",
                  "weapon", "shoot", "bullet", "projectile", "shield", "armor", "block", "dodge",
                  "melee", "ranged", "invincible"],
    "animation": ["anim", "animation", "sprite", "frame", "blend", "transition", "state_machine",
                  "skeleton", "pose", "flip_h", "modulate", "tween", "play"],
    "ui":        ["label", "button", "menu", "hud", "panel", "tooltip", "dialog", "popup",
                  "canvas", "progress_bar", "rich_text", "control", "visible"],
    "inventory": ["inventory", "item", "pickup", "drop", "equip", "slot", "stack", "loot",
                  "collect", "consumable", "gear", "backpack"],
    "audio":     ["audio", "sound", "music", "sfx", "play_sound", "stream", "volume", "bus",
                  "pitch", "audiosource"],
    "ai":        ["enemy", "npc", "pathfind", "navigate", "behavior", "patrol", "chase", "flee",
                  "waypoint", "blackboard", "decision", "aggro"],
    "camera":    ["camera", "zoom", "pan", "follow_target", "shake", "viewport", "cam_offset"],
    "save":      ["save", "load", "persist", "serialize", "config_file", "file_access",
                  "save_game", "load_game"],
    "input":     ["input", "is_action", "key_pressed", "mouse_button", "gamepad", "controller",
                  "just_pressed", "get_axis"],
    "network":   ["multiplayer", "peer", "rpc", "sync", "client", "server", "network",
                  "authority"],
}

_MIN_KEYWORD_HITS = 2


def infer_systems(content: str, topic: str = "") -> list:
    """Score chunk content against game-system keywords.
    Returns sorted list of matched system names (a chunk can belong to multiple)."""
    text = (content + " " + topic).lower()
    matched = []
    for system, keywords in SYSTEM_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= _MIN_KEYWORD_HITS:
            matched.append(system)
    return sorted(matched)


# Maximum lines per chunk before sub-chunking kicks in.
# Tuned for embedding models: ~60 lines ≈ 300-500 tokens, good retrieval granularity.
_MAX_CHUNK_LINES = 60


def _sub_chunk(chunk):
    """Split an oversized function chunk into smaller pieces at logical boundaries.

    Splits at blank lines or comment blocks within the function body,
    preserving the function signature in each sub-chunk for context.
    Returns a list of chunks (may be just the original if it's small enough).
    """
    content = chunk["content"]
    lines = content.split('\n')
    if len(lines) <= _MAX_CHUNK_LINES:
        return [chunk]

    # Extract the function signature (first line: "func name(...):")
    sig_line = lines[0] if lines else ""
    body_lines = lines[1:]

    # Find split points: blank lines or comment-only lines
    split_points = []
    for idx, line in enumerate(body_lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            split_points.append(idx)

    # Build sub-chunks of roughly _MAX_CHUNK_LINES each
    sub_chunks = []
    start = 0
    base_line = chunk["line_start"]
    part_num = 0

    for sp in split_points:
        # +1 for the sig_line offset
        if (sp - start) >= (_MAX_CHUNK_LINES - 1):
            segment = body_lines[start:sp]
            if any(ln.strip() for ln in segment):  # skip empty segments
                part_num += 1
                sub_content = sig_line + "  # (continued)\n" + "\n".join(segment)
                sub_chunks.append({
                    **chunk,
                    "name": f"{chunk['name']}_part{part_num}",
                    "content": sub_content.strip(),
                    "line_start": base_line + start + 1,
                    "type": "function_part",
                })
            start = sp + 1

    # Remainder
    remainder = body_lines[start:]
    if any(ln.strip() for ln in remainder):
        if sub_chunks:
            part_num += 1
            sub_content = sig_line + "  # (continued)\n" + "\n".join(remainder)
            sub_chunks.append({
                **chunk,
                "name": f"{chunk['name']}_part{part_num}",
                "content": sub_content.strip(),
                "line_start": base_line + start + 1,
                "type": "function_part",
            })
        else:
            # Didn't find good split points — keep original
            return [chunk]

    # Always keep the original full chunk as well (for Tier 1 full-detail display)
    # but mark sub-chunks for retrieval
    return sub_chunks if sub_chunks else [chunk]


def parse_file(file_path, code):
    """The Master Router: Sends files to their specific parsers based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # 🛡️ DEFENSIVE: Ensure we ALWAYS return the structure app.py expects
    default_return = {"identity": "", "global_state": "", "chunks": [], "signals": [], "topic": "Unknown"}
    
    try:
        if ext == '.gd': return parse_gdscript(code, file_path)
        elif ext == '.md': return parse_markdown(code, file_path)
        elif ext == '.tscn': return parse_tscn(code, file_path)
        elif ext == '.json': return parse_json(code, file_path)
        else: return default_return
    except Exception:
        return default_return

# ---------------------------------------------------------
# VARIABLE EXTRACTION (for graph Variable nodes)
# ---------------------------------------------------------
_VAR_PATTERN = re.compile(
    r'^[ \t]*'
    r'(?:(@export(?:\s*\([^)]*\))?)\s+)?'   # group 1: @export
    r'(?:(@onready)\s+)?'                     # group 2: @onready
    r'(var|const)\s+'                          # group 3: var or const
    r'([a-zA-Z_][a-zA-Z0-9_]*)'              # group 4: variable name
    r'(?:\s*:\s*([a-zA-Z0-9_\[\]]+))?',      # group 5: optional type hint
    re.MULTILINE
)


def _extract_variables(code):
    """Extract variable declarations from GDScript top-level scope.

    Returns list of {"name", "kind", "type_hint", "line"}.
    """
    variables = []
    for i, line in enumerate(code.split('\n'), 1):
        m = _VAR_PATTERN.match(line)
        if m:
            export_tag, onready_tag, decl_type, name, type_hint = m.groups()
            if decl_type == 'const':
                kind = 'const'
            elif export_tag:
                kind = 'export'
            elif onready_tag:
                kind = 'onready'
            else:
                kind = 'var'
            variables.append({"name": name, "kind": kind, "type_hint": type_hint or "", "line": i})
    return variables


def _detect_var_access(func_body, var_names):
    """Scan a function body for variable reads and writes.

    Returns (reads: set, writes: set).
    """
    if not var_names:
        return set(), set()
    reads = set()
    writes = set()
    escaped = [re.escape(n) for n in var_names]
    names_alt = '|'.join(escaped)
    write_pat = re.compile(r'\b(' + names_alt + r')\s*(?:\.[a-zA-Z_]\w*\s*)?(?:[\+\-\*/]?=(?!=))')
    read_pat = re.compile(r'\b(' + names_alt + r')\b')
    for line in func_body.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for wm in write_pat.finditer(line):
            writes.add(wm.group(1))
        for rm in read_pat.finditer(line):
            reads.add(rm.group(1))
    return reads, writes


# ---------------------------------------------------------
# 1. GDSCRIPT PARSER
# ---------------------------------------------------------
def parse_gdscript(code, file_path):
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    state_lines = re.findall(r'^[ \t]*(?:@export.*|@onready.*|var\s+.*|const\s+.*)', code, flags=re.MULTILINE)
    global_state = "\n".join(state_lines)

    parts = re.split(r'(^[ \t]*(?:static\s+)?func\s+)', code, flags=re.MULTILINE)
    identity = parts[0].strip() if parts else ""

    topic = "Unnamed Script"
    class_match = re.search(r'^[ \t]*class_name\s+([a-zA-Z0-9_]+)', identity, flags=re.MULTILINE)
    if class_match: topic = class_match.group(1)
    else:
        extends_match = re.search(r'^[ \t]*extends\s+([a-zA-Z0-9_]+)', identity, flags=re.MULTILINE)
        if extends_match: topic = f"Extends {extends_match.group(1)}"

    signals = re.findall(r'^[ \t]*signal\s+([a-zA-Z0-9_]+)', identity, flags=re.MULTILINE)

    # Extract extends parent for EXTENDS relationship
    extends_parent = ""
    extends_match_full = re.search(r'^[ \t]*extends\s+([a-zA-Z0-9_]+)', identity, flags=re.MULTILINE)
    if extends_match_full:
        extends_parent = extends_match_full.group(1)

    # Extract cross-script references for CALLS relationship
    # Patterns: get_node("Path"), $Path, ClassName.method(), preload("res://path.gd")
    node_refs = re.findall(r'get_node\s*\(\s*["\']([^"\']+)["\']', code)
    dollar_refs = re.findall(r'\$([A-Za-z0-9_/]+)', code)
    preload_refs = re.findall(r'(?:preload|load)\s*\(\s*["\']res://([^"\']+\.gd)["\']', code)
    # Static calls: ClassName.method() where ClassName starts with uppercase
    static_calls = re.findall(r'\b([A-Z][a-zA-Z0-9_]+)\s*\.\s*[a-z_][a-zA-Z0-9_]*\s*\(', code)
    cross_refs = list(set(node_refs + dollar_refs + preload_refs + static_calls))

    # Track cumulative line count through split parts so each chunk gets a real line_start.
    # parts[0] = identity preamble, then alternating (header, body) pairs.
    line_cursor = parts[0].count('\n') + 1 if parts else 1

    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if (i+1) < len(parts) else ""
        content = (header + body).strip()

        func_line = line_cursor  # line where this "func" keyword starts
        # Advance cursor past header + body for the next iteration
        line_cursor += header.count('\n') + body.count('\n')

        name_match = re.match(r'([a-zA-Z0-9_]+)', body.strip())
        name = name_match.group(1) if name_match else "anon_func"

        # Detect emits: Godot 4 style (signal.emit()) AND Godot 3 style (emit_signal("name"))
        emits_g4 = re.findall(r'([a-zA-Z0-9_]+)\s*\.\s*emit\s*\(', body)
        emits_g3 = re.findall(r'emit_signal\s*\(\s*["\']([a-zA-Z0-9_]+)["\']', body)
        all_emits = list(set(emits_g4 + emits_g3))

        chunks.append({
            "name": name,
            "content": content,
            "line_start": func_line,
            "type": "function",
            "emits": all_emits,
            "is_static": "static" in header,
            "topic": topic,
            "systems": infer_systems(content, topic),
        })

    # Sub-chunk oversized functions for better retrieval granularity
    expanded_chunks = []
    for chunk in chunks:
        expanded_chunks.extend(_sub_chunk(chunk))
    chunks = expanded_chunks

    if not chunks and identity.strip():
        chunks.append({
            "name": "_script_data", "content": f"# [Data/Utility Script]\n{identity}",
            "line_start": 1, "type": "script_identity", "emits": [], "is_static": False, "topic": topic
        })

    # Variable extraction + reads/writes detection
    variables = _extract_variables(code)
    var_names = [v["name"] for v in variables]
    for chunk in chunks:
        r, w = _detect_var_access(chunk["content"], var_names)
        chunk["reads_vars"] = sorted(r)
        chunk["writes_vars"] = sorted(w)

    # Extract class_name explicitly for Class node creation
    class_name_val = class_match.group(1) if class_match else ""

    return {
        "identity": identity, "global_state": global_state, "chunks": chunks,
        "signals": signals, "topic": topic,
        "extends": extends_parent,
        "cross_refs": cross_refs,
        "variables": variables,
        "class_name": class_name_val,
    }

# ---------------------------------------------------------
# 2. MARKDOWN PARSER
# ---------------------------------------------------------
def parse_markdown(code, file_path):
    code = code.replace('\r\n', '\n')
    parts = re.split(r'(^#{1,3}\s+.*)', code, flags=re.MULTILINE)
    topic = os.path.basename(file_path).replace('.md', '').replace('_', ' ').title()
    identity = parts[0].strip() if parts else "Markdown Document"
    
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if (i+1) < len(parts) else ""
        if not body: continue
        clean_name = re.sub(r'^#+\s*', '', header).replace(' ', '_').lower()
        
        chunks.append({
            "name": f"doc_{clean_name}",
            "content": f"{header}\n{body}",
            "line_start": 0, "type": "markdown_section", "emits": [], "is_static": False, "topic": topic
        })
        
    if not chunks and code.strip():
        chunks.append({
            "name": "full_document", "content": code.strip(),
            "line_start": 0, "type": "markdown_section", "emits": [], "is_static": False, "topic": topic
        })

    return {"identity": identity, "global_state": "", "chunks": chunks, "signals": [], "topic": topic}

# ---------------------------------------------------------
# 3. GODOT SCENE PARSER (.tscn) — with hierarchy + script bindings
# ---------------------------------------------------------
_TSCN_NODE_PATTERN = re.compile(
    r'\[node\s+'
    r'name="([^"]+)"'                          # name (required)
    r'(?:\s+type="([^"]*)")?'                   # type (optional)
    r'(?:\s+parent="([^"]*)")?'                 # parent path (optional; absent = root)
    r'[^\]]*\]'                                 # rest of header
    r'((?:\n(?!\[)[^\n]*)*)',                    # body lines until next section
    re.MULTILINE
)


def parse_tscn(code, file_path):
    code = code.replace('\r\n', '\n')
    topic = f"Scene: {os.path.basename(file_path)}"
    ext_resources = re.findall(r'\[ext_resource.*?path="(.*?)".*?\]', code)
    global_state = "Dependencies:\n" + "\n".join(ext_resources)

    scene_nodes = []
    for m in _TSCN_NODE_PATTERN.finditer(code):
        name, ntype, parent_path, body = m.groups()
        ntype = ntype or "Node"
        parent_path = parent_path if parent_path is not None else ""

        # Compute full node path for parent-child resolution
        if parent_path == "":
            node_path = "."  # root
        elif parent_path == ".":
            node_path = name
        else:
            node_path = f"{parent_path}/{name}"

        # Extract attached script from body (ExtResource reference)
        script_path = ""
        if body:
            # Godot 4 format: ExtResource("id")
            sm = re.search(r'script\s*=\s*ExtResource\(\s*"([^"]+)"\s*\)', body)
            if sm:
                res_id = sm.group(1)
                rp = re.search(
                    rf'\[ext_resource[^\]]*id="?{re.escape(res_id)}"?[^\]]*path="([^"]+)"'
                    rf'|\[ext_resource[^\]]*path="([^"]+)"[^\]]*id="?{re.escape(res_id)}"?',
                    code
                )
                if rp:
                    script_path = (rp.group(1) or rp.group(2) or "")
            else:
                # Godot 3 format: ExtResource( 2 )
                sm3 = re.search(r'script\s*=\s*ExtResource\(\s*(\d+)\s*\)', body)
                if sm3:
                    res_id = sm3.group(1)
                    rp = re.search(
                        rf'\[ext_resource[^\]]*id="?{re.escape(res_id)}"?[^\]]*path="([^"]+)"'
                        rf'|\[ext_resource[^\]]*path="([^"]+)"[^\]]*id="?{re.escape(res_id)}"?',
                        code
                    )
                    if rp:
                        script_path = (rp.group(1) or rp.group(2) or "")

        scene_nodes.append({
            "name": name,
            "type": ntype,
            "parent_path": parent_path,
            "node_path": node_path,
            "script_path": script_path.replace("res://", ""),
        })

    # Build chunk content (backwards-compatible format for vector embeddings)
    node_summaries = [f"- {n['name']} (Type: {n['type']})" for n in scene_nodes]
    content = "Scene Nodes:\n" + "\n".join(node_summaries)

    chunks = [{
        "name": "scene_hierarchy",
        "content": content,
        "line_start": 0, "type": "scene_data", "emits": [], "is_static": False, "topic": topic
    }]

    return {
        "identity": "Godot Scene File", "global_state": global_state,
        "chunks": chunks, "signals": [], "topic": topic,
        "scene_nodes": scene_nodes,
    }

# ---------------------------------------------------------
# 4. JSON PARSER
# ---------------------------------------------------------
def parse_json(code, file_path):
    topic = f"Data: {os.path.basename(file_path)}"
    chunks = []
    try:
        data = json.loads(code)
        content = json.dumps(data, indent=2)
        chunks.append({
            "name": "json_root",
            "content": content,
            "line_start": 0, "type": "json_data", "emits": [], "is_static": False, "topic": topic
        })
        identity = "Valid JSON Data"
    except json.JSONDecodeError:
        identity = "Invalid JSON File"

    return {"identity": identity, "global_state": "", "chunks": chunks, "signals": [], "topic": topic}