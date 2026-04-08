# core/symcode.py — SymCode: Compressed symbolic representation of code
#
# Converts raw GDScript function bodies into a token-efficient DSL
# for use in Tier 2 (BRIEF) and Tier 3 (GRAPH) context packing.
#
# Syntax key (injected into prompt as legend):
#   ƒ name(args)→ret     = function signature
#   ⚡signal_name          = emit_signal / signal.emit()
#   §VarName              = variable declaration (var/const/export)
#   ◆ClassName            = class reference / extends
#   ↦ path                = get_node / $ node reference
#   ⟳ func_name()         = function call
#   ⊳ condition { ... }   = if/elif/else block (collapsed)
#   ↻ collection { ... }  = for/while loop (collapsed)
#   ⇐ value               = return statement
import re

# Legend text injected once into the prompt so the LLM can decode SymCode
SYMCODE_LEGEND = (
    "[SymCode Legend] "
    "ƒ=func ⚡=emit_signal §=var/const ◆=class ↦=node_ref "
    "⟳=call ⊳=if ↻=loop ⇐=return"
)


def encode_function(content, emits=None, global_state=""):
    """Compress a GDScript function into SymCode notation.

    Args:
        content:      Raw function text (including `func` header).
        emits:        List of signal names this function emits.
        global_state: Raw state lines from the script (for § annotations).

    Returns:
        Compressed SymCode string.
    """
    emits = set(emits or [])
    lines = content.split('\n')
    if not lines:
        return content

    # --- Signature line ---
    sig = _compress_signature(lines[0])
    body_lines = lines[1:]

    # --- Compress body ---
    compressed = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # emit_signal / signal.emit → ⚡
        emit_match = (
            re.search(r'emit_signal\s*\(\s*["\']([a-zA-Z0-9_]+)["\']', line) or
            re.search(r'([a-zA-Z0-9_]+)\s*\.\s*emit\s*\(', line)
        )
        if emit_match:
            sig_name = emit_match.group(1)
            # Extract args if present
            args_match = re.search(r'emit\s*\(([^)]*)\)', line)
            if args_match and args_match.group(1).strip():
                compressed.append(f"⚡{sig_name}({args_match.group(1).strip()})")
            else:
                compressed.append(f"⚡{sig_name}")
            i += 1
            continue

        # return → ⇐
        ret_match = re.match(r'return\s*(.*)', line)
        if ret_match:
            val = ret_match.group(1).strip()
            compressed.append(f"⇐ {val}" if val else "⇐")
            i += 1
            continue

        # if/elif/else → ⊳ (collapse the block to one line)
        if_match = re.match(r'(if|elif)\s+(.+?):', line)
        if if_match:
            kw, cond = if_match.groups()
            # Gather the body until dedent
            block_stmts = _collect_block_stmts(body_lines, i + 1)
            block_sym = '; '.join(_compress_stmt(s, emits) for s in block_stmts if s.strip())
            prefix = "⊳" if kw == "if" else "⊳elif"
            compressed.append(f"{prefix} {cond} {{{block_sym}}}")
            i += 1 + len(block_stmts)
            continue

        if line.startswith('else:'):
            block_stmts = _collect_block_stmts(body_lines, i + 1)
            block_sym = '; '.join(_compress_stmt(s, emits) for s in block_stmts if s.strip())
            compressed.append(f"⊳else {{{block_sym}}}")
            i += 1 + len(block_stmts)
            continue

        # for/while → ↻
        loop_match = re.match(r'(for|while)\s+(.+?):', line)
        if loop_match:
            _, cond = loop_match.groups()
            block_stmts = _collect_block_stmts(body_lines, i + 1)
            block_sym = '; '.join(_compress_stmt(s, emits) for s in block_stmts if s.strip())
            compressed.append(f"↻ {cond} {{{block_sym}}}")
            i += 1 + len(block_stmts)
            continue

        # var declaration inside function → §
        var_match = re.match(r'var\s+([a-zA-Z_]\w*)(?:\s*:\s*(\w+))?\s*=\s*(.*)', line)
        if var_match:
            vname, vtype, val = var_match.groups()
            type_ann = f":{vtype}" if vtype else ""
            compressed.append(f"§{vname}{type_ann}={val.strip()}")
            i += 1
            continue

        # General statement — light compression
        compressed.append(_compress_stmt(line, emits))
        i += 1

    # --- State summary (compact) ---
    state_sym = ""
    if global_state:
        state_parts = []
        for sline in global_state.split('\n'):
            sline = sline.strip()
            if not sline:
                continue
            # @export var x: Type = val → §x:Type=val
            em = re.match(
                r'(?:@export(?:\s*\([^)]*\))?\s+)?(?:@onready\s+)?'
                r'(var|const)\s+([a-zA-Z_]\w*)(?:\s*:\s*(\w+))?\s*(?:=\s*(.+))?',
                sline
            )
            if em:
                kw, name, typ, val = em.groups()
                prefix = "§" if kw == "var" else "§const "
                type_ann = f":{typ}" if typ else ""
                val_ann = f"={val.strip()}" if val else ""
                state_parts.append(f"{prefix}{name}{type_ann}{val_ann}")
        if state_parts:
            state_sym = " | ".join(state_parts) + "\n"

    body_str = "; ".join(compressed)
    return f"{state_sym}{sig} {{{body_str}}}"


def encode_skeleton(file_path, functions, global_state=""):
    """Encode a Tier 3 skeleton: file + function signatures + state summary.

    Args:
        file_path:    Relative path of the script.
        functions:    List of function name strings.
        global_state: Raw state lines (optional).

    Returns:
        SymCode skeleton string.
    """
    fn_list = ", ".join(f"ƒ {f}()" for f in functions) if functions else "(empty)"
    state_sym = ""
    if global_state:
        vars_found = re.findall(
            r'(?:var|const)\s+([a-zA-Z_]\w*)', global_state
        )
        if vars_found:
            state_sym = f" §[{', '.join(vars_found)}]"
    return f"{file_path}{state_sym} :: {fn_list}"


# --- Internal helpers ---

def _compress_signature(sig_line):
    """Convert `func name(args) -> ret:` to `ƒ name(args)→ret`."""
    sig_line = sig_line.strip()
    # Remove static prefix
    sig_line = re.sub(r'^static\s+', '', sig_line)
    # Remove func keyword
    sig_line = re.sub(r'^func\s+', '', sig_line)
    # Remove trailing colon
    sig_line = sig_line.rstrip(':').strip()
    # Convert -> to →
    sig_line = sig_line.replace(' -> ', '→').replace('->', '→')
    return f"ƒ {sig_line}"


def _collect_block_stmts(lines, start_idx):
    """Collect indented block statements starting from start_idx.

    Returns the lines belonging to the block (stops at dedent or end).
    """
    if start_idx >= len(lines):
        return []
    # Determine indent level of first block line
    first = lines[start_idx] if start_idx < len(lines) else ""
    if not first.strip():
        # Skip blank lines at block start
        for j in range(start_idx, len(lines)):
            if lines[j].strip():
                first = lines[j]
                start_idx = j
                break
        else:
            return []

    block_indent = len(first) - len(first.lstrip())
    if block_indent == 0:
        return []  # Not actually indented — not a block

    stmts = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if not line.strip():
            stmts.append("")
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent < block_indent:
            break
        stmts.append(line.strip())
    return stmts


def _compress_stmt(stmt, emits=None):
    """Light compression of a single statement."""
    emits = emits or set()
    stmt = stmt.strip()

    # emit_signal → ⚡
    stmt = re.sub(
        r'emit_signal\s*\(\s*["\']([a-zA-Z0-9_]+)["\']\s*(?:,\s*([^)]*))?\)',
        lambda m: f"⚡{m.group(1)}({m.group(2).strip()})" if m.group(2) else f"⚡{m.group(1)}",
        stmt
    )
    # signal.emit() → ⚡
    stmt = re.sub(
        r'([a-zA-Z0-9_]+)\s*\.\s*emit\s*\(([^)]*)\)',
        lambda m: f"⚡{m.group(1)}({m.group(2).strip()})" if m.group(2).strip() else f"⚡{m.group(1)}",
        stmt
    )

    # get_node("path") or $path → ↦path
    stmt = re.sub(r'get_node\s*\(\s*["\']([^"\']+)["\']\s*\)', r'↦\1', stmt)
    stmt = re.sub(r'\$([A-Za-z0-9_/]+)', r'↦\1', stmt)

    # return → ⇐
    stmt = re.sub(r'^return\s+(.*)', r'⇐ \1', stmt)
    stmt = re.sub(r'^return$', '⇐', stmt)

    return stmt
