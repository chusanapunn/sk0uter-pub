# core/verifier.py — Hallucination Verification via Graph-Constrained Checking
#
# After an LLM generates a response, this module scans it for references to
# code entities (functions, signals, variables, files, classes) and cross-checks
# each against the Kuzu knowledge graph. Entities not found in the graph are
# flagged as potential hallucinations.
#
# Also computes mathematical verification metrics:
#   - Grounding Score: % of referenced entities that exist in the graph
#   - Entity Precision: per-type breakdown (functions, signals, vars, files)
#   - Context Coverage: % of prompt context actually referenced in the response
import re
from collections import defaultdict


class VerificationResult:
    """Structured result from hallucination verification."""

    def __init__(self):
        self.verified = defaultdict(list)     # type -> [entity_name, ...]
        self.hallucinated = defaultdict(list)  # type -> [entity_name, ...]
        self.context_files_referenced = set()  # files the LLM actually cited
        self.context_files_provided = set()    # files that were in the prompt

    @property
    def total_entities(self):
        return sum(len(v) for v in self.verified.values()) + \
               sum(len(v) for v in self.hallucinated.values())

    @property
    def total_verified(self):
        return sum(len(v) for v in self.verified.values())

    @property
    def total_hallucinated(self):
        return sum(len(v) for v in self.hallucinated.values())

    @property
    def grounding_score(self):
        """Percentage of referenced entities that exist in the graph (0.0-1.0)."""
        total = self.total_entities
        if total == 0:
            return 1.0  # No entities referenced = nothing to hallucinate
        return self.total_verified / total

    @property
    def context_coverage(self):
        """Percentage of provided context files actually referenced in response."""
        if not self.context_files_provided:
            return 0.0
        return len(self.context_files_referenced & self.context_files_provided) / \
               len(self.context_files_provided)

    def entity_precision(self):
        """Per-type breakdown: {type: {"verified": n, "hallucinated": n, "precision": float}}."""
        all_types = set(self.verified.keys()) | set(self.hallucinated.keys())
        breakdown = {}
        for etype in sorted(all_types):
            v = len(self.verified.get(etype, []))
            h = len(self.hallucinated.get(etype, []))
            total = v + h
            breakdown[etype] = {
                "verified": v,
                "hallucinated": h,
                "precision": v / total if total > 0 else 1.0,
            }
        return breakdown

    def summary(self):
        """One-line summary for UI display."""
        score = self.grounding_score
        total_h = self.total_hallucinated
        if total_h == 0:
            return f"Grounding: {score:.0%} — all entities verified"
        return (
            f"Grounding: {score:.0%} — "
            f"{self.total_verified} verified, {total_h} unverified"
        )

    def full_report(self):
        """Multi-line report for expander display."""
        lines = []
        score = self.grounding_score
        coverage = self.context_coverage

        lines.append(f"=== Grounding Score: {score:.1%} ===")
        lines.append(f"Context Coverage: {coverage:.1%} "
                      f"({len(self.context_files_referenced & self.context_files_provided)}"
                      f"/{len(self.context_files_provided)} files referenced)")
        lines.append("")

        precision = self.entity_precision()
        if precision:
            lines.append("Entity Precision:")
            for etype, stats in precision.items():
                bar = "+" * stats["verified"] + "-" * stats["hallucinated"]
                lines.append(f"  {etype:12s} [{bar}] {stats['precision']:.0%}")
            lines.append("")

        if self.hallucinated:
            lines.append("Potential Hallucinations:")
            for etype, entities in sorted(self.hallucinated.items()):
                for e in entities:
                    lines.append(f"  [{etype}] {e}")
        else:
            lines.append("No hallucinations detected.")

        return "\n".join(lines)


def verify_response(response_text, graph_db, context_files=None, pipeline_log=None):
    """Verify an LLM response against the knowledge graph.

    Args:
        response_text: The LLM's generated response string.
        graph_db:      An open ScoutGraph instance (read-only).
        context_files: Set of file paths that were provided in the prompt context.
        pipeline_log:  Optional PipelineLog for error collection.

    Returns:
        VerificationResult with grounding score and hallucination details.
    """
    result = VerificationResult()
    result.context_files_provided = set(context_files or [])

    # --- Extract entity references from the response ---
    # Function names: look for func_name(), _func_name, common patterns
    func_refs = set(re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', response_text))
    # Remove common non-function patterns
    func_refs -= {"if", "for", "while", "match", "return", "print", "str",
                  "int", "float", "bool", "len", "min", "max", "abs", "range",
                  "typeof", "is_instance_of", "preload", "load", "var", "null",
                  "true", "false", "not", "and", "or", "in", "as", "class",
                  # Godot builtins that appear as function-like patterns
                  "emit_signal", "connect", "disconnect", "get_node", "get_tree",
                  "queue_free", "add_child", "remove_child", "set_process",
                  "is_instance", "push_error", "push_warning", "assert",
                  "yield", "await", "super", "self", "ready", "process",
                  "physics_process", "input", "unhandled_input"}

    # Signal names: emit_signal("name"), signal.emit(), ⚡name
    signal_refs = set()
    signal_refs.update(re.findall(r'emit_signal\s*\(\s*["\']([a-zA-Z0-9_]+)["\']', response_text))
    signal_refs.update(re.findall(r'([a-zA-Z0-9_]+)\s*\.\s*emit\s*\(', response_text))
    signal_refs.update(re.findall(r'signal\s+([a-zA-Z0-9_]+)', response_text))

    # File paths: anything.gd, anything.tscn
    file_refs = set(re.findall(r'([a-zA-Z0-9_/\-]+\.(?:gd|tscn|cs|cpp|h))', response_text))
    result.context_files_referenced = file_refs

    # Variable names from backtick references: `var_name`
    var_refs = set(re.findall(r'`([a-z_][a-z0-9_]*)`', response_text))

    # Class names: PascalCase references
    class_refs = set(re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:2D|3D)?)\b', response_text))
    # Remove common non-class words
    class_refs -= {"GDScript", "Godot", "NOTE", "TODO", "FIXME", "WARNING",
                   "OK", "ERROR", "CRITICAL", "HIGH", "MEDIUM", "LOW",
                   "Code", "The", "This", "That", "Here", "Each", "When",
                   "If", "For", "Bug", "Fix", "None", "True", "False",
                   "JSON", "HTML", "CSS", "API", "URL", "HTTP", "HTTPS"}

    # --- Cross-check against the graph ---
    graph_functions = _query_set(graph_db, "MATCH (f:Function) RETURN f.name",
                                  pipeline_log)
    graph_signals = _query_set(graph_db, "MATCH (s:Signal) RETURN s.name",
                                pipeline_log)
    graph_scripts = _query_set(graph_db, "MATCH (s:Script) RETURN s.path",
                                pipeline_log)
    graph_variables = _query_set(graph_db, "MATCH (v:Variable) RETURN v.name",
                                  pipeline_log)
    graph_classes = _query_set(graph_db, "MATCH (c:Class) RETURN c.name",
                                pipeline_log)

    # Also collect script basenames for fuzzy file matching
    graph_basenames = {p.split('/')[-1].split('\\')[-1] for p in graph_scripts}

    # --- Classify each reference ---
    for fn in func_refs:
        if fn in graph_functions:
            result.verified["function"].append(fn)
        else:
            result.hallucinated["function"].append(fn)

    for sig in signal_refs:
        if sig in graph_signals:
            result.verified["signal"].append(sig)
        else:
            result.hallucinated["signal"].append(sig)

    for fp in file_refs:
        basename = fp.split('/')[-1].split('\\')[-1]
        if fp in graph_scripts or basename in graph_basenames:
            result.verified["file"].append(fp)
        else:
            result.hallucinated["file"].append(fp)

    for var in var_refs:
        if var in graph_variables:
            result.verified["variable"].append(var)
        # Don't flag vars as hallucinated — backtick patterns are too noisy

    for cls in class_refs:
        if cls in graph_classes:
            result.verified["class"].append(cls)
        # Don't flag classes — too many false positives from Godot builtins not in graph

    return result


def compute_prompt_efficiency(prompt_text, response_text, token_budget):
    """Compute mathematical metrics for prompt quality.

    Returns dict with:
        - compression_ratio: prompt chars / original context chars (lower = better)
        - budget_utilization: tokens used / budget (closer to 1.0 = better)
        - response_density: unique entities in response / tokens in response
    """
    prompt_tokens = max(1, len(prompt_text) // 3)  # rough estimate
    response_tokens = max(1, len(response_text) // 4)

    # Extract unique entity references from response
    entities = set()
    entities.update(re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', response_text))
    entities.update(re.findall(r'([a-zA-Z0-9_/\-]+\.gd)', response_text))

    return {
        "budget_utilization": min(1.0, prompt_tokens / token_budget),
        "response_density": len(entities) / response_tokens if response_tokens > 0 else 0,
        "prompt_tokens_est": prompt_tokens,
        "response_tokens_est": response_tokens,
        "unique_entities_cited": len(entities),
    }


def _query_set(graph_db, query, pipeline_log=None):
    """Run a graph query and return a set of the first column values."""
    results = set()
    try:
        res = graph_db.conn.execute(query)
        while res.has_next():
            results.add(res.get_next()[0])
    except Exception as e:
        if pipeline_log:
            pipeline_log.warn("verifier", f"Graph query failed: {query[:60]}", e)
    return results
