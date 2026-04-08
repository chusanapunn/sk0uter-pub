# core/pipeline_log.py — Structured logging for pipeline runs
#
# Lightweight collector for warnings/errors during indexing and querying.
# The UI reads this after a pipeline completes to show actionable feedback.


class PipelineLog:
    def __init__(self):
        self._entries = []

    def warn(self, stage, message, detail=""):
        self._entries.append({"level": "warn", "stage": stage, "message": message, "detail": str(detail)})

    def error(self, stage, message, detail=""):
        self._entries.append({"level": "error", "stage": stage, "message": message, "detail": str(detail)})

    def info(self, stage, message):
        self._entries.append({"level": "info", "stage": stage, "message": message, "detail": ""})

    @property
    def warnings(self):
        return [e for e in self._entries if e["level"] == "warn"]

    @property
    def errors(self):
        return [e for e in self._entries if e["level"] == "error"]

    @property
    def has_issues(self):
        return any(e["level"] in ("warn", "error") for e in self._entries)

    def summary(self):
        w = len(self.warnings)
        e = len(self.errors)
        if not self.has_issues:
            return ""
        parts = []
        if e:
            parts.append(f"{e} error(s)")
        if w:
            parts.append(f"{w} warning(s)")
        return f"Pipeline completed with {', '.join(parts)}. Context may be incomplete."

    def full_report(self):
        lines = []
        for entry in self._entries:
            prefix = {"error": "ERROR", "warn": "WARN", "info": "INFO"}[entry["level"]]
            lines.append(f"[{prefix}] [{entry['stage']}] {entry['message']}")
            if entry["detail"]:
                lines.append(f"         {entry['detail']}")
        return "\n".join(lines)
