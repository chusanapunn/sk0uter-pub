# core/manager.py
import kuzu
import os
import json
from config import SCOUT_DATA, EMBED_DIM
from core.qdrant_client import get_qdrant_client, ensure_collection


SETTINGS_FILE = os.path.expanduser("~/.scout_registry.json")

class ProjectManager:
    @staticmethod
    def connect_project(path):
        """Creates DBs and folders for a new project connection."""
        project_name = os.path.basename(path).replace("-", "_").lower()
        project_dir = os.path.join(SCOUT_DATA, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # 1. Setup Kuzu Graph (Embedded)
        db = kuzu.Database(os.path.join(project_dir, "graph"))

        return project_name, project_dir

    @staticmethod
    def get_graph_connection(project_dir):
        db = kuzu.Database(os.path.join(project_dir, "graph"))
        return kuzu.Connection(db)

    @staticmethod
    def get_project_dirs(project_path):
        """Generates the internal paths for a project's metadata."""
        project_id = os.path.basename(project_path).replace("-", "_").lower()
        scout_dir = os.path.join(project_path, ".scout")
        os.makedirs(scout_dir, exist_ok=True)
        return project_id, scout_dir

    @staticmethod
    def init_databases(project_id, scout_dir, qdrant_mode="local", host=None, port=None):
        """Initializes Kuzu and Qdrant for this specific project."""
        qc = get_qdrant_client(mode=qdrant_mode, scout_dir=scout_dir, host=host, port=port)
        ensure_collection(qc, project_id, EMBED_DIM)
        qc.close()



import requests
from config import OLLAMA_URL

class OllamaEmbedder:
    """A lightweight wrapper that replaces SentenceTransformers."""
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.host = OLLAMA_URL

    def _model_exists(self):
        """Check if our model is already pulled. Returns (exists, model_list)."""
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        if r.status_code != 200:
            return False, []
        models = [m.get("name", "") for m in r.json().get("models", [])]
        match = any(m == self.model_name or m.startswith(self.model_name + ":") for m in models)
        return match, models

    def pull_model(self, progress_cb=None):
        """Pull the model from Ollama. Streams progress via optional callback."""
        r = requests.post(
            f"{self.host}/api/pull",
            json={"name": self.model_name},
            stream=True,
            timeout=600
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if line and progress_cb:
                import json as _json
                try:
                    data = _json.loads(line)
                    progress_cb(data.get("status", ""))
                except Exception:
                    pass

    def ensure_model(self, progress_cb=None):
        """Check Ollama is reachable; pull model if missing. Returns (ok, message)."""
        try:
            exists, models = self._model_exists()
        except Exception as e:
            return False, f"Cannot reach Ollama at {self.host}: {e}"
        if exists:
            return True, f"Model '{self.model_name}' ready."
        # Auto-pull
        if progress_cb:
            progress_cb(f"Pulling '{self.model_name}' from Ollama...")
        try:
            self.pull_model(progress_cb=progress_cb)
        except Exception as e:
            return False, f"Failed to pull '{self.model_name}': {e}"
        # Verify after pull
        try:
            exists, _ = self._model_exists()
        except Exception as e:
            return False, f"Post-pull check failed: {e}"
        if exists:
            return True, f"Model '{self.model_name}' pulled and ready."
        return False, f"Model '{self.model_name}' still not available after pull."

    def _try_embed(self, sub_batch):
        """Try /api/embed (batch), then /api/embeddings (legacy single-prompt)."""
        # Modern batch endpoint
        r = requests.post(f"{self.host}/api/embed",
                          json={"model": self.model_name, "input": sub_batch},
                          timeout=120)
        if r.status_code == 200:
            return r.json().get("embeddings", [])

        print(f"[WARN] /api/embed returned {r.status_code}: {r.text[:300]}")

        # Legacy single-prompt fallback
        results = []
        for text in sub_batch:
            r2 = requests.post(f"{self.host}/api/embeddings",
                               json={"model": self.model_name, "prompt": text},
                               timeout=120)
            if r2.status_code != 200:
                print(f"[ERROR] /api/embeddings also failed ({r2.status_code}): {r2.text[:300]}")
                return []
            vec = r2.json().get("embedding")
            if not vec:
                print(f"[ERROR] /api/embeddings returned no 'embedding' key: {r2.text[:300]}")
                return []
            results.append(vec)
        return results

    def encode(self, texts, batch_size=512, show_progress_bar=False):
        """Encode texts into embeddings. Large batch_size is faster — Ollama parallelizes internally."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i:i + batch_size]
            sub_idx = i // batch_size + 1
            try:
                embeddings = self._try_embed(sub_batch)
                if not embeddings:
                    print(f"[ERROR] No embeddings returned for sub-batch {sub_idx}.")
                    return []
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"[ERROR] Ollama Embedding Error (sub-batch {sub_idx}): {e}")
                return []

        return all_embeddings

def get_embedder(model_path):
    """Replaces the old caching function from app.py"""
    return OllamaEmbedder(model_name=model_path)