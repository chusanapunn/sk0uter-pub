# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCOUT_DATA = os.path.join(BASE_DIR, "scout_metadata")
os.makedirs(SCOUT_DATA, exist_ok=True)

LOCAL_PATH = "D:/Projects/flooded-cafe" # Update to your actual path
BASE_COLLECTION = "flooded_scout"

PROJECT_NAME = "flooded-cafe"
REPO_URL = "https://gitea.chusa.xyz/chusanapunn/flooded-cafe.git"
LOCAL_PATH = f"./repo_{PROJECT_NAME}"

# Qdrant mode: "local" uses embedded storage per project (.scout/vectors/).
# Set to "server" and configure QDRANT_HOST/PORT to use a remote Qdrant instance.
QDRANT_MODE = "local"  # "local" or "server"
QDRANT_HOST = "192.168.1.44"
QDRANT_PORT = 6333
OLLAMA_URL = "http://localhost:11434"

LLM_MODEL = "qwen2.5-coder:7b"

# --- THE OLLAMA EMBEDDING SWITCH ---
# Nomic Embed Text uses 768 dimensions. Mxbai uses 1024.
EMBED_DIM = 768 

SUPPORTED_MODELS = {
    "⚡ Nomic Embed (Fast, 768d)": "nomic-embed-text",
    "🧠 Mxbai Embed (Accurate, 1024d)": "mxbai-embed-large",
    "💎 All-MiniLM (Light, 384d)": "all-minilm"
}

MODEL_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}

def get_model_dim(model_name):
    m = model_name.lower()
    for key, dim in MODEL_DIMS.items():
        if key in m:
            return dim
    return 768  # default