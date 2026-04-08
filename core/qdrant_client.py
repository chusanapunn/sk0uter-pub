# core/qdrant_client.py — Centralized Qdrant client factory
#
# Supports two modes:
#   "local"  — embedded Qdrant stored in .scout/vectors/ (no server needed)
#   "server" — connects to a remote Qdrant instance via host:port
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance


def get_qdrant_client(mode="local", scout_dir=None, host=None, port=None):
    """Create a QdrantClient based on mode.

    Args:
        mode:      "local" for embedded storage, "server" for remote.
        scout_dir: Required for local mode — path to .scout/ directory.
        host/port: Required for server mode.
    """
    if mode == "local":
        if not scout_dir:
            raise ValueError("scout_dir is required for local Qdrant mode")
        vec_path = os.path.join(scout_dir, "vectors")
        os.makedirs(vec_path, exist_ok=True)
        return QdrantClient(path=vec_path)
    else:
        return QdrantClient(url=f"http://{host}:{port}")


def create_hybrid_collection(client, collection_name, dim):
    """Create a Qdrant collection with named dense + sparse vectors."""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(),
        },
    )


def ensure_collection(client, collection_name, dim):
    """Create collection if it doesn't exist. Returns True if created."""
    try:
        client.get_collection(collection_name)
        return False
    except Exception:
        create_hybrid_collection(client, collection_name, dim)
        return True


def wipe_vector_db(client, collection_name, dim):
    """Delete and recreate a collection."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    create_hybrid_collection(client, collection_name, dim)
    return dim
