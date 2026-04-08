import os, shutil
from config import get_model_dim
from core.qdrant_client import get_qdrant_client, create_hybrid_collection, wipe_vector_db as _wipe_vec


def wipe_graph_db(scout_dir, collection_name):
    """Delete the Kuzu graph database folder for the given collection."""
    safe_name = "".join([c for c in collection_name if c.isalnum() or c in ('_', '-')])
    db_path = os.path.join(scout_dir, f"graph_{safe_name}")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)


def nuke_and_reset(scout_dir, collection_name, model_name,
                   qdrant_mode="local", host=None, port=None):
    """Wipe both graph and vector DBs. Returns (graph_ok, vector_ok, dim, errors)."""
    errors = []
    dim = get_model_dim(model_name)

    try:
        wipe_graph_db(scout_dir, collection_name)
        graph_ok = True
    except Exception as e:
        graph_ok = False
        errors.append(f"Graph Error: {e}")

    try:
        qc = get_qdrant_client(mode=qdrant_mode, scout_dir=scout_dir, host=host, port=port)
        _wipe_vec(qc, collection_name, dim)
        qc.close()
        vector_ok = True
    except Exception as e:
        vector_ok = False
        errors.append(f"Vector Error: {e}")

    return graph_ok, vector_ok, dim, errors
