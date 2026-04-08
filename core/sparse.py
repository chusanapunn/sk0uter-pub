# core/sparse.py — Lightweight BM25 sparse vector builder for Qdrant hybrid search
#
# No external dependencies — uses stdlib only.
# Generates sparse vectors compatible with Qdrant's SparseVector model.
import math
import re
from collections import Counter


# Deterministic hash to map tokens → sparse vector indices.
# Qdrant sparse vectors use integer indices; we hash tokens to a fixed range.
_SPARSE_DIM = 50_000  # collision-tolerant hash space


def _tokenize(text):
    """Code-aware tokenizer: splits on whitespace, punctuation, and camelCase/snake_case."""
    # Expand camelCase: "moveAndSlide" → "move And Slide"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split on non-alphanumeric (keeps underscores as separators too)
    tokens = re.findall(r'[a-zA-Z0-9_]{2,}', text.lower())
    # Also split snake_case: "move_and_slide" → ["move", "and", "slide"]
    expanded = []
    for t in tokens:
        parts = t.split('_')
        expanded.extend(p for p in parts if len(p) >= 2)
        if len(parts) > 1:
            expanded.append(t)  # keep the full token too
    return expanded


def _token_hash(token):
    """Deterministic hash → sparse index. Uses djb2 for speed."""
    h = 5381
    for ch in token:
        h = ((h << 5) + h) + ord(ch)
    return h % _SPARSE_DIM


def build_bm25_vocab(corpus_texts, k1=1.5, b=0.75):
    """Build BM25 statistics from a corpus of texts.

    Args:
        corpus_texts: list of raw text strings (one per chunk).
        k1, b: BM25 hyperparameters.

    Returns:
        A dict with idf scores and avgdl, used by `encode_sparse()`.
    """
    doc_count = len(corpus_texts)
    if doc_count == 0:
        return {"idf": {}, "avgdl": 1.0, "k1": k1, "b": b}

    df = Counter()  # document frequency per token
    total_length = 0

    for text in corpus_texts:
        tokens = _tokenize(text)
        total_length += len(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1

    avgdl = total_length / doc_count

    # IDF with smoothing (prevents zero/negative IDF for common tokens)
    idf = {}
    for token, freq in df.items():
        idf[token] = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    return {"idf": idf, "avgdl": avgdl, "k1": k1, "b": b}


def encode_sparse(text, vocab):
    """Encode a single text into a BM25-weighted sparse vector.

    Args:
        text:  Raw text string.
        vocab: BM25 vocabulary from `build_bm25_vocab()`.

    Returns:
        (indices, values) — parallel lists for Qdrant SparseVector.
    """
    idf = vocab["idf"]
    avgdl = vocab["avgdl"]
    k1 = vocab["k1"]
    b = vocab["b"]

    tokens = _tokenize(text)
    doc_len = len(tokens)
    if doc_len == 0:
        return [], []

    tf = Counter(tokens)
    sparse = {}  # hash_index → bm25_weight

    for token, count in tf.items():
        token_idf = idf.get(token, 0.0)
        if token_idf <= 0:
            continue
        # BM25 TF component
        tf_score = (count * (k1 + 1)) / (count + k1 * (1 - b + b * doc_len / avgdl))
        weight = token_idf * tf_score

        idx = _token_hash(token)
        # Accumulate (hash collisions add up — acceptable at 50k range)
        sparse[idx] = sparse.get(idx, 0.0) + weight

    indices = sorted(sparse.keys())
    values = [sparse[i] for i in indices]
    return indices, values


def encode_sparse_query(text, vocab):
    """Encode a query for BM25 sparse search.

    Query encoding uses raw IDF weights (no length normalization)
    since queries are short.
    """
    idf = vocab["idf"]
    tokens = _tokenize(text)
    if not tokens:
        return [], []

    seen = set()
    sparse = {}
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        token_idf = idf.get(token, 0.0)
        if token_idf <= 0:
            continue
        idx = _token_hash(token)
        sparse[idx] = sparse.get(idx, 0.0) + token_idf

    indices = sorted(sparse.keys())
    values = [sparse[i] for i in indices]
    return indices, values
