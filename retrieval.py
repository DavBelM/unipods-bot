"""Retrieval layer: embed a question and fetch the closest meeting chunks."""

from sentence_transformers import SentenceTransformer
import chromadb

_DB_PATH = "./unipods_db"
_COLLECTION = "meetings"

# Max distance for a chunk to count as relevant. Set from distance_check.py.
# Lower = stricter.
_RELEVANCE_THRESHOLD = 1.4

# Loaded once at import time so repeated calls to retrieve() are fast.
_model = SentenceTransformer("all-MiniLM-L6-v2")
_collection = chromadb.PersistentClient(path=_DB_PATH).get_collection(_COLLECTION)


def retrieve(question: str, k: int = 4) -> dict:
    """Return the top-k chunks for a question, with relevance info.

    Returns a dict:
      {
        "relevant": bool,        # True if the best chunk cleared the threshold
        "chunks": [ {"text","session","timestamp","distance"}, ... ]
      }
    """
    embedding = _model.encode(question).tolist()
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "session": meta["session"],
            "timestamp": meta["timestamp"],
            "distance": dist,
        })

    best = chunks[0]["distance"] if chunks else float("inf")
    return {"relevant": best <= _RELEVANCE_THRESHOLD, "chunks": chunks}
