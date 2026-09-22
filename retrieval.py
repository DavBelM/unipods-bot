"""Retrieval layer: embed a question and fetch the closest meeting chunks.

Backed by Supabase (pgvector) instead of the local Chroma index, and Gemini
embeddings instead of local sentence-transformers — see migrate_to_supabase.py
for the one-time migration that populated the 'chunks' table.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 768

# Max cosine distance for a chunk to count as relevant. Set from
# distance_check_supabase.py. Lower = stricter. Not comparable to the old
# Chroma/sentence-transformers threshold (1.4) — different model, different
# distance scale.
_RELEVANCE_THRESHOLD = 0.40

# Loaded once at import time so repeated calls to retrieve() are fast.
_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def retrieve(question: str, k: int = 4) -> dict:
    """Return the top-k chunks for a question, with relevance info.

    Returns a dict:
      {
        "relevant": bool,        # True if the best chunk cleared the threshold
        "chunks": [ {"text","session","timestamp","distance"}, ... ]
      }
    """
    response = _gemini.models.embed_content(
        model=_EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
    )
    embedding = response.embeddings[0].values

    result = _supabase.rpc(
        "match_chunks", {"query_embedding": embedding, "match_count": k}
    ).execute()

    chunks = [
        {
            "text": row["text"],
            "session": row["session"],
            "timestamp": row["ts"],
            "distance": row["distance"],
        }
        for row in result.data
    ]

    best = chunks[0]["distance"] if chunks else float("inf")
    return {"relevant": best <= _RELEVANCE_THRESHOLD, "chunks": chunks}
