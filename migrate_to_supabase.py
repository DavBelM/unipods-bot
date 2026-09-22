"""One-time migration: dump the curated chunks out of the local Chroma
'meetings' collection, re-embed them with Gemini's embedding model, and
load them into Supabase (table 'chunks', pgvector column 'embedding').

Run supabase_schema.sql in the Supabase SQL Editor first. This script does
not touch the local Chroma index — it only reads from it.
"""

import os
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

DB_PATH = "./unipods_db"
COLLECTION = "meetings"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20


def dump_chunks() -> list[dict]:
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        chunks.append({
            "id": id_,
            "text": doc,
            "session": meta["session"],
            "timestamp": meta["timestamp"],
        })
    return chunks


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in response.embeddings]


def main():
    chunks = dump_chunks()
    print(f"Dumped {len(chunks)} chunks from local Chroma collection '{COLLECTION}'.")

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    start = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        embeddings = embed_batch(gemini, [c["text"] for c in batch])

        rows = [
            {
                "id": c["id"],
                "text": c["text"],
                "session": c["session"],
                "ts": c["timestamp"],
                "embedding": emb,
            }
            for c, emb in zip(batch, embeddings)
        ]
        supabase.table("chunks").upsert(rows).execute()

        done = min(i + BATCH_SIZE, len(chunks))
        elapsed = time.time() - start
        print(f"  {done}/{len(chunks)} chunks embedded and upserted "
              f"({elapsed:.1f}s elapsed)")

    print(f"Done. {len(chunks)} chunks loaded into Supabase 'chunks' table.")


if __name__ == "__main__":
    main()
