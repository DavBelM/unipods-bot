"""Chunk a single new [MM:SS] transcript file, embed it with Gemini, and
append it to the Supabase 'chunks' table — for adding one more session
without re-running the full migration.

Usage: python3 add_transcript.py <file.txt> "<Session Label>" <id_prefix>
Example: python3 add_transcript.py wadhwani_ignite_module2.txt "Wadhwani Ignite Module 2" ignite_m2
"""

import os
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20
WINDOW = 40  # lines per chunk, matching build_index.py's convention


def load_chunks(path: str, session: str, id_prefix: str) -> list[dict]:
    lines = open(path, encoding="utf-8").read().splitlines()
    entries = []
    for ln in lines:
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", ln)
        if m:
            mm, ss, txt = int(m.group(1)), int(m.group(2)), m.group(3)
            entries.append((mm * 60 + ss, txt))

    chunks = []
    i = 0
    while i < len(entries):
        group = entries[i:i + WINDOW]
        start = group[0][0]
        text = " ".join(t for _, t in group)
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
        m, s = divmod(start, 60)
        chunks.append({
            "id": f"{id_prefix}_{i}",
            "text": text,
            "session": session,
            "timestamp": f"{m:02d}:{s:02d}",
        })
        i += WINDOW
    return chunks


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in response.embeddings]


def main():
    path, session, id_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    chunks = load_chunks(path, session, id_prefix)
    print(f"{len(chunks)} chunks from {path}")

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

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
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} embedded and upserted")

    print(f"Done. {len(chunks)} chunks added under session '{session}'.")


if __name__ == "__main__":
    main()
