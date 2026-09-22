"""Add the (separate) Wadhwani UniPod group chat as its own session — new
content, no overlap with the main UniPods group, so this is a plain append.
Reuses the Android-format parser and filters from ingest_chat_v2.py.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

from ingest_chat import build_chunks
from ingest_chat_v2 import parse_messages

load_dotenv()

CHAT_PATH = "WhatsApp Chat with Wadhwani UniPod AI Program Africa without media/WhatsApp Chat with Wadhwani UniPod AI Program Africa.txt"
SESSION_LABEL = "Wadhwani Group Chat"
ID_PREFIX = "wadhwani_chat"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in response.embeddings]


def main():
    messages = parse_messages(CHAT_PATH)
    print(f"Kept {len(messages)} messages after filtering.")

    chunks = build_chunks(messages)
    print(f"Grouped into {len(chunks)} chunks.")

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        embeddings = embed_batch(gemini, [c["text"] for c in batch])
        rows = [
            {
                "id": f"{ID_PREFIX}_{i + j}",
                "text": c["text"],
                "session": SESSION_LABEL,
                "ts": c["timestamp"],
                "embedding": emb,
            }
            for j, (c, emb) in enumerate(zip(batch, embeddings))
        ]
        supabase.table("chunks").upsert(rows).execute()
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} embedded and upserted")

    print(f"Done. {len(chunks)} '{SESSION_LABEL}' chunks added.")


if __name__ == "__main__":
    main()
