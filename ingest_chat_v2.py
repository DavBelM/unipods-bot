"""Parse the newer Android-style WhatsApp export ("DD/MM/YYYY, HH:MM -
sender: text", no brackets, no AM/PM) into cleaned messages and chunks,
reusing the same strict informational filter and chunking logic as
ingest_chat.py (which targets the older iOS-style bracketed export).

This is a full REPLACE of the WhatsApp Chat chunks in Supabase, not an
append — this export covers the entire group history (including
everything the original chat.txt already covered) plus new messages
since, so re-processing from scratch avoids any duplicate/dual-format
mess. Old "whatsapp_*" rows are deleted before the new ones are inserted.
"""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

from ingest_chat import _is_informational, _is_skippable, build_chunks

load_dotenv()

CHAT_PATH = "WhatsApp Chat with UniPods METI AI Program 2026 Cohort without media/WhatsApp Chat with UniPods METI AI Program 2026 Cohort.txt"
SESSION_LABEL = "WhatsApp Chat"
ID_PREFIX = "whatsapp"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20

# "DD/MM/YYYY, HH:MM - <rest>" — no brackets, 24-hour time, no seconds.
_MESSAGE_START = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{4}), (?P<time>\d{1,2}:\d{2}) - (?P<rest>.*)$"
)


def parse_messages(path: str) -> list[dict]:
    raw = []
    current = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _MESSAGE_START.match(line)
            if m:
                rest = m.group("rest")
                # No system notice in this export format contains ": " —
                # only real "sender: text" messages do.
                if ": " not in rest:
                    current = None
                    continue
                sender, text = rest.split(": ", 1)
                current = [m.group("date"), m.group("time"), sender, [text]]
                raw.append(current)
            else:
                if current is not None:
                    current[3].append(line)

    messages = []
    for date, time, sender, text_lines in raw:
        text = "\n".join(text_lines).strip()
        if _is_skippable(text):
            continue
        if not _is_informational(text):
            continue
        messages.append({"sender": sender, "date": date, "time": time, "text": text})
    return messages


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

    deleted = supabase.table("chunks").delete().like("id", f"{ID_PREFIX}_%").execute()
    print(f"Deleted {len(deleted.data)} old '{ID_PREFIX}_*' chunks.")

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

    print(f"Done. {len(chunks)} WhatsApp Chat chunks now in Supabase.")


if __name__ == "__main__":
    main()
