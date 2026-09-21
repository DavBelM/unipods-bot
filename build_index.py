import re, glob
from sentence_transformers import SentenceTransformer
import chromadb

# --- load transcripts, split into timestamped chunks ---
def load_chunks(path, session, window=40):
    lines = open(path, encoding="utf-8").read().splitlines()
    entries = []
    for ln in lines:
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", ln)
        if m:
            mm, ss, txt = int(m.group(1)), int(m.group(2)), m.group(3)
            entries.append((mm*60+ss, txt))
    chunks = []
    i = 0
    while i < len(entries):
        group = entries[i:i+window]
        start = group[0][0]
        text = " ".join(t for _, t in group)
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # kill "the the the"
        m, s = divmod(start, 60)
        chunks.append({
            "id": f"{session}_{i}",
            "text": text,
            "session": session,
            "timestamp": f"{m:02d}:{s:02d}",
        })
        i += window
    return chunks

sessions = {
    "welcome_module0.txt": "Welcome + Module 0",
    "module1_class.txt": "Module 1 Class",
    "problem_statement_qa.txt": "Problem Statement Q&A",
    "mit_onboarding.txt": "MIT Onboarding",
}

all_chunks = []
for fname, label in sessions.items():
    all_chunks += load_chunks(fname, label)

print(f"{len(all_chunks)} chunks")

# --- embed and store ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode([c["text"] for c in all_chunks], show_progress_bar=True)

client = chromadb.PersistentClient(path="./unipods_db")
try:
    client.delete_collection("meetings")
except Exception:
    pass
col = client.create_collection("meetings")
col.add(
    ids=[c["id"] for c in all_chunks],
    documents=[c["text"] for c in all_chunks],
    embeddings=[e.tolist() for e in embs],
    metadatas=[{"session": c["session"], "timestamp": c["timestamp"]} for c in all_chunks],
)
print("Index built ->", col.count(), "chunks in ./unipods_db")
