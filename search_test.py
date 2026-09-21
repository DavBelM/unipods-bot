from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="./unipods_db").get_collection("meetings")

while True:
    q = input("\nQuestion (or 'quit'): ")
    if q.strip().lower() == "quit":
        break
    emb = model.encode(q).tolist()
    res = col.query(query_embeddings=[emb], n_results=3)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(f"\n[{meta['session']} @ {meta['timestamp']}]")
        print(doc[:300], "...")
