from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="./unipods_db").get_collection("meetings")

tests = [
    "Who is going to pay for solving the problem?",
    "What is a customer problem?",
    "How do I write a good problem statement?",
    "What is the difference between primary and secondary customers?",
    "What is the capital of France?",
    "How do I bake bread?",
    "What is the weather today?",
]

for q in tests:
    emb = model.encode(q).tolist()
    res = col.query(query_embeddings=[emb], n_results=1,
                    include=["distances"])
    d = res["distances"][0][0]
    print(f"{d:6.3f}   {q}")
