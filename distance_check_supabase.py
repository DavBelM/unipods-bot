import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

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
    resp = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=q,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    emb = resp.embeddings[0].values
    res = supabase.rpc("match_chunks", {"query_embedding": emb, "match_count": 1}).execute()
    d = res.data[0]["distance"]
    print(f"{d:6.3f}   {q}")
