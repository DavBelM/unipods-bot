"""FastAPI app definition: /ask and /queries. No static file mounting here —
main.py (local dev) and api/index.py (Vercel) each wire up static serving
their own way, since Vercel serves static files directly rather than
through the Python function.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

from retrieval import retrieve
from answer import generate_answer

load_dotenv()

_supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

app = FastAPI(title="UniPods AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    session: str
    timestamp: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _log_query(question: str, answer: str, relevant: bool, sources: list[dict]) -> None:
    # Logging must never break a user-facing answer, so failures are swallowed.
    try:
        _supabase.table("queries").insert({
            "question": question,
            "answer": answer,
            "relevant": relevant,
            "sources": sources,
        }).execute()
    except Exception as e:
        print(f"Query logging failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = retrieve(request.question)
    answer = generate_answer(request.question, result)
    sources = (
        [{"session": c["session"], "timestamp": c["timestamp"]} for c in result["chunks"]]
        if result["relevant"]
        else []
    )
    _log_query(request.question, answer, result["relevant"], sources)
    return {"answer": answer, "sources": sources}


@app.get("/queries")
def list_queries(limit: int = 200):
    res = (
        _supabase.table("queries")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data
