"""FastAPI app: single /ask endpoint plus the static chat UI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval import retrieve
from answer import generate_answer

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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = retrieve(request.question)
    answer = generate_answer(request.question, result)
    sources = (
        [{"session": c["session"], "timestamp": c["timestamp"]} for c in result["chunks"]]
        if result["relevant"]
        else []
    )
    return {"answer": answer, "sources": sources}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
