"""Answer generation: turn retrieved chunks into a grounded, cited answer."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-flash-latest"

_SYSTEM_INSTRUCTIONS = """You are the UniPods AI programme assistant. You answer questions from members \
using ONLY the meeting transcript excerpts provided below. Rules:

1. Answer only using information contained in the excerpts. Do not use outside knowledge.
2. Every claim you make must be followed by a citation to its source, in the form \
(Session Name @ timestamp), taken from the excerpt it came from.
3. If the excerpts do not contain enough information to answer the question, respond \
exactly with: "This wasn't covered in the sessions I have." Do not guess.
4. Be concise and direct.
"""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "(no relevant excerpts found)"
    else:
        context = "\n\n".join(
            f"[{c['session']} @ {c['timestamp']}]\n{c['text']}"
            for c in chunks
        )

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"--- Transcript excerpts ---\n{context}\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Answer ---\n"
    )


def _call_llm(prompt: str) -> str:
    last_err = None
    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,  # low, so it stays faithful to the chunks
                    max_output_tokens=600,
                    # Without this, the model's internal "thinking" tokens
                    # eat the whole budget and the visible answer gets cut
                    # off mid-sentence (confirmed via finish_reason=MAX_TOKENS
                    # with thoughts_token_count in the hundreds). Not needed
                    # for straightforward grounded-answer synthesis anyway.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"Gemini call failed after 3 attempts: {last_err}")
    return "The assistant is temporarily unavailable. Please try again in a moment."


def generate_answer(question: str, result: dict) -> str:
    """Build a grounded prompt from the retrieval result and return the answer.

    `result` is the dict returned by retrieval.retrieve(): {"relevant", "chunks"}.
    When nothing relevant was found, no chunks are used and a fixed message
    is returned instead of attempting an answer.
    """
    if not result["relevant"] or not result["chunks"]:
        return "That wasn't covered in the coaching sessions I have access to."

    prompt = _build_prompt(question, result["chunks"])
    return _call_llm(prompt)
