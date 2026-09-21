"""Answer generation: turn retrieved chunks into a grounded, cited answer.

Phase 1 has no LLM wired in yet. generate_answer() builds the prompt and
returns a placeholder. Tomorrow, fill in _call_llm() with a single API call
(OpenAI or Anthropic) and nothing else in this file needs to change.
"""

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
    # TODO: LLM call — replace this with a real OpenAI or Anthropic API call.
    # e.g. response = client.messages.create(model=..., messages=[{"role": "user", "content": prompt}])
    #      return response.content[0].text
    return "[placeholder] LLM not connected yet — this is where the generated answer will appear."


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
