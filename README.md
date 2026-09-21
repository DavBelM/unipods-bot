# UniPods AI Chatbot — Phase 1

A RAG chatbot that answers member questions from Wadhwani coaching session
transcripts, with citations to the session name and timestamp. Phase 1 has
retrieval and the UI working end-to-end; the LLM call is stubbed out until
an API key is available.

## Run it

```bash
source unipods/bin/activate
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

Type a question in the box and hit Send. You'll see your question appear,
then a bot reply. Right now the reply text is a placeholder
(`[placeholder] LLM not connected yet...`) because no LLM is wired in yet —
but the retrieval underneath it is real: the sources listed under the
answer are the actual top-k chunks pulled from `./unipods_db` for your
question, each shown as "Session @ timestamp".

## Wiring in the LLM (tomorrow)

Open [answer.py](answer.py) and fill in `_call_llm(prompt)` with a single
OpenAI or Anthropic API call that sends `prompt` and returns the text
response. That's the only place that needs to change — `generate_answer()`
already builds the grounded, cite-your-source prompt from the retrieved
chunks.

## How it fits together

- [retrieval.py](retrieval.py) — loads the `all-MiniLM-L6-v2` model once,
  embeds the question, queries the existing `meetings` collection in
  `./unipods_db`, and returns the top-k chunks (text + session + timestamp).
- [answer.py](answer.py) — builds a prompt instructing the model to answer
  only from the given chunks, cite session + timestamp, and say
  "This wasn't covered in the sessions I have" when it doesn't know.
  The actual LLM call is a `# TODO` stub for now.
- [main.py](main.py) — FastAPI app with `POST /ask` (`{"question": "..."}`
  → `{"answer": ..., "sources": [...]}`), CORS enabled, and serves the
  `static/` UI.
- [static/](static/) — plain HTML/CSS/JS chat page. No framework, no build
  step.

## Not built yet (by design)

- **Phase 2 (personalization):** tracking who asked what and what they've
  missed. This would plug in as a user identifier passed to `/ask` (e.g. a
  member ID or WhatsApp number) and a small store mapping users to sessions
  attended/missed, used to tailor retrieval or flag "you missed this" in
  the response. No code for this yet.
- **Phase 3 (WhatsApp integration):** this would be a separate webhook
  endpoint (e.g. `/whatsapp`) that receives incoming messages from a
  WhatsApp Business API/webhook provider, calls the same `retrieve()` /
  `generate_answer()` functions used by `/ask`, and sends the reply back
  through that provider's API. `main.py` is structured so this can be
  added as another route without touching retrieval or answer logic.

## Constraints honored

- Existing venv (`./unipods`), existing Chroma index (`./unipods_db`), and
  the transcript `.txt` files are untouched.
- `fetch_transcripts.py`, `build_index.py`, and `search_test.py` are
  untouched.
- No new dependencies beyond what was already installed.
- No auth, no extra database, no Docker.
