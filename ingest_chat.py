"""Parse chat.txt (a WhatsApp export) into cleaned messages, group them into
chunks, and append those chunks to the existing 'meetings' Chroma collection
alongside the meeting transcripts. This does NOT rebuild the collection —
it only appends. Run once; running twice will duplicate chunks.
"""

import re

import chromadb
from sentence_transformers import SentenceTransformer

CHAT_PATH = "chat.txt"
DB_PATH = "./unipods_db"
COLLECTION = "meetings"
SESSION_LABEL = "WhatsApp Chat"

MESSAGES_PER_CHUNK = 10
WORDS_PER_CHUNK = 600

# A line starts a new message only if it opens with a WhatsApp timestamp.
_MESSAGE_START = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), (?P<time>\d{1,2}:\d{2}:\d{2}\s?[AP]M)\] (?P<rest>.*)$"
)

_URL_ONLY = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)

# Broad emoji / pictograph / symbol ranges, plus the joiners/variation
# selectors that decorate them (skin tones, ZWJ sequences, keycaps).
_EMOJI_CHARS = re.compile(
    "[\U0001F000-\U0001FFFF☀-➿←-⇿⬀-⯿"
    "️‍⃣]"
)

# Message bodies that are just a media/deletion placeholder carry no
# retrievable content, so they're treated the same as an empty message.
_CONTENTLESS_BODIES = {
    "<image omitted>",
    "<video omitted>",
    "<audio omitted>",
    "<sticker omitted>",
    "<gif omitted>",
    "<document omitted>",
    "<contact card omitted>",
    "this message was deleted",
    "you deleted this message",
}

_MIN_WORDS = 8

_QUESTION_START = re.compile(
    r"^(what|when|how|where|why|who|can|is|are|should|does)\b", re.IGNORECASE
)

# Substring phrases that mark self-introductions / greetings (case-insensitive).
_GREETING_PHRASES = [
    "my name is",
    "glad to",
    "happy to join",
    "excited to",
    "hello everyone",
    "looking forward",
]
_IM_WORD = re.compile(r"\bi'?m\b", re.IGNORECASE)

_SOCIAL_URL = re.compile(r"(linkedin\.com|github\.com)", re.IGNORECASE)
_PROFILE_PHRASES = ["here's my profile", "heres my profile"]
_URL_ANY = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

_REACTION_WORDS = {
    "thanks", "thank", "you", "so", "much", "noted", "ok", "okay", "great",
    "nice", "lol", "haha", "hahaha", "same", "here", "exactly", "true",
    "agreed", "well", "said", "got", "it", "cheers", "bye", "welcome",
    "good", "alright", "fine", "cool", "yes", "no", "yeah", "yep", "sure",
}


def _is_system_line(rest: str) -> bool:
    # System/notice lines have "- " right after the timestamp instead of
    # "sender: ". Covers joins, leaves, adds, removes, subject changes,
    # the encryption notice, and "[System notification]" lines.
    return rest.startswith("- ")


def _is_skippable(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _URL_ONLY.match(stripped):
        return True
    if stripped.lower() in _CONTENTLESS_BODIES:
        return True
    if not _EMOJI_CHARS.sub("", stripped).strip():
        return True
    return False


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if not re.search(r"[A-Za-z]", stripped):
        return False  # punctuation/symbols only, e.g. "???" — not a real question
    if "?" in stripped:
        return True
    return bool(_QUESTION_START.match(stripped))


def _is_greeting_or_social(text: str) -> bool:
    lowered = text.lower()
    if _IM_WORD.search(lowered):
        return True
    if any(phrase in lowered for phrase in _GREETING_PHRASES):
        return True
    # Mostly hashtags/emoji: strip both out and see if hardly anything's left.
    no_emoji = _EMOJI_CHARS.sub("", text)
    no_hashtags = re.sub(r"#\S+", "", no_emoji).strip()
    if len(no_hashtags.split()) <= 2 and no_hashtags != text.strip():
        return True
    return False


def _is_social_profile_post(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _PROFILE_PHRASES):
        return True
    if not _SOCIAL_URL.search(lowered):
        return False
    # A LinkedIn/GitHub link with barely any text around it is a profile drop.
    remainder = _URL_ANY.sub("", text).strip()
    return len(remainder.split()) <= 6


def _is_pure_reaction(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False
    return len(words) <= 10 and all(w in _REACTION_WORDS for w in words)


def _is_informational(text: str) -> bool:
    """Stricter pass: keep questions and substantive answers, drop
    self-intros/greetings, social-profile posts, pure reactions, and
    very short non-question chatter.
    """
    if _is_question(text):
        return True
    if _is_greeting_or_social(text):
        return False
    if _is_social_profile_post(text):
        return False
    if _is_pure_reaction(text):
        return False
    if len(text.split()) < _MIN_WORDS:
        return False
    return True


def parse_messages(path: str) -> list[dict]:
    """Parse the raw export into a list of {sender, date, time, text} dicts,
    with system lines and skippable (empty/URL-only/emoji-only) messages
    already filtered out.
    """
    raw = []  # (date, time, sender, [text_lines]) before content filtering
    current = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _MESSAGE_START.match(line)
            if m:
                rest = m.group("rest")
                if _is_system_line(rest):
                    current = None
                    continue
                if ": " not in rest:
                    # Malformed line, not a real sender message — skip.
                    current = None
                    continue
                sender, text = rest.split(": ", 1)
                current = [m.group("date"), m.group("time"), sender, [text]]
                raw.append(current)
            else:
                # Continuation line: only belongs to a message if one is
                # currently open (drops stray pre-message lines, e.g. the
                # leading encryption banner with no timestamp at all).
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


def build_chunks(messages: list[dict]) -> list[dict]:
    """Group consecutive messages into ~10-message (or ~600-word) chunks,
    preserving order. Each chunk carries the date of its first message.
    """
    chunks = []
    current = []
    current_words = 0

    def flush():
        if not current:
            return
        text = "\n".join(f"{m['sender']} ({m['time']}): {m['text']}" for m in current)
        chunks.append({"text": text, "timestamp": current[0]["date"]})

    for msg in messages:
        word_count = len(msg["text"].split())
        would_exceed = current and (
            len(current) >= MESSAGES_PER_CHUNK or current_words + word_count > WORDS_PER_CHUNK
        )
        if would_exceed:
            flush()
            current = []
            current_words = 0
        current.append(msg)
        current_words += word_count

    flush()
    return chunks


def main():
    messages = parse_messages(CHAT_PATH)
    print(f"Kept {len(messages)} messages after filtering.")

    chunks = build_chunks(messages)
    print(f"Grouped into {len(chunks)} chunks.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION)

    collection.add(
        ids=[f"whatsapp_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"session": SESSION_LABEL, "timestamp": c["timestamp"]} for c in chunks],
    )

    print(f"Added {len(chunks)} chunks to '{COLLECTION}'.")
    print(f"Collection total is now {collection.count()} chunks.")


if __name__ == "__main__":
    main()
