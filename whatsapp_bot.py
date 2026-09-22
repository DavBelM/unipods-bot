"""WhatsApp group listener for the UniPods AI bot.

Uses neonize (unofficial WhatsApp Web automation, not the official Business
API) to watch a group for messages that mention the bot, answer them with
the same retrieve() + generate_answer() pipeline the web chat uses, and
reply in-thread. Every answered question is also logged to Supabase, same
as the web UI, so it shows up on /dashboard.html too.

This is meant to run LOCALLY during a scheduled test window in the group,
not as a permanently deployed service — the underlying WhatsApp session is
a persistent, stateful connection that doesn't fit a serverless backend,
and running it only during your slot limits exposure on the number used.

Setup (one-time):
  1. In .env, set WHATSAPP_PAIR_PHONE to the bot's number, digits only,
     country code first, no "+" — e.g. 15551234567 for a US number.
     Use a secondary/burner number, not your personal one.
  2. Optionally set WHATSAPP_BOT_TRIGGER (default: "@unipods") — the text
     members type to address the bot, e.g. "@unipods what's the deadline?"

Run:
    source unipods/bin/activate
    python3 whatsapp_bot.py

First run prints a pairing code. On the BOT'S phone (the number you set
above): WhatsApp -> Settings -> Linked Devices -> Link a Device -> "Link
with phone number instead" -> enter the code. After that first pairing,
the session is saved locally and reconnects automatically on future runs.
"""

import os
import re

from dotenv import load_dotenv
from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv
from neonize.utils import extract_text
from supabase import create_client

from answer import generate_answer
from retrieval import retrieve

load_dotenv()

BOT_TRIGGER = os.environ.get("WHATSAPP_BOT_TRIGGER", "@unipods")
WEB_APP_URL = os.environ.get("WEB_APP_URL", "https://unipods-bot.vercel.app")

client = NewClient("unipods-bot")
_supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _log_query(question: str, answer: str, relevant: bool, sources: list[dict]) -> None:
    try:
        _supabase.table("queries").insert({
            "question": question,
            "answer": answer,
            "relevant": relevant,
            "sources": sources,
        }).execute()
    except Exception as e:
        print(f"Query logging failed: {e}")


@client.event(ConnectedEv)
def on_connected(client: NewClient, _):
    print("Connected to WhatsApp. Listening for messages mentioning "
          f"'{BOT_TRIGGER}'...")


@client.event(MessageEv)
def on_message(client: NewClient, message: MessageEv):
    source = message.Info.MessageSource
    if source.IsFromMe:
        return  # never reply to ourselves

    text = extract_text(message.Message)
    if not text or BOT_TRIGGER.lower() not in text.lower():
        return  # only respond when explicitly addressed

    question = re.sub(re.escape(BOT_TRIGGER), "", text, flags=re.IGNORECASE).strip()
    if not question:
        return

    who = message.Info.Pushname or "someone"
    print(f"Q from {who}: {question}")

    result = retrieve(question)
    answer = generate_answer(question, result)
    sources = (
        [{"session": c["session"], "timestamp": c["timestamp"]} for c in result["chunks"]]
        if result["relevant"]
        else []
    )

    # Point people to the full app for more depth / to ask follow-ups there.
    reply_text = f"{answer}\n\n💬 Full chat + sources: {WEB_APP_URL}"

    client.reply_message(reply_text, message, to=source.Chat)
    _log_query(question, answer, result["relevant"], sources)
    print(f"Replied: {answer[:80]}...")


def main():
    phone = os.environ["WHATSAPP_PAIR_PHONE"]
    if not client.is_logged_in:
        code = client.PairPhone(phone, show_push_notification=True)
        print(f"\nPairing code: {code}")
        print("On the bot's WhatsApp: Settings -> Linked Devices -> Link a "
              "Device -> 'Link with phone number instead' -> enter this code.\n")
    client.connect()


if __name__ == "__main__":
    main()
