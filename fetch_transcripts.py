from youtube_transcript_api import YouTubeTranscriptApi

videos = {
    "welcome_module0": "yVji4ZQECVw",
    "module1_class": "6q4uPBO_sDc",
    "problem_statement_qa": "-6G7LXiu47o",
}

api = YouTubeTranscriptApi()

for name, vid in videos.items():
    try:
        fetched = api.fetch(vid)
    except Exception as e:
        print(f"FAILED {name} ({vid}): {e}")
        continue

    # timestamped version, for "watch from 15:27" answers
    with open(f"{name}.txt", "w", encoding="utf-8") as f:
        for snip in fetched:
            m, s = divmod(int(snip.start), 60)
            f.write(f"[{m:02d}:{s:02d}] {snip.text}\n")

    print(f"OK {name}: {len(fetched.snippets)} segments -> {name}.txt")
