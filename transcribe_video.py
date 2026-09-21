"""Transcribe a local video to the same [MM:SS] text transcript format
used by the existing .txt files, so it can be indexed by build_index.py.

Usage:
    python3 transcribe_video.py
"""

import time
from faster_whisper import WhisperModel

VIDEO_PATH = "MIT Universal AI Welcome and onboarding Call-20260916_140218-Meeting Recording.mp4"
OUTPUT_PATH = "mit_onboarding.txt"

model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe(VIDEO_PATH, beam_size=5)

print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")
print(f"Audio duration: {info.duration / 60:.1f} minutes")
print("Transcribing... this will take a while on CPU.")

start_time = time.time()
count = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for segment in segments:
        mm, ss = divmod(int(segment.start), 60)
        text = segment.text.strip()
        f.write(f"[{mm:02d}:{ss:02d}] {text}\n")
        f.flush()

        count += 1
        if count % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  {count} segments written, {elapsed / 60:.1f} min elapsed, "
                  f"at [{mm:02d}:{ss:02d}] in the audio")

elapsed = time.time() - start_time
print(f"Done. {count} segments written to {OUTPUT_PATH} in {elapsed / 60:.1f} minutes.")
