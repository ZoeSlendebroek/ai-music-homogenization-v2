#!/usr/bin/env python3
"""
generate_lyria.py

Automated Lyria 3 Clip generation for Experiment 2.

Setup:
    1. Add GEMINI_API_KEY to .env (from aistudio.google.com)
    2. Enable billing on your Google account
    3. pip install google-genai pydub

Usage:
    python src/generate_lyria.py --genre metal
    python src/generate_lyria.py --genre metal --resume
    python src/generate_lyria.py --genre metal --test
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from google import genai
except ImportError:
    sys.exit("pip install google-genai")

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("pip install pydub")

ROOT       = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "data" / "prompts"
OUT_DIR    = ROOT / "data" / "audio" / "lyria"

CROP_MS       = 30_000
SLEEP_BETWEEN = 5


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()


def generate_lyria_clip(prompt: str, client) -> tuple:
    response = client.models.generate_content(
        model="lyria-3-clip-preview",
        contents=prompt,
    )
    candidates = getattr(response, "candidates", None)
    if not candidates or candidates[0].content is None:
        reason = getattr(candidates[0], "finish_reason", "unknown") if candidates else "no candidates"
        raise ValueError(f"Blocked/empty response (finish_reason={reason})")
    for part in candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            data = part.inline_data.data
            mime = part.inline_data.mime_type
            if isinstance(data, str):
                data = base64.b64decode(data)
            return data, mime
    raise ValueError(f"No audio in response: {response}")


def save_as_30s_mp3(audio_bytes: bytes, mime: str, out_path: Path):
    fmt = "wav" if "wav" in mime else ("ogg" if "ogg" in mime else "mp3")
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    dur = len(audio)
    if dur <= CROP_MS:
        audio = audio + AudioSegment.silent(duration=CROP_MS - dur)
        audio.export(str(out_path), format="mp3", bitrate="192k")
    else:
        mid   = dur // 2
        start = max(0, mid - CROP_MS // 2)
        audio[start: start + CROP_MS].export(str(out_path), format="mp3", bitrate="192k")


def generate_genre(genre_key: str, genre_display: str, resume: bool, test: bool):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in your .env file.")

    prompt_csv = PROMPT_DIR / f"{genre_key}_exp2_prompts.csv"
    if not prompt_csv.exists():
        sys.exit(f"No prompts found for '{genre_key}'. Run 2_generate_prompts.py first.")

    df = pd.read_csv(prompt_csv)
    if test:
        df = df.head(3)
        print(f"  [TEST MODE] Running on first 3 tracks only")

    out_dir = OUT_DIR / genre_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Lyria generation: {genre_display}  ({len(df)} tracks)")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}")

    client = genai.Client(api_key=api_key)
    failed = []

    for i, row in df.iterrows():
        spotify_id  = str(row["spotify_id"])
        prompt_text = str(row["prompt_lyria"])
        out_path    = out_dir / f"{spotify_id}_lyria.mp3"

        if resume and out_path.exists():
            print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42} [cached]")
            continue

        print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42}", end=" ... ", flush=True)

        success = False
        for attempt in range(3):
            try:
                audio_bytes, mime = generate_lyria_clip(prompt_text, client)
                save_as_30s_mp3(audio_bytes, mime, out_path)
                print("[ok]")
                success = True
                break
            except Exception as e:
                if attempt < 2:
                    print(f"[retry {attempt+1}]", end=" ... ", flush=True)
                    time.sleep(5)
                else:
                    print(f"[ERROR] {e}")
                    failed.append(spotify_id)

        time.sleep(SLEEP_BETWEEN)

    n_done = len(df) - len(failed)
    print(f"\n{'='*55}")
    print(f"  Done: {n_done}/{len(df)} generated successfully")
    if failed:
        print(f"  Failed ({len(failed)}): {failed}")
        print(f"  Re-run with --resume to retry failed tracks")
    print(f"  Files saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", required=True,
                        help="Genre key: metal / afrobeats / dancepop / kpop")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated files")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: generate first 3 tracks only")
    args = parser.parse_args()

    genre_map = {
        "metal":     "Heavy Metal",
        "afrobeats": "Afrobeats",
        "dancepop":  "Dance Pop",
        "kpop":      "K-pop",
    }

    if args.genre not in genre_map:
        sys.exit(f"Unknown genre '{args.genre}'. Options: {list(genre_map.keys())}")

    generate_genre(args.genre, genre_map[args.genre], args.resume, args.test)


if __name__ == "__main__":
    main()
