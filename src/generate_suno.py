#!/usr/bin/env python3
"""
generate_suno.py

Automated Suno generation for Experiment 2 using the SunoAI
unofficial Python library (cookie-based auth).

Setup:
    1. pip install SunoAI pydub
    2. Get your Suno cookie (instructions below) and add to .env:
       SUNO_COOKIE=your_cookie_here

How to get your Suno cookie:
    1. Go to suno.com and log in
    2. Open DevTools (F12) → Network tab
    3. Refresh the page
    4. Find any request to suno.com, click it → Headers tab
    5. Copy the full value of the 'Cookie' header
    6. Paste into .env as SUNO_COOKIE=...

Usage:
    python src/generate_suno.py --genre metal
    python src/generate_suno.py --genre metal --resume   # skip existing files
    python src/generate_suno.py --genre metal --test     # test with 3 tracks only
"""

import argparse
import os
import sys
import time
from pathlib import Path


import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = "https://studio-api-prod.suno.com/api"


def _make_headers(token: str) -> dict:
    import base64, json as _json
    browser_token = base64.b64encode(
        _json.dumps({"timestamp": int(time.time() * 1000)}).encode()
    ).decode()
    device_id = os.environ.get("SUNO_DEVICE_ID", "5f293098-ae8a-4653-aa3a-1a21d4691acc")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "browser-token": browser_token,
        "device-id": device_id,
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
        "origin": "https://suno.com",
        "referer": "https://suno.com/",
    }


def generate_suno_tracks(prompt: str, token: str):
    headers = _make_headers(token)

    # Step 1: create generation job
    r = requests.post(
        f"{BASE_URL}/generate/v2/",
        headers=headers,
        json={
            "gpt_description_prompt": prompt,
            "make_instrumental": True,
            "mv": "chirp-v3-5",
            "prompt": "",
        },
        timeout=60,
    )
    if not r.ok:
        print(f"\n    [API error {r.status_code}] {r.text[:400]}")
        if r.status_code == 422:
            sys.exit("\n  [TOKEN EXPIRED] Get a fresh Bearer token from the Suno Network tab, update .env, and re-run with --resume.")
        r.raise_for_status()
    data = r.json()
    print(f"\n    [generate response] {str(data)[:300]}")

    ids = [clip["id"] for clip in data["clips"]]

    # Step 2: poll until ready via POST /feed/v3
    while True:
        r = requests.post(
            f"{BASE_URL}/feed/v3",
            headers=_make_headers(token),
            json={"ids": ids},
            timeout=60,
        )
        r.raise_for_status()
        clips = r.json()["clips"]
        print(f"\n    [feed response] {str(clips)[:300]}")

        if all(c["status"] == "complete" for c in clips):
            return clips

        time.sleep(5)
try:
    from pydub import AudioSegment
    import requests
except ImportError:
    sys.exit("pip install pydub requests")

ROOT       = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "data" / "prompts"
OUT_DIR    = ROOT / "data" / "audio" / "suno"

# Middle-crop target
CROP_MS = 30_000   # 30 seconds in milliseconds

# Rate limiting — Suno generates 2 tracks per call, be polite
SLEEP_BETWEEN = 8   # seconds between generation calls


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()


def middle_crop_mp3(in_path: Path, out_path: Path):
    """Load audio, take middle 30s, save as MP3."""
    audio = AudioSegment.from_file(str(in_path))
    dur   = len(audio)
    if dur <= CROP_MS:
        # shorter than 30s — pad with silence
        audio = audio + AudioSegment.silent(duration=CROP_MS - dur)
        audio.export(str(out_path), format="mp3", bitrate="192k")
    else:
        mid   = dur // 2
        start = max(0, mid - CROP_MS // 2)
        crop  = audio[start: start + CROP_MS]
        crop.export(str(out_path), format="mp3", bitrate="192k")


def download_audio(url: str, dest: Path) -> bool:
    """Download audio file from URL."""
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [download error] {e}")
        return False


def generate_genre(genre_key: str, genre_display: str,
                   resume: bool, test: bool):

    token = os.environ.get("SUNO_BEARER")
    if not token:
        sys.exit("Set SUNO_BEARER in your .env file.")
    
    prompt_csv = PROMPT_DIR / f"{genre_key}_exp2_prompts.csv"
        
    
    if not prompt_csv.exists():
        sys.exit(f"No prompts found for '{genre_key}'. Run 2_generate_prompts.py first.")

    df = pd.read_csv(prompt_csv)
    if test:
        df = df.head(3)
        print(f"  [TEST MODE] Running on first 3 tracks only")

    out_dir = OUT_DIR / genre_key
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Suno generation: {genre_display}  ({len(df)} tracks)")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}")

    # Check credits before starting

    print("  Using bearer auth (direct API mode)")   
    failed = []

    for i, row in df.iterrows():
        spotify_id  = str(row["spotify_id"])
        prompt_text = str(row["prompt_suno"])
        out_path    = out_dir / f"{spotify_id}_suno.mp3"

        if resume and out_path.exists():
            print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42} [cached]")
            continue

        print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42}", end=" ... ")

        try:
            # Generate — returns 2 clips by default, we take the first
            clips = generate_suno_tracks(prompt_text, token)

            if not clips:
                print("[FAIL — no clips returned]")
                failed.append(spotify_id)
                continue

            audio_url = clips[0].get("audio_url")

            if not audio_url:
                print("[FAIL — no audio URL]")
                failed.append(spotify_id)
                continue

            tmp_path = tmp_dir / f"{spotify_id}_raw.mp3"
            ok = download_audio(audio_url, tmp_path)
            if not ok:
                print("[FAIL — download error]")
                failed.append(spotify_id)
                continue

            # Middle-crop to 30s
            middle_crop_mp3(tmp_path, out_path)
            tmp_path.unlink(missing_ok=True)

            print(f"[ok]")

        except Exception as e:
            print(f"[ERROR] {e}")
            failed.append(spotify_id)

        # Rate limit pause
        time.sleep(SLEEP_BETWEEN)

    # Summary
    n_done   = len(df) - len(failed)
    print(f"\n{'='*55}")
    print(f"  Done: {n_done}/{len(df)} generated successfully")
    if failed:
        print(f"  Failed ({len(failed)}): {failed}")
        print(f"  Re-run with --resume to retry failed tracks")
    print(f"  Files saved to: {out_dir}")

    # Clean up tmp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", required=True,
                        help="Genre key: metal / afrobeats / dancepop")
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
