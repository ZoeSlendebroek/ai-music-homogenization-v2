#!/usr/bin/env python3
"""
generate_udio.py

Automated Udio generation for Experiment 2.

Setup:
    1. Go to udio.com and log in
    2. Open DevTools → Network tab → generate a song
    3. Find the POST request to udio.com/api/generate-proxy
    4. Copy the Authorization header value (JWT only, no "Bearer" prefix)
    5. Add to .env: UDIO_BEARER=your_token_here

Usage:
    python src/generate_udio.py --genre metal
    python src/generate_udio.py --genre metal --resume
    python src/generate_udio.py --genre metal --test
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("pip install pydub")

ROOT       = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "data" / "prompts"
OUT_DIR    = ROOT / "data" / "audio" / "udio"

BASE_URL       = "https://www.udio.com/api"
CROP_MS        = 30_000
SLEEP_BETWEEN  = 10


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()


def _make_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
        "origin": "https://www.udio.com",
        "referer": "https://www.udio.com/",
    }


def generate_udio_tracks(prompt: str, token: str) -> list:
    headers = _make_headers(token)

    # Step 1: submit generation job
    r = requests.post(
        f"{BASE_URL}/generate-proxy",
        headers=headers,
        json={
            "prompt": prompt,
            "samplerOptions": {"seed": -1},
        },
        timeout=60,
        impersonate="chrome",
    )
    if not r.ok:
        print(f"\n    [API error {r.status_code}] {r.text[:400]}")
        r.raise_for_status()

    data = r.json()
    track_ids = data.get("track_ids", [])
    if not track_ids:
        raise ValueError(f"No track_ids in response: {data}")

    # Step 2: poll until complete
    while True:
        r = requests.get(
            f"{BASE_URL}/songs",
            headers=headers,
            params={"songIds": ",".join(track_ids)},
            timeout=60,
            impersonate="chrome",
        )
        r.raise_for_status()
        songs = r.json().get("songs", [])

        if all(s.get("finished") for s in songs):
            return songs

        time.sleep(5)


def download_audio(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=60, impersonate="chrome")
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    [download error] {e}")
        return False


def middle_crop_mp3(in_path: Path, out_path: Path):
    audio = AudioSegment.from_file(str(in_path))
    dur   = len(audio)
    if dur <= CROP_MS:
        audio = audio + AudioSegment.silent(duration=CROP_MS - dur)
        audio.export(str(out_path), format="mp3", bitrate="192k")
    else:
        mid   = dur // 2
        start = max(0, mid - CROP_MS // 2)
        audio[start: start + CROP_MS].export(str(out_path), format="mp3", bitrate="192k")


def _save_token(token: str):
    env_path = ROOT / ".env"
    env_text = re.sub(r"^UDIO_BEARER=.*$", f"UDIO_BEARER={token}",
                      env_path.read_text(), flags=re.MULTILINE)
    env_path.write_text(env_text)


def generate_genre(genre_key: str, genre_display: str, resume: bool, test: bool):
    token = os.environ.get("UDIO_BEARER")
    if not token:
        sys.exit("Set UDIO_BEARER in your .env file.")

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
    print(f"  Udio generation: {genre_display}  ({len(df)} tracks)")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}")

    failed = []

    for i, row in df.iterrows():
        spotify_id  = str(row["spotify_id"])
        prompt_text = str(row["prompt_udio"])
        out_path    = out_dir / f"{spotify_id}_udio.mp3"

        if resume and out_path.exists():
            print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42} [cached]")
            continue

        print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42}", end=" ... ", flush=True)

        while True:
            try:
                songs = generate_udio_tracks(prompt_text, token)

                if not songs:
                    print("[FAIL — no songs returned]")
                    failed.append(spotify_id)
                    break

                song_url = songs[0].get("song_path")
                if not song_url:
                    print("[FAIL — no audio URL]")
                    failed.append(spotify_id)
                    break

                tmp_path = tmp_dir / f"{spotify_id}_raw.mp3"
                if not download_audio(song_url, tmp_path):
                    print("[FAIL — download error]")
                    failed.append(spotify_id)
                    break

                middle_crop_mp3(tmp_path, out_path)
                tmp_path.unlink(missing_ok=True)
                print("[ok]")
                break

            except Exception as e:
                if "422" in str(e) or "401" in str(e):
                    print("[TOKEN EXPIRED]")
                    print("\n  Paste fresh Udio Bearer token (Network tab, no 'Bearer' prefix), then Enter:")
                    new_token = input("  > ").strip()
                    if not new_token:
                        failed.append(spotify_id)
                        break
                    token = new_token
                    os.environ["UDIO_BEARER"] = new_token
                    _save_token(new_token)
                    print(f"  [{i+1:03d}/{len(df)}] {str(row['title'])[:40]:<42}", end=" ... ", flush=True)
                else:
                    print(f"[ERROR] {e}")
                    failed.append(spotify_id)
                    break

        time.sleep(SLEEP_BETWEEN)

    n_done = len(df) - len(failed)
    print(f"\n{'='*55}")
    print(f"  Done: {n_done}/{len(df)} generated successfully")
    if failed:
        print(f"  Failed ({len(failed)}): {failed}")
        print(f"  Re-run with --resume to retry failed tracks")
    print(f"  Files saved to: {out_dir}")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass


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
