#!/usr/bin/env python3
"""
0_collect_human_corpus.py

Builds the human reference corpus for each genre.

Pipeline per genre:
  1. Pull tracks from Spotify playlists (metadata only)
  2. Try Spotify 30s preview URL first
  3. Fall back to Deezer /search for any track missing a Spotify preview
  4. Download audio, middle-crop to 30s, save as MP3
  5. Write manifest CSV: track metadata + audio path + source

Outputs:
    data/human_corpus/{genre}/audio/*.mp3
    data/human_corpus/{genre}/manifest.csv   (before stratified sampling)

Usage:
    python src/0_collect_human_corpus.py --genre kpop
    python src/0_collect_human_corpus.py --genre all
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd

# ── auth ──────────────────────────────────────────────────────────
def _load_env():
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:
    sys.exit("pip install spotipy")

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("pip install pydub  (also requires ffmpeg on PATH)")

# ── config ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres import GENRES, CANDIDATE_POOL_SIZE

ROOT       = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "human_corpus"

DEEZER_SEARCH = "https://api.deezer.com/search"
HEADERS       = {"User-Agent": "MusicHomogenizationResearch/2.0"}
SLEEP_DEEZER  = 0.4   # polite rate limit


# ── Spotify client ────────────────────────────────────────────────
def get_spotify():
    cid    = os.environ.get("SPOTIFY_CLIENT_ID")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=cid, client_secret=secret))


# ── Spotify playlist scraper ──────────────────────────────────────
def collect_spotify_tracks(sp, playlist_ids: list, year_range: tuple,
                            limit: int) -> list[dict]:
    """Pull up to `limit` unique tracks from a list of playlist IDs."""
    seen   = set()
    tracks = []

    for pid in playlist_ids:
        offset = 0
        while len(tracks) < limit:
            try:
                res   = sp.playlist_tracks(pid, offset=offset, limit=100,
                                           fields="items(track(id,name,artists,album,"
                                                  "preview_url,duration_ms)),next")
                items = res.get("items", [])
            except Exception as e:
                print(f"  [WARN] Playlist {pid} offset {offset}: {e}")
                break

            if not items:
                break

            for item in items:
                t = item.get("track")
                if not t or not t.get("id"):
                    continue
                if t["id"] in seen:
                    continue

                # year filter via album release
                rel = t.get("album", {}).get("release_date", "")
                try:
                    year = int(rel[:4])
                except (ValueError, TypeError):
                    year = 0
                if year and not (year_range[0] <= year <= year_range[1]):
                    continue

                seen.add(t["id"])
                tracks.append({
                    "spotify_id":   t["id"],
                    "title":        t["name"],
                    "artist":       t["artists"][0]["name"] if t["artists"] else "",
                    "year":         year,
                    "duration_ms":  t.get("duration_ms", 0),
                    "preview_url":  t.get("preview_url") or "",
                    "source":       "spotify",
                })

            if len(tracks) >= limit:
                break
            offset += 100

    return tracks[:limit]


# ── Deezer fallback ───────────────────────────────────────────────
def deezer_preview(artist: str, title: str) -> Optional[str]:
    """Return a 30s Deezer preview URL, or None if not found."""
    try:
        r = requests.get(DEEZER_SEARCH,
                         params={"q": f"{artist} {title}", "limit": 5},
                         headers=HEADERS, timeout=10)
        r.raise_for_status()
        for hit in r.json().get("data", []):
            url = hit.get("preview")
            if url:
                return url
    except Exception as e:
        print(f"    [WARN] Deezer lookup failed for '{artist} – {title}': {e}")
    time.sleep(SLEEP_DEEZER)
    return None


# ── audio download + 30s middle crop ─────────────────────────────
def download_and_crop(url: str, out_path: Path) -> bool:
    """Download audio from URL, middle-crop to 30s, save as MP3."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        r.raise_for_status()

        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        audio = AudioSegment.from_file(str(tmp))
        dur   = len(audio)             # milliseconds
        target = 30_000                # 30s in ms

        if dur >= target:
            mid   = dur // 2
            start = max(0, mid - target // 2)
            crop  = audio[start: start + target]
        else:
            # pad with silence if shorter than 30s (rare)
            crop = audio + AudioSegment.silent(duration=target - dur)

        crop.export(str(out_path), format="mp3", bitrate="192k")
        tmp.unlink(missing_ok=True)
        return True

    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")
        if out_path.with_suffix(".tmp").exists():
            out_path.with_suffix(".tmp").unlink()
        return False


# ── deduplication: max 4 songs per artist, no duplicate titles ────
def deduplicate_by_artist(tracks: list[dict]) -> list[dict]:
    artist_counts: dict[str, int] = {}
    seen_songs: set[tuple[str, str]] = set()
    out = []
    for t in tracks:
        a = t["artist"].lower().strip()
        title = t["title"].lower().strip()
        key = (a, title)
        if key in seen_songs:
            continue
        if artist_counts.get(a, 0) >= 5:
            continue
        seen_songs.add(key)
        artist_counts[a] = artist_counts.get(a, 0) + 1
        out.append(t)
    return out


# ── main per-genre routine ────────────────────────────────────────
def collect_genre(genre_key: str):
    cfg      = GENRES[genre_key]
    out_dir  = CORPUS_DIR / genre_key / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = CORPUS_DIR / genre_key / "manifest.csv"

    print(f"\n{'='*60}")
    print(f"  {cfg['display']}  (target pool: {CANDIDATE_POOL_SIZE})")
    print(f"{'='*60}")

    sp     = get_spotify()
    tracks = collect_spotify_tracks(sp, cfg["spotify_playlists"],
                                    cfg["year_range"], CANDIDATE_POOL_SIZE)
    tracks = deduplicate_by_artist(tracks)
    print(f"  Collected {len(tracks)} unique tracks from Spotify playlists")

    rows = []
    for i, t in enumerate(tracks):
        safe_title  = "".join(c for c in t["title"]  if c.isalnum() or c in " -_")[:40]
        safe_artist = "".join(c for c in t["artist"] if c.isalnum() or c in " -_")[:30]
        fname       = f"{genre_key}_{i:04d}_{safe_artist}_{safe_title}.mp3"
        fname       = fname.replace(" ", "_")
        out_path    = out_dir / fname

        print(f"  [{i+1:03d}/{len(tracks)}] {t['artist']} – {t['title']}", end="")

        if out_path.exists():
            print("  [cached]")
            t["audio_path"] = str(out_path)
            t["downloaded"] = True
            rows.append(t)
            continue

        # try Spotify preview first
        preview_url = t["preview_url"]
        source      = "spotify"

        # fall back to Deezer if no Spotify preview
        if not preview_url:
            print("  [no Spotify preview → Deezer]", end="")
            preview_url = deezer_preview(t["artist"], t["title"])
            source      = "deezer"

        if not preview_url:
            print("  [SKIP — no preview found]")
            t["audio_path"] = ""
            t["downloaded"] = False
            rows.append(t)
            continue

        ok = download_and_crop(preview_url, out_path)
        if ok:
            print(f"  [ok · {source}]")
        else:
            print(f"  [FAIL]")

        t["audio_path"] = str(out_path) if ok else ""
        t["downloaded"] = ok
        t["source"]     = source
        rows.append(t)

    df = pd.DataFrame(rows)
    df.to_csv(manifest, index=False)

    n_ok   = df["downloaded"].sum()
    n_fail = len(df) - n_ok
    print(f"\n  Done: {n_ok} downloaded, {n_fail} failed → {manifest}")
    if n_ok < 80:
        print(f"  [WARN] Only {n_ok} tracks for {cfg['display']}."
              f" Consider additional playlists. Statistical power may be reduced.")
    return df


# ── CLI ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="all",
                        help="Genre key (afrobeats/kpop/dancepop/metal) or 'all'")
    args = parser.parse_args()

    targets = list(GENRES.keys()) if args.genre == "all" else [args.genre]
    for g in targets:
        if g not in GENRES:
            sys.exit(f"Unknown genre '{g}'. Options: {list(GENRES.keys())}")
        collect_genre(g)


if __name__ == "__main__":
    main()
