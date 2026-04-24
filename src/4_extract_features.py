#!/usr/bin/env python3
"""
4_extract_features.py

Extract 67 MIR features from all AI-generated and human tracks.
Matches v1 pipeline exactly (same FeatureExtractor, same 30s middle crop).

Inputs:
    data/audio/{system}/{genre}/*.{mp3,wav,m4a,flac}
    data/human_corpus/{genre}/corpus_final.csv

Outputs:
    data/features/{genre}_ai_features.csv      (AI tracks)
    data/features/{genre}_human_features.csv   (human tracks, re-extracted for consistency)
    data/features/all_features.csv             (combined, with condition/system/genre cols)

Usage:
    python src/4_extract_features.py --genre kpop
    python src/4_extract_features.py --genre all
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres    import GENRES, SYSTEMS
from extract_features_shared import FeatureExtractor, FEATURE_COLS

ROOT        = Path(__file__).resolve().parent.parent
AUDIO_DIR   = ROOT / "data" / "audio"
CORPUS_DIR  = ROOT / "data" / "human_corpus"
FEATURE_DIR = ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

# Suno Afrobeats version note — flagged as covariate
SUNO_AFROBEATS_VERSION_NOTE = (
    "Suno Afrobeats data collected on v3 (Dec 2025 / Jan 2026). "
    "All other data collected on current version. "
    "Version treated as a covariate in cross-genre analysis."
)


def extract_ai_tracks(genre_key: str, extractor: FeatureExtractor) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        folder = AUDIO_DIR / system / genre_key
        if not folder.exists():
            print(f"  [SKIP] {folder} not found")
            continue

        files = sorted([f for f in folder.iterdir() if f.suffix.lower() in EXTS])
        print(f"  {system:<8} {len(files):3d} files", end="")

        for f in files:
            try:
                feats          = extractor.extract(f)
                feats["filename"] = f.name
                feats["system"]   = system
                feats["genre"]    = genre_key
                feats["condition"] = "ai"
                # version flag
                if system == "suno" and genre_key == "afrobeats":
                    feats["version_note"] = "suno_v3_legacy"
                else:
                    feats["version_note"] = "current"
                rows.append(feats)
            except Exception as e:
                print(f"\n    [ERROR] {f.name}: {e}", end="")

        print()
    return rows


def extract_human_tracks(genre_key: str, extractor: FeatureExtractor) -> list[dict]:
    corpus = CORPUS_DIR / genre_key / "corpus_final.csv"
    if not corpus.exists():
        print(f"  [SKIP] No corpus_final.csv for {genre_key}")
        return []

    df   = pd.read_csv(corpus)
    rows = []
    print(f"  human   {len(df):3d} tracks", end="")

    for _, row in df.iterrows():
        p = Path(row["audio_path"])
        if not p.exists():
            continue
        try:
            feats               = extractor.extract(p)
            feats["filename"]   = p.name
            feats["system"]     = "human"
            feats["genre"]      = genre_key
            feats["condition"]  = "human"
            feats["spotify_id"] = row.get("spotify_id", "")
            feats["artist"]     = row.get("artist", "")
            feats["title"]      = row.get("title", "")
            feats["version_note"] = "n/a"
            rows.append(feats)
        except Exception as e:
            print(f"\n    [ERROR] {p.name}: {e}", end="")

    print()
    return rows


def extract_genre(genre_key: str):
    cfg       = GENRES[genre_key]
    extractor = FeatureExtractor()

    print(f"\n{'='*55}")
    print(f"  {cfg['display']}")
    print(f"{'='*55}")

    ai_rows    = extract_ai_tracks(genre_key, extractor)
    human_rows = extract_human_tracks(genre_key, extractor)

    ai_df    = pd.DataFrame(ai_rows)
    human_df = pd.DataFrame(human_rows)

    meta = ["filename", "system", "genre", "condition", "version_note"]
    def reorder(df):
        cols = [c for c in meta if c in df.columns]
        rest = [c for c in df.columns if c not in meta]
        return df[cols + rest]

    if not ai_df.empty:
        ai_df = reorder(ai_df)
        ai_df.to_csv(FEATURE_DIR / f"{genre_key}_ai_features.csv", index=False)
        print(f"  AI features:    {len(ai_df)} rows → {genre_key}_ai_features.csv")

    if not human_df.empty:
        human_df = reorder(human_df)
        human_df.to_csv(FEATURE_DIR / f"{genre_key}_human_features.csv", index=False)
        print(f"  Human features: {len(human_df)} rows → {genre_key}_human_features.csv")

    return ai_df, human_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="all")
    args = parser.parse_args()

    targets = list(GENRES.keys()) if args.genre == "all" else [args.genre]

    all_dfs = []
    for g in targets:
        if g not in GENRES:
            sys.exit(f"Unknown genre '{g}'")
        ai_df, human_df = extract_genre(g)
        all_dfs.extend([ai_df, human_df])

    combined = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
    out      = FEATURE_DIR / "all_features.csv"
    combined.to_csv(out, index=False)
    print(f"\nCombined: {len(combined)} rows → {out}")
    print(combined.groupby(["genre","condition","system"]).size().to_string())

    if "suno" in combined.get("system", pd.Series()).values:
        print(f"\nVersion note: {SUNO_AFROBEATS_VERSION_NOTE}")


if __name__ == "__main__":
    main()
