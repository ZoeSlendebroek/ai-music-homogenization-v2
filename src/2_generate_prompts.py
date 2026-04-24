#!/usr/bin/env python3
"""
2_generate_prompts.py

For each track in corpus_final.csv, extract the 4 Exp 2 prompt features
and render prompt strings for Suno, Lyria 3, and Udio.

NOTE: IOI CV and RMS energy are deliberately excluded from prompts.
      They are free outcome variables measured post-generation.

Outputs:
    data/prompts/{genre}_exp2_prompts.csv
      columns: spotify_id, title, artist, audio_path,
               tempo, onset_density, harmonic_ratio, self_similarity_mean,
               [tempo_label, energy_label, rhythm_label, structure_label, hp_label]
               prompt_suno, prompt_lyria, prompt_udio, seed_lyria

Usage:
    python src/2_generate_prompts.py --genre kpop
    python src/2_generate_prompts.py --genre all
"""

import argparse
import hashlib
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres import GENRES
from extract_features_shared import FeatureExtractor, PROMPT_FEATURE_COLS

ROOT       = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "human_corpus"
PROMPT_DIR = ROOT / "data" / "prompts"
PROMPT_DIR.mkdir(parents=True, exist_ok=True)


# ── feature → natural language binning ───────────────────────────

def tempo_label(bpm: float) -> str:
    if bpm < 66:   return "Largo"
    if bpm < 76:   return "Adagio"
    if bpm < 108:  return "Andante"
    if bpm < 120:  return "Moderato"
    if bpm < 156:  return "Allegro"
    return "Presto"

def energy_label(onset_density: float) -> str:
    # onsets per second
    if onset_density < 3:   return "relaxed"
    if onset_density < 6:   return "moderate"
    if onset_density < 10:  return "driving"
    return "intense"

def hp_label(harmonic_ratio: float) -> str:
    # harmonic_ratio = harmonic energy / total energy
    if harmonic_ratio < 0.35: return "drum-forward, percussion-heavy"
    if harmonic_ratio < 0.60: return "balanced drums and melody"
    return "melodic, harmonic-forward"

def rhythm_label(onset_density: float, harmonic_ratio: float) -> str:
    if onset_density >= 8:    return "dense, fast-hitting"
    if onset_density >= 5:    return "syncopated, groove-forward"
    if harmonic_ratio < 0.4:  return "sparse, percussive"
    return "steady groove"

def structure_label(self_sim: float) -> str:
    if self_sim > 0.75: return "loop-based and hypnotic, highly repetitive"
    if self_sim > 0.55: return "moderately repetitive with variation"
    return "through-composed, structurally varied"


# ── genre-specific instrumentation hints ─────────────────────────
INSTRUMENTATION = {
    "afrobeats": "talking drum, shekere, bass guitar, Afropop synths",
    "kpop":      "synth pads, trap hi-hats, orchestral strings, electronic bass",
    "dancepop":  "four-on-the-floor kick, synth bass, layered pads, claps",
    "metal":     "distorted electric guitar, double-kick drums, bass guitar, no synths",
}


# ── prompt renderers ──────────────────────────────────────────────

def render_suno(genre_key: str, bpm: float, t_label: str, e_label: str,
                r_label: str, hp: str, struct: str) -> str:
    genre_display = GENRES[genre_key]["display"]
    instr         = INSTRUMENTATION.get(genre_key, "acoustic and electronic instrumentation")
    return (
        f"Instrumental {genre_display} at {bpm:.0f} BPM ({t_label}), "
        f"{e_label}, {r_label}. "
        f"{hp.capitalize()}. {struct.capitalize()}. "
        f"Instrumentation: {instr}. No vocals."
    )

def render_lyria(genre_key: str, bpm: float, t_label: str, e_label: str,
                 r_label: str, hp: str, struct: str, seed: int) -> str:
    genre_display = GENRES[genre_key]["display"]
    instr         = INSTRUMENTATION.get(genre_key, "acoustic and electronic instrumentation")
    prompt = (
        f"An instrumental {genre_display} track. "
        f"Mood: {e_label}. "
        f"Tempo: {bpm:.0f} BPM, {t_label}. "
        f"Rhythm: {r_label}, {hp}. "
        f"Instrumentation: {instr}. "
        f"Structure: {struct}."
    )
    negative = "vocals, lyrics, singing, spoken word"
    return f'prompt: "{prompt}" | negative_prompt: "{negative}" | seed: {seed}'

def render_udio(genre_key: str, bpm: float, t_label: str, e_label: str,
                r_label: str, hp: str, struct: str) -> str:
    # Udio follows similar structure to Suno; trim to middle 30s post-download
    genre_display = GENRES[genre_key]["display"]
    instr         = INSTRUMENTATION.get(genre_key, "acoustic and electronic instrumentation")
    return (
        f"Instrumental {genre_display}, {bpm:.0f} BPM ({t_label}), "
        f"{e_label} energy. {r_label.capitalize()}, {hp}. "
        f"{struct.capitalize()}. "
        f"Instruments: {instr}. No vocals, no lyrics."
    )


# ── deterministic seed for Lyria reproducibility ─────────────────
def make_seed(spotify_id: str) -> int:
    return int(hashlib.md5(spotify_id.encode()).hexdigest(), 16) % (2**31)


# ── main ──────────────────────────────────────────────────────────

def generate_prompts(genre_key: str):
    cfg     = GENRES[genre_key]
    corpus  = CORPUS_DIR / genre_key / "corpus_final.csv"
    if not corpus.exists():
        sys.exit(f"No corpus for '{genre_key}'. Run 1_stratified_sample.py first.")

    df = pd.read_csv(corpus)
    print(f"\n{cfg['display']}: generating Exp 2 prompts for {len(df)} tracks")

    extractor = FeatureExtractor()
    rows      = []

    for _, row in df.iterrows():
        p = Path(row["audio_path"])
        if not p.exists():
            print(f"  [SKIP] {p.name} not found")
            continue

        try:
            feats = extractor.extract(p)
        except Exception as e:
            print(f"  [WARN] Feature extraction failed for {p.name}: {e}")
            continue

        bpm     = feats["tempo"]
        od      = feats["onset_density"]
        hr      = feats["harmonic_ratio"]
        ss      = feats["self_similarity_mean"]

        t_lbl   = tempo_label(bpm)
        e_lbl   = energy_label(od)
        hp      = hp_label(hr)
        r_lbl   = rhythm_label(od, hr)
        s_lbl   = structure_label(ss)
        seed    = make_seed(str(row["spotify_id"]))

        rows.append({
            "spotify_id":          row["spotify_id"],
            "title":               row["title"],
            "artist":              row["artist"],
            "audio_path":          row["audio_path"],
            # raw features (for diversity check)
            "tempo":               bpm,
            "onset_density":       od,
            "harmonic_ratio":      hr,
            "self_similarity_mean": ss,
            # binned labels
            "tempo_label":         t_lbl,
            "energy_label":        e_lbl,
            "rhythm_label":        r_lbl,
            "hp_label":            hp,
            "structure_label":     s_lbl,
            "seed_lyria":          seed,
            # rendered prompts
            "prompt_suno":  render_suno(genre_key,  bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl),
            "prompt_lyria": render_lyria(genre_key, bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl, seed),
            "prompt_udio":  render_udio(genre_key,  bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl),
        })

    out_df  = pd.DataFrame(rows)
    out_csv = PROMPT_DIR / f"{genre_key}_exp2_prompts.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"  Saved {len(out_df)} prompts → {out_csv}")
    print(f"  Tempo labels:     {out_df['tempo_label'].value_counts().to_dict()}")
    print(f"  Energy labels:    {out_df['energy_label'].value_counts().to_dict()}")
    print(f"  Structure labels: {out_df['structure_label'].value_counts().to_dict()}")
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="all")
    args = parser.parse_args()

    targets = list(GENRES.keys()) if args.genre == "all" else [args.genre]
    for g in targets:
        if g not in GENRES:
            sys.exit(f"Unknown genre '{g}'")
        generate_prompts(g)


if __name__ == "__main__":
    main()
