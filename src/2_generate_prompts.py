#!/usr/bin/env python3
"""
2_generate_prompts.py

For each track in corpus_final.csv, extract the 4 Exp 2 prompt features
and render prompt strings for Suno, Lyria 3, and Udio.

Feature split
─────────────
PROMPT features (steering — included in prompts):
    tempo, onset_density, harmonic_ratio, self_similarity_mean, year

OUTCOME features (measurement — never put in prompts):
    All spectral features, MFCCs, chroma, tonnetz, ZCR,
    harmonic_ratio/percussive_ratio raw values, dynamic range, RMS, IOI CV.

Instrumentation is intentionally omitted: it is a direct proxy for timbre
(MFCCs / spectral centroid) which is the primary homogenisation outcome.
Specifying it would pre-determine the very signal we want to observe freely.
Era labels are safe to include because year is not a MIR measurement feature.

Outputs:
    data/prompts/{genre}_exp2_prompts.csv
      columns: spotify_id, title, artist, audio_path, year,
               tempo, onset_density, harmonic_ratio, self_similarity_mean,
               tempo_label, energy_label, rhythm_label, hp_label,
               structure_label, era_label,
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres import GENRES

ROOT       = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "human_corpus"
PROMPT_DIR = ROOT / "data" / "prompts"
PROMPT_DIR.mkdir(parents=True, exist_ok=True)


# ── feature → natural language binning ───────────────────────────
#
# Thresholds calibrated against the observed feature ranges across the
# human corpora (onset_density: 0–5.8, harmonic_ratio: 0.45–0.95,
# self_similarity_mean: 0.085–0.32). Original thresholds had upper
# bounds outside the data range (e.g. onset >= 6, self_sim > 0.55),
# making several label buckets permanently unreachable.

def tempo_label(bpm: float) -> str:
    if bpm < 66:   return "Largo"
    if bpm < 76:   return "Adagio"
    if bpm < 108:  return "Andante"
    if bpm < 120:  return "Moderato"
    if bpm < 156:  return "Allegro"
    return "Presto"

def energy_label(onset_density: float) -> str:
    # Calibrated to observed range 0–7.4 onsets/sec (afrobeats extends to 7.4)
    if onset_density < 1.5: return "sparse"
    if onset_density < 3.0: return "relaxed"
    if onset_density < 4.5: return "moderate"
    if onset_density < 6.0: return "driving"
    return "intense"

def hp_label(harmonic_ratio: float) -> str:
    # HPSS classifies sustained distortion as harmonic, so the observed
    # floor is ~0.45 across all genres. Thresholds reflect that reality.
    if harmonic_ratio < 0.65: return "balanced harmonic and percussive content"
    if harmonic_ratio < 0.78: return "melodic, harmonic-leaning"
    return "strongly melodic, harmonic-forward"

def rhythm_label(onset_density: float) -> str:
    # harmonic_ratio dependency removed: its branch (< 0.40) was never
    # triggered and the feature is already captured in hp_label.
    if onset_density >= 6.0: return "relentless, high-density groove"
    if onset_density >= 5.0: return "dense, fast-hitting groove"
    if onset_density >= 3.0: return "syncopated, groove-forward"
    if onset_density >= 1.5: return "steady groove"
    return "sparse, open feel"

def structure_label(self_sim: float) -> str:
    # Calibrated to observed self_similarity_mean range (0.085–0.32).
    # Prior thresholds (0.55, 0.75) were entirely outside the data range,
    # collapsing all 200 tracks into a single "through-composed" bucket.
    if self_sim > 0.130: return "structured with repeating sections"
    if self_sim > 0.107: return "some recurring elements, mostly varied"
    return "through-composed, minimal repetition"


# ── era labels (safe steering: year is not a MIR outcome feature) ─
# Each entry: (year_cutoff_inclusive, label).
# Labels reference production era / sound aesthetics, not just dates.

ERA_LABELS: dict[str, list[tuple[int, str]]] = {
    "kpop": [
        (2018, "3rd-generation K-pop sound (lush synth arrangements, anthemic structure)"),
        (2021, "4th-generation K-pop, early wave (intense concept-driven production)"),
        (2024, "4th-generation K-pop, peak era (Y2K minimalism meets maximalist pop)"),
        (9999, "current-wave K-pop (contemporary idol pop production)"),
    ],
    "afrobeats": [
        (2017, "mid-2010s Afrobeats (Lagos party sound, percussion-led grooves)"),
        (2021, "global Afrobeats crossover era (dancehall and R&B fusion)"),
        (2023, "early-2020s Afrobeats/Afropop (post-pandemic global surge)"),
        (2024, "contemporary Afrobeats (Amapiano crossover, drill-influenced production)"),
        (9999, "current-wave Afrobeats (2025–2026, hyperpop and electronic fusion)"),
    ],
    "dancepop": [
        (2013, "early-2010s dance pop (electro-pop and EDM boom era)"),
        (2017, "mid-2010s dance pop (tropical house and EDM crossover)"),
        (2020, "late-2010s dance pop (future bass, clean production)"),
        (2023, "early-2020s dance pop (Y2K revival, hyperpop-tinged production)"),
        (9999, "current-wave dance pop (maximalist layering, AI-era production)"),
    ],
    "metal": [
        (2009, "2000s heavy metal (nu-metal and metalcore wave)"),
        (2019, "2010s metal (djent and progressive metal influence)"),
        (9999, "contemporary heavy metal (modern production, post-metal elements)"),
    ],
}

def get_era_label(genre_key: str, year: int) -> str:
    brackets = ERA_LABELS.get(genre_key)
    if not brackets:
        return f"{year} production style"
    for cutoff, label in brackets:
        if year <= cutoff:
            return label
    return brackets[-1][1]


# ── prompt renderers ──────────────────────────────────────────────

def render_suno(genre_key: str, bpm: float, t_label: str, e_label: str,
                r_label: str, hp: str, struct: str, hr: float, era: str) -> str:
    genre_display = GENRES[genre_key]["display"]
    return (
        f"Instrumental {genre_display} at {bpm:.1f} BPM ({t_label}), "
        f"{e_label}, {r_label}. "
        f"{hp.capitalize()} (harmonic ratio {hr:.2f}). {struct.capitalize()}. "
        f"{era}. No vocals."
    )

def render_lyria(genre_key: str, bpm: float, t_label: str, e_label: str,
                 r_label: str, hp: str, struct: str, seed: int,
                 hr: float, era: str) -> str:
    genre_display = GENRES[genre_key]["display"]
    prompt = (
        f"An instrumental {genre_display} track. "
        f"Era: {era}. "
        f"Mood: {e_label}. "
        f"Tempo: {bpm:.1f} BPM, {t_label}. "
        f"Rhythm: {r_label}, {hp} (harmonic ratio {hr:.2f}). "
        f"Structure: {struct}."
    )
    negative = "vocals, lyrics, singing, spoken word"
    return f'prompt: "{prompt}" | negative_prompt: "{negative}" | seed: {seed}'

def render_udio(genre_key: str, bpm: float, t_label: str, e_label: str,
                r_label: str, hp: str, struct: str, hr: float, era: str) -> str:
    genre_display = GENRES[genre_key]["display"]
    return (
        f"Instrumental {genre_display}, {bpm:.1f} BPM ({t_label}), "
        f"{e_label} energy. {r_label.capitalize()}, {hp} (harmonic ratio {hr:.2f}). "
        f"{struct.capitalize()}. {era}. No vocals, no lyrics."
    )


# ── deterministic seed for Lyria reproducibility ─────────────────
def make_seed(spotify_id: str) -> int:
    return int(hashlib.md5(spotify_id.encode()).hexdigest(), 16) % (2**31)


# ── main ──────────────────────────────────────────────────────────

def generate_prompts(genre_key: str):
    cfg    = GENRES[genre_key]
    corpus = CORPUS_DIR / genre_key / "corpus_final.csv"
    if not corpus.exists():
        sys.exit(f"No corpus for '{genre_key}'. Run 1_stratified_sample.py first.")

    df = pd.read_csv(corpus)
    print(f"\n{cfg['display']}: generating Exp 2 prompts for {len(df)} tracks")

    # Features are already extracted and stored in corpus_final.csv by
    # 1_stratified_sample.py — read them directly rather than re-extracting
    # from audio. This keeps labels consistent with the clustering step.
    rows = []

    for _, row in df.iterrows():
        bpm  = float(row["tempo"])
        od   = float(row["onset_density"])
        hr   = float(row["harmonic_ratio"])
        ss   = float(row["self_similarity_mean"])
        year = int(row["year"])

        t_lbl = tempo_label(bpm)
        e_lbl = energy_label(od)
        hp    = hp_label(hr)
        r_lbl = rhythm_label(od)
        s_lbl = structure_label(ss)
        era   = get_era_label(genre_key, year)
        seed  = make_seed(str(row["spotify_id"]))

        rows.append({
            "spotify_id":           row["spotify_id"],
            "title":                row["title"],
            "artist":               row["artist"],
            "audio_path":           row["audio_path"],
            "year":                 year,
            # raw features (for diversity check)
            "tempo":                bpm,
            "onset_density":        od,
            "harmonic_ratio":       hr,
            "self_similarity_mean": ss,
            # binned labels
            "tempo_label":          t_lbl,
            "energy_label":         e_lbl,
            "rhythm_label":         r_lbl,
            "hp_label":             hp,
            "structure_label":      s_lbl,
            "era_label":            era,
            "seed_lyria":           seed,
            # rendered prompts
            "prompt_suno":  render_suno(genre_key,  bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl, hr, era),
            "prompt_lyria": render_lyria(genre_key, bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl, seed, hr, era),
            "prompt_udio":  render_udio(genre_key,  bpm, t_lbl, e_lbl, r_lbl, hp, s_lbl, hr, era),
        })

    out_df  = pd.DataFrame(rows)
    out_csv = PROMPT_DIR / f"{genre_key}_exp2_prompts.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"  Saved {len(out_df)} prompts → {out_csv}")
    print(f"  Tempo labels:     {out_df['tempo_label'].value_counts().to_dict()}")
    print(f"  Energy labels:    {out_df['energy_label'].value_counts().to_dict()}")
    print(f"  Structure labels: {out_df['structure_label'].value_counts().to_dict()}")
    print(f"  Era labels:       {out_df['era_label'].value_counts().to_dict()}")
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
