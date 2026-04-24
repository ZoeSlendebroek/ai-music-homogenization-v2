#!/usr/bin/env python3
"""
3_check_prompt_diversity.py

Before generating any AI audio, verify that Exp 2 prompts are
sufficiently diverse. Flags if > 15% of prompt pairs are identical
(after normalisation) — sign that the human corpus lacks acoustic range.

Also reports: distribution of tempo/energy/structure labels, and
pairwise Jaccard similarity on prompt token sets.

Usage:
    python src/3_check_prompt_diversity.py --genre kpop
    python src/3_check_prompt_diversity.py --genre all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres import GENRES, PROMPT_DIVERSITY_THRESHOLD

ROOT       = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "data" / "prompts"


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def check_genre(genre_key: str):
    cfg = GENRES[genre_key]
    f   = PROMPT_DIR / f"{genre_key}_exp2_prompts.csv"
    if not f.exists():
        sys.exit(f"No prompts for '{genre_key}'. Run 2_generate_prompts.py first.")

    df     = pd.read_csv(f)
    n      = len(df)
    prompts = df["prompt_suno"].tolist()

    print(f"\n{'='*55}")
    print(f"  {cfg['display']}  —  {n} prompts")
    print(f"{'='*55}")

    # ── exact duplicates ──────────────────────────────────────────
    n_pairs    = n * (n - 1) / 2
    n_exact    = sum(1 for i in range(n) for j in range(i+1, n)
                     if prompts[i] == prompts[j])
    pct_exact  = n_exact / n_pairs if n_pairs > 0 else 0.0

    print(f"  Exact duplicate pairs:  {n_exact} / {int(n_pairs)}  ({pct_exact:.1%})")

    if pct_exact > PROMPT_DIVERSITY_THRESHOLD:
        print(f"  [WARN] Exceeds threshold ({PROMPT_DIVERSITY_THRESHOLD:.0%}).")
        print(f"         Consider finer BPM bins or additional playlist sources.")
    else:
        print(f"  [OK]   Below threshold ({PROMPT_DIVERSITY_THRESHOLD:.0%}).")

    # ── label distributions ───────────────────────────────────────
    for col in ["tempo_label", "energy_label", "structure_label", "hp_label"]:
        if col in df.columns:
            print(f"\n  {col}:")
            for val, cnt in df[col].value_counts().items():
                bar = "█" * cnt
                print(f"    {val:<35} {cnt:3d}  {bar}")

    # ── pairwise Jaccard sample (capped at 500 pairs for speed) ──
    sample_pairs = min(500, int(n_pairs))
    idx_pairs    = [(i, j) for i in range(n) for j in range(i+1, n)]
    if len(idx_pairs) > sample_pairs:
        rng      = np.random.default_rng(42)
        idx_pairs = [idx_pairs[k] for k in rng.choice(len(idx_pairs),
                                                        sample_pairs, replace=False)]
    sims = [jaccard(prompts[i], prompts[j]) for i, j in idx_pairs]
    print(f"\n  Jaccard similarity (sample of {len(sims)} pairs):")
    print(f"    mean={np.mean(sims):.3f}  median={np.median(sims):.3f}"
          f"  p90={np.percentile(sims, 90):.3f}  max={np.max(sims):.3f}")
    if np.percentile(sims, 90) > 0.85:
        print(f"  [WARN] p90 Jaccard > 0.85 — prompts are very similar."
              f" Consider encoding additional features (e.g. chroma variety).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="all")
    args = parser.parse_args()

    targets = list(GENRES.keys()) if args.genre == "all" else [args.genre]
    for g in targets:
        if g not in GENRES:
            sys.exit(f"Unknown genre '{g}'")
        check_genre(g)


if __name__ == "__main__":
    main()
