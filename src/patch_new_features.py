"""
patch_new_features.py

Appends 5 new homogenization features to the existing all_features.csv
WITHOUT re-extracting the 67 original features.

New features:
    chroma_entropy       : Shannon entropy of mean chroma vector
    key_clarity          : max/mean of normalised chroma vector
    spectral_centroid_cv : CV of frame-level spectral centroid (intra-clip timbral variation)
    onset_strength_cv    : CV of onset strength envelope (rhythmic energy variation)
    rms_cv               : CV of RMS frames (dynamic variation within clip)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

ROOT   = Path(__file__).resolve().parent.parent
FEAT   = ROOT / "data" / "features" / "all_features.csv"
AUDIO  = ROOT / "data" / "audio"
HUMAN  = ROOT / "data" / "human_corpus"

SR     = 22050
TARGET = 30 * SR

NEW_COLS = ["chroma_entropy", "key_clarity", "spectral_centroid_cv",
            "onset_strength_cv", "rms_cv"]

GENRE_DIRS = {
    "afrobeats":  "afrobeats",
    "kpop":       "kpop",
    "dancepop":   "dancepop",
    "metal":      "metal",
}

def load_30s(path):
    y, sr = librosa.load(str(path), sr=SR)
    if len(y) > TARGET:
        mid   = len(y) // 2
        start = max(0, mid - TARGET // 2)
        end   = start + TARGET
        if end > len(y):
            end   = len(y)
            start = end - TARGET
        y = y[start:end]
    elif len(y) < TARGET:
        pad = TARGET - len(y)
        y   = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    return y

def compute_new(y):
    ch      = librosa.feature.chroma_stft(y=y, sr=SR)
    mean_ch = np.abs(np.mean(ch, axis=1))
    p       = mean_ch / (mean_ch.sum() + 1e-10)
    chroma_ent  = float(-np.sum(p * np.log2(p + 1e-10)))
    key_clarity = float(np.max(p)) / (float(np.mean(p)) + 1e-10)

    sc    = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    sc_cv = float(np.std(sc) / (np.mean(sc) + 1e-10))

    oe    = librosa.onset.onset_strength(y=y, sr=SR)
    oe_cv = float(np.std(oe) / (np.mean(oe) + 1e-10))

    rms    = librosa.feature.rms(y=y)[0]
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))

    return chroma_ent, key_clarity, sc_cv, oe_cv, rms_cv

def find_audio(row):
    fname = row["filename"]
    genre = GENRE_DIRS[row["genre"]]
    if row["condition"] == "human":
        p = HUMAN / genre / "audio" / fname
    else:
        system = row["system"]
        p = AUDIO / system / genre / fname
    return p

def main():
    df = pd.read_csv(FEAT)

    if all(c in df.columns for c in NEW_COLS):
        print("All new features already present. Nothing to do.")
        return

    results = {c: [None] * len(df) for c in NEW_COLS}
    failed  = []

    for i, row in df.iterrows():
        path = find_audio(row)
        try:
            y = load_30s(path)
            vals = compute_new(y)
            for j, col in enumerate(NEW_COLS):
                results[col][i] = vals[j]
        except Exception as e:
            failed.append((i, str(path), str(e)))
            for col in NEW_COLS:
                results[col][i] = np.nan

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} done", flush=True)

    for col in NEW_COLS:
        df[col] = results[col]

    df.to_csv(FEAT, index=False)
    print(f"\nSaved {FEAT}  shape={df.shape}")

    if failed:
        print(f"\n{len(failed)} failures:")
        for idx, p, e in failed[:10]:
            print(f"  row {idx}: {p} — {e}")

    # Also patch the per-genre CSVs
    for genre in GENRE_DIRS:
        for tag in ("ai", "human"):
            p = ROOT / "data" / "features" / f"{genre}_{tag}_features.csv"
            if not p.exists():
                continue
            sub = pd.read_csv(p)
            if all(c in sub.columns for c in NEW_COLS):
                continue
            idx = df[df["genre"] == genre]
            if tag == "human":
                idx = idx[idx["condition"] == "human"]
            else:
                idx = idx[idx["condition"] == "ai"]
            for col in NEW_COLS:
                sub[col] = idx[col].values
            sub.to_csv(p, index=False)
            print(f"Patched {p.name}  shape={sub.shape}")

if __name__ == "__main__":
    main()
