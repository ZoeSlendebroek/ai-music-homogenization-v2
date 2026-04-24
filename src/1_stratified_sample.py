#!/usr/bin/env python3
"""
1_stratified_sample.py

Takes the raw candidate pool from script 0, extracts MIR features,
runs k-means (k=10) on the 67-feature space, samples proportionally
across clusters → final corpus of n≤100 per genre.

Outputs:
    data/human_corpus/{genre}/corpus_final.csv   (sampled tracks + features)
    data/human_corpus/{genre}/cluster_report.txt (cluster sizes + sample counts)

Usage:
    python src/1_stratified_sample.py --genre kpop
    python src/1_stratified_sample.py --genre all
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.genres import GENRES, KMEANS_K

ROOT       = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "human_corpus"

# Import the feature extractor from script 4 (shared — no duplication)
sys.path.insert(0, str(ROOT / "src"))
from extract_features_shared import FeatureExtractor, FEATURE_COLS


def sample_genre(genre_key: str):
    cfg      = GENRES[genre_key]
    manifest = CORPUS_DIR / genre_key / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"No manifest for '{genre_key}'. Run 0_collect_human_corpus.py first.")

    df = pd.read_csv(manifest)
    df = df[df["downloaded"] == True].copy()
    print(f"\n{cfg['display']}: {len(df)} downloaded tracks in pool")

    # ── extract features from all candidate tracks ────────────────
    extractor = FeatureExtractor()
    feat_rows = []
    failed    = []

    for _, row in df.iterrows():
        p = Path(row["audio_path"])
        if not p.exists():
            failed.append(row["spotify_id"])
            continue
        try:
            feats = extractor.extract(p)
            feats["spotify_id"] = row["spotify_id"]
            feat_rows.append(feats)
        except Exception as e:
            print(f"  [WARN] Feature extraction failed for {p.name}: {e}")
            failed.append(row["spotify_id"])

    if failed:
        print(f"  [WARN] {len(failed)} tracks failed feature extraction — excluded")

    feat_df = pd.DataFrame(feat_rows)

    # ── k-means stratified sampling ───────────────────────────────
    X = feat_df[FEATURE_COLS].values
    imp   = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imp.fit_transform(X))

    k      = min(KMEANS_K, len(feat_df))
    target = cfg["target_n"]

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df["cluster"] = km.fit_predict(X_scaled)

    # proportional allocation across clusters
    cluster_counts = feat_df["cluster"].value_counts().sort_index()
    total          = len(feat_df)
    allocations    = {}
    allocated      = 0

    for c, cnt in cluster_counts.items():
        alloc = max(1, round(cnt / total * target))
        allocations[c] = alloc
        allocated += alloc

    # trim/add to hit exact target
    while allocated > target:
        biggest = max(allocations, key=allocations.get)
        allocations[biggest] -= 1
        allocated -= 1
    while allocated < target and allocated < len(feat_df):
        smallest_cluster = min(
            (c for c, a in allocations.items()
             if a < cluster_counts[c]),
            key=allocations.get, default=None
        )
        if smallest_cluster is None:
            break
        allocations[smallest_cluster] += 1
        allocated += 1

    # sample from each cluster
    sampled = []
    for c, n in allocations.items():
        cluster_df = feat_df[feat_df["cluster"] == c]
        n_sample   = min(n, len(cluster_df))
        sampled.append(cluster_df.sample(n=n_sample, random_state=42))

    sampled_df = pd.concat(sampled).reset_index(drop=True)

    # merge back with metadata
    final_df = sampled_df.merge(df[["spotify_id","title","artist","year",
                                     "audio_path","source"]],
                                 on="spotify_id", how="left")

    out_csv    = CORPUS_DIR / genre_key / "corpus_final.csv"
    report_txt = CORPUS_DIR / genre_key / "cluster_report.txt"

    final_df.to_csv(out_csv, index=False)

    # write cluster report
    lines = [f"{cfg['display']} — stratified sampling report",
             f"Pool size: {len(feat_df)}",
             f"Target n: {target}",
             f"Sampled n: {len(final_df)}",
             f"k = {k}",
             ""]
    for c in sorted(cluster_counts.index):
        lines.append(f"  Cluster {c:2d}: pool={cluster_counts[c]:3d}  "
                     f"sampled={allocations.get(c,0):3d}")

    if len(final_df) < 80:
        lines.append(f"\n[WARN] n={len(final_df)} is below 80. "
                     f"Consider adding more playlists. "
                     f"Report statistical power in paper.")

    report_txt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved → {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="all")
    args = parser.parse_args()

    targets = list(GENRES.keys()) if args.genre == "all" else [args.genre]
    for g in targets:
        if g not in GENRES:
            sys.exit(f"Unknown genre '{g}'")
        sample_genre(g)


if __name__ == "__main__":
    main()
