#!/usr/bin/env python3
"""
6_diagnostic_plots.py

Diagnostic investigation of AI music homogenization.
Ordered investigation answering:
  1. What are AI systems actually outputting vs human? (feature mean heatmap)
  2. Is there within-genre homogenization? (the methodologically correct test)
  3. Did AI follow the prompts, or converge to its own defaults? (fidelity)
  4. What is the BPM distribution — does AI respect the full BPM range?
  5. What are the distributions of the most AI-distinctive features?
  6. Does within-genre pairwise distance confirm homogenization per genre?

All plots saved to data/analysis/diagnostics/

Usage:
    python src/6_diagnostic_plots.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

ROOT     = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data" / "features"
DIAG_DIR = ROOT / "data" / "analysis" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_COLORS  = {"human": "#2C7BB6", "suno": "#D7191C", "lyria": "#1A9641"}
GENRE_DISPLAY  = {"metal": "Heavy Metal", "afrobeats": "Afrobeats",
                  "dancepop": "Dance Pop", "kpop": "K-pop"}
GENRES         = ["metal", "afrobeats", "dancepop", "kpop"]
SYSTEMS        = ["human", "suno", "lyria"]

# Features directly encoded in prompts
PROMPT_ENCODED = ["tempo", "harmonic_ratio", "onset_density", "self_similarity_mean"]
# Production artifacts
PRODUCTION     = ["rms_mean", "rms_std", "rms_var", "dynamic_range_db", "crest_factor",
                  "percussive_ratio"]
ALL_67_FEATS = [
    "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_contrast_mean", "spectral_contrast_std",
    "zero_crossing_rate_mean", "zero_crossing_rate_std",
    "harmonic_ratio", "percussive_ratio",
    "tempo", "onset_density", "onset_strength_mean", "onset_strength_std",
    "ioi_mean", "ioi_std", "ioi_cv",
    "chroma_stft_mean", "chroma_stft_std", "chroma_stft_var",
    "chroma_cqt_mean", "chroma_cqt_std",
    "tonnetz_mean", "tonnetz_std",
    "repetition_score", "self_similarity_mean", "self_similarity_std",
    "rms_mean", "rms_std", "rms_var",
    "dynamic_range_db", "crest_factor",
] + [f"mfcc_{i}_mean" for i in range(13)] \
  + [f"mfcc_{i}_std"  for i in range(13)] \
  + ["mfcc_delta_mean", "mfcc_delta_std", "mfcc_delta2_mean", "mfcc_delta2_std"]

INTERPRETABLE = [f for f in ALL_67_FEATS
                 if f not in PROMPT_ENCODED and f not in PRODUCTION]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(FEAT_DIR / "all_features.csv")
    # Extract spotify_id from AI filenames for pairing
    df["sid"] = df.apply(
        lambda r: r["spotify_id"] if r["condition"] == "human"
                  else r["filename"].replace("_suno.mp3", "").replace("_lyria.mp3", ""),
        axis=1
    )
    return df


def load_prompt_targets():
    """Prompt CSVs hold the exact feature values that were encoded."""
    dfs = []
    for g in GENRES:
        p = ROOT / "data" / "prompts" / f"{g}_exp2_prompts.csv"
        if p.exists():
            tmp = pd.read_csv(p)[["spotify_id"] + PROMPT_ENCODED]
            tmp["genre"] = g
            dfs.append(tmp)
    targets = pd.concat(dfs, ignore_index=True)
    targets.columns = ["spotify_id"] + [f"target_{f}" for f in PROMPT_ENCODED] + ["genre"]
    return targets


# ── Figure A: Feature mean heatmap ───────────────────────────────────────────

def fig_feature_heatmap(df):
    """
    Z-scored feature means per (system × genre).
    Immediately shows: what does each system systematically over/under-produce?
    Rows = features, columns = system×genre combos.
    """
    print("[Fig A] Feature mean heatmap...")

    # Use a selection of representative features to keep readable
    feat_groups = {
        "Spectral": ["spectral_centroid_mean", "spectral_bandwidth_mean",
                     "spectral_rolloff_mean", "spectral_flatness_mean",
                     "spectral_contrast_mean"],
        "Timbral / MFCC": ["zero_crossing_rate_mean", "mfcc_0_mean", "mfcc_1_mean",
                            "mfcc_2_mean", "mfcc_delta_mean", "mfcc_delta2_mean"],
        "Rhythmic": ["tempo", "onset_density", "onset_strength_mean",
                     "ioi_mean", "ioi_cv"],
        "Harmonic": ["harmonic_ratio", "chroma_stft_mean", "chroma_cqt_mean",
                     "tonnetz_mean"],
        "Structural": ["self_similarity_mean", "self_similarity_std",
                       "repetition_score"],
        "Dynamic": ["rms_mean", "dynamic_range_db", "crest_factor"],
    }
    feats = [f for grp in feat_groups.values() for f in grp]

    # Z-score across all tracks so heatmap is relative
    scaler = StandardScaler().fit(df[feats].values)
    df2 = df.copy()
    df2[feats] = scaler.transform(df[feats].values)

    # Compute means per system × genre
    cols, col_labels = [], []
    for sys in SYSTEMS:
        for g in GENRES:
            sub = df2[(df2["system"] == sys) & (df2["genre"] == g)]
            cols.append(sub[feats].mean().values)
            col_labels.append(f"{sys[:3].upper()}\n{g[:3]}")

    mat = np.array(cols).T  # shape (n_feats, n_cols)

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)

    # Feature row labels with group separators
    row_labels, row_colors = [], []
    group_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#888888"]
    for gi, (grp, gfeats) in enumerate(feat_groups.items()):
        for f in gfeats:
            short = (f.replace("_mean","").replace("_std","±")
                      .replace("_","  ").replace("mfcc","M").replace("spectral","Spec"))
            row_labels.append(short)
            row_colors.append(group_colors[gi])

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for tick, col in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(col)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)

    # Vertical lines between systems
    for i in [3.5, 7.5]:
        ax.axvline(i, color="white", lw=2)

    # System labels on top
    for xi, sys in enumerate(SYSTEMS):
        ax.text(xi * 4 + 1.5, -1.8, sys.capitalize(),
                ha="center", fontsize=11, fontweight="bold",
                color=SYSTEM_COLORS[sys])

    plt.colorbar(im, ax=ax, label="Z-scored mean (relative to all tracks)", pad=0.01)
    ax.set_title("Feature Means by System × Genre  (z-scored across all 1200 tracks)\n"
                 "Blue = below average  |  Red = above average",
                 fontsize=12, fontweight="bold", pad=30)

    # Group legend
    patches = [mpatches.Patch(color=group_colors[i], label=list(feat_groups.keys())[i])
               for i in range(len(feat_groups))]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.18, 1.0),
              fontsize=8, title="Feature group")

    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figA_feature_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figA_feature_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figA_feature_heatmap")


# ── Figure B: Within-genre variance ratios ────────────────────────────────────

def fig_within_genre_variance(df):
    """
    THE METHODOLOGICALLY CORRECT HOMOGENIZATION TEST.

    Pools across genres inflate AI variance (a system that correctly
    differentiates metal from dancepop will show high cross-genre variance
    even if it's homogeneous within each genre).

    This figure shows var(AI_within_genre) / var(Human_within_genre)
    separately for each feature and genre.
    """
    print("[Fig B] Within-genre variance ratios...")

    feats = INTERPRETABLE
    results = {sys: pd.DataFrame(index=feats, columns=GENRES, dtype=float)
               for sys in ("suno", "lyria")}

    for g in GENRES:
        h_vals = df[(df["system"] == "human") & (df["genre"] == g)][feats]
        for sys in ("suno", "lyria"):
            ai_vals = df[(df["system"] == sys) & (df["genre"] == g)][feats]
            h_var   = h_vals.var(ddof=1)
            ai_var  = ai_vals.var(ddof=1)
            ratio   = ai_var / h_var.replace(0, np.nan)
            results[sys][g] = ratio.values

    # Summary stats
    for sys in ("suno", "lyria"):
        mat = results[sys].values.astype(float)
        n_lower = np.nansum(mat < 1)
        n_total = np.sum(~np.isnan(mat))
        print(f"  {sys}: {n_lower}/{n_total} feature×genre cells show AI < Human variance "
              f"({100*n_lower/n_total:.0f}%)")
        print(f"  {sys}: median within-genre var ratio = {np.nanmedian(mat):.3f}")

    # Plot: two heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 12))
    for ax, sys in zip(axes, ("suno", "lyria")):
        mat = np.log2(results[sys].values.astype(float).clip(0.05, 20))
        # log2 scale: 0 = equal variance, negative = AI more homogeneous
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(GENRES)))
        ax.set_xticklabels([GENRE_DISPLAY[g] for g in GENRES], fontsize=10)
        ax.set_yticks(range(len(feats)))
        short = [f.replace("_mean","").replace("_std","±").replace("_"," ")
                 for f in feats]
        ax.set_yticklabels(short, fontsize=6)
        ax.set_title(f"{sys.capitalize()}\nlog₂(Var ratio: AI/Human)\n"
                     f"Blue < 0 = AI more homogeneous",
                     fontsize=11, fontweight="bold", color=SYSTEM_COLORS[sys])
        plt.colorbar(im, ax=ax, label="log₂(AI var / Human var)", shrink=0.6)

    fig.suptitle("Within-Genre Variance Ratios\n"
                 "(correct homogenization test — computed separately per genre)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figB_within_genre_variance.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figB_within_genre_variance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save summary table
    for sys in ("suno", "lyria"):
        results[sys].to_csv(DIAG_DIR / f"within_genre_var_{sys}.csv")

    return results


# ── Figure C: Prompt fidelity ─────────────────────────────────────────────────

def fig_prompt_fidelity(df, targets):
    """
    Did the AI systems actually follow the encoded feature targets?
    Scatter: target value (x) vs AI output value (y).
    Perfect fidelity = points on the diagonal.
    Convergence to a default = horizontal band (AI ignores the prompt range).
    """
    print("[Fig C] Prompt fidelity scatters...")

    # Merge AI rows with their targets
    ai = df[df["condition"] == "ai"].copy()
    ai = ai.merge(targets[["spotify_id"] + [f"target_{f}" for f in PROMPT_ENCODED]],
                  left_on="sid", right_on="spotify_id", how="left")

    feat_labels = {
        "tempo":               "Tempo (BPM)",
        "harmonic_ratio":      "Harmonic Ratio (H/P balance)",
        "onset_density":       "Onset Density (onsets/sec)",
        "self_similarity_mean": "Self-similarity Mean (structure)",
    }
    n_feats = len(PROMPT_ENCODED)

    fig, axes = plt.subplots(n_feats, 2, figsize=(11, 14))

    for fi, feat in enumerate(PROMPT_ENCODED):
        for si, sys in enumerate(("suno", "lyria")):
            ax    = axes[fi, si]
            sub   = ai[ai["system"] == sys].dropna(subset=[f"target_{feat}", feat])
            x     = sub[f"target_{feat}"].values
            y     = sub[feat].values

            # Color by genre
            for g in GENRES:
                mask = sub["genre"] == g
                if mask.sum() == 0:
                    continue
                ax.scatter(x[mask.values], y[mask.values],
                           c=plt.cm.Set1(GENRES.index(g) / 4),
                           s=18, alpha=0.6, zorder=3, label=GENRE_DISPLAY[g])

            # Perfect fidelity line
            lo = min(x.min(), y.min()) * 0.95
            hi = max(x.max(), y.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5, label="Perfect match")

            # Correlation
            r, p = stats.pearsonr(x, y)
            ax.text(0.05, 0.93, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9, va="top",
                    color=SYSTEM_COLORS[sys], fontweight="bold")

            ax.set_xlabel(f"Target {feat_labels[feat]}", fontsize=8)
            ax.set_ylabel(f"AI output", fontsize=8)
            ax.set_title(f"{sys.capitalize()} — {feat_labels[feat]}",
                         fontsize=9, color=SYSTEM_COLORS[sys], fontweight="bold")
            if fi == 0 and si == 0:
                ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Prompt Fidelity: Target Feature Value vs AI Output\n"
                 "(diagonal = perfect match; horizontal band = AI ignores range)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figC_prompt_fidelity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figC_prompt_fidelity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figC_prompt_fidelity")


# ── Figure D: BPM distributions ──────────────────────────────────────────────

def fig_bpm_distributions(df, targets):
    """
    BPM is the most concrete prompt-encoded feature.
    Show: what range of BPMs did human tracks span (the targets)?
    What did Suno / Lyria actually produce?
    Per genre so we can see if AI flattens the BPM range.
    """
    print("[Fig D] BPM distributions...")

    ai = df[df["condition"] == "ai"].copy()
    ai = ai.merge(targets[["spotify_id", "target_tempo"]],
                  left_on="sid", right_on="spotify_id", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for ax, g in zip(axes, GENRES):
        h_sub  = df[(df["system"] == "human") & (df["genre"] == g)]["tempo"].dropna()
        s_sub  = ai[(ai["system"] == "suno")  & (ai["genre"] == g)]["tempo"].dropna()
        l_sub  = ai[(ai["system"] == "lyria") & (ai["genre"] == g)]["tempo"].dropna()

        bw = 4
        for vals, sys, ls in [(h_sub, "human", "-"),
                               (s_sub, "suno",  "--"),
                               (l_sub, "lyria", ":")]:
            if len(vals) < 5:
                continue
            kde = stats.gaussian_kde(vals, bw_method=bw / vals.std(ddof=1)
                                     if vals.std(ddof=1) > 0 else 0.3)
            xs  = np.linspace(vals.min() - 10, vals.max() + 10, 300)
            ax.plot(xs, kde(xs), color=SYSTEM_COLORS[sys], lw=2, ls=ls,
                    label=f"{sys.capitalize()} (mean={vals.mean():.0f}, "
                          f"std={vals.std():.0f})")
            ax.axvline(vals.mean(), color=SYSTEM_COLORS[sys], lw=1, alpha=0.4)

        ax.set_xlabel("Tempo (BPM)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(GENRE_DISPLAY[g], fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)

    fig.suptitle("BPM Distributions: Human Target vs AI Output\n"
                 "(narrower AI distribution = tempo convergence even with explicit BPM encoding)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figD_bpm_distributions.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figD_bpm_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary stats
    print("\n  BPM summary (mean ± std, range):")
    print(f"  {'':12}  {'Human target':>20}  {'Suno output':>20}  {'Lyria output':>20}")
    for g in GENRES:
        h = df[(df["system"] == "human") & (df["genre"] == g)]["tempo"].dropna()
        s = ai[(ai["system"] == "suno")  & (ai["genre"] == g)]["tempo"].dropna()
        l = ai[(ai["system"] == "lyria") & (ai["genre"] == g)]["tempo"].dropna()
        def fmt(v): return f"{v.mean():.0f}±{v.std():.0f} [{v.min():.0f}–{v.max():.0f}]"
        print(f"  {GENRE_DISPLAY[g]:<12}  {fmt(h):>20}  {fmt(s):>20}  {fmt(l):>20}")


# ── Figure E: Key feature KDE comparison ─────────────────────────────────────

def fig_key_features(df):
    """
    Overlapping KDEs for the features that matter most for homogenization
    claims — selected to span spectral, rhythmic, harmonic, and structural.
    """
    print("[Fig E] Key feature KDEs...")

    key_feats = [
        ("ioi_cv",                "IOI Coefficient of Variation\n(rhythmic regularity)"),
        ("self_similarity_std",   "Self-similarity Std\n(structural consistency)"),
        ("spectral_centroid_mean","Spectral Centroid Mean\n(brightness/timbre)"),
        ("chroma_stft_mean",      "Chroma Mean\n(harmonic content)"),
        ("mfcc_delta2_mean",      "MFCC Δ² Mean\n(timbral change rate)"),
        ("spectral_flatness_mean","Spectral Flatness\n(noise vs tonal)"),
    ]

    fig, axes = plt.subplots(len(key_feats), len(GENRES),
                             figsize=(15, 3 * len(key_feats)))

    for fi, (feat, flabel) in enumerate(key_feats):
        for gi, g in enumerate(GENRES):
            ax = axes[fi, gi]
            all_vals = []
            for sys in SYSTEMS:
                vals = df[(df["system"] == sys) & (df["genre"] == g)][feat].dropna()
                all_vals.append(vals)
            lo = min(v.min() for v in all_vals if len(v) > 0)
            hi = max(v.max() for v in all_vals if len(v) > 0)
            xs = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 300)
            for vals, sys in zip(all_vals, SYSTEMS):
                if len(vals) < 5 or vals.std() == 0:
                    continue
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.fill_between(xs, kde(xs), alpha=0.18, color=SYSTEM_COLORS[sys])
                ax.plot(xs, kde(xs), color=SYSTEM_COLORS[sys], lw=1.8,
                        label=sys.capitalize())
            if fi == 0:
                ax.set_title(GENRE_DISPLAY[g], fontsize=10, fontweight="bold")
            if gi == 0:
                ax.set_ylabel(flabel, fontsize=8)
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)
            if fi == 0 and gi == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Distribution of Key Features by System and Genre\n"
                 "(shaded fill = same color as line for readability)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figE_key_features.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figE_key_features.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figE_key_features")


# ── Figure F: Within-genre pairwise distances ─────────────────────────────────

def fig_within_genre_distances(df):
    """
    Intra-group pairwise distances computed WITHIN each genre.
    This is the correct unit for homogenization — not pooled across genres.
    """
    print("[Fig F] Within-genre pairwise distances...")

    scaler = StandardScaler().fit(df[INTERPRETABLE].values)
    records = []
    for g in GENRES:
        for sys in SYSTEMS:
            sub = df[(df["system"] == sys) & (df["genre"] == g)]
            if len(sub) < 5:
                continue
            mat = scaler.transform(sub[INTERPRETABLE].values)
            dv  = pdist(mat, "euclidean")
            records.append({"genre": g, "system": sys,
                             "mean": np.mean(dv), "std": np.std(dv),
                             "distances": dv})

    fig, axes = plt.subplots(1, len(GENRES), figsize=(14, 5), sharey=True)
    for ax, g in zip(axes, GENRES):
        g_recs = [r for r in records if r["genre"] == g]
        data   = [r["distances"] for r in g_recs]
        colors = [SYSTEM_COLORS[r["system"]] for r in g_recs]
        labels = [r["system"].capitalize() for r in g_recs]
        vp = ax.violinplot(data, positions=range(len(data)),
                           showmedians=True, showextrema=False)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)
        means = [r["mean"] for r in g_recs]
        ax.scatter(range(len(means)), means, color="black", s=30, zorder=5, marker="x")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(GENRE_DISPLAY[g], fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Pairwise Distance\n(z-scored interpretable features)", fontsize=10)

    # Print table
    print(f"\n  Within-genre mean pairwise distance:")
    print(f"  {'Genre':<12}  {'Human':>8}  {'Suno':>8}  {'Lyria':>8}  "
          f"{'Suno/Hum':>9}  {'Lyria/Hum':>10}")
    for g in GENRES:
        g_recs = {r["system"]: r for r in records if r["genre"] == g}
        h = g_recs.get("human", {}).get("mean", np.nan)
        s = g_recs.get("suno",  {}).get("mean", np.nan)
        l = g_recs.get("lyria", {}).get("mean", np.nan)
        print(f"  {GENRE_DISPLAY[g]:<12}  {h:>8.3f}  {s:>8.3f}  {l:>8.3f}  "
              f"  {s/h:>7.3f}    {l/h:>7.3f}")

    fig.suptitle("Within-Genre Intra-Group Pairwise Distance\n"
                 "(× = mean; lower = more homogeneous within genre)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figF_within_genre_distances.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figF_within_genre_distances.png", dpi=300, bbox_inches="tight")
    plt.close()


# ── Figure G: Prompt-encoded feature fidelity — all 4, per genre, KDE ────────

def fig_encoded_distributions(df, targets):
    """
    For each prompt-encoded feature, show the distribution of:
      (a) human target (= what was encoded into the prompt)
      (b) suno output
      (c) lyria output
    in a 4×4 grid (features × genres).
    If AI distributions are narrower than human targets → homogenization DESPITE encoding.
    """
    print("[Fig G] Encoded feature distributions...")

    ai = df[df["condition"] == "ai"].copy()
    ai = ai.merge(targets[["spotify_id"] + [f"target_{f}" for f in PROMPT_ENCODED]],
                  left_on="sid", right_on="spotify_id", how="left")

    feat_labels = {
        "tempo":               "Tempo (BPM)",
        "harmonic_ratio":      "Harmonic Ratio",
        "onset_density":       "Onset Density (onsets/s)",
        "self_similarity_mean":"Self-similarity",
    }

    fig, axes = plt.subplots(len(PROMPT_ENCODED), len(GENRES),
                             figsize=(15, 3.5 * len(PROMPT_ENCODED)))
    for fi, feat in enumerate(PROMPT_ENCODED):
        for gi, g in enumerate(GENRES):
            ax = axes[fi, gi]
            # Human target
            h_vals = targets[targets["genre"] == g][f"target_{feat}"].dropna()
            s_vals = ai[(ai["system"] == "suno")  & (ai["genre"] == g)][feat].dropna()
            l_vals = ai[(ai["system"] == "lyria") & (ai["genre"] == g)][feat].dropna()

            all_v = pd.concat([h_vals, s_vals, l_vals])
            lo, hi = all_v.min(), all_v.max()
            xs = np.linspace(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo), 300)

            for vals, sys, label in [(h_vals, "human", "Human target"),
                                      (s_vals, "suno",  "Suno output"),
                                      (l_vals, "lyria", "Lyria output")]:
                if len(vals) < 5 or vals.std() == 0:
                    continue
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.fill_between(xs, kde(xs), alpha=0.15, color=SYSTEM_COLORS[sys])
                ax.plot(xs, kde(xs), color=SYSTEM_COLORS[sys], lw=2,
                        label=f"{label} (σ={vals.std():.2f})")

            if fi == 0:
                ax.set_title(GENRE_DISPLAY[g], fontsize=10, fontweight="bold")
            if gi == 0:
                ax.set_ylabel(feat_labels[feat], fontsize=9)
            ax.tick_params(labelsize=7)
            if fi == 0 and gi == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Prompt-Encoded Feature Distributions: Target vs AI Output\n"
                 "(σ = std dev — narrower AI std vs human std despite encoding = homogenization)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "figG_encoded_distributions.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(DIAG_DIR / "figG_encoded_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print std comparison table
    print(f"\n  Std deviation of prompt-encoded features (target vs output):")
    for feat in PROMPT_ENCODED:
        print(f"\n  {feat}:")
        print(f"    {'Genre':<12}  {'Target std':>12}  {'Suno std':>10}  "
              f"{'Lyria std':>10}  {'Suno/Tgt':>9}  {'Lyria/Tgt':>10}")
        for g in GENRES:
            t  = targets[targets["genre"] == g][f"target_{feat}"].dropna().std()
            s  = ai[(ai["system"] == "suno")  & (ai["genre"] == g)][feat].dropna().std()
            l  = ai[(ai["system"] == "lyria") & (ai["genre"] == g)][feat].dropna().std()
            print(f"    {GENRE_DISPLAY[g]:<12}  {t:>12.3f}  {s:>10.3f}  "
                  f"{l:>10.3f}    {s/t:>7.3f}    {l/t:>7.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df      = load_data()
    targets = load_prompt_targets()

    print(f"\nRunning diagnostic plots ({len(df)} tracks, {len(targets)} prompt targets)\n")

    fig_feature_heatmap(df)
    var_results = fig_within_genre_variance(df)
    fig_prompt_fidelity(df, targets)
    fig_bpm_distributions(df, targets)
    fig_key_features(df)
    fig_within_genre_distances(df)
    fig_encoded_distributions(df, targets)

    print(f"\nAll diagnostic plots saved to {DIAG_DIR}/")


if __name__ == "__main__":
    main()
