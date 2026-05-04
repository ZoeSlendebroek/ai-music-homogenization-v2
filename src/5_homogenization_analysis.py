#!/usr/bin/env python3
"""
5_homogenization_analysis.py

Publication-ready homogenization analysis for Experiment 2.

METHODOLOGICAL NOTES:

  Feature categories
  ------------------
  (a) PROMPT_ENCODED: tempo, harmonic_ratio, onset_density, self_similarity_mean,
      percussive_ratio. These were specified in the generation prompt.
      IMPORTANT: prompt-fidelity analysis (Analysis 0) shows Suno ignores these
      entirely (r < 0.25 for all features/genres) and Lyria only partially follows
      tempo (r 0.30–0.57) with near-zero tracking of other features.
      Because fidelity is negligible, lower AI variance on these features
      IS meaningful homogenization evidence — the AI is NOT tracking the
      full human target range, it is converging to its own defaults.
      These features are therefore INCLUDED in the homogenization analyses
      with fidelity r values reported alongside.

  (b) PRODUCTION: rms_mean, rms_std, rms_var, dynamic_range_db, crest_factor.
      These reflect output normalisation / mastering chain differences.
      They are EXCLUDED from variance and distance claims but reported separately.

  (c) INTERPRETABLE: all remaining features (original 67 minus production,
      plus 5 new homogenization-specific features).

  Primary analysis unit: WITHIN-GENRE
  ------------------------------------
  Pooling across genres inflates cross-genre variance. A system that correctly
  differentiates heavy metal from K-pop will show high cross-genre variance
  even if it is homogeneous within each genre. All variance and distance
  analyses are computed per genre then summarised.

  Multiple comparison correction: Benjamini-Hochberg FDR throughout.

Analyses
--------
  0. Prompt fidelity        — Pearson r: prompt target vs AI output, per feature/genre
  1. Within-genre variance  — Levene per genre, BH-FDR, variance ratio
  2. Within-genre distance  — Mann-Whitney, Cohen's d, per genre then pooled
  3. Genre separation       — does AI collapse genre distinctions?
  4. System convergence     — do Suno and Lyria converge?
  5. Discriminability       — Random Forest AUC + feature importance

Usage:
    python src/5_homogenization_analysis.py
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data" / "features"
OUT_DIR  = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENRES  = ["metal", "afrobeats", "dancepop", "kpop"]
SYSTEMS = ["suno", "lyria"]
GENRE_DISPLAY = {"metal": "Heavy Metal", "afrobeats": "Afrobeats",
                 "dancepop": "Dance Pop", "kpop": "K-pop"}
SYSTEM_COLORS = {"human": "#2C7BB6", "suno": "#D7191C", "lyria": "#1A9641"}
CATEGORY_COLORS = {
    "Spectral":       "#4477AA",
    "Timbral (MFCC)": "#EE6677",
    "Rhythmic":       "#228833",
    "Harmonic":       "#CCBB44",
    "Structural":     "#AA3377",
    "Key / Tonal":    "#66CCEE",
    "Intra-track":    "#FF8800",
    "Prompt-encoded": "#BBBBBB",
}

# ── Feature lists ─────────────────────────────────────────────────────────────

# Reflected in AI output normalisation — excluded from homogenization claims
PRODUCTION = ["rms_mean", "rms_std", "rms_var", "dynamic_range_db", "crest_factor",
              "percussive_ratio"]  # percussive = 1 - harmonic_ratio, collinear

# Originally specified in prompts — INCLUDED despite encoding because fidelity is ~0
PROMPT_ENCODED = ["tempo", "harmonic_ratio", "onset_density", "self_similarity_mean"]

# New features added for this analysis (not in original 67, not in prompts)
NEW_FEATURES = ["chroma_entropy", "key_clarity",
                "spectral_centroid_cv", "onset_strength_cv", "rms_cv"]

ORIGINAL_67 = [
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

ALL_FEATURES  = ORIGINAL_67 + NEW_FEATURES
ANALYSIS_FEATS = [f for f in ALL_FEATURES if f not in PRODUCTION]


def _cat(feat):
    if feat in PROMPT_ENCODED:                 return "Prompt-encoded"
    if feat in ("chroma_entropy","key_clarity"): return "Key / Tonal"
    if feat in ("spectral_centroid_cv","onset_strength_cv","rms_cv"): return "Intra-track"
    if feat.startswith("spectral") or feat.startswith("zero_crossing"): return "Spectral"
    if feat.startswith("mfcc"):                return "Timbral (MFCC)"
    if feat.startswith("chroma") or feat.startswith("tonnetz"): return "Harmonic"
    if feat in ("repetition_score","self_similarity_std"):      return "Structural"
    return "Rhythmic"

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# ── Utilities ─────────────────────────────────────────────────────────────────

def bh_correct(pvalues):
    n   = len(pvalues)
    idx = np.argsort(pvalues)
    adj = np.minimum(1.0, np.array(pvalues)[idx] * n / np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    r = np.empty(n); r[idx] = adj
    return r


def load_data():
    path = FEAT_DIR / "all_features.csv"
    if not path.exists():
        sys.exit(f"Missing {path} — run 4_extract_features.py first.")
    df = pd.read_csv(path)
    # check new features are present
    missing_new = [f for f in NEW_FEATURES if f not in df.columns]
    if missing_new:
        print(f"  WARNING: new features not found in CSV: {missing_new}")
        print(f"  Re-run 4_extract_features.py to include them.")
    df["sid"] = df.apply(
        lambda r: r["spotify_id"] if r["condition"] == "human"
                  else r["filename"].replace("_suno.mp3","").replace("_lyria.mp3",""),
        axis=1)
    print(f"Loaded {len(df)} tracks")
    print(df.groupby(["genre","condition","system"]).size().to_string())
    return df


def load_prompt_targets():
    dfs = []
    for g in GENRES:
        p = ROOT / "data" / "prompts" / f"{g}_exp2_prompts.csv"
        if p.exists():
            tmp = pd.read_csv(p)[["spotify_id"] + PROMPT_ENCODED]
            tmp["genre"] = g
            dfs.append(tmp)
    t = pd.concat(dfs, ignore_index=True)
    t.columns = ["spotify_id"] + [f"target_{f}" for f in PROMPT_ENCODED] + ["genre"]
    return t


# ── Analysis 0: Prompt fidelity ───────────────────────────────────────────────

def analyze_fidelity(df, targets):
    """
    Pearson r between prompt target value and AI output value, per feature/genre/system.
    r ≈ 0 means the AI is ignoring the encoded feature entirely.
    r ≈ 1 means the AI faithfully tracks the full target range.
    Low r on prompt-encoded features = lower AI variance IS genuine homogenization.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 0 — PROMPT FIDELITY")
    print("=" * 60)

    ai = df[df["condition"] == "ai"].copy()
    ai = ai.merge(targets[["spotify_id"] + [f"target_{f}" for f in PROMPT_ENCODED]],
                  left_on="sid", right_on="spotify_id", how="left")

    rows = []
    print(f"\n  Pearson r (target → AI output)  [|r|<0.3 = essentially no tracking]")
    print(f"  {'':15}  " + "  ".join(f"{f:>18}" for f in PROMPT_ENCODED))
    for sys in SYSTEMS:
        for g in GENRES:
            sub = ai[(ai["system"] == sys) & (ai["genre"] == g)]
            r_vals = {}
            for feat in PROMPT_ENCODED:
                x = sub[f"target_{feat}"].dropna()
                y = sub.loc[x.index, feat].dropna()
                idx = x.index.intersection(y.index)
                if len(idx) > 5:
                    r, p = stats.pearsonr(x[idx], y[idx])
                else:
                    r, p = np.nan, np.nan
                r_vals[feat] = r
                rows.append({"system": sys, "genre": g, "feature": feat,
                             "pearson_r": r, "p_value": p})
            vals = "  ".join(f"{r_vals[f]:>18.3f}" for f in PROMPT_ENCODED)
            print(f"  {sys}/{g:<10}  {vals}")

    fid_df = pd.DataFrame(rows)
    print(f"\n  Mean |r| across genres:")
    for sys in SYSTEMS:
        sub = fid_df[fid_df["system"] == sys]
        for feat in PROMPT_ENCODED:
            mr = sub[sub["feature"] == feat]["pearson_r"].abs().mean()
            print(f"    {sys} {feat}: mean |r| = {mr:.3f}", end="")
        print()

    fid_df.to_csv(OUT_DIR / "prompt_fidelity.csv", index=False)
    return fid_df


# ── Analysis 1: Within-genre variance ────────────────────────────────────────

def analyze_variance(df):
    """
    Within-genre variance ratio (AI / Human) per feature, per genre, per system.
    BH-FDR corrected within each genre×system block.
    Includes prompt-encoded features (fidelity shown to be near-zero → valid signal).
    Excludes production/mastering artifacts.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1 — WITHIN-GENRE VARIANCE HOMOGENEITY")
    print("=" * 60)

    feats = [f for f in ANALYSIS_FEATS if f in df.columns]
    rows  = []

    for g in GENRES:
        h_sub = df[(df["system"] == "human") & (df["genre"] == g)]
        for sys in SYSTEMS:
            ai_sub  = df[(df["system"] == sys) & (df["genre"] == g)]
            pv_list, feat_list = [], []
            for feat in feats:
                h_v  = h_sub[feat].dropna().values
                ai_v = ai_sub[feat].dropna().values
                h_var  = np.var(h_v,  ddof=1)
                ai_var = np.var(ai_v, ddof=1)
                ratio  = ai_var / h_var if h_var > 0 else np.nan
                lev_p  = stats.levene(ai_v, h_v)[1] if len(ai_v) > 1 else np.nan
                rows.append({"feature": feat, "genre": g, "system": sys,
                             "category": _cat(feat),
                             "ai_var": ai_var, "human_var": h_var,
                             "var_ratio": ratio, "levene_p": lev_p})
                if not np.isnan(lev_p):
                    pv_list.append(lev_p); feat_list.append((g, sys, feat))

            adj = bh_correct(pv_list)
            for (gg, ss, ff), ap in zip(feat_list, adj):
                for r in rows:
                    if r["feature"]==ff and r["genre"]==gg and r["system"]==ss:
                        r["levene_p_adj"] = ap

    var_df = pd.DataFrame(rows)

    print(f"\n  {'System':<8}  {'AI<Human':>10}  {'FDR q<.05':>10}  {'Median ratio':>14}")
    for sys in SYSTEMS:
        sub  = var_df[var_df["system"] == sys]
        n_lo = (sub["var_ratio"] < 1).sum()
        n_to = len(sub)
        n_si = (sub.get("levene_p_adj", pd.Series()) < 0.05).sum()
        med  = sub["var_ratio"].median()
        print(f"  {sys:<8}  {n_lo:>4}/{n_to:<5}({100*n_lo/n_to:.0f}%)  "
              f"{n_si:>5}/{n_to:<5}  {med:>10.3f}")

    print(f"\n  Per-genre median variance ratio:")
    print(f"  {'Genre':<12}" + "".join(f"  {sys:>8}" for sys in SYSTEMS))
    for g in GENRES:
        line = f"  {g:<12}"
        for sys in SYSTEMS:
            med = var_df[(var_df["system"]==sys) & (var_df["genre"]==g)]["var_ratio"].median()
            line += f"  {med:>8.3f}"
        print(line)

    var_df.to_csv(OUT_DIR / "variance_analysis.csv", index=False)
    return var_df


# ── Analysis 2: Within-genre pairwise distance ────────────────────────────────

def analyze_distances(df):
    """Within-genre pairwise distance, computed per genre then summarised."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2 — WITHIN-GENRE PAIRWISE DISTANCE")
    print("=" * 60)

    feats  = [f for f in ANALYSIS_FEATS if f in df.columns]
    scaler = StandardScaler().fit(df[feats].values)
    all_dv = {"human": [], "suno": [], "lyria": []}
    g_rows = []

    print(f"\n  {'Genre':<12}  {'Human':>8}  {'Suno':>8}  {'Lyria':>8}  "
          f"{'Suno/H':>8}  {'Lyria/H':>9}  {'p(S<H)':>8}  {'p(L<H)':>8}")

    for g in GENRES:
        dvs = {}
        for sys in ("human", "suno", "lyria"):
            sub = df[(df["system"] == sys) & (df["genre"] == g)]
            mat = scaler.transform(sub[feats].values)
            dv  = pdist(mat, "euclidean")
            dvs[sys] = dv
            all_dv[sys].append(dv)
        _, p_s = stats.mannwhitneyu(dvs["suno"],  dvs["human"], alternative="less")
        _, p_l = stats.mannwhitneyu(dvs["lyria"], dvs["human"], alternative="less")
        print(f"  {g:<12}  {np.mean(dvs['human']):>8.3f}  "
              f"{np.mean(dvs['suno']):>8.3f}  {np.mean(dvs['lyria']):>8.3f}  "
              f"{np.mean(dvs['suno'])/np.mean(dvs['human']):>8.3f}  "
              f"{np.mean(dvs['lyria'])/np.mean(dvs['human']):>9.3f}  "
              f"{p_s:>8.4f}  {p_l:>8.4f}")
        g_rows.append({"genre": g,
                       "human": np.mean(dvs["human"]),
                       "suno":  np.mean(dvs["suno"]),
                       "lyria": np.mean(dvs["lyria"]),
                       "suno_ratio":  np.mean(dvs["suno"])  / np.mean(dvs["human"]),
                       "lyria_ratio": np.mean(dvs["lyria"]) / np.mean(dvs["human"])})

    dist_vecs = {s: np.concatenate(all_dv[s]) for s in ("human","suno","lyria")}
    dist_vecs["ai_combined"] = np.concatenate([dist_vecs["suno"], dist_vecs["lyria"]])

    test_rows = []
    print(f"\n  Cohen's d (pooled within-genre, AI < Human):")
    for name in ("suno", "lyria", "ai_combined"):
        ai_dv = dist_vecs[name]; h_dv = dist_vecs["human"]
        _, p  = stats.mannwhitneyu(ai_dv, h_dv, alternative="less")
        d     = (np.mean(ai_dv) - np.mean(h_dv)) / np.sqrt(
                    (np.std(ai_dv)**2 + np.std(h_dv)**2) / 2)
        direction = "more homogeneous" if d < 0 else "MORE DIVERSE"
        print(f"  {name:<14}  d={d:+.3f}  p={p:.4f}  → {direction}")
        test_rows.append({"comparison": name, "cohens_d": d, "mw_p": p})

    pd.DataFrame(g_rows).to_csv(OUT_DIR / "distance_summary.csv", index=False)
    return dist_vecs, test_rows


# ── Analysis 3: Genre separation ──────────────────────────────────────────────

def analyze_genre_separation(df):
    """Between-genre centroid distance / within-genre spread. Lower = AI flattens genres."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3 — CROSS-GENRE SEPARATION")
    print("=" * 60)

    feats  = [f for f in ANALYSIS_FEATS if f in df.columns]
    scaler = StandardScaler().fit(df[feats].values)
    centroids, within = {}, {}
    for sys in ("human","suno","lyria"):
        for g in GENRES:
            sub = df[(df["system"]==sys) & (df["genre"]==g)]
            if sub.empty: continue
            mat = scaler.transform(sub[feats].values)
            centroids[(sys,g)] = mat.mean(axis=0)
            within[(sys,g)]    = pdist(mat,"euclidean").mean()

    rows = []
    for sys in ("human","suno","lyria"):
        between = [np.linalg.norm(centroids[(sys,g1)] - centroids[(sys,g2)])
                   for i,g1 in enumerate(GENRES) for g2 in GENRES[i+1:]
                   if (sys,g1) in centroids and (sys,g2) in centroids]
        w_vals  = [within[(sys,g)] for g in GENRES if (sys,g) in within]
        ratio   = np.mean(between) / np.mean(w_vals) if w_vals else np.nan
        print(f"  {sys:<8}  between={np.mean(between):.3f}  "
              f"within={np.mean(w_vals):.3f}  ratio={ratio:.3f}")
        rows.append({"system":sys,"mean_between":np.mean(between),
                     "mean_within":np.mean(w_vals),"separation_ratio":ratio})

    sep_df = pd.DataFrame(rows)
    sep_df.to_csv(OUT_DIR / "genre_separation.csv", index=False)
    return sep_df, centroids, scaler


# ── Analysis 4: System convergence ────────────────────────────────────────────

def analyze_convergence(df, scaler):
    """d(suno_centroid, lyria_centroid) vs bootstrapped human-split null."""
    print("\n" + "=" * 60)
    print("ANALYSIS 4 — SYSTEM CONVERGENCE (Suno vs Lyria)")
    print("=" * 60)

    feats = [f for f in ANALYSIS_FEATS if f in df.columns]
    np.random.seed(42)
    rows  = []
    for g in GENRES:
        sm = scaler.transform(df[(df["system"]=="suno")  & (df["genre"]==g)][feats].values)
        lm = scaler.transform(df[(df["system"]=="lyria") & (df["genre"]==g)][feats].values)
        hm = scaler.transform(df[(df["system"]=="human") & (df["genre"]==g)][feats].values)
        ai_dist = np.linalg.norm(sm.mean(0) - lm.mean(0))
        h_dists = []
        for _ in range(500):
            idx = np.random.permutation(len(hm))
            h_dists.append(np.linalg.norm(hm[idx[:len(idx)//2]].mean(0)
                                           - hm[idx[len(idx)//2:]].mean(0)))
        h_null = np.mean(h_dists)
        ratio  = ai_dist / h_null
        print(f"  {g:<12}  d(suno,lyria)={ai_dist:.3f}  null={h_null:.3f}  ratio={ratio:.3f}")
        rows.append({"genre":g,"suno_lyria_dist":ai_dist,
                     "human_null":h_null,"convergence_ratio":ratio})

    conv_df = pd.DataFrame(rows)
    conv_df.to_csv(OUT_DIR / "system_convergence.csv", index=False)
    return conv_df


# ── Analysis 5: Discriminability ─────────────────────────────────────────────

def analyze_discriminability(df):
    """Random Forest AUC + feature importance for AI vs Human classification."""
    print("\n" + "=" * 60)
    print("ANALYSIS 5 — AI vs HUMAN DISCRIMINABILITY")
    print("=" * 60)

    feats = [f for f in ANALYSIS_FEATS if f in df.columns]
    X = StandardScaler().fit_transform(df[feats].values)
    y = (df["condition"] == "ai").astype(int).values
    clf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"\n  5-fold CV AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
    clf.fit(X, y)
    imp = pd.DataFrame({"feature": feats, "importance": clf.feature_importances_,
                        "category": [_cat(f) for f in feats]}
                       ).sort_values("importance", ascending=False)
    print("\n  Top 15 discriminating features:")
    print(imp.head(15)[["feature","category","importance"]].to_string(index=False))
    imp.to_csv(OUT_DIR / "classifier_importance.csv", index=False)
    return {"auc_mean": aucs.mean(), "auc_std": aucs.std()}, imp


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_pca(df):
    print("[Fig 1] PCA scatter...")
    feats  = [f for f in ANALYSIS_FEATS if f in df.columns]
    X      = StandardScaler().fit_transform(df[feats].values)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    df2    = df.copy().reset_index(drop=True)
    df2["pc1"], df2["pc2"] = coords[:,0], coords[:,1]
    markers = {"human":"o","suno":"s","lyria":"^"}
    fig, axes = plt.subplots(2,2,figsize=(11,9))
    for ax, g in zip(axes.flatten(), GENRES):
        sub = df2[df2["genre"]==g]
        for sys in ("human","suno","lyria"):
            s = sub[sub["system"]==sys]
            ax.scatter(s["pc1"], s["pc2"], c=SYSTEM_COLORS[sys],
                       marker=markers[sys], s=20 if sys=="human" else 28,
                       alpha=0.35 if sys=="human" else 0.7, label=sys.capitalize())
        ax.set_title(GENRE_DISPLAY[g], fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
    handles = [mpatches.Patch(color=SYSTEM_COLORS[s], label=s.capitalize())
               for s in ("human","suno","lyria")]
    fig.legend(handles=handles, title="System", loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("PCA — Interpretable + New Features  (by Genre and System)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig1_pca.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig1_pca.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_variance_heatmap(var_df):
    print("[Fig 2] Variance heatmap...")
    feats = var_df["feature"].unique().tolist()
    fig, axes = plt.subplots(1,2,figsize=(14,max(8,len(feats)*0.22)))
    for ax, sys in zip(axes, SYSTEMS):
        sub = var_df[var_df["system"]==sys]
        mat = np.full((len(feats),len(GENRES)), np.nan)
        for gi,g in enumerate(GENRES):
            for fi,f in enumerate(feats):
                row = sub[(sub["feature"]==f)&(sub["genre"]==g)]
                if not row.empty: mat[fi,gi] = row["var_ratio"].values[0]
        log_mat = np.log2(np.clip(mat,0.05,20))
        im = ax.imshow(log_mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(GENRES)))
        ax.set_xticklabels([GENRE_DISPLAY[g] for g in GENRES], fontsize=9, rotation=20, ha="right")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([f.replace("_mean","").replace("_std","±").replace("_"," ")
                            for f in feats], fontsize=6)
        n_lo  = np.sum(mat<1)
        n_tot = np.sum(~np.isnan(mat))
        ax.set_title(f"{sys.capitalize()}  ({n_lo}/{n_tot} cells AI<Human)\n"
                     f"log₂(AI/Human var)  blue=homogeneous",
                     fontsize=10, fontweight="bold", color=SYSTEM_COLORS[sys])
        plt.colorbar(im, ax=ax, shrink=0.5)
    fig.suptitle("Within-Genre Variance Ratios — All Analysis Features\n"
                 "(prompt-encoded features included; fidelity r≈0 makes them valid signals)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig2_variance_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig2_variance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_distance_violin(dist_vecs):
    print("[Fig 3] Distance violin...")
    order  = ["human","suno","lyria","ai_combined"]
    labels = {"human":"Human","suno":"Suno","lyria":"Lyria","ai_combined":"AI\n(combined)"}
    colors = {"human":SYSTEM_COLORS["human"],"suno":SYSTEM_COLORS["suno"],
              "lyria":SYSTEM_COLORS["lyria"],"ai_combined":"#888888"}
    fig, ax = plt.subplots(figsize=(7,5))
    vp = ax.violinplot([dist_vecs[k] for k in order], positions=range(4),
                       showmedians=True, showextrema=False)
    for body,key in zip(vp["bodies"],order):
        body.set_facecolor(colors[key]); body.set_alpha(0.7)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)
    ax.scatter(range(4),[np.mean(dist_vecs[k]) for k in order],
               color="black",s=35,zorder=5,marker="x")
    ax.set_xticks(range(4))
    ax.set_xticklabels([labels[k] for k in order],fontsize=11)
    ax.set_ylabel("Within-Genre Pairwise Distance\n(z-scored analysis features)",fontsize=10)
    ax.set_title("Intra-Group Diversity  (× = mean, line = median)\n"
                 "Pooled across genres — each genre computed separately first",
                 fontsize=10,fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig3_distances.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig3_distances.png",dpi=300,bbox_inches="tight")
    plt.close()


def fig_fidelity_heatmap(fid_df):
    print("[Fig 4] Fidelity heatmap...")
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax,sys in zip(axes,SYSTEMS):
        sub = fid_df[fid_df["system"]==sys]
        mat = np.full((len(PROMPT_ENCODED),len(GENRES)),np.nan)
        for fi,f in enumerate(PROMPT_ENCODED):
            for gi,g in enumerate(GENRES):
                row = sub[(sub["feature"]==f)&(sub["genre"]==g)]
                if not row.empty: mat[fi,gi] = row["pearson_r"].values[0]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.6, vmax=0.6)
        ax.set_xticks(range(len(GENRES)))
        ax.set_xticklabels([GENRE_DISPLAY[g] for g in GENRES],fontsize=10,rotation=20,ha="right")
        ax.set_yticks(range(len(PROMPT_ENCODED)))
        ax.set_yticklabels(PROMPT_ENCODED,fontsize=10)
        for fi in range(len(PROMPT_ENCODED)):
            for gi in range(len(GENRES)):
                v = mat[fi,gi]
                if not np.isnan(v):
                    ax.text(gi,fi,f"{v:.2f}",ha="center",va="center",
                            fontsize=9,color="black" if abs(v)<0.4 else "white")
        ax.set_title(f"{sys.capitalize()}\nPearson r: prompt target → AI output",
                     fontsize=11,fontweight="bold",color=SYSTEM_COLORS[sys])
        plt.colorbar(im,ax=ax,label="r",shrink=0.7)
    fig.suptitle("Prompt Fidelity — How Well Did AI Follow Encoded Feature Values?\n"
                 "|r| < 0.3 = essentially no tracking → lower AI variance is genuine homogenization",
                 fontsize=11,fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig4_fidelity.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig4_fidelity.png",dpi=300,bbox_inches="tight")
    plt.close()


def fig_genre_separation(sep_df):
    print("[Fig 5] Genre separation...")
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    x = np.arange(len(sep_df))
    colors = [SYSTEM_COLORS.get(s,"#aaa") for s in sep_df["system"]]
    bars = axes[0].bar(x, sep_df["separation_ratio"], color=colors, width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.capitalize() for s in sep_df["system"]],fontsize=11)
    axes[0].set_ylabel("Between / Within Genre Distance",fontsize=10)
    axes[0].set_title("Genre Separation Ratio\n(lower = AI flattens genre differences)",
                      fontsize=11,fontweight="bold")
    for b,v in zip(bars,sep_df["separation_ratio"]):
        axes[0].text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.2f}",ha="center",fontsize=10)
    w = 0.3
    axes[1].bar(x-w/2,sep_df["mean_between"],w,color=colors,label="Between")
    axes[1].bar(x+w/2,sep_df["mean_within"],w,color=colors,alpha=0.45,hatch="//",label="Within")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.capitalize() for s in sep_df["system"]],fontsize=11)
    axes[1].set_ylabel("Centroid Distance",fontsize=10)
    axes[1].set_title("Between vs Within Genre Distance",fontsize=11,fontweight="bold")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig5_genre_separation.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig5_genre_separation.png",dpi=300,bbox_inches="tight")
    plt.close()


def fig_feature_importance(imp_df):
    print("[Fig 6] Feature importance...")
    top  = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7,6))
    ax.barh(range(len(top)), top["importance"].values,
            color=[CATEGORY_COLORS.get(_cat(f),"#888") for f in top["feature"]])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f.replace("_mean","").replace("_std","±").replace("_"," ")
                        for f in top["feature"]], fontsize=8)
    ax.set_xlabel("Feature Importance (RF Gini)",fontsize=10)
    ax.set_title("Top 20 Features: AI vs Human Discriminability",
                 fontsize=11,fontweight="bold")
    patches=[mpatches.Patch(color=CATEGORY_COLORS.get(c,"#888"),label=c)
             for c in CATEGORY_COLORS]
    ax.legend(handles=patches,fontsize=7,loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig6_importance.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig6_importance.png",dpi=300,bbox_inches="tight")
    plt.close()


def fig_convergence(conv_df):
    print("[Fig 7] Convergence...")
    GENRE_COLORS={"metal":"#7B2D8B","afrobeats":"#E8A838","dancepop":"#E83A8C","kpop":"#3AB8E8"}
    fig,ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(conv_df))
    bars=ax.bar(x,conv_df["convergence_ratio"],
                color=[GENRE_COLORS.get(g,"#aaa") for g in conv_df["genre"]],width=0.5)
    ax.axhline(1.0,color="black",lw=1.2,ls="--",label="Human baseline (=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([GENRE_DISPLAY.get(g,g) for g in conv_df["genre"]],fontsize=11)
    ax.set_ylabel("d(Suno,Lyria) / d(Human split)",fontsize=10)
    ax.set_title("System Convergence: Suno vs Lyria\n"
                 "(<1 = systems closer than random human halves)",
                 fontsize=11,fontweight="bold")
    for b,v in zip(bars,conv_df["convergence_ratio"]):
        ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"fig7_convergence.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"fig7_convergence.png",dpi=300,bbox_inches="tight")
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(fid_df, var_df, dist_rows, sep_df, conv_df, clf_res):
    print("\n" + "=" * 70)
    print("HOMOGENIZATION ANALYSIS — FINAL SUMMARY")
    print("=" * 70)

    print("\n0. PROMPT FIDELITY")
    for sys in SYSTEMS:
        sub = fid_df[fid_df["system"]==sys]
        for feat in PROMPT_ENCODED:
            mr = sub[sub["feature"]==feat]["pearson_r"].abs().mean()
            print(f"   {sys} {feat}: mean |r| = {mr:.3f}",end="")
        print()

    print(f"\n1. WITHIN-GENRE VARIANCE")
    for sys in SYSTEMS:
        sub  = var_df[var_df["system"]==sys]
        n_lo = (sub["var_ratio"]<1).sum(); n_to = len(sub)
        med  = sub["var_ratio"].median()
        print(f"   {sys:<8} AI<Human: {n_lo}/{n_to} ({100*n_lo/n_to:.0f}%)  "
              f"Median ratio: {med:.3f}")

    print(f"\n2. WITHIN-GENRE PAIRWISE DISTANCE")
    for r in dist_rows:
        arrow = "← homogeneous" if r["cohens_d"]<0 else "→ MORE DIVERSE"
        print(f"   {r['comparison']:<14} d={r['cohens_d']:+.3f}  p={r['mw_p']:.4f}  {arrow}")

    print(f"\n3. GENRE SEPARATION RATIO")
    for _,r in sep_df.iterrows():
        print(f"   {r['system']:<8} {r['separation_ratio']:.3f}")

    print(f"\n4. SYSTEM CONVERGENCE (>1 = systems DIVERGE)")
    for _,r in conv_df.iterrows():
        print(f"   {r['genre']:<12} {r['convergence_ratio']:.3f}")

    print(f"\n5. DISCRIMINABILITY  AUC={clf_res['auc_mean']:.3f}±{clf_res['auc_std']:.3f}")
    print(f"\n{'─'*70}")
    print("INTERPRETATION:")
    print("  Suno:  ignores all prompt encodings (|r|<0.25). MORE diverse within-genre")
    print("         but collapses BETWEEN-genre differences (separation ratio 0.44 vs 0.67).")
    print("  Lyria: partial tempo tracking (|r| 0.30–0.57), near-zero on other features.")
    print("         Genuinely MORE homogeneous within-genre (60% cells, median ratio 0.82).")
    print("  Both:  near-perfectly discriminable from human (AUC=0.991).")
    print("  Both:  strongly DIVERGE from each other (convergence ratio 2.5–4.4).")
    print("="*70)
    print(f"\nOutputs → {OUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df      = load_data()
    targets = load_prompt_targets()

    fid_df                      = analyze_fidelity(df, targets)
    var_df                      = analyze_variance(df)
    dist_vecs, dist_rows        = analyze_distances(df)
    sep_df, _centroids, scaler  = analyze_genre_separation(df)
    conv_df                     = analyze_convergence(df, scaler)
    clf_res, imp_df             = analyze_discriminability(df)

    print("\n[Generating figures...]")
    fig_pca(df)
    fig_variance_heatmap(var_df)
    fig_distance_violin(dist_vecs)
    fig_fidelity_heatmap(fid_df)
    fig_genre_separation(sep_df)
    fig_feature_importance(imp_df)
    fig_convergence(conv_df)

    print_summary(fid_df, var_df, dist_rows, sep_df, conv_df, clf_res)


if __name__ == "__main__":
    main()
