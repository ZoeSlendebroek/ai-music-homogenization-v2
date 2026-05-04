"""
7_presentation_figures.py

Five clean, professor-ready figures + one results summary document.
Replaces the old fig1–fig3 and removes all diagnostic figures.

Figure layout:
  fig1  Prompt fidelity heatmap (Suno vs Lyria)
  fig2  Within-genre diversity: pairwise distance ratio vs human baseline
  fig3  Genre separation: Suno collapses genre boundaries
  fig4  AI discriminability: top features driving AUC=0.991
  fig5  System divergence: Suno and Lyria are further apart than random human halves
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
FEAT    = ROOT / "data" / "features" / "all_features.csv"
OUT     = ROOT / "data" / "analysis"
OUT.mkdir(exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────
C_HUMAN = "#4C72B0"
C_SUNO  = "#DD4949"
C_LYRIA = "#2CA02C"

GENRES      = ["metal", "afrobeats", "dancepop", "kpop"]
GENRE_LABEL = {"metal": "Heavy Metal", "afrobeats": "Afrobeats",
               "dancepop": "Dance Pop", "kpop": "K-pop"}

PRODUCTION     = ["rms_mean","rms_std","rms_var","dynamic_range_db","crest_factor","percussive_ratio"]
PROMPT_ENCODED = ["tempo","harmonic_ratio","onset_density","self_similarity_mean"]
PROMPT_LABEL   = {"tempo": "Tempo", "harmonic_ratio": "Harmonic ratio",
                  "onset_density": "Onset density", "self_similarity_mean": "Self-similarity"}

# ── helpers ────────────────────────────────────────────────────────────────
def savefig(name):
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:   ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=11)

def analysis_features(df):
    all_feats = [c for c in df.columns
                 if c not in ("filename","system","genre","condition",
                              "version_note","spotify_id","artist","title")
                 and c not in PRODUCTION]
    return all_feats

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Prompt Fidelity
# ══════════════════════════════════════════════════════════════════════════
def fig1_fidelity(df):
    print("Fig 1: Prompt fidelity …")

    # load prompt targets
    prompts_dir = ROOT / "data" / "prompts"
    targets = []
    for g in GENRES:
        for fname in (f"{g}_exp2_prompts.csv",):
            p = prompts_dir / fname
            if p.exists():
                tmp = pd.read_csv(p)[["spotify_id"] + PROMPT_ENCODED]
                tmp["genre"] = g
                targets.append(tmp)
    targets = pd.concat(targets, ignore_index=True)
    targets.columns = ["spotify_id"] + [f"t_{f}" for f in PROMPT_ENCODED] + ["genre"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, system, color, label in [
        (axes[0], "suno",  C_SUNO,  "Suno"),
        (axes[1], "lyria", C_LYRIA, "Lyria"),
    ]:
        mat = np.full((len(PROMPT_ENCODED), len(GENRES)), np.nan)
        for gi, g in enumerate(GENRES):
            ai = df[(df["system"] == system) & (df["genre"] == g)].copy()
            # extract spotify_id from filename (e.g. "ABCDE_suno.mp3" → "ABCDE")
            ai["spotify_id"] = ai["filename"].str.rsplit("_", n=1).str[0]
            ai = ai.merge(targets[targets["genre"] == g][["spotify_id"] + [f"t_{f}" for f in PROMPT_ENCODED]],
                          on="spotify_id", how="inner")
            for fi, feat in enumerate(PROMPT_ENCODED):
                if f"t_{feat}" in ai.columns and feat in ai.columns and len(ai) > 2:
                    mat[fi, gi] = ai[feat].corr(ai[f"t_{feat}"])

        vmax = 0.7
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(GENRES)))
        ax.set_xticklabels([GENRE_LABEL[g] for g in GENRES], fontsize=10)
        ax.set_yticks(range(len(PROMPT_ENCODED)))
        ax.set_yticklabels([PROMPT_LABEL[f] for f in PROMPT_ENCODED], fontsize=10)

        for fi in range(len(PROMPT_ENCODED)):
            for gi in range(len(GENRES)):
                v = mat[fi, gi]
                if not np.isnan(v):
                    ax.text(gi, fi, f"{v:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if abs(v) > 0.4 else "black")

        ax.set_title(label, fontsize=13, fontweight="bold", color=color, pad=8)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    fig.suptitle("Prompt Fidelity — Does AI Follow the Encoded Feature Values?\n"
                 "Pearson r: prompt target → AI output  (|r| < 0.3 = essentially no tracking)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("fig1_fidelity")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Within-genre diversity (distance ratio)
# ══════════════════════════════════════════════════════════════════════════
def fig2_diversity(df):
    print("Fig 2: Within-genre diversity …")

    feats   = analysis_features(df)
    scaler  = StandardScaler()
    df2     = df.copy()
    df2[feats] = scaler.fit_transform(df[feats].fillna(0))

    ratios_suno  = []
    ratios_lyria = []

    for g in GENRES:
        human = df2[(df2["genre"] == g) & (df2["condition"] == "human")][feats].values
        suno  = df2[(df2["genre"] == g) & (df2["system"]    == "suno" )][feats].values
        lyria = df2[(df2["genre"] == g) & (df2["system"]    == "lyria")][feats].values

        dh = cdist(human, human, "euclidean")
        ds = cdist(suno,  suno,  "euclidean")
        dl = cdist(lyria, lyria, "euclidean")

        iu = np.triu_indices(len(human), k=1)
        dh_vals = dh[iu]
        iu_s = np.triu_indices(len(suno), k=1)
        iu_l = np.triu_indices(len(lyria), k=1)

        ratios_suno.append(np.mean(ds[iu_s]) / np.mean(dh_vals))
        ratios_lyria.append(np.mean(dl[iu_l]) / np.mean(dh_vals))

    x      = np.arange(len(GENRES))
    width  = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_s = ax.bar(x - width/2, ratios_suno,  width, color=C_SUNO,  label="Suno",  alpha=0.88, zorder=3)
    bars_l = ax.bar(x + width/2, ratios_lyria, width, color=C_LYRIA, label="Lyria", alpha=0.88, zorder=3)

    ax.axhline(1.0, color=C_HUMAN, linewidth=2, linestyle="--", label="Human baseline (= 1.0)", zorder=4)

    for bar in list(bars_s) + list(bars_l):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([GENRE_LABEL[g] for g in GENRES], fontsize=11)
    ax.set_ylim(0, max(max(ratios_suno), max(ratios_lyria)) + 0.25)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.fill_between([-0.5, len(GENRES)-0.5], 0, 1.0,
                    color=C_HUMAN, alpha=0.06, zorder=0)

    style_ax(ax,
             title="Within-Genre Acoustic Diversity  (AI / Human pairwise distance)\n"
                   "Values > 1 = more diverse than human  |  < 1 = more homogeneous",
             ylabel="Diversity ratio  (AI / Human)")
    ax.spines["bottom"].set_visible(True)
    plt.tight_layout()
    savefig("fig2_diversity")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Genre separation
# ══════════════════════════════════════════════════════════════════════════
def fig3_genre_separation(_df):
    print("Fig 3: Genre separation …")

    p = ROOT / "data" / "analysis" / "genre_separation.csv"
    sep = pd.read_csv(p)
    sep["system"] = sep["system"].str.capitalize()
    sep.loc[sep["system"] == "Human", "system"] = "Human"

    labels = ["Human", "Suno", "Lyria"]
    colors = [C_HUMAN, C_SUNO, C_LYRIA]
    ratios = [sep.loc[sep["system"].str.lower() == s.lower(), "separation_ratio"].values[0]
              for s in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, ratios, color=colors, alpha=0.88, width=0.5, zorder=3)

    for bar, v in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(ratios) + 0.12)
    style_ax(ax,
             title="Genre Separation Ratio  (between-genre / within-genre distance)\n"
                   "Lower = genre boundaries are weaker — AI music sounds more genre-agnostic",
             ylabel="Separation ratio")
    ax.annotate("Suno collapses genre\ndistinctions (−36%)",
                xy=(1, ratios[1]), xytext=(1.35, ratios[1] + 0.06),
                fontsize=9, color=C_SUNO,
                arrowprops=dict(arrowstyle="->", color=C_SUNO, lw=1.4))
    plt.tight_layout()
    savefig("fig3_genre_separation")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Feature importance (AI vs human discriminability)
# ══════════════════════════════════════════════════════════════════════════
def fig4_importance():
    print("Fig 4: Feature importance …")

    p = ROOT / "data" / "analysis" / "classifier_importance.csv"
    if not p.exists():
        print("  classifier_importance.csv not found — skipping")
        return

    imp = pd.read_csv(p).sort_values("importance", ascending=False).head(12)

    CAT_COLOR = {
        "Timbral (MFCC)":  "#E07B39",
        "Rhythmic":        "#2E8B57",
        "Spectral":        "#4C72B0",
        "Structural":      "#9467BD",
        "Intra-track":     "#17BECF",
        "Prompt-encoded":  "#AAAAAA",
        "Key / Tonal":     "#D4AC0D",
        "Dynamic":         "#C0392B",
    }

    # clean labels
    def clean(s):
        s = s.replace("mfcc_delta2", "MFCC Δ² ").replace("mfcc_delta", "MFCC Δ ")
        s = s.replace("mfcc_0", "MFCC 0").replace("mfcc_2", "MFCC 2")
        s = s.replace("mfcc_5", "MFCC 5").replace("mfcc_6", "MFCC 6")
        s = s.replace("_mean","").replace("_std","±").replace("_cv"," CV")
        s = s.replace("ioi","IOI").replace("onset_density","Onset density")
        s = s.replace("zero_crossing_rate","Zero crossing rate")
        s = s.replace("onset_strength","Onset strength")
        s = s.replace("self_similarity","Self-similarity")
        s = s.replace("spectral_contrast","Spectral contrast")
        s = s.replace("spectral_centroid","Spectral centroid")
        return s.strip()

    labels = [clean(f) for f in imp["feature"]]
    colors = [CAT_COLOR.get(c, "#888888") for c in imp["category"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(range(len(imp)), imp["importance"].values[::-1],
                   color=colors[::-1], alpha=0.88, zorder=3)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Feature importance (Random Forest Gini)", fontsize=11)

    # legend
    seen = {}
    for cat, col in CAT_COLOR.items():
        if cat in imp["category"].values:
            seen[cat] = col
    patches = [mpatches.Patch(color=c, label=l) for l, c in seen.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right", framealpha=0.9)

    style_ax(ax,
             title=f"Top {len(imp)} Features — AI vs Human Discriminability  (AUC = 0.991 ± 0.003)\n"
                   "What makes AI music acoustically identifiable?")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig("fig4_importance")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — System divergence
# ══════════════════════════════════════════════════════════════════════════
def fig5_convergence():
    print("Fig 5: System convergence …")

    p = ROOT / "data" / "analysis" / "system_convergence.csv"
    if not p.exists():
        print("  system_convergence.csv not found — skipping")
        return

    conv = pd.read_csv(p)
    genre_order = ["metal", "afrobeats", "dancepop", "kpop"]
    conv["genre"] = pd.Categorical(conv["genre"], categories=genre_order, ordered=True)
    conv = conv.sort_values("genre")

    labels = [GENRE_LABEL[g] for g in conv["genre"]]
    ratios = conv["convergence_ratio"].values

    GENRE_COLORS = ["#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, ratios, color=GENRE_COLORS, alpha=0.88, width=0.55, zorder=3)

    ax.axhline(1.0, color="black", linewidth=1.8, linestyle="--", zorder=4)
    ax.text(3.4, 1.06, "Human baseline\n(= 1.0)", fontsize=9, ha="right",
            va="bottom", color="black")

    for bar, v in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(ratios) + 0.7)
    style_ax(ax,
             title="Suno vs Lyria — Do AI Systems Converge on a Shared Sound?\n"
                   "Ratio = d(Suno, Lyria) / d(Human split)  |  > 1 = systems DIVERGE",
             ylabel="Convergence ratio")

    ax.annotate("2.7–4.6× further apart\nthan random human halves",
                xy=(0, ratios[0]), xytext=(0.5, ratios[0] + 0.4),
                fontsize=9.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    plt.tight_layout()
    savefig("fig5_convergence")


# ══════════════════════════════════════════════════════════════════════════
# RESULTS DOCUMENT
# ══════════════════════════════════════════════════════════════════════════
def write_results_doc(df):
    print("Writing results document …")

    feats = analysis_features(df)
    scaler = StandardScaler()
    df2 = df.copy()
    df2[feats] = scaler.fit_transform(df[feats].fillna(0))

    # distance ratios
    dist_rows = []
    for g in GENRES:
        human = df2[(df2["genre"] == g) & (df2["condition"] == "human")][feats].values
        suno  = df2[(df2["genre"] == g) & (df2["system"]    == "suno" )][feats].values
        lyria = df2[(df2["genre"] == g) & (df2["system"]    == "lyria")][feats].values
        dh = np.mean(cdist(human, human, "euclidean")[np.triu_indices(len(human), k=1)])
        ds = np.mean(cdist(suno,  suno,  "euclidean")[np.triu_indices(len(suno),  k=1)])
        dl = np.mean(cdist(lyria, lyria, "euclidean")[np.triu_indices(len(lyria), k=1)])
        dist_rows.append((GENRE_LABEL[g], round(dh,2), round(ds,2), round(dl,2),
                          round(ds/dh,3), round(dl/dh,3)))

    doc = f"""# Experiment 2 — Results Summary for Supervisor Meeting

**Project:** AI Music Homogenization
**Corpora:** 4 genres × 100 human tracks + 100 Suno + 100 Lyria = 1,200 audio clips
**Genres:** Heavy Metal · Afrobeats · Dance Pop · K-pop
**Features:** 72 acoustic features (spectral, timbral, rhythmic, harmonic, structural, dynamic)

---

## Core Question
Do AI music generation systems (Suno, Lyria) produce more homogeneous music than humans,
and do they converge on a shared "AI sound"?

---

## Finding 1 — AI Systems Do Not Follow Prompt Encodings  *(Fig 1)*

Each AI track was generated from a prompt encoding 4 acoustic features of a specific human track
(tempo, harmonic ratio, onset density, self-similarity). Track-level Pearson r between target and output:

| Feature | Suno mean |r| | Lyria mean |r| |
|---|---|---|
| Tempo | 0.15 | **0.42** |
| Harmonic ratio | 0.15 | 0.17 |
| Onset density | 0.08 | **0.32** |
| Self-similarity | 0.08 | 0.16 |

**Suno** ignores all encoded features (|r| < 0.25 for everything, including negative
correlations for harmonic ratio in Metal).
**Lyria** shows moderate tempo and onset density tracking but near-zero fidelity for
harmonic content and structure.

**Why this matters:** Because AI systems ignore the prompts, any variance compression
in AI outputs is genuine homogenization — the model clustering to its own preferred
values, not reflecting the diversity we asked for.

---

## Finding 2 — Lyria Homogenizes Within-Genre; Suno Does the Opposite  *(Fig 2)*

Within-genre pairwise acoustic distance (AI / Human ratio — Human baseline = 1.0):

| Genre | Human | Suno | Lyria | Suno/Human | Lyria/Human |
|---|---|---|---|---|---|
"""
    for row in dist_rows:
        doc += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n"

    doc += f"""
**Lyria:** significantly MORE homogeneous (Cohen's d = −0.265, p < 0.0001).
**Suno:** significantly MORE diverse (Cohen's d = +0.646).

Variance analysis (Levene + BH-FDR correction, 264 feature×genre cells):
- Suno AI < Human in only **17%** of cells (median variance ratio 1.67)
- Lyria AI < Human in **58%** of cells (median variance ratio 0.84)

*Note: Suno's higher diversity is not musical diversity — it reflects structureless
acoustic variation. Suno outputs wander frame-to-frame in timbral space without
genre-coherent musical development.*

---

## Finding 3 — Suno Collapses Genre Boundaries  *(Fig 3)*

Genre separation ratio = between-genre centroid distance / within-genre spread.
Higher = genres are more distinct.

| System | Separation ratio |
|---|---|
| Human | 0.669 |
| **Suno** | **0.429** (−36%) |
| Lyria | 0.676 (≈ human) |

Suno produces music that sounds genre-agnostic: its Heavy Metal and K-pop are more
similar to each other than human Heavy Metal and K-pop are. This is homogenization
at the categorical level — an erasure of genre identity, not just variance compression.

Lyria, by contrast, preserves genre distinctiveness at the human level.

---

## Finding 4 — AI Music is Near-Perfectly Discriminable from Human  *(Fig 4)*

5-fold cross-validated Random Forest:
**AUC = 0.991 ± 0.003**

Top discriminating features:
1. MFCC Δ² mean — *timbral change rate over time (AI timbre stays flat)*
2. MFCC 0 mean — *overall energy character*
3. IOI mean & std — *inter-onset interval (AI rhythms are metronomic)*
4. MFCC Δ mean
5. Onset strength CV — *rhythmic energy variation within clip*

**Interpretation:** AI music is not identified by sounding "wrong" for a genre — it is
identified by lacking the natural within-track evolution of human music. AI timbre stays
constant throughout; human timbre develops. AI rhythms are perfectly metronomic; human
rhythms breathe.

*Caveat: the AI tracks are instrumental; many human tracks contain vocals. Vocals
affect MFCC and spectral features strongly. The AUC may partially reflect this
vocal/instrumental difference, not only AI-specific properties.*

---

## Finding 5 — Suno and Lyria Do Not Converge on a Shared AI Sound  *(Fig 5)*

Convergence ratio = d(Suno centroid, Lyria centroid) / d(random human split).
Values > 1 mean the two AI systems are *further apart* than random halves of the human corpus.

| Genre | Convergence ratio |
|---|---|
| Heavy Metal | 2.70 |
| Afrobeats | 4.64 |
| Dance Pop | 4.38 |
| K-pop | 4.47 |

The two systems are **2.7–4.6× more different from each other than human variation**.
There is no single "AI sound." Suno and Lyria have strongly distinct sonic signatures.

---

## Summary

| Claim | Suno | Lyria |
|---|---|---|
| Homogenizes within-genre? | No — MORE diverse | **Yes** |
| Preserves genre boundaries? | **No — collapses them** | Yes |
| Discriminable from human? | **Yes (AUC 0.991)** | **Yes (AUC 0.991)** |
| Converges with other AI system? | **No — strongly diverges** | **No — strongly diverges** |
| Follows prompt encodings? | Essentially no | Partially (tempo, onset) |

**Homogenization is real but system-specific.** Lyria narrows within-genre acoustic
space; Suno erases categorical genre identity. Both produce music that is structurally
distinct from human — timbrally rigid and rhythmically flat — but they do not converge
on a shared sound.

---

## Key Limitations

1. **Vocal/instrumental confound** — AI tracks are instrumental; human corpus contains vocals.
   This affects spectral and MFCC features and may inflate the AUC.
2. **Only 2 AI systems** — Udio was planned but not collected.
3. **30-second middle clips** — structural diversity across full tracks not captured.
4. **Librosa BPM estimation** is noisy (octave-doubling errors affect tempo features).
5. **No perceptual validation** — acoustic homogenization ≠ perceived homogenization.

---

*Figures: fig1_fidelity · fig2_diversity · fig3_genre_separation · fig4_importance · fig5_convergence*
*Full methodology and observations: OBSERVATIONS.md*
"""

    out_path = ROOT / "RESULTS_SUMMARY.md"
    out_path.write_text(doc)
    print(f"  saved RESULTS_SUMMARY.md")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import shutil

    # delete old / diagnostic figures
    old = [
        "fig1_pca", "fig2_variance_heatmap", "fig2_variance_forest",
        "fig3_distances", "fig4_fidelity", "fig4_genre_separation",
        "fig5_genre_separation", "fig5_importance",
        "fig6_importance", "fig6_convergence", "fig7_convergence",
    ]
    for name in old:
        for ext in ("png","pdf"):
            f = OUT / f"{name}.{ext}"
            if f.exists():
                f.unlink()

    diag = OUT / "diagnostics"
    if diag.exists():
        shutil.rmtree(diag)
        print("  deleted diagnostics/")

    df = pd.read_csv(FEAT)
    print(f"Loaded {len(df)} tracks\n")

    fig1_fidelity(df)
    fig2_diversity(df)
    fig3_genre_separation(None)
    fig4_importance()
    fig5_convergence()
    write_results_doc(df)

    print("\nDone. 5 figures + RESULTS_SUMMARY.md saved.")
