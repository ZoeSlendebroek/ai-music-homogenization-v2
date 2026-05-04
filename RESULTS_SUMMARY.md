# Experiment 2 — Results Summary for Supervisor Meeting

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
| Heavy Metal | 9.43 | 9.47 | 7.29 | 1.004 | 0.773 |
| Afrobeats | 8.81 | 13.94 | 7.33 | 1.582 | 0.832 |
| Dance Pop | 7.64 | 9.72 | 8.55 | 1.272 | 1.119 |
| K-pop | 7.99 | 11.04 | 7.77 | 1.382 | 0.972 |

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
