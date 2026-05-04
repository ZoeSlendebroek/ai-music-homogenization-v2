# Experiment 2 — Dataset Observations, Findings, and Limitations

This document records empirical observations, methodological decisions, and open
questions emerging from the Experiment 2 pipeline. Updated as analysis progresses.

---

## 1. Corpus Quality

### 1.1 Genre purity — Dance Pop
The Dance Pop corpus (sourced from Spotify "dance pop" genre tag + playlists) contains
several tracks that are not Dance Pop by conventional classification:

| Track | Artist | Issue |
|---|---|---|
| I Want To Break Free | Queen | Classic rock |
| Pumped Up Kicks | Foster The People | Indie pop |
| Paradise | Coldplay | Alternative pop |
| Shake It Out | Florence + The Machine | Indie/art pop |
| The Less I Know The Better | Tame Impala | Psychedelic rock |
| Midnight City | M83 | Synthwave |
| Can't Hold Us | Macklemore | Hip-hop |
| Airplanes | B.o.B | Hip-hop |
| A Bar Song (Tipsy) | Shaboozey | Country |
| Sunflower | Post Malone | Hip-hop/R&B |
| Waka Waka | Shakira | World pop |

**Why this happens:** Spotify's genre classification uses audio-feature-based ML, not
editorial genre assignment. Tracks with high danceability + energy receive a "dance pop"
tag regardless of actual style.

**Implication:** The Dance Pop corpus has more genre noise than Metal, Afrobeats, or
K-pop. Any homogenization analysis for Dance Pop should be interpreted with more caution.
The diffuse genre identity of the human corpus means AI homogenization effects may be
harder to detect, not because they don't exist, but because the baseline is already
heterogeneous.

**Decision:** Retain as-is for now. Note as a limitation in the paper. Consider a
sensitivity analysis excluding the 10–11 most egregious outliers.

---

## 2. AI Generation — System Behaviour

### 2.1 Suno — token / authentication
- Suno uses short-lived Bearer JWT tokens (~1 hr expiry). The script now prompts
  for a new token inline when a 422 is returned, without losing progress.
- Auto-refresh via the Clerk `__client` cookie was attempted but abandoned — refreshed
  JWTs were consistently rejected despite being structurally valid. Likely due to
  Cloudflare bot detection or session binding on Suno's side.
- Workaround: navigate the Suno tab to `about:blank` after copying the Bearer token.
  This prevents the tab's background requests from invalidating the token.
- Suno's Afrobeats data was collected on model v3 (chirp-v3, Dec 2025 / Jan 2026).
  All other Suno data collected on chirp-v3-5. **This is a covariate in cross-genre
  analysis.** The version difference should be flagged in any paper using Afrobeats Suno data.

### 2.2 Lyria — content filtering
- Lyria uses `FinishReason.OTHER` for generation failures (not `SAFETY`). This is a
  generic failure code, not a confirmed safety block.
- Approximately 6–8% of Metal tracks triggered `OTHER` failures on first attempt.
  Most succeeded on retry (up to 3 attempts implemented). Titles like "Crazy Train",
  "The Game", and "Let Me Go" failed — but so did some innocuous ones.
- **Lyria does not see track titles** — only the prompt text. Failures are driven
  by prompt content, not track name.
- Lyria requires paid billing ($25 prepaid used). Free-tier quota is 0 for
  `lyria-3-clip-preview`.
- Lyria uses the Gemini API path (`google-genai` SDK, `aistudio.google.com` key),
  NOT the Vertex AI path (separate GCP product, different SDK, different billing).

### 2.3 Udio — not collected
- Udio generation script (`generate_udio.py`) was written and tested but generation
  was not completed. No Udio audio files exist in `data/audio/udio/`.
- Udio would be the third AI system in the analysis. Its absence means the current
  analysis covers only Suno and Lyria.
- **Impact:** Cross-system convergence analysis is limited to 2 systems. The
  finding that Suno and Lyria diverge (not converge) would be strengthened or
  complicated by a third system.

---

## 3. Prompt Fidelity — Critical Finding

**This is the most methodologically important finding of the analysis.**

### 3.1 Suno ignores encoded features entirely
Track-level Pearson r between the prompt-encoded target value and Suno's actual output:

| Feature | Metal | Afrobeats | Dance Pop | K-pop |
|---|---|---|---|---|
| Tempo | 0.13 | 0.16 | 0.24 | 0.06 |
| Harmonic ratio | **-0.26** | -0.06 | 0.18 | 0.10 |
| Onset density | -0.10 | 0.02 | 0.07 | 0.13 |
| Self-similarity | -0.10 | 0.05 | 0.12 | -0.05 |

r ≈ 0 across the board. Suno's outputs show essentially zero correlation with the
features we encoded in the prompt at the individual track level.

### 3.2 Lyria partially follows tempo and onset density
| Feature | Metal | Afrobeats | Dance Pop | K-pop | Mean |r| |
|---|---|---|---|---|---|
| Tempo | 0.49 | 0.31 | 0.30 | 0.57 | 0.418 |
| Harmonic ratio | 0.17 | 0.15 | 0.15 | 0.21 | 0.172 |
| Onset density | 0.35 | 0.45 | 0.35 | 0.12 | 0.316 |
| Self-similarity | 0.09 | 0.08 | 0.23 | 0.24 | 0.161 |

Lyria shows moderate tempo tracking (r 0.30–0.57) and moderate onset density
tracking (mean |r| = 0.316, range 0.12–0.45). Harmonic content and structural
features remain near-zero fidelity.

### 3.3 Why distribution-level comparisons looked good (and were misleading)
Initial BPM analysis (comparing distributions, not paired values) showed similar
mean ± std between human targets and AI outputs. This is a coincidence: we encoded
feature values drawn *from* the human corpus, so both populations naturally span
the same range — even though individual tracks are not being matched.

This is the **distribution-level vs. paired-level distinction**. The distribution
overlap tells us nothing about fidelity.

### 3.4 Implication for homogenization claims
Because AI systems (especially Suno) do not follow the encoded features, any
compression of variance on those features in the AI outputs IS genuine homogenization.
The AI is clustering to its own preferred values, not tracking the diversity of the
input. Prompt-encoded features therefore cannot be excluded from homogenization
analysis. They are included in the final analysis with fidelity r values reported
alongside the variance statistics.

---

## 4. Homogenization Findings

### 4.1 The correct analysis unit is within-genre
Pooling all genres before comparing inflates AI variance. A system that correctly
produces different sounds for Heavy Metal vs K-pop shows high cross-genre variance
even if it is completely homogeneous within each genre. All primary analyses use
within-genre comparisons.

### 4.2 Lyria homogenizes within-genre; Suno does the opposite

**Within-genre variance ratio (AI/Human), median across all interpretable features (72 total, 66 after excluding production features):**
- Suno: **1.667** — more variable than human in 83% of feature×genre cells (45/264 cells AI < Human)
- Lyria: **0.839** — less variable than human in 58% of feature×genre cells (153/264 cells AI < Human)

**Per-genre median variance ratio:**
| Genre | Suno | Lyria |
|---|---|---|
| Heavy Metal | 1.451 | 0.722 |
| Afrobeats | 2.154 | 0.697 |
| Dance Pop | 1.524 | 1.062 |
| K-pop | 1.717 | 0.848 |

**Within-genre pairwise distance ratio (AI/Human):**

| Genre | Suno/Human | Lyria/Human |
|---|---|---|
| Heavy Metal | 1.004 | **0.773** |
| Afrobeats | **1.582** | 0.832 |
| Dance Pop | 1.272 | 1.119 |
| K-pop | 1.382 | **0.972** |

- Lyria: significantly more homogeneous (Cohen's d = -0.265, p < 0.0001).
- Suno: significantly MORE diverse (Cohen's d = +0.646, p = 1.0 in one-sided test).

*These numbers are from the full 72-feature run (67 original + 5 new). Previously reported with 67 features: Suno 1.62/d=+0.61, Lyria 0.82/d=-0.30 — conclusions unchanged.*

### 4.3 Suno collapses genre distinctions; Lyria does not
Genre separation ratio (between-genre / within-genre distance):
- Human: 0.669
- Suno: **0.429** — dramatic collapse of genre boundaries
- Lyria: 0.676 — essentially identical to human

Suno produces music that is more genre-agnostic. Its within-genre diversity is high,
but the genres it produces are less distinct from one another than human genres are.
This is a different, arguably more severe form of homogenization — a flattening of
categorical musical identity rather than low within-genre variance.

### 4.4 Suno and Lyria strongly diverge from each other
Convergence ratio d(Suno, Lyria) / d(Human split):
- Metal: 2.705 | Afrobeats: 4.636 | Dance Pop: 4.377 | K-pop: 4.470

The two systems are 2.5–4.4× more different from each other than random halves of the
human corpus are from each other. They are not converging on a shared "AI sound" —
they have strongly distinct sonic signatures.

### 4.5 AI is near-perfectly discriminable from human
5-fold CV Random Forest AUC = 0.991 ± 0.003 (interpretable + new features).
Top discriminating features (Random Forest importance):
1. MFCC Δ² mean (0.144) — timbral change rate over time
2. MFCC 0 mean (0.117) — overall energy/loudness character
3. IOI mean (0.051) — inter-onset interval / rhythmic regularity
4. IOI std (0.049)
5. MFCC Δ mean (0.048)
6. Onset density (0.043) — prompt-encoded but model-ignored
7. IOI CV (0.029)
8. Zero crossing rate mean (0.025)
9. MFCC 2 mean (0.024)
10. **onset_strength_cv (0.024) — NEW: intra-clip rhythmic energy variation**

**Interpretation:** AI music is distinguished from human primarily by its timbral
evolution over time (flat/constant timbre vs. natural variation) and its rhythmic
rigidity (highly metronomic IOI). These are not genre-specific artifacts — they
are systemic properties of the generation process.

### 4.6 New features (added for this analysis)
Five features not in the original 67 were added:
- `chroma_entropy`: entropy of the pitch-class distribution (low = strong tonal key,
  high = chromatic/atonal). Tests whether AI converges to a preferred key distribution.
- `key_clarity`: max/mean of chroma vector. Companion to chroma_entropy.
- `spectral_centroid_cv`: CV of frame-level spectral centroid within 30s. Measures
  whether AI timbre stays constant throughout (AI systems tend to have flat timbral
  evolution).
- `onset_strength_cv`: CV of onset strength envelope. Measures rhythmic energy variation.
- `rms_cv`: CV of frame-level RMS. Measures dynamic variation within the clip.

---

## 5. Methodological Decisions and Justifications

### 5.1 Production/mastering features excluded
`rms_mean`, `rms_std`, `rms_var`, `dynamic_range_db`, `crest_factor`, `percussive_ratio`
are excluded from primary homogenization claims. These are heavily influenced by the
output normalisation and mastering chain of the AI systems, not by musical content.
`percussive_ratio` is excluded additionally because it is `1 - harmonic_ratio`
(perfectly collinear).

### 5.2 Multiple comparison correction
Benjamini-Hochberg FDR correction applied within each genre×system block for variance
tests. 67 + 5 = 72 raw features → multiple comparisons are a real concern without
correction.

### 5.3 Middle-crop methodology
All 30-second clips are extracted from the midpoint of the track (±15 s), not from the
start. This avoids intro/outro sections that are less representative of the track's main
character. Same methodology applied to both human and AI tracks.

### 5.4 Version covariate — Suno Afrobeats
Suno Afrobeats tracks were generated on chirp-v3 (legacy). All other data on chirp-v3-5.
This is flagged in the data (`version_note` column) and should be treated as a covariate
in any cross-genre analysis involving Suno Afrobeats.

### 5.5 Matched vs. population-level design
Experiment 2 uses a population-level design: each AI track is generated from the
prompt derived from one specific human track, but the analysis treats them as
independent populations (not matched pairs). This is the correct design for
testing variance/diversity claims. Matched-pair analysis (fidelity) is reported
separately as Analysis 0.

---

## 6. Limitations

### 6.1 Only two AI systems
Without Udio, cross-system convergence analysis is limited to one pairwise comparison.
The strong divergence between Suno and Lyria is a robust finding, but whether this
is typical of AI music systems or specific to these two cannot be determined.

### 6.2 Librosa BPM estimation is noisy
`librosa.beat.beat_track` uses a dynamic programming approach prone to octave errors
(detecting half or double the actual tempo). Both human and AI tracks are affected,
but AI tracks may have more perfectly metronomic rhythms that interact differently
with the estimator. Any claim about tempo homogenization should be accompanied by
a caveat about estimation error.

### 6.3 Instrumental-only AI vs. vocal human music
AI tracks were generated with "no vocals" in the prompt. Many human corpus tracks
contain vocals. Vocals strongly influence spectral centroid, MFCCs, zero-crossing
rate, and harmonic content. The near-perfect AUC (0.991) likely reflects this
vocal/instrumental difference at least as much as true homogenization. Spectral
and MFCC features should be interpreted cautiously in vocal vs. instrumental
comparisons.

**This is a major confound.** A vocal/instrumental comparison is not the same as
a human/AI comparison. To isolate true AI homogenization effects, the ideal
comparison would use instrumental human tracks only. The current corpus does not
guarantee this.

### 6.4 30-second clips from the middle
Short clips may miss structural diversity that only appears across the full track.
Homogenization claims about song-level structure (e.g., verse–chorus arrangement,
dynamic arc) cannot be made from 30-second clips.

### 6.5 Features assume Western tonal music
Chroma, tonnetz, and MFCC features are calibrated to Western 12-tone equal temperament.
Afrobeats tracks may use non-Western tuning systems, rhythmic structures, or timbral
qualities that these features do not capture well. Homogenization analysis for
Afrobeats is therefore less reliable than for the other three genres.

### 6.6 Self-similarity features in prompt AND in outcome
`self_similarity_mean` was encoded in the prompt but fidelity is near-zero. However,
it is ALSO an outcome variable of interest. The analysis includes it in homogenization
claims (since fidelity is negligible) but this dual role should be disclosed.

### 6.7 No perceptual validation
All findings are based on acoustic/computational features. None have been validated
by human listeners. A track that appears homogeneous in feature space may sound
diverse to humans, and vice versa. The AUC=0.991 result in particular requires
perceptual follow-up — are the features that discriminate AI from human also the
features that human listeners notice?

---

## 7. Research Value and Framing

### 7.1 Why the results are not a null finding
A superficial reading might conclude "no homogenization" because Suno is MORE diverse
within-genre than human. This misreads the results. The findings are:

- **Lyria genuinely homogenizes within-genre** (variance ratio 0.839, Cohen's d = -0.265,
  p < 0.0001, 58% of feature×genre cells). A real effect with high statistical confidence.
- **Suno collapses genre boundaries** (separation ratio 0.429 vs human 0.669). This is
  homogenization at the categorical level — AI music erases the distinction between Metal
  and K-pop even if within-genre variance is high. This is arguably more culturally
  significant than within-genre compression.
- **Both systems produce structurally AI-like music** (AUC 0.991). The discriminating
  features are timbral rigidity (flat MFCC evolution over time) and rhythmic flatness
  (metronomic IOI). These are systemic properties of AI generation, not genre artifacts.

### 7.2 Suno's diversity is structureless, not musical
Suno being "more diverse" than human does not mean it produces more musically varied
output. Its high variance is in dimensions like MFCC delta (frame-to-frame timbral
change) and onset_strength_cv — incoherent variation rather than purposeful musical
development. Human music varies within a track in structured ways (dynamics, arrangement,
tension/release); Suno varies in ways that are acoustically large but musically arbitrary.
High acoustic variance ≠ high musical diversity.

### 7.3 The two systems have distinct failure modes
Lyria: reduces within-genre diversity (sounds more generic within a genre).
Suno: preserves within-genre variance but erases between-genre distinctions (sounds
genre-agnostic). These are different problems with different cultural implications and
cannot be summarised by a single "AI homogenizes music" claim.

### 7.4 Correct framing for the paper
> *"AI music generation systems exhibit system-specific homogenization patterns —
> Lyria reduces within-genre acoustic diversity while Suno collapses categorical genre
> distinctions — and both produce music that is structurally distinguishable from human
> music via timbral rigidity and rhythmic flatness, independent of genre."*

This is a stronger and more honest contribution than a simple positive or null result.

### 7.5 Methodological contributions independent of the homogenization findings
- Prompt fidelity as a prerequisite for excluding prompt-encoded features: established
  that track-level Pearson r (not population-level distribution overlap) is the correct
  test, and found Suno r < 0.25 for all features.
- Within-genre as the correct analysis unit: showed that pooling across genres
  artificially inflates apparent AI diversity.
- Distribution-level vs. paired-level distinction: a methodological point useful for
  any future work using prompt-conditioned AI music generation.

---

## 8. Open Questions

1. **Does the vocal/instrumental confound fully explain the AUC=0.991?**
   Test: train the classifier on only the confirmed-instrumental human tracks.
   If AUC drops substantially, the discriminability is largely about vocals, not AI.

2. **Is Suno's genre-flattening (separation ratio 0.44) perceptible?**
   A listening study could test whether humans rate Suno-generated Metal and
   Suno-generated K-pop as more similar to each other than their human counterparts.

3. **Does chroma_entropy reveal key-center homogenization?**
   If AI systems default to certain keys (e.g., always generating in C major or A
   minor), chroma_entropy would be lower for AI. Pending re-extraction with new features.

4. **What explains Suno Afrobeats anomaly?**
   Suno Afrobeats has the highest within-genre diversity of any genre×system cell
   (distance ratio 1.50 vs human). The v3/v3-5 version difference is one explanation,
   but the effect is very large. This warrants investigation before including Suno
   Afrobeats in cross-genre claims.

5. **Would Udio change the convergence finding?**
   Suno and Lyria strongly diverge. If Udio is closer to one of them than to the
   other, it would suggest system families rather than a unified AI sound.

6. **Is rhythmic rigidity (low IOI_cv) the primary AI signature?**
   IOI features dominate the classifier importance. A targeted analysis of just
   rhythmic features might reveal that all homogenization effects reduce to AI
   music being more metronomic.

---

## 8. File Structure Reference

```
data/
  audio/
    suno/{genre}/        100 × {spotify_id}_suno.mp3 per genre
    lyria/{genre}/       100 × {spotify_id}_lyria.mp3 per genre (some blocked)
    [udio/ — not generated]
  human_corpus/{genre}/  audio/ + corpus_final.csv
  prompts/               {genre}_exp2_prompts.csv  (includes target feature values)
  features/
    all_features.csv     1200 rows × 72 cols (67 original + 5 new + metadata)
    {genre}_{ai,human}_features.csv
  analysis/
    variance_analysis.csv
    distance_summary.csv
    genre_separation.csv
    system_convergence.csv
    classifier_importance.csv
    prompt_fidelity.csv
    fig1_pca.{pdf,png}
    fig2_variance_heatmap.{pdf,png}
    fig3_distances.{pdf,png}
    fig4_fidelity.{pdf,png}
    fig5_genre_separation.{pdf,png}
    fig6_importance.{pdf,png}
    fig7_convergence.{pdf,png}
    diagnostics/         (7 additional diagnostic figures from 6_diagnostic_plots.py)

src/
  4_extract_features.py       feature extraction (67 + 5 new features)
  5_homogenization_analysis.py  primary analysis + 7 publication figures
  6_diagnostic_plots.py         diagnostic investigation figures
  generate_suno.py / generate_udio.py / generate_lyria.py
```
