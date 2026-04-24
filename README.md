# AI Music Homogenization — Phase 2

Expansion of the FAccT '26 audit to 4 genres × 3 systems.

**Genres:** Afrobeats (v3 legacy data retained), K-pop, Dance Pop, Heavy Metal  
**Systems:** Suno (current), Lyria 3, Udio  
**Note:** Afrobeats Suno data was collected on v3 (Dec 2025 / Jan 2026). All new data collected on current versions. Version difference is flagged as a covariate in analysis, not treated as a confound.

## Pipeline

```
0_collect_human_corpus.py   # Spotify discovery → Spotify/Deezer audio → 30s MP3s
1_stratified_sample.py      # k-means (k=10) on 67 features → final n≤100 per genre
2_generate_prompts.py       # MIR features → Suno / Lyria / Udio prompt strings
3_check_prompt_diversity.py # Pairwise prompt similarity check before generation
4_extract_features.py       # 67-feature MIR extraction (matches v1 pipeline exactly)
5_homogenization.py         # D1–D5 diagnostics per genre × system cell
6_cross_genre_analysis.py   # Compare homogenization severity across genres
```

## Setup

```bash
pip install spotipy requests librosa pandas numpy scikit-learn scipy
```

Add a `.env` file (never committed):
```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

## Data notes

- Human corpus: soft n=100 floor. Genres below 80 tracks get a power analysis note.
- IOI CV and RMS energy are **not** included in Exp 2 prompts — they are free outcome variables.
- All audio standardized to 30s middle crop before feature extraction.
- Udio outputs trimmed to middle 30s post-download.
