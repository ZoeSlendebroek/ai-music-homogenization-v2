# config/genres.py
# Central config for all genres: Spotify playlist seeds, corpus targets, prompt vocab.
# Add new genres here only — scripts read from this file.

GENRES = {
    "afrobeats": {
        "display":  "Afrobeats",
        "target_n": 100,
        "year_range": (2015, 2025),
        # Spotify playlist IDs — instrumental/groove-heavy playlists preferred
        "spotify_playlists": [
            "37i9dQZF1DX3LDIBRoaCDQ",  # Afrobeats
            "37i9dQZF1DWYs9lGFdakKe",  # Afro hits
            "37i9dQZF1DX0F1O4H5nHOA",  # Afrobeats Instrumentals (if available)
        ],
        "note": "Legacy Suno v3 data retained. New Lyria/Udio data collected on current versions.",
    },
    "kpop": {
        "display":  "K-pop",
        "target_n": 100,
        "year_range": (2018, 2025),
        "spotify_playlists": [
            "37i9dQZF1DX9tPFwDMOaN1",  # K-pop hits
            "37i9dQZF1DWU7yZzS8s0lX",  # K-pop rising
            "37i9dQZF1DX4FcAKI5Nhzq",  # K-pop daebak
        ],
        "note": "Large-scale non-Western comparator. High streaming volume.",
    },
    "dancepop": {
        "display":  "Dance Pop",
        "target_n": 100,
        "year_range": (2018, 2025),
        "spotify_playlists": [
            "37i9dQZF1DX4dyzvuaRJ0n",  # mint (dance pop)
            "37i9dQZF1DX1g0iEXLFycr",  # Dance pop hits
            "37i9dQZF1DXaXB8fQg7xof",  # Dance pop rising
        ],
        "note": "Western high-resource anchor. Grammy Best Dance Pop Recording category.",
    },
    "metal": {
        "display":  "Heavy Metal",
        "target_n": 100,
        "year_range": (2015, 2025),
        "spotify_playlists": [
            "37i9dQZF1DWTcqUzwhNmKv",  # Heavy metal
            "37i9dQZF1DX08jcQJXDnEQ",  # Metal classics
            "37i9dQZF1DX9qNs32fujYe",  # Metal essentials
        ],
        "note": "Western niche genre. Maximally distinct MIR feature space from Afrobeats.",
    },
}

SYSTEMS = ["suno", "lyria", "udio"]

# Experiment 1: prompts per genre for naturalistic prompting
EXP1_N_PROMPTS      = 10
EXP1_GENS_PER_PROMPT = 2   # = 20 AI tracks per genre-system cell

# Experiment 2: one generation per human track
EXP2_GENS_PER_TRACK = 1

# Stratified sampling
KMEANS_K            = 10
CANDIDATE_POOL_SIZE = 200   # tracks to collect before sampling down to target_n

# Prompt diversity threshold — flag if > 15% of prompt pairs are identical
PROMPT_DIVERSITY_THRESHOLD = 0.15
