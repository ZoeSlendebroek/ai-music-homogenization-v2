# config/genres.py
# Central config for all genres: Spotify playlist seeds, corpus targets, prompt vocab.
# Add new genres here only — scripts read from this file.



GENRES = {
    "afrobeats": {
        "display":  "Afrobeats",
        "target_n": 100,
        "year_range": (2010, 2026),
        # Spotify playlist IDs — instrumental/groove-heavy playlists preferred
        "spotify_playlists": [
            # LOOK FOR INSTRUMENTAL AFROBEATS!!!!!!!!!!
            "2Fv4oN2mty17NfTTJarSOH",  # 168
            "1dUGRxuSyKCHsI3dnXYcYc",  # 208
            "1mHunEEIPUZwV6EfX1so5e",  # Afrobeats Instrumentals (if available)
            "2GeWzrGeU8MfxCw6MyQPxn",
            "0CRVC0hIRzwgeLLsITwRGs" # 113
        ],
        "note": "Legacy Suno v3 data retained. New Lyria/Udio data collected on current versions.",
    },
    "kpop": {
        "display":  "K-pop",
        "target_n": 100,
        "year_range": (2010, 2026),
        "spotify_playlists": [
            "37i9dQZF1DX9tPFwDMOaN1",  # K-pop ON! (from Spotify)
            "37i9dQZF1EQpesGsmIyqcW",  # K-pop mix
            "2EoheVFjqIxgJMb8VnDRtZ",  # KPOP Hits 2026
            "3Ir5YWemOTGRRfXgROrsDV", # KPOP 2010-2026
        ],
        "note": "Large-scale non-Western comparator. High streaming volume.",
    },
    "dancepop": {
        "display":  "Dance Pop",
        "target_n": 100,
        "year_range": (2010, 2026),
        "spotify_playlists": [
            "0NGrmRJhL59zivESZMrGFU",  # 57 songs Dancepop
            "38fdzlWsgSywPEf9HaXMYu",  # 92 dancepop
            "1Hu55P6Ah04dM5SRXNeEDI",  # ~500
        ],
        "note": "Western high-resource anchor. Grammy Best Dance Pop Recording category.",
    },
    "metal": {
        "display":  "Heavy Metal",
        "target_n": 100,
        "year_range": (2000, 2026),
        "spotify_playlists": [
            "1yMlpNGEpIVUIilZlrbdS0",
            "27gN69ebwiJRtXEboL12Ih",
            "50dc1OFb8lASN0AurBHKq0",
            "2w16bIXfyPacvwOQaI8YvT",
            "0jtl1YriX5cWmaIxu7mHss",
            "7CDGxii3HkuZjIYPdAZfVj",
            "7kveDCZaADORf7TtZqN0sC",

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
