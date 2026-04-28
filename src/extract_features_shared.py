"""
extract_features_shared.py

Single source of truth for the 67-feature MIR extractor.
Imported by:  1_stratified_sample.py
              2_generate_prompts.py
              4_extract_features.py

Matches the v1 pipeline exactly. Do not modify feature extraction
logic here without updating the v1 comparison note in the paper.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import librosa


class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13):
        self.sr     = sr
        self.n_mfcc = n_mfcc

    def load_30s(self, path):
        y, sr = librosa.load(str(path), sr=self.sr)
        target = int(30 * sr)
        if len(y) > target:
            mid   = len(y) // 2
            start = max(0, mid - target // 2)
            end   = start + target
            if end > len(y):
                end   = len(y)
                start = end - target
            y = y[start:end]
        elif len(y) < target:
            pad = target - len(y)
            y   = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
        return y, sr

    def extract(self, path):
        y, sr = self.load_30s(path)
        f = {}
        f.update(self._spectral(y, sr))
        f.update(self._timbral(y, sr))
        f.update(self._rhythmic(y, sr))
        f.update(self._harmonic(y, sr))
        f.update(self._structural(y, sr))
        f.update(self._dynamic(y, sr))
        f.update(self._mfcc(y, sr))
        return f

    def _spectral(self, y, sr):
        sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        ro  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        fl  = librosa.feature.spectral_flatness(y=y)[0]
        con = librosa.feature.spectral_contrast(y=y, sr=sr)
        return {
            "spectral_centroid_mean":  float(np.mean(sc)),
            "spectral_centroid_std":   float(np.std(sc)),
            "spectral_centroid_var":   float(np.var(sc)),
            "spectral_bandwidth_mean": float(np.mean(bw)),
            "spectral_bandwidth_std":  float(np.std(bw)),
            "spectral_rolloff_mean":   float(np.mean(ro)),
            "spectral_rolloff_std":    float(np.std(ro)),
            "spectral_flatness_mean":  float(np.mean(fl)),
            "spectral_flatness_std":   float(np.std(fl)),
            "spectral_contrast_mean":  float(np.mean(con)),
            "spectral_contrast_std":   float(np.std(con)),
        }

    def _timbral(self, y, sr):
        zcr     = librosa.feature.zero_crossing_rate(y)[0]
        yh, yp  = librosa.effects.hpss(y)
        h_en    = float(np.sum(yh**2))
        p_en    = float(np.sum(yp**2))
        tot     = h_en + p_en
        return {
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std":  float(np.std(zcr)),
            "harmonic_ratio":          h_en / tot if tot > 0 else 0.0,
            "percussive_ratio":        p_en / tot if tot > 0 else 0.0,
        }

    def _rhythmic(self, y, sr):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo    = float(np.asarray(tempo).flat[0])  # librosa 0.11 returns 1-d array; numpy 2.x requires scalar
        oe       = librosa.onset.onset_strength(y=y, sr=sr)
        of       = librosa.onset.onset_detect(onset_envelope=oe, sr=sr)
        if len(of) > 1:
            ot  = librosa.frames_to_time(of, sr=sr)
            ioi = np.diff(ot)
            od  = len(of) / (len(y) / sr)
        else:
            ioi = np.array([0.0])
            od  = 0.0
        ioi_m = float(np.mean(ioi))
        ioi_s = float(np.std(ioi))
        return {
            "tempo":               tempo,
            "onset_density":       float(od),
            "onset_strength_mean": float(np.mean(oe)),
            "onset_strength_std":  float(np.std(oe)),
            "ioi_mean":            ioi_m,
            "ioi_std":             ioi_s,
            "ioi_cv":              ioi_s / ioi_m if ioi_m > 0 else 0.0,
        }

    def _harmonic(self, y, sr):
        ch = librosa.feature.chroma_stft(y=y, sr=sr)
        cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        tn = librosa.feature.tonnetz(y=y, sr=sr)
        return {
            "chroma_stft_mean": float(np.mean(ch)),
            "chroma_stft_std":  float(np.std(ch)),
            "chroma_stft_var":  float(np.var(ch)),
            "chroma_cqt_mean":  float(np.mean(cq)),
            "chroma_cqt_std":   float(np.std(cq)),
            "tonnetz_mean":     float(np.mean(tn)),
            "tonnetz_std":      float(np.std(tn)),
        }

    def _structural(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        ms   = librosa.util.sync(mfcc, beats) if beats is not None and len(beats) >= 2 else mfcc
        R    = librosa.segment.recurrence_matrix(ms, mode="affinity")
        diag = float(np.mean(np.diag(R))) if R.shape[0] > 0 else 0.0
        return {
            "repetition_score":     float(np.mean(R) - diag),
            "self_similarity_mean": float(np.mean(R)) if R.size else 0.0,
            "self_similarity_std":  float(np.std(R))  if R.size else 0.0,
        }

    def _dynamic(self, y, sr):
        rms = librosa.feature.rms(y=y)[0]
        db  = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=1.0)
        msq = float(np.mean(y**2))
        return {
            "rms_mean":         float(np.mean(rms)),
            "rms_std":          float(np.std(rms)),
            "rms_var":          float(np.var(rms)),
            "dynamic_range_db": float(np.max(db) - np.min(db)) if db.size else 0.0,
            "crest_factor":     float(np.max(np.abs(y)) / np.sqrt(msq)) if msq > 0 else 0.0,
        }

    def _mfcc(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        d1   = librosa.feature.delta(mfcc)
        d2   = librosa.feature.delta(mfcc, order=2)
        out  = {}
        for i in range(self.n_mfcc):
            out[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            out[f"mfcc_{i}_std"]  = float(np.std(mfcc[i]))
        out["mfcc_delta_mean"]  = float(np.mean(d1))
        out["mfcc_delta_std"]   = float(np.std(d1))
        out["mfcc_delta2_mean"] = float(np.mean(d2))
        out["mfcc_delta2_std"]  = float(np.std(d2))
        return out


# The 67 acoustic feature column names — used for k-means and z-scoring.
# ioi_cv and rms_* are OUTCOME VARIABLES ONLY — not used in Exp 2 prompts.
FEATURE_COLS = [
    "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_contrast_mean", "spectral_contrast_std",
    "zero_crossing_rate_mean", "zero_crossing_rate_std",
    "harmonic_ratio", "percussive_ratio",
    "tempo", "onset_density", "onset_strength_mean", "onset_strength_std",
    "ioi_mean", "ioi_std", "ioi_cv",           # ioi_cv = outcome only
    "chroma_stft_mean", "chroma_stft_std", "chroma_stft_var",
    "chroma_cqt_mean", "chroma_cqt_std",
    "tonnetz_mean", "tonnetz_std",
    "repetition_score", "self_similarity_mean", "self_similarity_std",
    "rms_mean", "rms_std", "rms_var",          # rms_* = outcome only
    "dynamic_range_db", "crest_factor",
] + [f"mfcc_{i}_mean" for i in range(13)] \
  + [f"mfcc_{i}_std"  for i in range(13)] \
  + ["mfcc_delta_mean", "mfcc_delta_std", "mfcc_delta2_mean", "mfcc_delta2_std"]

assert len(FEATURE_COLS) == 67, f"Expected 67 features, got {len(FEATURE_COLS)}"

# Exp 2 prompt inputs — IOI CV and RMS dropped (free outcome variables)
PROMPT_FEATURE_COLS = [
    "tempo",
    "onset_density",
    "harmonic_ratio",         # → H/P balance
    "self_similarity_mean",   # → loop structure
]
