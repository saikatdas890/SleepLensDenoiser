"""
Stage 2 — Frame-Level Feature Extraction
=========================================
Extracts a rich set of acoustic features at 25 ms frames / 10 ms hop,
aligned so every feature array has the same time axis ``T``.

Features
--------
* **MFCC + Δ + ΔΔ**      shape ``(120, T)``  — 40 coefficients × 3 orders
* **Spectral centroid**   shape ``(1, T)``    — Hz, centre of spectral mass
* **Spectral rolloff**    shape ``(1, T)``    — 85th percentile frequency
* **Zero crossing rate**  shape ``(1, T)``    — proxy for noisiness / voicing
* **HNR (harmonic-to-noise ratio)** shape ``(1, T)``  — dB, via HPSS
* **Log-mel spectrogram** shape ``(128, T)``  — perceptual spectral envelope

Stacked feature dimension  D = 40*3 + 1 + 1 + 1 + 1 + 128 = **252**

Usage::

    from feature_utils import extract_features, stack_features, normalize_features
    import librosa, soundfile as sf

    audio, sr = sf.read("denoised.wav")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    feats = extract_features(audio.astype("float32"), sr)
    X = normalize_features(stack_features(feats))   # (252, T)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ── Default frame parameters ──────────────────────────────────────────────────
FRAME_MS: int = 25   # frame length in milliseconds
HOP_MS:   int = 10   # hop size in milliseconds
N_MFCC:   int = 40   # number of MFCC coefficients
N_MELS:   int = 128  # mel-filter banks
FFT_MS:   int = 50   # FFT window (>= FRAME_MS, rounded up to next power-of-2)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """
    All frame-level features time-aligned to the same ``T`` axis.

    Attributes
    ----------
    mfcc              : (n_mfcc*3, T)  MFCC + first + second delta
    spectral_centroid : (1, T)         Hz
    spectral_rolloff  : (1, T)         Hz  at 85th percentile
    zcr               : (1, T)         zero-crossing rate  [0, 1]
    hnr               : (1, T)         harmonic-to-noise ratio  dB
    log_mel           : (n_mels, T)    log-power mel spectrogram  dB
    times             : (T,)           centre time in seconds for each frame
    sr                : int            sample rate
    hop_length        : int            hop in samples
    frame_length      : int            window in samples
    """
    mfcc:               np.ndarray
    spectral_centroid:  np.ndarray
    spectral_rolloff:   np.ndarray
    zcr:                np.ndarray
    hnr:                np.ndarray
    log_mel:            np.ndarray
    times:              np.ndarray
    sr:                 int
    hop_length:         int
    frame_length:       int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def _compute_hnr(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    frame_length: int,
) -> np.ndarray:
    """
    Estimate per-frame Harmonic-to-Noise Ratio (HNR) in dB.

    Strategy
    --------
    1. Decompose the signal into harmonic and residual components using
       librosa's HPSS (Harmonic–Percussive Source Separation) on the full
       STFT.  The harmonic component captures voiced/periodic content
       (snoring resonances); the residual captures broadband noise.
    2. Frame both signals and compute per-frame RMS energy.
    3. HNR = 10 · log₁₀(E_harmonic² / E_residual²)

    This is a practical approximation that avoids the computational cost of
    pitch tracking while still distinguishing periodic from aperiodic sound.

    Returns
    -------
    np.ndarray  shape ``(1, T)``  in dB
    """
    eps = 1e-10

    # Full-signal harmonic/residual separation
    y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    y_residual = y - y_harmonic

    rms_h = librosa.feature.rms(
        y=y_harmonic, frame_length=frame_length, hop_length=hop_length
    )[0]  # (T,)
    rms_n = librosa.feature.rms(
        y=y_residual, frame_length=frame_length, hop_length=hop_length
    )[0]  # (T,)

    hnr_db = 10.0 * np.log10((rms_h ** 2 + eps) / (rms_n ** 2 + eps))
    return hnr_db.reshape(1, -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main feature-extraction function
# ---------------------------------------------------------------------------

def extract_features(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
    hop_ms:   int = HOP_MS,
    n_mfcc:   int = N_MFCC,
    n_mels:   int = N_MELS,
    compute_hnr: bool = True,
) -> FrameFeatures:
    """
    Extract all frame-level acoustic features from a mono audio array.

    Parameters
    ----------
    audio       : np.ndarray  1-D float32, shape ``(N,)``
    sr          : int          sample rate in Hz
    frame_ms    : int          analysis window length in ms  (default 25)
    hop_ms      : int          hop size in ms                (default 10)
    n_mfcc      : int          MFCC coefficients             (default 40)
    n_mels      : int          mel filter banks              (default 128)
    compute_hnr : bool         if False skip HNR (faster for very long files)

    Returns
    -------
    FrameFeatures
        All arrays share the same ``T`` (time frames) dimension.
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D, got shape {audio.shape}")

    frame_length = int(sr * frame_ms / 1000)
    hop_length   = int(sr * hop_ms  / 1000)
    n_fft        = _next_power_of_two(int(sr * FFT_MS / 1000))
    n_fft        = max(n_fft, frame_length)

    logger.info(
        "[Features] SR=%d | frame=%dms(%dsmp) | hop=%dms(%dsmp) | n_fft=%d",
        sr, frame_ms, frame_length, hop_ms, hop_length, n_fft,
    )

    # ── MFCC + delta + delta² ────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        n_mels=n_mels,
        fmin=20.0,
        fmax=float(sr // 2),
    ).astype(np.float32)                         # (n_mfcc, T)

    delta_mfcc   = librosa.feature.delta(mfcc)  # (n_mfcc, T)
    delta2_mfcc  = librosa.feature.delta(mfcc, order=2)
    mfcc_full    = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)  # (3*n_mfcc, T)

    # ── Spectral centroid ────────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
    ).astype(np.float32)                         # (1, T)

    # ── Spectral rolloff (85th percentile) ───────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
        roll_percent=0.85,
    ).astype(np.float32)                         # (1, T)

    # ── Zero crossing rate ───────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float32)                         # (1, T)

    # ── Harmonic-to-noise ratio ──────────────────────────────────────────────
    if compute_hnr:
        hnr = _compute_hnr(audio, sr, hop_length, frame_length)  # (1, T)
    else:
        # Placeholder: zeros aligned to MFCC time axis
        hnr = np.zeros((1, mfcc_full.shape[1]), dtype=np.float32)

    # ── Log-mel spectrogram ──────────────────────────────────────────────────
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        n_mels=n_mels,
        fmin=20.0,
        fmax=float(sr // 2),
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0).astype(np.float32)  # (n_mels, T)

    # ── Align all arrays to the shortest T ───────────────────────────────────
    T = min(
        mfcc_full.shape[1], centroid.shape[1], rolloff.shape[1],
        zcr.shape[1], hnr.shape[1], log_mel.shape[1],
    )
    mfcc_full = mfcc_full[:, :T]
    centroid  = centroid[:, :T]
    rolloff   = rolloff[:, :T]
    zcr       = zcr[:, :T]
    hnr       = hnr[:, :T]
    log_mel   = log_mel[:, :T]

    times = librosa.frames_to_time(
        np.arange(T), sr=sr, hop_length=hop_length
    ).astype(np.float32)

    logger.info(
        "[Features] Done: T=%d frames (%.1f s) | "
        "MFCC=%s | log-mel=%s | HNR=%s",
        T, float(times[-1]) if T > 0 else 0.0,
        mfcc_full.shape, log_mel.shape, hnr.shape,
    )

    return FrameFeatures(
        mfcc=mfcc_full,
        spectral_centroid=centroid,
        spectral_rolloff=rolloff,
        zcr=zcr,
        hnr=hnr,
        log_mel=log_mel,
        times=times,
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def stack_features(feats: FrameFeatures) -> np.ndarray:
    """
    Concatenate all feature arrays into a single ``(D, T)`` matrix.

    D = n_mfcc*3 + 1 + 1 + 1 + 1 + n_mels  (default: 120+1+1+1+1+128 = 252)
    """
    return np.concatenate(
        [
            feats.mfcc,
            feats.spectral_centroid,
            feats.spectral_rolloff,
            feats.zcr,
            feats.hnr,
            feats.log_mel,
        ],
        axis=0,
    )


def normalize_features(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize each feature dimension independently.

    Parameters
    ----------
    features : np.ndarray  shape ``(D, T)``
    eps      : float        numerical stability constant

    Returns
    -------
    np.ndarray  shape ``(D, T)``, float32, zero-mean unit-variance per row
    """
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True)
    return ((features - mean) / (std + eps)).astype(np.float32)
