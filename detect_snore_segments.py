"""
Stage 4 — Snore Event Detection & Grouping
===========================================
Converts the continuous snore-probability trace from the AST model into a
clean list of timestamped snore *events* suitable for audio cutting.

Pipeline
--------
1. Load the denoised WAV (mono).
2. Run ``SnoreDetector.predict_framewise`` → per-frame snore probability.
3. **Smooth** the trace with a median filter (removes isolated spikes) followed
   by a uniform / moving-average filter (softens edges).
4. **Threshold** the smoothed trace → binary snore/non-snore mask.
5. **Find contiguous ON regions** → raw candidate events.
6. **Merge** events whose gap is smaller than ``merge_gap_s``.
7. **Pad** event boundaries outward by ``padding_s`` to avoid cutting off
   the onset/offset of each snore.
8. **Filter** events shorter than ``min_duration_s``.
9. Return ``List[Dict]`` with keys ``start``, ``end``, ``confidence``.

Typical call::

    from snore_model import SnoreDetector
    from detect_snore_segments import detect_snore_segments
    import torch

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = SnoreDetector(device=device)
    segments = detect_snore_segments("output/denoised.wav", detector)
    # [{'start': 12.3, 'end': 14.1, 'confidence': 0.72}, ...]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter, uniform_filter1d

from snore_model import SnoreDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame grouping
# ---------------------------------------------------------------------------

def group_snore_frames(
    frame_probs: np.ndarray,
    frame_times: np.ndarray,
    threshold:      float = 0.30,
    min_duration_s: float = 0.40,
    merge_gap_s:    float = 0.30,
    padding_s:      float = 0.10,
) -> List[Dict[str, float]]:
    """
    Convert a per-frame probability array into a list of snore event dicts.

    Parameters
    ----------
    frame_probs     : (T,) float32  snore probability per frame ∈ [0, 1]
    frame_times     : (T,) float32  centre time (seconds) of each frame
    threshold       : float  frames with prob ≥ threshold are labelled snore
    min_duration_s  : float  events shorter than this are discarded
    merge_gap_s     : float  consecutive events separated by less than this
                             gap are merged into one
    padding_s       : float  each surviving event is extended by this amount
                             on both sides (clamped to recording boundaries)

    Returns
    -------
    list of dicts  [{'start': float, 'end': float, 'confidence': float}, …]
    """
    if len(frame_probs) == 0:
        return []

    duration = float(frame_times[-1])

    # ── Step 1: Binary mask ───────────────────────────────────────────────────
    binary = (frame_probs >= threshold).astype(np.int8)

    # ── Step 2: Find contiguous ON regions (index-based) ─────────────────────
    raw_events: List[Tuple[int, int]] = []
    in_event = False
    start_idx = 0

    for i, val in enumerate(binary):
        if val == 1 and not in_event:
            in_event  = True
            start_idx = i
        elif val == 0 and in_event:
            in_event = False
            raw_events.append((start_idx, i - 1))

    if in_event:
        raw_events.append((start_idx, len(binary) - 1))

    if not raw_events:
        logger.info("[GroupFrames] No frames above threshold %.2f.", threshold)
        return []

    # Convert index ranges to (start_s, end_s, prob_sum, n_frames).
    # Storing sum + count allows correct weighted-mean confidence after merging,
    # avoiding the double-averaging bug that mean-of-means introduces.
    timed: List[Tuple[float, float, float, int]] = [
        (
            float(frame_times[s]),
            float(frame_times[e]),
            float(frame_probs[s : e + 1].sum()),
            e - s + 1,
        )
        for s, e in raw_events
    ]

    # ── Step 3: Merge nearby events ───────────────────────────────────────────
    merged: List[Tuple[float, float, float, int]] = []
    cur_start, cur_end, cur_sum, cur_n = timed[0]

    for start, end, prob_sum, n in timed[1:]:
        gap = start - cur_end
        if gap <= merge_gap_s:
            cur_end  = end
            cur_sum += prob_sum
            cur_n   += n
        else:
            merged.append((cur_start, cur_end, cur_sum, cur_n))
            cur_start, cur_end, cur_sum, cur_n = start, end, prob_sum, n

    merged.append((cur_start, cur_end, cur_sum, cur_n))

    # ── Step 4: Pad boundaries ────────────────────────────────────────────────
    padded: List[Tuple[float, float, float]] = [
        (
            max(0.0, s - padding_s),
            min(duration, e + padding_s),
            cur_sum / max(cur_n, 1),          # true frame-weighted mean
        )
        for s, e, cur_sum, cur_n in merged
    ]

    # ── Step 5: Filter by minimum duration ───────────────────────────────────
    filtered: List[Dict[str, float]] = [
        {
            "start":      round(s, 4),
            "end":        round(e, 4),
            "confidence": round(c, 4),
        }
        for s, e, c in padded
        if (e - s) >= min_duration_s
    ]

    logger.info(
        "[GroupFrames] %d raw → %d merged → %d after duration filter "
        "(min_dur=%.2fs, merge_gap=%.2fs, pad=%.2fs)",
        len(raw_events), len(merged), len(filtered),
        min_duration_s, merge_gap_s, padding_s,
    )
    return filtered


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def smooth_probabilities(
    frame_probs: np.ndarray,
    frame_hop_s: float,
    median_ms:  float = 100.0,
    uniform_ms: float = 200.0,
) -> np.ndarray:
    """
    Apply two-stage smoothing to the raw frame-probability trace.

    1. **Median filter** (``median_ms`` window): removes isolated spikes and
       single-frame false positives without blurring real event boundaries.
    2. **Uniform (moving-average) filter** (``uniform_ms`` window): softens
       the trace to produce clean, non-jagged probability envelopes that
       threshold into contiguous ON regions.

    Parameters
    ----------
    frame_probs : (T,) float32
    frame_hop_s : float  hop size (seconds) used when computing ``frame_probs``
    median_ms   : float  median filter window in milliseconds
    uniform_ms  : float  uniform filter window in milliseconds

    Returns
    -------
    (T,) float32  smoothed probabilities
    """
    # Convert ms windows to frame counts (minimum 1)
    med_frames = max(1, int(round(median_ms  / 1000.0 / frame_hop_s)))
    uni_frames = max(1, int(round(uniform_ms / 1000.0 / frame_hop_s)))

    smoothed = median_filter(frame_probs, size=med_frames).astype(np.float32)
    smoothed = uniform_filter1d(smoothed, size=uni_frames).astype(np.float32)
    return np.clip(smoothed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_snore_segments(
    audio_path:     str,
    detector:       SnoreDetector,
    threshold:      float = 0.30,
    min_duration_s: float = 0.40,
    merge_gap_s:    float = 0.30,
    padding_s:      float = 0.10,
    frame_hop_ms:   float = 10.0,
    median_smooth_ms:  float = 100.0,
    uniform_smooth_ms: float = 200.0,
) -> List[Dict[str, float]]:
    """
    Full snore-segment detection pipeline.

    Parameters
    ----------
    audio_path          : str    path to the denoised WAV
    detector            : SnoreDetector  pre-loaded model wrapper
    threshold           : float  snore probability threshold  (default 0.30)
    min_duration_s      : float  discard events shorter than this (s)
    merge_gap_s         : float  merge events with gap < this (s)
    padding_s           : float  extend each event boundary (s)
    frame_hop_ms        : float  output frame resolution in ms (default 10)
    median_smooth_ms    : float  median filter width in ms
    uniform_smooth_ms   : float  uniform filter width in ms

    Returns
    -------
    list of dicts  [{'start': float, 'end': float, 'confidence': float}, …]
                   sorted by start time
    """
    # ── Load audio ─────────────────────────────────────────────────────────
    logger.info("[Detect] Loading denoised audio: %s", audio_path)
    audio_data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    # To mono
    audio_mono = audio_data.mean(axis=1)
    duration = len(audio_mono) / sr
    logger.info(
        "[Detect] Loaded: SR=%d Hz | duration=%.1f s | channels=%d→1(mono)",
        sr, duration, audio_data.shape[1],
    )

    frame_hop_s = frame_hop_ms / 1000.0

    # ── Stage 3: AST frame-level probabilities ─────────────────────────────
    logger.info("[Detect] Running AST snore detection …")
    raw_probs, frame_times = detector.predict_framewise(
        audio_mono, sr, frame_hop_s=frame_hop_s
    )

    # ── Smooth ─────────────────────────────────────────────────────────────
    logger.info("[Detect] Smoothing probability trace …")
    smoothed = smooth_probabilities(
        raw_probs, frame_hop_s,
        median_ms=median_smooth_ms,
        uniform_ms=uniform_smooth_ms,
    )
    logger.info(
        "[Detect] After smoothing: max=%.4f | mean=%.5f | "
        "frames≥threshold(%s)=%d/%d",
        smoothed.max(), smoothed.mean(), threshold,
        int((smoothed >= threshold).sum()), len(smoothed),
    )

    # ── Stage 4: Group frames into events ──────────────────────────────────
    segments = group_snore_frames(
        frame_probs=smoothed,
        frame_times=frame_times,
        threshold=threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        padding_s=padding_s,
    )

    # Summarise detected events
    if segments:
        total_snore = sum(s["end"] - s["start"] for s in segments)
        logger.info(
            "[Detect] Detected %d snore event(s) | "
            "total snore time: %.1f s (%.1f%% of recording)",
            len(segments), total_snore, 100.0 * total_snore / duration,
        )
        for i, seg in enumerate(segments[:20]):
            logger.info(
                "  [%3d] %7.2f s – %7.2f s  dur=%5.2f s  conf=%.3f",
                i + 1, seg["start"], seg["end"],
                seg["end"] - seg["start"], seg["confidence"],
            )
        if len(segments) > 20:
            logger.info("  … and %d more events", len(segments) - 20)
    else:
        logger.warning(
            "[Detect] No snore events detected.  "
            "Consider lowering --threshold (current: %.2f).", threshold
        )

    return segments
