"""
Stage 5 — Snore Clip Extraction
================================
Cuts each detected snore event from the **original, pre-denoised** recording
and saves it as a standalone WAV file.

Rationale for using original audio
------------------------------------
The denoised audio is optimised for *detection* (cleaner SNR for the AST
model).  However, for downstream ML training it is better to keep the raw
recording, which preserves the full acoustic character of snoring sounds,
including room acoustics and microphone colour.

Output layout::

    output/
    ├── denoised.wav            ← from Stage 1
    ├── snore_segments.csv      ← manifest written by this module
    └── snore_clips/
        ├── snore_0001.wav
        ├── snore_0002.wav
        └── …

CSV columns::

    clip_file, start_time, end_time, duration, confidence

Usage::

    from cut_snore_clips import cut_snore_clips
    csv_path = cut_snore_clips(
        original_audio_path="recording.wav",
        segments=[{'start': 12.3, 'end': 14.1, 'confidence': 0.72}, …],
        output_dir="output/",
    )
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def cut_snore_clips(
    original_audio_path: str,
    segments: List[Dict[str, float]],
    output_dir: str,
    clips_subdir: str = "snore_clips",
    csv_filename: str = "snore_segments.csv",
    min_rms: float = 1e-5,
) -> str:
    """
    Extract snore segments from the original recording and save them as WAVs.

    Parameters
    ----------
    original_audio_path : str
        Path to the **raw** (pre-denoised) WAV file.
    segments : list of dict
        Each element must have keys ``'start'``, ``'end'``, ``'confidence'``
        (seconds and probability, as returned by ``detect_snore_segments``).
    output_dir : str
        Root directory for output artefacts.
    clips_subdir : str
        Sub-folder under *output_dir* where clip WAVs are written.
    csv_filename : str
        Name of the manifest CSV written under *output_dir*.
    min_rms : float
        Clips whose RMS amplitude is below this value are considered silent
        and are skipped (avoids saving near-empty files).

    Returns
    -------
    str
        Absolute path to the written CSV manifest.
    """
    output_dir = Path(output_dir)
    clips_dir  = output_dir / clips_subdir
    clips_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_filename

    # ── Load original audio once (preserves native SR and channel count) ───
    logger.info("[Cut] Loading original audio: %s", original_audio_path)
    audio, sr = sf.read(original_audio_path, always_2d=True, dtype="float32")
    # audio: (n_samples, n_channels)
    n_samples = audio.shape[0]
    n_channels = audio.shape[1]
    duration  = n_samples / sr
    logger.info(
        "[Cut] Duration=%.1f s | SR=%d Hz | Channels=%d | Segments to cut=%d",
        duration, sr, n_channels, len(segments),
    )

    if not segments:
        logger.warning("[Cut] No segments provided — CSV will be empty.")

    saved: List[Dict] = []
    skipped_silent = 0
    clip_number = 0

    for seg in segments:
        start_s = float(seg["start"])
        end_s   = float(seg["end"])
        conf    = float(seg["confidence"])

        # ── Convert to sample indices, clamp to valid range ────────────────
        start_smp = int(np.round(start_s * sr))
        end_smp   = int(np.round(end_s   * sr))
        start_smp = max(0, min(start_smp, n_samples - 1))
        end_smp   = max(start_smp + 1, min(end_smp, n_samples))

        clip = audio[start_smp:end_smp, :]    # (T_clip, channels)

        # ── Skip silent clips ──────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms < min_rms:
            skipped_silent += 1
            logger.debug(
                "[Cut] Skipped silent clip at %.2f s (RMS=%.2e < %.2e)",
                start_s, rms, min_rms,
            )
            continue

        # ── Write WAV ──────────────────────────────────────────────────────
        clip_number += 1
        clip_name = f"snore_{clip_number:04d}.wav"
        clip_path = clips_dir / clip_name

        sf.write(str(clip_path), clip.astype(np.float32), sr)

        clip_duration = clip.shape[0] / sr
        saved.append(
            {
                "clip_file":  clip_name,
                "start_time": round(start_s,     4),
                "end_time":   round(end_s,        4),
                "duration":   round(clip_duration, 4),
                "confidence": round(conf,          4),
            }
        )
        logger.debug(
            "[Cut] Saved %s  %.2f–%.2f s (%.2f s)  conf=%.3f  RMS=%.4f",
            clip_name, start_s, end_s, clip_duration, conf, rms,
        )

    # ── Write CSV manifest ─────────────────────────────────────────────────
    fieldnames = ["clip_file", "start_time", "end_time", "duration", "confidence"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(saved)

    logger.info(
        "[Cut] Done — saved %d clip(s), skipped %d silent clip(s).",
        len(saved), skipped_silent,
    )
    logger.info("[Cut] Clips directory : %s", clips_dir)
    logger.info("[Cut] CSV manifest    : %s", csv_path)

    # ── Print summary table to INFO ────────────────────────────────────────
    if saved:
        total_snore_s = sum(r["duration"] for r in saved)
        logger.info(
            "[Cut] Total snore audio saved: %.1f s across %d file(s)",
            total_snore_s, len(saved),
        )

    return str(csv_path)
