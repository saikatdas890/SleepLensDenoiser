"""
Dataset Preparation for Snore Encoder Training
===============================================
Builds a balanced, labelled train / val / test dataset from:

  1. Your own snore clips  (output/snore_clips/)
  2. Background clips auto-extracted from the gaps between snore events
     in your original recording  (guided by snore_segments.csv)
  3. (Optional) Kaggle snoring dataset clips for extra diversity

Output folder structure::

    data/
    ├── train/
    │   ├── snore/        snore_0001.wav …
    │   └── background/   bg_0000.wav …
    ├── val/
    │   ├── snore/
    │   └── background/
    ├── test/
    │   ├── snore/
    │   └── background/
    └── dataset_manifest.csv   ← split, label, label_id, path

Usage::

    # Minimal — uses your own recording only
    python prepare_dataset.py \\
        --snore-clips  output/snore_clips/ \\
        --original-audio ~/Downloads/recording.wav \\
        --csv           output/snore_segments.csv

    # With Kaggle dataset added
    python prepare_dataset.py \\
        --snore-clips  output/snore_clips/ \\
        --original-audio ~/Downloads/recording.wav \\
        --csv           output/snore_segments.csv \\
        --kaggle-snore  ~/Downloads/Snoring_Dataset/1/ \\
        --kaggle-bg     ~/Downloads/Snoring_Dataset/0/
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_S    = 2.0    # clip duration fed to AST  (must match train_encoder.py)
MIN_GAP_S   = 2.0    # safety buffer around snore events when sampling background
MIN_RMS     = 1e-4   # discard near-silent background clips
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # remainder goes to test


# ---------------------------------------------------------------------------
# CSV / gap utilities
# ---------------------------------------------------------------------------

def load_segments(csv_path: str) -> List[Dict[str, float]]:
    """Load snore_segments.csv → list of {start, end} dicts sorted by start."""
    segments: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            segments.append(
                {"start": float(row["start_time"]), "end": float(row["end_time"])}
            )
    return sorted(segments, key=lambda x: x["start"])


def compute_gaps(
    segments: List[Dict[str, float]],
    duration: float,
    min_gap_s: float,
    window_s: float,
) -> List[Tuple[float, float]]:
    """
    Return list of (gap_start, gap_end) intervals that are safely away from
    any snore event and long enough to contain at least one window.
    """
    gaps: List[Tuple[float, float]] = []
    prev_end = 0.0

    for seg in segments:
        gap_s = prev_end + min_gap_s
        gap_e = seg["start"] - min_gap_s
        if gap_e - gap_s >= window_s:
            gaps.append((gap_s, gap_e))
        prev_end = seg["end"]

    # Final tail of recording
    tail_s = prev_end + min_gap_s
    if duration - tail_s >= window_s:
        gaps.append((tail_s, duration - 0.5))

    return gaps


# ---------------------------------------------------------------------------
# Background clip extraction
# ---------------------------------------------------------------------------

def extract_background_clips(
    audio_path: str,
    segments: List[Dict[str, float]],
    out_dir: Path,
    n_clips: int,
    window_s: float = WINDOW_S,
    min_gap_s: float = MIN_GAP_S,
    min_rms: float = MIN_RMS,
    seed: int = 42,
) -> List[Path]:
    """
    Randomly sample ``n_clips`` non-overlapping windows from the background
    (gaps between snore events) of the recording.

    Parameters
    ----------
    audio_path : str   path to the original (or denoised) recording
    segments   : list  snore segment dicts from load_segments()
    out_dir    : Path  where to write the extracted WAV clips
    n_clips    : int   number of background clips to extract

    Returns
    -------
    list[Path]  paths to the saved background clips
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    logger.info("[BgExtract] Loading audio from '%s' …", audio_path)
    audio, sr = sf.read(audio_path, always_2d=False, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    win_smp = int(window_s * sr)
    gaps = compute_gaps(segments, duration, min_gap_s, window_s)

    if not gaps:
        logger.warning("[BgExtract] No valid gaps found — cannot extract background clips.")
        return []

    logger.info("[BgExtract] Found %d gap intervals | Extracting %d clips …", len(gaps), n_clips)

    # Weighted sampling proportional to gap duration
    gap_weights = [e - s for s, e in gaps]
    total_w = sum(gap_weights)
    gap_weights = [w / total_w for w in gap_weights]

    saved: List[Path] = []
    attempts = 0
    max_attempts = n_clips * 30

    while len(saved) < n_clips and attempts < max_attempts:
        attempts += 1

        g_idx = random.choices(range(len(gaps)), weights=gap_weights, k=1)[0]
        gs, ge = gaps[g_idx]
        max_start = ge - window_s
        if max_start <= gs:
            continue

        clip_start_s = random.uniform(gs, max_start)
        s_smp = int(clip_start_s * sr)
        e_smp = s_smp + win_smp

        if e_smp > len(audio):
            continue

        clip = audio[s_smp:e_smp]
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms < min_rms:
            continue

        clip_path = out_dir / f"bg_{len(saved):04d}.wav"
        sf.write(str(clip_path), clip, sr)
        saved.append(clip_path)

    logger.info("[BgExtract] Extracted %d / %d background clips", len(saved), n_clips)
    return saved


# ---------------------------------------------------------------------------
# Dataset split + copy
# ---------------------------------------------------------------------------

def split_files(
    files: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Shuffle and split file list into train / val / test."""
    files = list(files)
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return files[:n_tr], files[n_tr : n_tr + n_val], files[n_tr + n_val :]


def copy_clips(files: List[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare snore/background dataset for encoder training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--snore-clips",     default="output/snore_clips/",
                   help="Folder of snore WAV clips from the main pipeline.")
    p.add_argument("--original-audio",  required=True,
                   help="Original (raw) recording WAV used to extract background clips.")
    p.add_argument("--csv",             default="output/snore_segments.csv",
                   help="snore_segments.csv produced by the main pipeline.")
    p.add_argument("--data-dir",        default="data/",
                   help="Root output directory for the prepared dataset.")
    p.add_argument("--kaggle-snore",    default=None,
                   help="(Optional) Path to Kaggle 'snoring' folder (class 1/).")
    p.add_argument("--kaggle-bg",       default=None,
                   help="(Optional) Path to Kaggle 'not snoring' folder (class 0/).")
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect snore clips ──────────────────────────────────────────────
    snore_files = sorted(Path(args.snore_clips).glob("*.wav"))
    logger.info("Snore clips from pipeline : %d", len(snore_files))

    if args.kaggle_snore:
        kaggle_snore = sorted(Path(args.kaggle_snore).glob("*.wav"))
        snore_files = list(snore_files) + kaggle_snore
        logger.info("Kaggle snore clips added  : %d", len(kaggle_snore))

    if not snore_files:
        logger.error(
            "No snore clips found in '%s'.  "
            "Run main.py first to generate them.", args.snore_clips
        )
        return

    # ── 2. Extract background clips ─────────────────────────────────────────
    segments = load_segments(args.csv)
    logger.info("Loaded %d snore segment timestamps from CSV", len(segments))

    bg_raw_dir = data_dir / "_raw_background"
    n_bg_target = len(snore_files)   # balanced dataset
    bg_files = extract_background_clips(
        args.original_audio,
        segments,
        bg_raw_dir,
        n_clips=n_bg_target,
        seed=args.seed,
    )

    if args.kaggle_bg:
        kaggle_bg = sorted(Path(args.kaggle_bg).glob("*.wav"))
        bg_files = list(bg_files) + kaggle_bg
        logger.info("Kaggle background clips   : %d", len(kaggle_bg))

    if not bg_files:
        logger.error("No background clips extracted.  Cannot build dataset.")
        return

    # ── 3. Train / val / test split ─────────────────────────────────────────
    s_tr, s_val, s_te = split_files(snore_files, seed=args.seed)
    b_tr, b_val, b_te = split_files(bg_files,    seed=args.seed)

    logger.info(
        "Split — train: snore=%d bg=%d | val: snore=%d bg=%d | test: snore=%d bg=%d",
        len(s_tr), len(b_tr), len(s_val), len(b_val), len(s_te), len(b_te),
    )

    # ── 4. Copy into folder structure ───────────────────────────────────────
    for split, sf_list, bf_list in [
        ("train", s_tr, b_tr),
        ("val",   s_val, b_val),
        ("test",  s_te,  b_te),
    ]:
        copy_clips(sf_list, data_dir / split / "snore")
        copy_clips(bf_list, data_dir / split / "background")

    # ── 5. Write manifest CSV ───────────────────────────────────────────────
    rows: List[Dict] = []
    for split, sf_list, bf_list in [
        ("train", s_tr, b_tr),
        ("val",   s_val, b_val),
        ("test",  s_te,  b_te),
    ]:
        for f in sf_list:
            rows.append({
                "split": split, "label": "snore", "label_id": 1,
                "file": f.name,
                "path": str(data_dir / split / "snore" / f.name),
            })
        for f in bf_list:
            rows.append({
                "split": split, "label": "background", "label_id": 0,
                "file": f.name,
                "path": str(data_dir / split / "background" / f.name),
            })

    manifest_path = data_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["split", "label", "label_id", "file", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Dataset ready at: %s", data_dir)
    logger.info("Total clips     : %d  (%d snore / %d background)",
                len(rows), len(snore_files), len(bg_files))
    logger.info("Manifest        : %s", manifest_path)
    logger.info("")
    logger.info("Next step:")
    logger.info("  python train_encoder.py --data-dir %s", data_dir)


if __name__ == "__main__":
    main()
