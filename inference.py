"""
Snore Encoder — Inference
==========================
Runs the fine-tuned snore encoder on any WAV file and outputs snore timestamps.

This is **your personal model** trained specifically on your recording — it
knows exactly what your snoring sounds like vs. your room background.

Inference is fast: ~100 ms per 2-second window on CPU.

Usage::

    # Detect snoring in a new recording
    python inference.py --input new_recording.wav --model models/snore_encoder_best.pt

    # Save results to CSV
    python inference.py --input new_recording.wav \\
                        --model models/snore_encoder_best.pt \\
                        --output-csv results.csv

    # More sensitive (lower threshold catches quieter snores)
    python inference.py --input new_recording.wav \\
                        --model models/snore_encoder_best.pt \\
                        --threshold 0.35

    # Faster (less overlap between windows)
    python inference.py --input new_recording.wav \\
                        --model models/snore_encoder_best.pt \\
                        --window-stride 2.0
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

logger = logging.getLogger(__name__)

AST_SR   = 16_000
LABELS   = ["background", "snore"]
SNORE_ID = 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ASTForAudioClassification, ASTFeatureExtractor, dict]:
    """
    Reconstruct the fine-tuned encoder from a saved checkpoint.

    The checkpoint only stores the small classifier head weights; the AST
    backbone is re-downloaded from HuggingFace (or loaded from cache).

    Returns
    -------
    model             : ASTForAudioClassification with binary head
    feature_extractor : ASTFeatureExtractor
    config            : dict  with model metadata
    """
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    logger.info("Loading base AST model: %s", config["model_id"])
    model = ASTForAudioClassification.from_pretrained(config["model_id"])

    # Rebuild the exact same head used during training
    hidden = config["hidden_size"]
    model.classifier = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(p=0.30),
        nn.Linear(hidden, 256),
        nn.GELU(),
        nn.Dropout(p=0.20),
        nn.Linear(256, 2),
    )
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    model.config.num_labels = 2
    model.config.id2label   = {0: "background", 1: "snore"}

    model.to(device)
    model.eval()

    feature_extractor = ASTFeatureExtractor.from_pretrained(config["model_id"])

    logger.info(
        "Encoder loaded — trained val_acc=%.4f  epoch=%d",
        ckpt["val_acc"], ckpt["epoch"],
    )
    return model, feature_extractor, config


# ---------------------------------------------------------------------------
# Window-level inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_file(
    audio_path: str,
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    device: torch.device,
    window_s: float = 2.0,
    stride_s: float = 1.0,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the encoder on overlapping windows of an audio file.

    Parameters
    ----------
    audio_path : str    path to any WAV file (any SR, mono or stereo)
    window_s   : float  window length in seconds  (should match training)
    stride_s   : float  hop between windows — smaller = smoother but slower
    batch_size : int    windows per inference batch

    Returns
    -------
    times       : (W,) float32  centre time (s) for each window
    snore_probs : (W,) float32  snore probability ∈ [0, 1]
    """
    # ── Load & pre-process ────────────────────────────────────────────────
    logger.info("Loading: %s", audio_path)
    audio, sr = sf.read(audio_path, always_2d=False, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != AST_SR:
        logger.info("Resampling %d → %d Hz …", sr, AST_SR)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AST_SR, res_type="soxr_hq")

    duration  = len(audio) / AST_SR
    win_smp   = int(window_s * AST_SR)
    stride_smp = int(stride_s * AST_SR)
    logger.info("Duration: %.1f s | windows: %d (stride=%.1f s)",
                duration, int(np.ceil(duration / stride_s)), stride_s)

    # ── Slice into windows ────────────────────────────────────────────────
    windows: List[np.ndarray] = []
    times:   List[float]      = []
    pos = 0
    while pos < len(audio):
        seg = audio[pos : pos + win_smp]
        if len(seg) < win_smp:
            seg = np.pad(seg, (0, win_smp - len(seg)))
        windows.append(seg)
        times.append((pos + win_smp / 2) / AST_SR)
        pos += stride_smp

    # ── Batched inference ─────────────────────────────────────────────────
    all_probs: List[np.ndarray] = []

    for i in tqdm(range(0, len(windows), batch_size),
                  desc="Encoder inference", unit="batch"):
        batch = windows[i : i + batch_size]
        inputs = feature_extractor(
            [w.tolist() for w in batch],
            sampling_rate=AST_SR,
            padding=True,
            return_tensors="pt",
        )
        logits = model(input_values=inputs["input_values"].to(device)).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()   # (B, 2)
        all_probs.append(probs)

    all_probs_arr = np.concatenate(all_probs, axis=0)   # (W, 2)
    snore_probs   = all_probs_arr[:, SNORE_ID]           # (W,)

    logger.info(
        "Inference done | max_prob=%.4f  mean_prob=%.5f  "
        "windows_above_0.5=%d/%d",
        snore_probs.max(), snore_probs.mean(),
        int((snore_probs > 0.5).sum()), len(snore_probs),
    )
    return np.array(times, dtype=np.float32), snore_probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Event grouping
# ---------------------------------------------------------------------------

def group_events(
    times:       np.ndarray,
    snore_probs: np.ndarray,
    threshold:      float = 0.50,
    min_duration_s: float = 0.40,
    merge_gap_s:    float = 0.50,
    padding_s:      float = 0.10,
) -> List[Dict[str, float]]:
    """
    Convert window-level snore probabilities into timestamped events.

    1. Threshold → binary ON/OFF per window
    2. Find contiguous ON runs
    3. Merge runs separated by < merge_gap_s
    4. Pad event boundaries
    5. Discard events shorter than min_duration_s

    Returns
    -------
    list of {'start', 'end', 'confidence', 'duration'}
    """
    duration_total = float(times[-1]) if len(times) > 0 else 0.0
    binary = (snore_probs >= threshold).astype(np.int8)

    # Find contiguous ON runs
    raw: List[Tuple[float, float, List[float]]] = []
    in_ev = False
    s_t, confs = 0.0, []

    for t, flag, prob in zip(times, binary, snore_probs):
        if flag == 1 and not in_ev:
            in_ev = True
            s_t, confs = float(t), [float(prob)]
        elif flag == 1 and in_ev:
            confs.append(float(prob))
        elif flag == 0 and in_ev:
            in_ev = False
            raw.append((s_t, float(t), confs))

    if in_ev:
        raw.append((s_t, float(times[-1]), confs))

    if not raw:
        return []

    # Merge close events
    merged: List[Tuple[float, float, List[float]]] = []
    cs, ce, cc = raw[0]
    for s, e, c in raw[1:]:
        if s - ce <= merge_gap_s:
            ce = e
            cc = cc + c
        else:
            merged.append((cs, ce, cc))
            cs, ce, cc = s, e, c
    merged.append((cs, ce, cc))

    # Pad + filter
    events: List[Dict[str, float]] = []
    for s, e, c in merged:
        s = max(0.0, s - padding_s)
        e = min(duration_total, e + padding_s)
        dur = e - s
        if dur >= min_duration_s:
            events.append({
                "start":      round(s, 3),
                "end":        round(e, 3),
                "duration":   round(dur, 3),
                "confidence": round(float(np.mean(c)), 4),
            })

    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Snore Encoder — run inference on a sleep recording",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",         "-i", required=True,
                   help="Path to the input WAV file.")
    p.add_argument("--model",         "-m", default="models/snore_encoder_best.pt",
                   help="Path to trained encoder checkpoint.")
    p.add_argument("--threshold",     type=float, default=0.50,
                   help="Snore probability threshold (lower = more sensitive).")
    p.add_argument("--window-stride", type=float, default=1.0,
                   help="Hop between analysis windows in seconds.")
    p.add_argument("--min-duration",  type=float, default=0.40,
                   help="Minimum snore event duration (s).")
    p.add_argument("--merge-gap",     type=float, default=0.50,
                   help="Merge events separated by less than this (s).")
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--output-csv",    default=None,
                   help="(Optional) Save detected events to a CSV file.")
    p.add_argument("--device",        default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    if not Path(args.model).exists():
        logger.error(
            "Checkpoint not found: %s\n"
            "Train the encoder first: python train_encoder.py --data-dir data/",
            args.model,
        )
        return

    # ── Load encoder ──────────────────────────────────────────────────────
    model, feature_extractor, config = load_encoder(args.model, device)
    window_s = config.get("window_s", 2.0)

    # ── Inference ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    times, snore_probs = predict_file(
        args.input, model, feature_extractor, device,
        window_s=window_s,
        stride_s=args.window_stride,
        batch_size=args.batch_size,
    )

    # ── Event grouping ────────────────────────────────────────────────────
    events = group_events(
        times, snore_probs,
        threshold=args.threshold,
        min_duration_s=args.min_duration,
        merge_gap_s=args.merge_gap,
    )
    elapsed = time.perf_counter() - t0

    # ── Print results ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  Results")
    logger.info("=" * 55)
    logger.info("Inference time    : %.1f s", elapsed)
    logger.info("Snore events found: %d", len(events))

    if events:
        total_snore = sum(e["duration"] for e in events)
        audio_dur   = float(times[-1]) if len(times) > 0 else 0.0
        logger.info("Total snore time  : %.1f s (%.1f%% of recording)",
                    total_snore, 100.0 * total_snore / max(audio_dur, 1))
        logger.info("")
        for i, ev in enumerate(events):
            m_s = int(ev["start"] // 60);  s_s = ev["start"] % 60
            m_e = int(ev["end"]   // 60);  s_e = ev["end"]   % 60
            logger.info(
                "  [%4d]  %d:%05.2f → %d:%05.2f  (%5.2f s)  conf=%.3f",
                i + 1, m_s, s_s, m_e, s_e, ev["duration"], ev["confidence"],
            )
    else:
        logger.info(
            "No snore events detected.  "
            "Try lowering --threshold (current: %.2f).", args.threshold
        )

    # ── Save CSV ──────────────────────────────────────────────────────────
    if args.output_csv and events:
        out_path = Path(args.output_csv)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["start_time", "end_time", "duration", "confidence"]
            )
            writer.writeheader()
            for ev in events:
                writer.writerow({
                    "start_time": ev["start"],
                    "end_time":   ev["end"],
                    "duration":   ev["duration"],
                    "confidence": ev["confidence"],
                })
        logger.info("Results saved to: %s", out_path)


if __name__ == "__main__":
    main()
