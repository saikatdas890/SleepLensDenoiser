"""
SleepLensDenoiser — Main Pipeline
===================================
End-to-end orchestrator for sleep-audio snore extraction.

Pipeline stages
---------------
1. **Denoise**  (``denoise_audio.py``)
   Remove fan / AC / traffic noise via Demucs or spectral gating.

2. **Feature extraction**  (``feature_utils.py``)
   Compute MFCCs, spectral features, HNR and log-mel spectrograms at
   25 ms / 10 ms frame resolution (available for downstream analysis).

3. **Snore detection**  (``snore_model.py``)
   AST fine-tuned on AudioSet produces per-frame snore probabilities.

4. **Event grouping**  (``detect_snore_segments.py``)
   Smooth → threshold → merge → filter → timestamped snore events.

5. **Clip extraction**  (``cut_snore_clips.py``)
   Cut snore segments from the *original* (pre-denoised) recording and
   save them as individual WAVs with a CSV manifest.

Usage
-----
.. code-block:: bash

    python main.py --input recording.wav --output output/

    # Use spectral gating denoiser (no model download):
    python main.py --input recording.wav --output output/ \\
                   --denoise-method spectral

    # Skip denoising if you already ran it once:
    python main.py --input recording.wav --output output/ \\
                   --skip-denoise

    # Tune detection sensitivity:
    python main.py --input recording.wav --output output/ \\
                   --threshold 0.25 --min-duration 0.3 --merge-gap 0.5

    # Use your own fine-tuned encoder instead of the generic AST model:
    python main.py --input recording.wav --output output/ \\
                   --encoder-model models/snore_encoder_best.pt

Output layout
-------------
.. code-block::

    output/
    ├── denoised.wav
    ├── snore_segments.csv
    └── snore_clips/
        ├── snore_0001.wav
        └── …
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy third-party loggers at DEBUG level
    for noisy in ("transformers", "torch", "demucs", "librosa", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sleeplens",
        description="SleepLensDenoiser — snore extraction from sleep recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument(
        "--input", "-i", required=True,
        help="Path to the raw sleep recording WAV file.",
    )
    io.add_argument(
        "--output", "-o", default="output/",
        help="Root directory for all output artefacts.",
    )

    denoise = p.add_argument_group("Stage 1 — Denoising")
    denoise.add_argument(
        "--denoise-method", choices=["demucs", "spectral"], default="demucs",
        help=(
            "'demucs'   : Demucs htdemucs deep-learning separator (recommended).\n"
            "'spectral' : noisereduce adaptive spectral gating (no GPU needed)."
        ),
    )
    denoise.add_argument(
        "--skip-denoise", action="store_true",
        help=(
            "Skip Stage 1 and reuse an existing denoised.wav in --output.  "
            "Useful when re-running detection with different parameters."
        ),
    )

    detect = p.add_argument_group("Stage 3–4 — Detection")
    detect.add_argument(
        "--encoder-model", default=None,
        metavar="PATH",
        help=(
            "Path to a fine-tuned encoder checkpoint produced by train_encoder.py "
            "(e.g. models/snore_encoder_best.pt).  "
            "When supplied this replaces the generic AudioSet AST with your "
            "personalised model — more accurate for your specific recording setup."
        ),
    )
    detect.add_argument(
        "--threshold", type=float, default=0.30,
        help="Snore probability threshold (lower = more sensitive).",
    )
    detect.add_argument(
        "--min-duration", type=float, default=0.40,
        help="Minimum snore event duration in seconds.",
    )
    detect.add_argument(
        "--merge-gap", type=float, default=0.30,
        help="Merge snore events separated by less than this gap (seconds).",
    )
    detect.add_argument(
        "--padding", type=float, default=0.10,
        help="Pad each event boundary by this amount (seconds).",
    )
    detect.add_argument(
        "--window-size", type=float, default=2.0,
        help="AST analysis window size (seconds).",
    )
    detect.add_argument(
        "--window-stride", type=float, default=0.5,
        help="AST analysis window stride / hop (seconds).",
    )
    detect.add_argument(
        "--batch-size", type=int, default=8,
        help="AST inference batch size (reduce if OOM).",
    )

    hw = p.add_argument_group("Hardware")
    hw.add_argument(
        "--device", default=None,
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'.  "
             "Auto-detected if omitted.",
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    misc.add_argument(
        "--skip-features", action="store_true",
        help=(
            "Skip Stage 2 frame-feature extraction.  "
            "Features are not needed for detection but can be useful for analysis."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Stage helpers (with wall-clock timing)
# ---------------------------------------------------------------------------

def _stage(name: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", name)
    logger.info("=" * 60)


def _elapsed(t0: float) -> str:
    secs = time.perf_counter() - t0
    if secs < 60:
        return f"{secs:.1f} s"
    return f"{int(secs // 60)}m {int(secs % 60)}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    """
    Run the full SleepLensDenoiser pipeline.

    Returns
    -------
    int  exit code (0 = success)
    """
    parser  = build_parser()
    args    = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger  = logging.getLogger(__name__)

    # ── Resolve device ───────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("SleepLensDenoiser  |  device=%s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  VRAM: %.1f GB",
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_properties(device).total_memory / 1e9,
        )

    # ── Validate inputs ──────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    denoised_path = str(output_dir / "denoised.wav")

    pipeline_t0 = time.perf_counter()

    # ════════════════════════════════════════════════════════════════════════
    # Stage 1: Denoise
    # ════════════════════════════════════════════════════════════════════════
    _stage("Stage 1 — Denoising")

    if args.skip_denoise:
        if not Path(denoised_path).exists():
            logger.error(
                "--skip-denoise specified but %s does not exist.  "
                "Run without --skip-denoise first.", denoised_path
            )
            return 1
        logger.info("Skipping denoising — using existing: %s", denoised_path)
    else:
        from denoise_audio import denoise_audio

        t0 = time.perf_counter()
        denoise_audio(
            input_path=str(input_path),
            output_path=denoised_path,
            method=args.denoise_method,
            device=device,
        )
        logger.info("Stage 1 complete in %s", _elapsed(t0))

    # ════════════════════════════════════════════════════════════════════════
    # Stage 2: Feature Extraction (optional — analytical, not required for detection)
    # ════════════════════════════════════════════════════════════════════════
    _stage("Stage 2 — Frame-Level Feature Extraction")

    if args.skip_features:
        logger.info("Skipping feature extraction (--skip-features).")
    else:
        from feature_utils import extract_features, stack_features, normalize_features

        t0 = time.perf_counter()
        logger.info("Loading denoised audio for feature extraction …")
        audio_data, sr = sf.read(denoised_path, always_2d=True, dtype="float32")
        audio_mono = audio_data.mean(axis=1)

        feats = extract_features(audio_mono, sr)
        X = normalize_features(stack_features(feats))  # (D, T)

        feat_path = str(output_dir / "frame_features.npz")
        np.savez_compressed(
            feat_path,
            features=X,
            times=feats.times,
            sr=np.array(feats.sr),
            hop_length=np.array(feats.hop_length),
        )
        logger.info(
            "Stage 2 complete in %s — features shape=%s saved to %s",
            _elapsed(t0), X.shape, feat_path,
        )

    # ════════════════════════════════════════════════════════════════════════
    # Stage 3: Load Snore Detection Model
    # ════════════════════════════════════════════════════════════════════════
    using_personal_encoder = bool(args.encoder_model)

    if using_personal_encoder:
        _stage("Stage 3 — Loading Personal Snore Encoder (fine-tuned)")
        from inference import load_encoder

        encoder_path = Path(args.encoder_model)
        if not encoder_path.exists():
            logger.error(
                "Encoder checkpoint not found: %s\n"
                "Train it first:  python train_encoder.py --data-dir data/",
                encoder_path,
            )
            return 1

        t0 = time.perf_counter()
        encoder_model, encoder_fx, encoder_cfg = load_encoder(str(encoder_path), device)
        logger.info("Personal encoder loaded in %s", _elapsed(t0))
    else:
        _stage("Stage 3 — Loading Snore Detection Model (AST/AudioSet zero-shot)")
        from snore_model import SnoreDetector

        t0 = time.perf_counter()
        detector = SnoreDetector(
            device=device,
            window_size_s=args.window_size,
            stride_s=args.window_stride,
            batch_size=args.batch_size,
        )
        logger.info("Model loaded in %s", _elapsed(t0))

    # ════════════════════════════════════════════════════════════════════════
    # Stage 4: Detect Snore Segments
    # ════════════════════════════════════════════════════════════════════════
    _stage("Stage 4 — Snore Event Detection")

    t0 = time.perf_counter()

    if using_personal_encoder:
        # ── Personal encoder path ─────────────────────────────────────────
        from inference import predict_file, group_events

        times, snore_probs = predict_file(
            denoised_path,
            encoder_model,
            encoder_fx,
            device,
            window_s=encoder_cfg.get("window_s", 2.0),
            stride_s=args.window_stride,
            batch_size=args.batch_size,
        )
        segments = group_events(
            times, snore_probs,
            threshold=args.threshold,
            min_duration_s=args.min_duration,
            merge_gap_s=args.merge_gap,
            padding_s=args.padding,
        )
    else:
        # ── Generic AST path ──────────────────────────────────────────────
        from detect_snore_segments import detect_snore_segments

        segments = detect_snore_segments(
            audio_path=denoised_path,
            detector=detector,
            threshold=args.threshold,
            min_duration_s=args.min_duration,
            merge_gap_s=args.merge_gap,
            padding_s=args.padding,
        )

    logger.info(
        "Stage 4 complete in %s — %d event(s) detected",
        _elapsed(t0), len(segments),
    )

    # ════════════════════════════════════════════════════════════════════════
    # Stage 5: Cut Snore Clips
    # ════════════════════════════════════════════════════════════════════════
    _stage("Stage 5 — Extracting Snore Clips")

    from cut_snore_clips import cut_snore_clips

    t0 = time.perf_counter()
    csv_path = cut_snore_clips(
        original_audio_path=str(input_path),
        segments=segments,
        output_dir=str(output_dir),
    )
    logger.info("Stage 5 complete in %s", _elapsed(t0))

    # ════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════
    _stage("Pipeline Complete")
    logger.info("Total wall-clock time : %s", _elapsed(pipeline_t0))
    logger.info("Snore events detected : %d", len(segments))
    logger.info("Clip directory        : %s", output_dir / "snore_clips")
    logger.info("CSV manifest          : %s", csv_path)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review clips in: %s", output_dir / "snore_clips")
    logger.info("  2. Check timestamps: %s", csv_path)
    logger.info(
        "  3. If too many/few clips, re-run with "
        "--threshold (lower=more sensitive, higher=stricter)"
    )
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
