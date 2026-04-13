"""
Stage 3 — Snore Detection Model
================================
Zero-shot snore detector built on the Audio Spectrogram Transformer (AST)
fine-tuned on AudioSet (``MIT/ast-finetuned-audioset-10-10-0.4593``).

Why AST + AudioSet?
-------------------
AudioSet contains **527 sound classes** including "Snoring", "Breathing",
and "Wheeze" — all directly relevant to sleep-audio analysis.  The pretrained
model can classify snoring without any additional labelled training data.

Architecture
------------
1. Raw audio at 16 kHz is converted to a log-mel spectrogram by
   ``ASTFeatureExtractor`` (128 mel bins, 10 ms hop, up to 1024 frames).
2. ``ASTForAudioClassification`` processes the spectrogram patch tokens and
   outputs logits over 527 AudioSet classes.
3. Softmax is applied; the probabilities for snoring-related classes are
   combined into a single *snore score* per audio window.
4. Windows are processed with configurable overlap and results are
   interpolated to a 10 ms frame grid via ``predict_framewise``.

GPU / CPU
---------
``SnoreDetector`` automatically moves the model to CUDA if available.
Batch inference amortises the per-call overhead.

Usage::

    from snore_model import SnoreDetector
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read("denoised.wav")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    detector = SnoreDetector()
    frame_probs, frame_times = detector.predict_framewise(audio.astype("float32"), sr)
    # frame_probs: (T,) in [0, 1],  frame_times: (T,) seconds
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

logger = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16_000   # AST was trained on 16 kHz audio

# AudioSet label keywords — searched case-insensitively in id2label
_SNORE_KEYWORDS    = ["snor"]
_BREATHING_KEYWORDS = ["breath", "wheez", "snort"]

# Weight applied to the snoring-class score vs. breathing classes
# (snoring gets 2× weight; breathing classes are supporting evidence)
_SNORE_WEIGHT    = 2.0
_BREATHING_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def _find_label_indices(
    id2label: Dict[int, str],
    keywords: List[str],
) -> List[int]:
    """
    Return all class indices whose label strings contain any keyword.

    Parameters
    ----------
    id2label : dict  mapping int → label string  (from model.config.id2label)
    keywords : list[str]  substrings to search for (case-insensitive)

    Returns
    -------
    list[int]  sorted class indices
    """
    matched: List[int] = []
    for idx, label in id2label.items():
        if any(kw in label.lower() for kw in keywords):
            matched.append(int(idx))
            logger.info("  class %4d → '%s'", int(idx), label)
    return sorted(matched)


# ---------------------------------------------------------------------------
# SnoreDetector
# ---------------------------------------------------------------------------

class SnoreDetector:
    """
    Zero-shot snore detector using AST fine-tuned on AudioSet.

    Parameters
    ----------
    model_id      : str    HuggingFace model identifier
    device        : torch.device  auto-detected if None
    window_size_s : float  analysis window length in seconds (default 2.0)
    stride_s      : float  hop between windows in seconds    (default 0.5)
    batch_size    : int    windows per inference batch        (default 8)
    """

    def __init__(
        self,
        model_id:      str                     = AST_MODEL_ID,
        device:        Optional[torch.device]  = None,
        window_size_s: float                   = 2.0,
        stride_s:      float                   = 0.5,
        batch_size:    int                     = 8,
    ) -> None:
        self.model_id      = model_id
        self.device        = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.window_size_s = window_size_s
        self.stride_s      = stride_s
        self.batch_size    = batch_size

        self._load_model()
        self._resolve_target_classes()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        logger.info("[SnoreDetector] Downloading / loading '%s' …", self.model_id)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_id)
        self.model = ASTForAudioClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "[SnoreDetector] Model ready on %s  (%d output classes)",
            self.device, self.model.config.num_labels,
        )

    def _resolve_target_classes(self) -> None:
        """
        Identify AudioSet class indices for snoring and related sounds.
        Falls back to breathing classes if no explicit snoring class exists.
        """
        id2label: Dict[int, str] = self.model.config.id2label

        logger.info("[SnoreDetector] Scanning for snoring-related AudioSet labels:")
        self.snore_indices    = _find_label_indices(id2label, _SNORE_KEYWORDS)
        logger.info("[SnoreDetector] Scanning for breathing-related AudioSet labels:")
        self.breathing_indices = _find_label_indices(id2label, _BREATHING_KEYWORDS)

        if not self.snore_indices:
            logger.warning(
                "[SnoreDetector] No explicit 'snoring' label found.  "
                "Using breathing-class indices as primary signal."
            )
            self.snore_indices = self.breathing_indices
            self.breathing_indices = []

        if not self.snore_indices and not self.breathing_indices:
            raise RuntimeError(
                "Could not find snoring or breathing classes in the AudioSet model.  "
                "Check the model's id2label mapping."
            )

        logger.info(
            "[SnoreDetector] Snore indices:    %s", self.snore_indices
        )
        logger.info(
            "[SnoreDetector] Breathing indices: %s", self.breathing_indices
        )

    # ── Window construction ───────────────────────────────────────────────────

    def _make_windows(
        self,
        audio_16k: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Slice 1-D 16 kHz audio into overlapping, zero-padded windows.

        Returns
        -------
        windows : list of 1-D arrays, each ``win_samples`` long
        centers : (W,) array of centre-times in seconds for each window
        """
        win_samples    = int(self.window_size_s * AST_SR)
        stride_samples = int(self.stride_s      * AST_SR)
        n              = len(audio_16k)

        windows: List[np.ndarray] = []
        centers: List[float]      = []

        pos = 0
        while pos < n:
            seg = audio_16k[pos : pos + win_samples]
            if len(seg) < win_samples:
                seg = np.pad(seg, (0, win_samples - len(seg)))
            windows.append(seg)
            centers.append((pos + win_samples / 2) / AST_SR)
            pos += stride_samples

        return windows, np.array(centers, dtype=np.float32)

    # ── Batched inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(self, windows: List[np.ndarray]) -> np.ndarray:
        """
        Run AST on a list of 1-D float32 arrays (all at 16 kHz).

        Returns
        -------
        np.ndarray  shape ``(B, n_classes)``  softmax probabilities
        """
        # ASTFeatureExtractor expects a list of Python lists or 1-D arrays
        inputs = self.feature_extractor(
            [w.tolist() for w in windows],
            sampling_rate=AST_SR,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        # input_values: (B, n_frames, n_mels)

        logits = self.model(input_values=input_values).logits  # (B, 527)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()   # (B, 527)
        return probs

    # ── Score aggregation ─────────────────────────────────────────────────────

    def _aggregate_score(self, probs: np.ndarray) -> float:
        """
        Compute a single snore score from a ``(527,)`` probability vector.

        Combines the snoring class probability (weighted 2×) with supporting
        breathing-class probabilities (weighted 1×), then normalises to [0, 1].
        """
        score = 0.0
        total_weight = 0.0

        if self.snore_indices:
            score        += _SNORE_WEIGHT * float(probs[self.snore_indices].max())
            total_weight += _SNORE_WEIGHT

        if self.breathing_indices:
            score        += _BREATHING_WEIGHT * float(probs[self.breathing_indices].max())
            total_weight += _BREATHING_WEIGHT

        if total_weight == 0:
            return 0.0
        return min(score / total_weight, 1.0)

    # ── Public prediction methods ─────────────────────────────────────────────

    def predict_windows(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run AST-based snore detection on overlapping windows.

        Parameters
        ----------
        audio : np.ndarray  1-D float32, arbitrary sample rate
        sr    : int          sample rate of *audio*

        Returns
        -------
        snore_probs : (W,) float32  snore score per window
        centers     : (W,) float32  centre time (seconds) per window
        """
        # Resample to AST's required 16 kHz
        if sr != AST_SR:
            logger.info("[SnoreDetector] Resampling %d → %d Hz …", sr, AST_SR)
            audio_16k = librosa.resample(
                audio, orig_sr=sr, target_sr=AST_SR, res_type="soxr_hq"
            )
        else:
            audio_16k = audio

        windows, centers = self._make_windows(audio_16k)
        n_windows = len(windows)
        logger.info(
            "[SnoreDetector] Processing %d windows "
            "(window=%.1fs, stride=%.1fs) …",
            n_windows, self.window_size_s, self.stride_s,
        )

        all_snore_probs: List[float] = []

        for batch_start in tqdm(
            range(0, n_windows, self.batch_size),
            desc="AST inference",
            unit="batch",
        ):
            batch  = windows[batch_start : batch_start + self.batch_size]
            probs  = self._infer_batch(batch)          # (B, 527)
            for row in probs:
                all_snore_probs.append(self._aggregate_score(row))

        snore_probs = np.array(all_snore_probs, dtype=np.float32)
        logger.info(
            "[SnoreDetector] Window stats: max=%.4f  mean=%.4f  "
            "windows_above_0.3=%d/%d",
            snore_probs.max(), snore_probs.mean(),
            int((snore_probs > 0.3).sum()), n_windows,
        )
        return snore_probs, centers

    def predict_framewise(
        self,
        audio: np.ndarray,
        sr: int,
        frame_hop_s: float = 0.010,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce per-frame snore probabilities at ``frame_hop_s`` resolution.

        Window-level scores are linearly interpolated onto the target
        time grid.  This produces a smooth probability trace suitable
        for thresholding and event grouping.

        Parameters
        ----------
        audio       : np.ndarray  1-D float32 at *sr* Hz
        sr          : int
        frame_hop_s : float  output frame resolution in seconds (default 10 ms)

        Returns
        -------
        frame_probs : (T,) float32  snore probability per frame ∈ [0, 1]
        frame_times : (T,) float32  time in seconds for each frame
        """
        duration_s = len(audio) / sr

        snore_probs, centers = self.predict_windows(audio, sr)

        # Build output time grid
        frame_times = np.arange(0.0, duration_s, frame_hop_s, dtype=np.float32)

        # Edge-fill: clamp interpolation to first/last window centre
        #   np.interp uses the boundary values outside [xp[0], xp[-1]]
        frame_probs = np.interp(frame_times, centers, snore_probs).astype(np.float32)
        frame_probs = np.clip(frame_probs, 0.0, 1.0)

        logger.info(
            "[SnoreDetector] Frame probs: T=%d | max=%.4f | mean=%.5f",
            len(frame_probs), frame_probs.max(), frame_probs.mean(),
        )
        return frame_probs, frame_times
