"""
Stage 1 — Audio Denoising
=========================
Removes background noise (fan, AC, traffic) from a raw sleep recording.

Primary method  : Demucs ``htdemucs``
  Demucs is a deep neural-network source-separator trained on music and speech.
  For sleep recordings it works as a denoiser by extracting the *vocals* stem,
  which captures biological sounds (breathing, snoring) while discarding the
  stationary ambient stems (drums ≈ transients, bass ≈ low rumble, other ≈
  diffuse noise).  The ``apply_model`` chunked-processing mode keeps GPU/RAM
  usage bounded even for 8-hour files.

Fallback method : ``noisereduce`` adaptive spectral gating
  Uses the first N seconds of the recording as a noise reference and applies
  a non-stationary spectral gate that adapts over time.  No model download
  required; works fully offline.

Usage (standalone)::

    from denoise_audio import denoise_audio
    out_path, sr = denoise_audio("recording.wav", "output/denoised.wav",
                                  method="demucs")

CLI (via main.py)::

    python main.py --input recording.wav --output output/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_audio_native(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file at its native sample rate without any resampling.

    Returns
    -------
    audio : np.ndarray, shape ``(channels, samples)``, float32
    sr    : int  — native sample rate
    """
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile returns (samples, channels); transpose to (channels, samples)
    return audio.T, sr


def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """
    Write a float32 audio array to a WAV file.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(channels, samples)`` or ``(samples,)``
    sr    : int
    path  : str
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = audio.T if audio.ndim == 2 else audio   # soundfile wants (samples, channels)
    sf.write(path, out.astype(np.float32), sr)
    logger.info("Saved audio → %s  (shape=%s, sr=%d Hz)", path, out.shape, sr)


# ---------------------------------------------------------------------------
# Demucs denoiser
# ---------------------------------------------------------------------------

def denoise_with_demucs(
    audio: np.ndarray,
    sr: int,
    device: torch.device,
    model_name: str = "htdemucs",
    segment_s: float = 10.0,
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Separate biological sounds from background noise using Demucs.

    Demucs splits audio into four stems:
    ``drums | bass | other | vocals``
    The **vocals** stem retains breathing and snoring while discarding
    fan/AC/traffic noise spread across the other stems.

    Parameters
    ----------
    audio      : np.ndarray  ``(channels, samples)``, float32, arbitrary SR
    sr         : int          native sample rate of *audio*
    device     : torch.device
    model_name : str          Demucs model tag (default ``htdemucs``)
    segment_s  : float        chunk duration fed to ``apply_model`` to cap RAM
    overlap    : float        overlap ratio between consecutive chunks (0–0.5)

    Returns
    -------
    np.ndarray  ``(channels, samples)``, float32, same SR as input
    """
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as exc:
        raise ImportError(
            "demucs is not installed.  Run: pip install demucs>=4.0.0"
        ) from exc

    logger.info("[Demucs] Loading model '%s' …", model_name)
    model = get_model(model_name)
    model.to(device)
    model.eval()

    model_sr: int = model.samplerate   # htdemucs → 44100
    logger.info("[Demucs] Model SR: %d Hz", model_sr)

    # ── Resample to model's native SR ──────────────────────────────────────
    if sr != model_sr:
        logger.info("[Demucs] Resampling %d → %d Hz …", sr, model_sr)
        audio_r = librosa.resample(
            audio, orig_sr=sr, target_sr=model_sr, res_type="soxr_hq"
        )
    else:
        audio_r = audio.copy()

    # ── Demucs requires exactly 2 channels ────────────────────────────────
    if audio_r.shape[0] == 1:
        audio_r = np.concatenate([audio_r, audio_r], axis=0)   # mono → fake stereo
    elif audio_r.shape[0] > 2:
        audio_r = audio_r[:2]

    mix_tensor = torch.tensor(audio_r, dtype=torch.float32).unsqueeze(0).to(device)
    # shape: (1, 2, T)

    duration_s = mix_tensor.shape[-1] / model_sr
    logger.info("[Demucs] Separating %.1f s of audio …", duration_s)

    with torch.no_grad():
        sources = apply_model(
            model,
            mix_tensor,
            device=device,
            progress=True,
            segment=segment_s,
            overlap=overlap,
            num_workers=0,
        )
    # sources: (batch=1, n_stems, 2, T)

    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx].cpu().numpy()  # (2, T)
    logger.info("[Demucs] Extracted 'vocals' stem: %s", vocals.shape)

    # ── Resample back to original SR ───────────────────────────────────────
    if sr != model_sr:
        vocals = librosa.resample(
            vocals, orig_sr=model_sr, target_sr=sr, res_type="soxr_hq"
        )

    # ── Match original channel count ───────────────────────────────────────
    if audio.shape[0] == 1:
        vocals = vocals.mean(axis=0, keepdims=True)
    # else keep stereo (2, T) as-is

    return vocals.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral-gating fallback
# ---------------------------------------------------------------------------

def denoise_with_spectral_gating(
    audio: np.ndarray,
    sr: int,
    noise_duration_s: float = 2.0,
    prop_decrease: float = 0.80,
    n_std_thresh: float = 1.5,
    chunk_s: float = 30.0,
) -> np.ndarray:
    """
    Denoise using ``noisereduce``'s non-stationary adaptive spectral gate.

    The first ``noise_duration_s`` seconds are used as a noise reference
    profile.  The gate is applied channel-by-channel in chunks to limit RAM.

    Parameters
    ----------
    audio            : np.ndarray  ``(channels, samples)``, float32
    sr               : int
    noise_duration_s : float  seconds of initial recording to use as noise ref
    prop_decrease    : float  noise attenuation strength  0 = none, 1 = full
    n_std_thresh     : float  threshold multiplier for noise floor estimate
    chunk_s          : float  processing chunk size in seconds

    Returns
    -------
    np.ndarray  ``(channels, samples)``, float32
    """
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise ImportError(
            "noisereduce is not installed.  Run: pip install noisereduce>=3.0.0"
        ) from exc

    logger.info("[SpectralGate] Applying adaptive spectral gating …")
    noise_samples = int(noise_duration_s * sr)
    chunk_samples = int(chunk_s * sr)
    n_ch = audio.shape[0]
    denoised = np.empty_like(audio)

    for ch in range(n_ch):
        sig = audio[ch]
        noise_ref = sig[:noise_samples]
        ch_out_parts: list[np.ndarray] = []

        # Process in chunks with a small overlap to reduce boundary artefacts
        pad = min(int(sr * 0.5), noise_samples)
        start = 0
        while start < len(sig):
            end = min(start + chunk_samples, len(sig))
            chunk = sig[max(0, start - pad) : end]
            reduced = nr.reduce_noise(
                y=chunk,
                y_noise=noise_ref,
                sr=sr,
                prop_decrease=prop_decrease,
                stationary=False,
                n_std_thresh_stationary=n_std_thresh,
            )
            # Trim leading pad from all chunks except the first
            trim = pad if start > 0 else 0
            ch_out_parts.append(reduced[trim:])
            start += chunk_samples

        denoised[ch] = np.concatenate(ch_out_parts)[: len(sig)]

    logger.info("[SpectralGate] Done.")
    return denoised.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def denoise_audio(
    input_path: str,
    output_path: str,
    method: str = "demucs",
    device: Optional[torch.device] = None,
) -> Tuple[str, int]:
    """
    Denoise a sleep recording WAV and save the result.

    Parameters
    ----------
    input_path  : str          path to the raw WAV file
    output_path : str          where to write the denoised WAV
    method      : str          ``'demucs'`` (default) or ``'spectral'``
    device      : torch.device auto-detected if *None*

    Returns
    -------
    (output_path, sample_rate)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("[Denoise] method=%s | device=%s", method, device)

    audio, sr = load_audio_native(input_path)
    duration = audio.shape[-1] / sr
    logger.info(
        "[Denoise] Loaded '%s'  channels=%d  SR=%d Hz  duration=%.1f s",
        input_path, audio.shape[0], sr, duration,
    )

    if method == "demucs":
        try:
            denoised = denoise_with_demucs(audio, sr, device)
        except Exception as exc:
            logger.warning(
                "[Denoise] Demucs failed (%s).  Falling back to spectral gating.", exc
            )
            denoised = denoise_with_spectral_gating(audio, sr)

    elif method == "spectral":
        denoised = denoise_with_spectral_gating(audio, sr)

    else:
        raise ValueError(
            f"Unknown denoising method '{method}'.  Choose 'demucs' or 'spectral'."
        )

    save_audio(denoised, sr, output_path)
    return output_path, sr
