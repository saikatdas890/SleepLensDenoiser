"""
Snore Encoder — Fine-Tuning Script
====================================
Fine-tunes the AST (Audio Spectrogram Transformer) pretrained on AudioSet
with a new binary classification head for snore vs. background detection.

Strategy
--------
* The AST encoder backbone is **frozen** — its weights are not updated.
* Only the last ``--unfreeze-last-n`` encoder layers + the new head are trained.
* This makes training fast (~minutes on CPU) and prevents overfitting on small
  datasets.

Model architecture::

    Input WAV (2 s, 16 kHz)
         ↓
    ASTFeatureExtractor  →  log-mel spectrogram (n_frames × 128)
         ↓
    AST Transformer Encoder  (frozen)
         ↓
    Pooled CLS embedding  (768-dim)
         ↓
    LayerNorm → Dropout(0.3) → Linear(768→256) → GELU → Dropout(0.2) → Linear(256→2)
         ↓
    [background_prob, snore_prob]

Outputs
-------
    models/
    ├── snore_encoder_best.pt     ← best checkpoint by val accuracy
    └── training_history.json    ← per-epoch metrics

Usage::

    python train_encoder.py --data-dir data/

    # More epochs, unfreeze last 4 layers for better accuracy:
    python train_encoder.py --data-dir data/ --epochs 30 --unfreeze-last-n 4

    # Smaller batch if running low on RAM:
    python train_encoder.py --data-dir data/ --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Constants (must match prepare_dataset.py and inference.py) ───────────────
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR       = 16_000
WINDOW_S     = 2.0
LABELS       = ["background", "snore"]   # index 0 = background, 1 = snore


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SnoreAudioDataset(Dataset):
    """
    Loads WAV clips from::

        data_dir/
          {split}/snore/       → label 1
          {split}/background/  → label 0

    Applies optional data augmentation at the audio level.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        feature_extractor: ASTFeatureExtractor,
        augment: bool = False,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []

        for label_id, label_name in enumerate(LABELS):
            label_dir = Path(data_dir) / split / label_name
            if not label_dir.exists():
                logger.warning("Directory not found: %s", label_dir)
                continue
            for wav in sorted(label_dir.glob("*.wav")):
                self.samples.append((str(wav), label_id))

        counts = {l: sum(1 for _, li in self.samples if li == i)
                  for i, l in enumerate(LABELS)}
        logger.info("[%s] %d samples — %s", split, len(self.samples), counts)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        """Load WAV → mono float32 at AST_SR, padded/truncated to WINDOW_S."""
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != AST_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AST_SR,
                                     res_type="soxr_hq")
        target = int(WINDOW_S * AST_SR)
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        return audio

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Light augmentations that preserve snore identity:
          - Random amplitude scaling     (±30 %)
          - Small Gaussian noise         (SNR ≈ 40 dB)
          - Random circular time shift   (±200 ms)
        """
        # Amplitude scaling
        audio = audio * random.uniform(0.70, 1.30)

        # Additive noise
        if random.random() < 0.4:
            noise_std = random.uniform(1e-4, 5e-3)
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_std

        # Time shift
        if random.random() < 0.4:
            max_shift = int(0.20 * AST_SR)
            shift = random.randint(-max_shift, max_shift)
            audio = np.roll(audio, shift)

        return np.clip(audio, -1.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        audio = self._load(path)
        if self.augment:
            audio = self._augment(audio)

        inputs = self.feature_extractor(
            [audio.tolist()],
            sampling_rate=AST_SR,
            padding=True,
            return_tensors="pt",
        )
        return inputs["input_values"].squeeze(0), label   # (n_frames, n_mels), int


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    model_id: str,
    unfreeze_last_n: int = 2,
) -> ASTForAudioClassification:
    """
    Load pretrained AST, freeze the encoder, and attach a new binary head.

    Parameters
    ----------
    model_id        : str  HuggingFace model identifier
    unfreeze_last_n : int  number of last encoder layers to unfreeze (0 = head only)

    Returns
    -------
    ASTForAudioClassification with trainable binary classifier head
    """
    logger.info("Loading pretrained AST: %s", model_id)
    model = ASTForAudioClassification.from_pretrained(model_id)

    # ── Freeze entire encoder ────────────────────────────────────────────
    for param in model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False

    # ── Selectively unfreeze last N transformer layers ───────────────────
    if unfreeze_last_n > 0:
        encoder_layers = model.audio_spectrogram_transformer.encoder.layer
        n_total = len(encoder_layers)
        for layer in encoder_layers[max(0, n_total - unfreeze_last_n):]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info("Unfrozen last %d / %d encoder layers", unfreeze_last_n, n_total)

    # ── Replace classification head with binary head ─────────────────────
    hidden = model.config.hidden_size   # 768 for AST-base
    model.classifier = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(p=0.30),
        nn.Linear(hidden, 256),
        nn.GELU(),
        nn.Dropout(p=0.20),
        nn.Linear(256, 2),
    )

    # Update model config so it's self-consistent
    model.config.num_labels = 2
    model.config.id2label   = {0: "background", 1: "snore"}
    model.config.label2id   = {"background": 0, "snore": 1}

    # ── Parameter summary ────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s  (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )
    return model


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for input_values, labels in tqdm(loader, desc="  train", leave=False, unit="batch"):
        input_values = input_values.to(device)
        labels       = torch.as_tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(input_values=input_values).logits
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / len(loader), correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for input_values, labels in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        input_values = input_values.to(device)
        labels_t     = torch.as_tensor(labels, dtype=torch.long).to(device)

        logits = model(input_values=input_values).logits
        total_loss += criterion(logits, labels_t).item()

        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels_t.cpu().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return total_loss / len(loader), acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune AST encoder for binary snore classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",        default="data/",
                   help="Dataset root (output of prepare_dataset.py).")
    p.add_argument("--output-dir",      default="models/",
                   help="Where to save the trained model and history.")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--unfreeze-last-n", type=int,   default=2,
                   help="Unfreeze this many final AST encoder layers.")
    p.add_argument("--patience",        type=int,   default=5,
                   help="Early-stopping patience (epochs without val improvement).")
    p.add_argument("--device",          default=None,
                   help="'cuda', 'cpu', etc.  Auto-detected if omitted.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature extractor (shared, no weights to train) ───────────────────
    logger.info("Loading AST feature extractor …")
    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SnoreAudioDataset(args.data_dir, "train", feature_extractor, augment=True)
    val_ds   = SnoreAudioDataset(args.data_dir, "val",   feature_extractor, augment=False)
    test_ds  = SnoreAudioDataset(args.data_dir, "test",  feature_extractor, augment=False)

    if len(train_ds) == 0:
        logger.error(
            "Training set is empty.  Run prepare_dataset.py first:\n"
            "  python prepare_dataset.py --original-audio <path>"
        )
        return

    # Class-weighted loss to handle any residual imbalance
    train_labels = [li for _, li in train_ds.samples]
    n_bg = train_labels.count(0)
    n_sn = train_labels.count(1)
    n_total = n_bg + n_sn
    class_weights = torch.tensor(
        [n_total / (2.0 * max(n_bg, 1)), n_total / (2.0 * max(n_sn, 1))],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Class weights: background=%.3f  snore=%.3f",
                class_weights[0].item(), class_weights[1].item())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(AST_MODEL_ID, unfreeze_last_n=args.unfreeze_last_n)
    model.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc    = 0.0
    best_epoch      = 0
    patience_ctr    = 0
    history         = []

    logger.info(
        "Starting training: %d epochs | batch=%d | lr=%.0e | patience=%d",
        args.epochs, args.batch_size, args.lr, args.patience,
    )
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d | train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f | "
            "lr=%.2e | %.1fs",
            epoch, args.epochs,
            tr_loss, tr_acc,
            va_loss, va_acc,
            scheduler.get_last_lr()[0],
            elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss":   round(va_loss, 6), "val_acc":   round(va_acc, 6),
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            patience_ctr = 0

            # Save checkpoint — only the new trainable weights (small file)
            ckpt_path = output_dir / "snore_encoder_best.pt"
            torch.save(
                {
                    "classifier_state_dict": model.classifier.state_dict(),
                    "config": {
                        "model_id":       AST_MODEL_ID,
                        "hidden_size":    model.config.hidden_size,
                        "labels":         LABELS,
                        "window_s":       WINDOW_S,
                        "unfreeze_last_n": args.unfreeze_last_n,
                    },
                    "val_acc": round(va_acc, 6),
                    "epoch":   epoch,
                },
                ckpt_path,
            )
            logger.info("  ✓ New best — saved checkpoint (val_acc=%.4f)", va_acc)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info(
                    "Early stopping at epoch %d  (best epoch=%d, val_acc=%.4f)",
                    epoch, best_epoch, best_val_acc,
                )
                break

    # ── Save training history ─────────────────────────────────────────────
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)

    # ── Test-set evaluation ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Final evaluation on TEST set")
    logger.info("=" * 60)

    # Reload best checkpoint
    ckpt = torch.load(output_dir / "snore_encoder_best.pt", map_location=device)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    logger.info("Test accuracy : %.4f", test_acc)
    logger.info("\n%s", classification_report(test_labels, test_preds, target_names=LABELS))

    cm = confusion_matrix(test_labels, test_preds)
    logger.info("Confusion matrix (rows=true, cols=pred):")
    logger.info("             background  snore")
    logger.info("  background   %6d   %6d", cm[0, 0], cm[0, 1])
    logger.info("  snore        %6d   %6d", cm[1, 0], cm[1, 1])

    logger.info("")
    logger.info("Model saved to  : %s", output_dir / "snore_encoder_best.pt")
    logger.info("Training history: %s", hist_path)
    logger.info("")
    logger.info("Run inference:")
    logger.info(
        "  python inference.py --input recording.wav --model %s",
        output_dir / "snore_encoder_best.pt",
    )


if __name__ == "__main__":
    main()
