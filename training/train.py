"""
training/train.py

Full training loop for brain shift prediction model.
Features:
  - Multi-GPU support
  - Mixed precision (fp16)
  - Gradient clipping
  - Learning rate scheduler
  - TensorBoard + W&B logging
  - Best model checkpointing
  - Early stopping
"""

import os
import sys
import json
import time
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter  # Import later to avoid issues

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import BrainShiftDataset
from models.transmorph_brain_shift import build_model
from training.losses import RegistrationLoss
from evaluation.metrics import compute_tre, compute_dice


# ─── DEFAULT CONFIG ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "data_dir":       "data/processed",
    "split_file":     None,
    "target_shape":   [160, 192, 160],
    "us_stage":       "before",
    "use_t1":         True,
    "use_flair":      True,
    "augment":        True,
    # Model
    "model_type":     "diffeomorphic",
    "base_features":  32,
    "use_attention":  True,
    "int_steps":      7,
    # Training
    "batch_size":     1,
    "epochs":         200,
    "lr":             1e-4,
    "lr_decay":       0.95,
    "weight_decay":   1e-5,
    "grad_clip":      1.0,
    "mixed_precision": True,
    "device":          "auto",
    # Loss
    "lambda_sim":     1.0,
    "lambda_reg":     2.0,
    "lambda_lm":      5.0,
    "lambda_jac":     0.5,
    "similarity":     "mi",
    "regulariser":    "bending",
    # Output
    "output_dir":     "outputs",
    "log_dir":        "logs",
    "save_every":     10,
    "patience":       30,   # early stopping patience
    "seed":           42,
}


# ─── HELPERS ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_python(obj):
    """Convert PyTorch tensors to Python numbers for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(item) for item in obj]
    else:
        return obj


def load_config(config_path: str) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            override = yaml.safe_load(f)
        cfg.update(override)
    return cfg


def collate_fn(batch):
    """Custom collate that handles missing landmarks."""
    out = {}
    for key in batch[0]:
        if key == "landmarks":
            # Pad landmarks to same count
            lm_list = [b[key] for b in batch if key in b]
            if lm_list:
                max_n = max(lm.shape[0] for lm in lm_list)
                padded = [torch.cat([lm, torch.zeros(max_n - lm.shape[0], 6)], dim=0) for lm in lm_list]
                out[key] = torch.stack(padded)
        elif key == "case_id":
            out[key] = [b[key] for b in batch]
        else:
            vals = [b[key] for b in batch if key in b]
            if vals:
                out[key] = torch.stack(vals)
    return out


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, cfg):
    model.train()
    epoch_losses = {k: 0.0 for k in ["total", "sim", "reg", "jac", "landmark"]}

    for batch in tqdm(loader, desc="Train", leave=False):
        mri = batch.get("mri", None)
        us  = batch.get("us",  None)
        if mri is None or us is None:
            continue

        mri = mri.to(device)
        us  = us.to(device)
        lm  = batch.get("landmarks", None)
        if lm is not None:
            lm = lm.to(device)

        optimiser.zero_grad()

        with autocast(enabled=cfg["mixed_precision"]):
            flow, mri_warped = model(mri, us)
            loss_dict = criterion(mri_warped, us, flow, lm)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimiser)
        scaler.update()

        for k in epoch_losses:
            epoch_losses[k] += loss_dict.get(k, 0.0)

    n = max(len(loader), 1)
    return {k: v / n for k, v in epoch_losses.items()}


# ─── VALIDATION STEP ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    val_losses = {k: 0.0 for k in ["total", "sim", "reg", "jac", "landmark"]}
    tre_list   = []

    for batch in tqdm(loader, desc="Val  ", leave=False):
        mri = batch.get("mri", None)
        us  = batch.get("us",  None)
        if mri is None or us is None:
            continue

        mri = mri.to(device)
        us  = us.to(device)
        lm  = batch.get("landmarks", None)
        if lm is not None:
            lm = lm.to(device)

        with autocast(enabled=cfg["mixed_precision"]):
            flow, mri_warped = model(mri, us)
            loss_dict = criterion(mri_warped, us, flow, lm)

        for k in val_losses:
            val_losses[k] += loss_dict.get(k, 0.0)

        # Compute TRE if landmarks available
        if lm is not None and lm.shape[1] > 0:
            tre = compute_tre(flow, lm, tuple(cfg["target_shape"]))
            tre_list.append(tre)

    n = max(len(loader), 1)
    metrics = {k: v / n for k, v in val_losses.items()}
    if tre_list:
        metrics["TRE_mean"] = np.mean(tre_list)
        metrics["TRE_std"]  = np.std(tre_list)
    return metrics


# ─── MAIN TRAIN FUNCTION ───────────────────────────────────────────────────

def train(cfg: dict, args=None):
    set_seed(cfg["seed"])

    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False")
    else:
        device = torch.device(requested_device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    except ImportError as e:
        print(f"Warning: TensorBoard not available ({e}), logging disabled")
        writer = None

    # ── Datasets ──
    train_ds = BrainShiftDataset(
        data_dir=cfg["data_dir"], split="train",
        split_file=cfg["split_file"],
        target_shape=tuple(cfg["target_shape"]),
        use_t1=cfg["use_t1"], use_flair=cfg["use_flair"],
        augment=cfg["augment"], us_stage=cfg["us_stage"],
        segmentation_dir=cfg.get("segmentation_dir"),
        landmark_dir=cfg.get("landmark_dir")
    )
    val_ds = BrainShiftDataset(
        data_dir=cfg["data_dir"], split="val",
        split_file=cfg["split_file"],
        target_shape=tuple(cfg["target_shape"]),
        use_t1=cfg["use_t1"], use_flair=cfg["use_flair"],
        augment=False, us_stage=cfg["us_stage"],
        segmentation_dir=cfg.get("segmentation_dir"),
        landmark_dir=cfg.get("landmark_dir")
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1,
                              shuffle=False, num_workers=2, collate_fn=collate_fn,
                              pin_memory=True)

    # ── Model ──
    in_channels = int(cfg["use_t1"]) + int(cfg["use_flair"]) + 1  # MRI channels + US
    model_cfg = {
        "model_type":    cfg["model_type"],
        "in_channels":   in_channels,
        "base_features": cfg["base_features"],
        "img_size":      cfg["target_shape"],
        "use_attention": cfg["use_attention"],
        "int_steps":     cfg["int_steps"],
    }
    model = build_model(model_cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ── Loss ──
    criterion = RegistrationLoss(
        lambda_sim=cfg["lambda_sim"],
        lambda_reg=cfg["lambda_reg"],
        lambda_lm=cfg["lambda_lm"],
        lambda_jac=cfg["lambda_jac"],
        similarity=cfg["similarity"],
        regulariser=cfg["regulariser"],
        img_size=tuple(cfg["target_shape"]),
    )

    # ── Optimiser + Scheduler ──
    optimiser = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=cfg["lr_decay"])
    scaler    = GradScaler(enabled=cfg["mixed_precision"])

    # ── Resume from checkpoint if provided ──
    start_epoch = 1
    best_val_loss = float("inf")
    best_tre = float("inf")
    no_improve = 0
    history = []

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimiser.load_state_dict(checkpoint["optimiser"])
            start_epoch = checkpoint["epoch"] + 1
            if "val_loss" in checkpoint:
                best_val_loss = checkpoint["val_loss"]
            # Adjust scheduler for resumed epochs
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"Resumed from epoch {checkpoint['epoch']}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

    print(f"\nStarting training: {cfg['epochs']} epochs\n")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimiser, scaler, device, cfg)
        val_metrics   = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        elapsed = time.time() - t0
        tre_str = f"  TRE={val_metrics.get('TRE_mean', 0):.3f}mm" if "TRE_mean" in val_metrics else ""
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train: {train_metrics['total']:.4f} | "
              f"Val: {val_metrics['total']:.4f}{tre_str} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")

        # TensorBoard logging
        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Checkpoint
        record = {"epoch": epoch, "train": tensor_to_python(train_metrics), "val": tensor_to_python(val_metrics)}
        history.append(record)

        val_loss = val_metrics["total"]
        is_best  = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimiser":  optimiser.state_dict(),
                "val_loss":   val_loss,
                "config":     cfg,
            }, str(out_dir / "best_model.pth"))
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1

        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "config": cfg
            }, str(out_dir / f"checkpoint_epoch{epoch:04d}.pth"))

        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg['patience']} epochs)")
            break

    # Save training history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {out_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    # Allow command-line overrides
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--lambda_lm", type=float, default=None,
                        help="Landmark supervision loss weight")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                        help="Training device. Use cuda to require GPU.")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction,
                        default=None, help="Enable/disable CUDA mixed precision")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: data_dir = {cfg['data_dir']}")
    for k in ["data_dir", "epochs", "lr", "output_dir", "log_dir", "lambda_lm",
              "device", "mixed_precision"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    train(cfg, args)


# Find this block near the bottom:
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--data_dir", default=None)
parser.add_argument("--epochs",   type=int, default=None)
parser.add_argument("--lr",       type=float, default=None)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--log_dir", default=None)
parser.add_argument("--lambda_lm", type=float, default=None)
parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=None)

# ADD THIS LINE:
parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
