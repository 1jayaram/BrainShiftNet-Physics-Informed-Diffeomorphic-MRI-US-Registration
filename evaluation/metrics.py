"""
evaluation/metrics.py + evaluate.py

Metrics:
  - TRE  (Target Registration Error) — primary metric (mm)
  - Dice overlap of warped tumour segmentation vs ground truth
  - Jacobian determinant stats (folds, singularities)
  - Hausdorff distance
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List


# ─── METRICS ───────────────────────────────────────────────────────────────

def compute_tre(
    flow: torch.Tensor,
    landmarks: torch.Tensor,
    img_size: tuple,
    voxel_spacing_mm: tuple = (1.0, 1.0, 1.0)
) -> float:
    """
    Target Registration Error in mm.
    
    flow: (B, 3, D, H, W)
    landmarks: (B, N, 6) — [x1,y1,z1, x2,y2,z2] (source, target) in voxel coords
    Returns: mean TRE across all landmarks and batch (float, in mm)
    """
    B, N, _ = landmarks.shape
    src_lm = landmarks[:, :, :3]   # (B, N, 3)
    tgt_lm = landmarks[:, :, 3:]

    # Normalise source landmarks to [-1, 1]
    norm_lm = src_lm.clone()
    for i, s in enumerate(img_size):
        norm_lm[:, :, i] = 2.0 * src_lm[:, :, i] / (s - 1) - 1.0

    # Sample flow at landmark locations
    grid = norm_lm[:, :, [2, 1, 0]].unsqueeze(1).unsqueeze(1)  # (B,1,1,N,3)
    sampled = F.grid_sample(flow, grid, align_corners=True, mode="bilinear")
    sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1)   # (B, N, 3)

    warped_lm = src_lm + sampled

    # Euclidean distance in voxels
    diff    = warped_lm - tgt_lm
    spacing = torch.tensor(voxel_spacing_mm, device=diff.device).view(1, 1, 3)
    diff_mm = diff * spacing
    tre     = torch.sqrt((diff_mm ** 2).sum(dim=-1) + 1e-8)
    return tre.mean().item()


def compute_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """
    Dice similarity coefficient for binary tumour masks.
    pred_mask, gt_mask: (B, 1, D, H, W) — float in [0, 1]
    """
    pred = (pred_mask > threshold).float()
    gt   = (gt_mask   > threshold).float()
    intersection = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if denom < 1e-5:
        return 1.0
    return (2.0 * intersection / denom).item()


def compute_hausdorff(pred_mask: np.ndarray, gt_mask: np.ndarray,
                       percentile: float = 95.0) -> float:
    """95th percentile Hausdorff distance in voxels."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import directed_hausdorff

    pred_surf = pred_mask ^ binary_erosion(pred_mask)
    gt_surf   = gt_mask   ^ binary_erosion(gt_mask)

    pred_pts = np.argwhere(pred_surf)
    gt_pts   = np.argwhere(gt_surf)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)


def jacobian_stats(flow: torch.Tensor) -> dict:
    """
    Compute statistics of the Jacobian determinant.
    Values <= 0 indicate folding (non-invertible deformation).
    """
    B, _, D, H, W = flow.shape
    dy = flow[:, :, 1:, :, :]  - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :]  - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:]  - flow[:, :, :, :, :-1]
    s  = (slice(None), slice(None), slice(0,D-1), slice(0,H-1), slice(0,W-1))
    det = ((1 + dy[s][:,0]) * ((1 + dx[s][:,1]) * (1 + dz[s][:,2]) - dx[s][:,2]*dz[s][:,1])
         - dy[s][:,1]       * (dx[s][:,0]       * (1 + dz[s][:,2]) - dx[s][:,2]*dz[s][:,0])
         + dy[s][:,2]       * (dx[s][:,0]       * dz[s][:,1]       - (1 + dx[s][:,1])*dz[s][:,0]))
    det_np = det.detach().cpu().numpy()
    return {
        "jac_det_mean":    float(det_np.mean()),
        "jac_det_std":     float(det_np.std()),
        "jac_det_min":     float(det_np.min()),
        "jac_det_max":     float(det_np.max()),
        "pct_neg_jac":     float((det_np <= 0).mean() * 100),
    }


# ─── FULL EVALUATION SCRIPT ────────────────────────────────────────────────

def evaluate(
    model_path: str,
    data_dir: str,
    split: str = "test",
    output_dir: str = "eval_results",
    device: str = "cuda",
    config_path: str = None,
):
    import yaml
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset import BrainShiftDataset
    from models.transmorph_brain_shift import build_model

    # Load config
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # Try loading from checkpoint
        ckpt = torch.load(model_path, map_location="cpu")
        cfg  = ckpt.get("config", {})

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    target_shape = tuple(cfg.get("target_shape", [160, 192, 160]))
    in_ch = int(cfg.get("use_t1", True)) + int(cfg.get("use_flair", True)) + 1
    model = build_model({
        "model_type":    cfg.get("model_type", "diffeomorphic"),
        "in_channels":   in_ch,
        "base_features": cfg.get("base_features", 32),
        "img_size":      list(target_shape),
        "use_attention": cfg.get("use_attention", True),
    }).to(device)

    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt.get("model", ckpt)
    # Handle DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Dataset
    dataset = BrainShiftDataset(
        data_dir=data_dir, split=split,
        target_shape=target_shape,
        use_t1=cfg.get("use_t1", True),
        use_flair=cfg.get("use_flair", True),
        augment=False,
        us_stage=cfg.get("us_stage", "before"),
        segmentation_dir=cfg.get("segmentation_dir"),
        landmark_dir=cfg.get("landmark_dir"),
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    tre_all  = []

    print(f"\nEvaluating on {len(dataset)} cases...")
    for i in range(len(dataset)):
        sample  = dataset[i]
        case_id = sample["case_id"]

        mri = sample["mri"].unsqueeze(0).to(device)
        us  = sample["us"].unsqueeze(0).to(device)
        lm  = sample.get("landmarks")
        if lm is not None:
            lm = lm.unsqueeze(0).to(device)

        with torch.no_grad():
            flow, mri_warped = model(mri, us)

        # Metrics
        case_result = {"case_id": case_id}

        if lm is not None and lm.shape[1] > 0:
            tre = compute_tre(flow, lm, target_shape)
            case_result["TRE_mm"] = tre
            tre_all.append(tre)
            print(f"  {case_id}: TRE = {tre:.3f} mm")
        else:
            print(f"  {case_id}: (no landmarks)")

        # Jacobian stats
        jac = jacobian_stats(flow)
        case_result.update(jac)
        results.append(case_result)

    # Summary
    summary = {
        "n_cases":   len(results),
        "TRE_mean":  float(np.mean(tre_all)) if tre_all else None,
        "TRE_std":   float(np.std(tre_all))  if tre_all else None,
        "TRE_median":float(np.median(tre_all)) if tre_all else None,
        "TRE_95th":  float(np.percentile(tre_all, 95)) if tre_all else None,
        "cases":     results,
    }

    out_file = out_path / f"eval_{split}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Results ({split} set, {len(results)} cases)")
    print(f"{'─'*50}")
    if tre_all:
        print(f"TRE (mm): mean={summary['TRE_mean']:.3f}  "
              f"std={summary['TRE_std']:.3f}  "
              f"median={summary['TRE_median']:.3f}  "
              f"95th={summary['TRE_95th']:.3f}")
    print(f"Saved to: {out_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain shift model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir",   required=True, help="Processed data directory")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    evaluate(
        model_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        config_path=args.config,
    )
