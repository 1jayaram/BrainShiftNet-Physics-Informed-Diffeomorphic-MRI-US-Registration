"""
scripts/infer.py

Run inference on a new patient case.
Outputs:
  - Predicted deformation field (.nii.gz)
  - Warped MRI (aligned to US space)
  - Updated tumour mask position
  - Visualisation PNG

Usage:
  python scripts/infer.py \
    --mri     path/to/T1_processed.nii.gz  path/to/FLAIR_processed.nii.gz \
    --us      path/to/US_before.nii.gz \
    --seg     path/to/tumour_mask.nii.gz \
    --model   outputs/best_model.pth \
    --outdir  results/patient_01
"""
import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_nifti_as_tensor(path: str, target_shape=None):
    """Load NIfTI, pad/crop to target, return tensor + affine."""
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    if target_shape is not None:
        from data.dataset import pad_or_crop
        data = pad_or_crop(data, target_shape)

    # Normalise to [0, 1]
    lo = np.percentile(data[data > 0], 1) if (data > 0).sum() > 0 else 0
    hi = np.percentile(data[data > 0], 99) if (data > 0).sum() > 0 else 1
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo + 1e-8)

    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return tensor, affine, img.header


def save_nifti_from_tensor(tensor: torch.Tensor, path: str, affine: np.ndarray):
    """Save a 3D tensor as NIfTI."""
    data = tensor.squeeze().cpu().numpy()
    img  = nib.Nifti1Image(data, affine)
    nib.save(img, path)
    print(f"  Saved: {path}")


def _infer_case_id(*paths) -> str | None:
    """Infer a RESECT case id from common path components."""
    for raw_path in paths:
        if not raw_path:
            continue
        for part in Path(raw_path).parts:
            if part.lower().startswith("case") and part[4:].isdigit():
                return part
    return None


def resolve_segmentation_path(seg_path, mri_paths, us_path, cfg):
    """Resolve explicit or project-standard tumor segmentation paths."""
    if not seg_path:
        return None

    seg_path = Path(seg_path)
    if seg_path.exists():
        return seg_path

    case_id = _infer_case_id(*(mri_paths or []), us_path, seg_path)
    seg_dirs = []
    cfg_seg_dir = cfg.get("segmentation_dir")
    if cfg_seg_dir:
        seg_dirs.append(Path(cfg_seg_dir))
    seg_dirs.append(Path("RESECT_segmentation"))

    patterns = []
    if case_id:
        patterns.extend([
            f"{case_id}-US-before-tumor.nii.gz",
            f"*{case_id}*before*tumor*.nii.gz",
            f"*{case_id}*before*tumour*.nii.gz",
            f"*{case_id}*tumor*.nii.gz",
            f"*{case_id}*tumour*.nii.gz",
        ])
    patterns.extend(["*before*tumor*.nii.gz", "*before*tumour*.nii.gz"])

    matches = []
    for seg_dir in dict.fromkeys(seg_dirs):
        search_root = seg_dir / case_id if case_id else seg_dir
        if not search_root.exists():
            continue
        for pattern in patterns:
            matches.extend(search_root.glob(pattern))

    matches = sorted(dict.fromkeys(matches))
    if matches:
        resolved = matches[0]
        print(f"  Segmentation not found at {seg_path}; using {resolved}")
        return resolved

    searched = ", ".join(str(p) for p in dict.fromkeys(seg_dirs))
    raise FileNotFoundError(
        f"Segmentation file not found: {seg_path}\n"
        f"No replacement mask found for {case_id or 'unknown case'} in: {searched}\n"
        "Either pass the correct --seg path or omit --seg to run without mask warping."
    )


def warp_segmentation(seg_tensor: torch.Tensor,
                       flow: torch.Tensor,
                       threshold: float = 0.5) -> torch.Tensor:
    """Apply deformation field to binary segmentation mask (nearest neighbour)."""
    from models.transmorph_brain_shift import SpatialTransformer
    img_size = flow.shape[2:]
    stn = SpatialTransformer(img_size, mode="nearest").to(flow.device)
    warped = stn(seg_tensor.float(), flow)
    return (warped > threshold).float()


def visualise_results(mri_orig, us_fixed, mri_warped, seg_orig, seg_warped,
                       flow, save_path: str):
    """Save a PNG summary of registration results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Take central axial slice
        def mid_slice(vol):
            arr = vol.squeeze().cpu().numpy()
            return arr[arr.shape[0] // 2]

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Brain Shift Prediction Results", fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

        titles = ["Pre-op MRI (original)", "Intra-op US (fixed)",
                  "MRI warped to US", "Difference before reg.",
                  "Difference after reg.", "Tumour (original)",
                  "Tumour (warped)", "Flow magnitude"]
        slices = [
            mid_slice(mri_orig[:, :1]),
            mid_slice(us_fixed),
            mid_slice(mri_warped[:, :1]),
            mid_slice(torch.abs(mri_orig[:, :1] - us_fixed)),
            mid_slice(torch.abs(mri_warped[:, :1] - us_fixed)),
            mid_slice(seg_orig) if seg_orig is not None else np.zeros((10, 10)),
            mid_slice(seg_warped) if seg_warped is not None else np.zeros((10, 10)),
            mid_slice(torch.sqrt((flow ** 2).sum(dim=1, keepdim=True))),
        ]
        cmaps = ["gray", "hot", "gray", "RdYlGn_r", "RdYlGn_r",
                 "Reds", "Reds", "jet"]

        for i, (title, sl, cmap) in enumerate(zip(titles, slices, cmaps)):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.imshow(sl, cmap=cmap, origin="lower")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualisation: {save_path}")
    except Exception as e:
        print(f"  (Visualisation skipped: {e})")


# ─── MAIN INFERENCE ────────────────────────────────────────────────────────

def infer(mri_paths, us_path, model_path, outdir,
          seg_path=None, device="cuda"):
    import yaml

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model config + weights
    ckpt    = torch.load(model_path, map_location=device)
    cfg     = ckpt.get("config", {})
    target_shape = tuple(cfg.get("target_shape", [160, 192, 160]))

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.transmorph_brain_shift import build_model

    in_ch = len(mri_paths) + 1
    model = build_model({
        "model_type":    cfg.get("model_type", "diffeomorphic"),
        "in_channels":   in_ch,
        "base_features": cfg.get("base_features", 32),
        "img_size":      list(target_shape),
        "use_attention": cfg.get("use_attention", True),
    }).to(device)

    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load inputs
    print("Loading volumes...")
    mri_tensors, affine, header = [], None, None
    for p in mri_paths:
        t, aff, hdr = load_nifti_as_tensor(p, target_shape)
        mri_tensors.append(t.squeeze(0))  # (1, D, H, W)
        if affine is None:
            affine, header = aff, hdr

    mri = torch.cat(mri_tensors, dim=0).unsqueeze(0).to(device)  # (1, C, D, H, W)
    us_t, us_aff, _ = load_nifti_as_tensor(us_path, target_shape)
    us  = us_t.to(device)

    # Optional segmentation
    seg = None
    if seg_path:
        seg_path = resolve_segmentation_path(seg_path, mri_paths, us_path, cfg)
        seg_img = nib.load(seg_path)
        seg_arr = seg_img.get_fdata(dtype=np.float32)
        from data.dataset import pad_or_crop
        seg_arr = pad_or_crop(seg_arr, target_shape)
        seg = torch.from_numpy(seg_arr).float().unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    print("Running model inference...")
    t0 = time.time()
    with torch.no_grad():
        flow, mri_warped = model(mri, us)
    elapsed = time.time() - t0
    print(f"Inference time: {elapsed:.2f}s")

    # Warp segmentation
    seg_warped = None
    if seg is not None:
        seg_warped = warp_segmentation(seg, flow)

    # Save outputs
    print("Saving outputs...")
    save_nifti_from_tensor(flow[:, 0],     str(outdir / "flow_x.nii.gz"),  affine)
    save_nifti_from_tensor(flow[:, 1],     str(outdir / "flow_y.nii.gz"),  affine)
    save_nifti_from_tensor(flow[:, 2],     str(outdir / "flow_z.nii.gz"),  affine)
    save_nifti_from_tensor(mri_warped[:, 0], str(outdir / "mri_warped.nii.gz"), affine)

    flow_mag = torch.sqrt((flow ** 2).sum(dim=1, keepdim=True))
    save_nifti_from_tensor(flow_mag, str(outdir / "flow_magnitude.nii.gz"), affine)

    if seg_warped is not None:
        save_nifti_from_tensor(seg_warped, str(outdir / "tumour_warped.nii.gz"), affine)

    # Stats
    flow_np = flow.cpu().numpy()
    mag_np  = np.sqrt((flow_np ** 2).sum(axis=1))
    stats = {
        "inference_time_sec": elapsed,
        "max_shift_mm":       float(mag_np.max()),
        "mean_shift_mm":      float(mag_np.mean()),
        "95th_shift_mm":      float(np.percentile(mag_np, 95)),
    }
    print(f"\nShift statistics:")
    print(f"  Max  shift: {stats['max_shift_mm']:.2f} mm")
    print(f"  Mean shift: {stats['mean_shift_mm']:.2f} mm")
    print(f"  95th shift: {stats['95th_shift_mm']:.2f} mm")

    import json
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Visualise
    visualise_results(
        mri, us, mri_warped, seg, seg_warped, flow,
        str(outdir / "result_overview.png")
    )

    print(f"\nAll results saved to: {outdir}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain shift inference")
    parser.add_argument("--mri",    nargs="+", required=True,
                        help="Pre-op MRI file(s) (T1, FLAIR)")
    parser.add_argument("--us",     required=True,
                        help="Intra-op US before resection")
    parser.add_argument("--model",  required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for results")
    parser.add_argument("--seg",    default=None,
                        help="Optional tumour segmentation mask to warp. If the path is missing, "
                             "the script tries RESECT_segmentation/<Case>/*before*tumor*.nii.gz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    infer(
        mri_paths=args.mri,
        us_path=args.us,
        model_path=args.model,
        outdir=args.outdir,
        seg_path=args.seg,
        device=args.device,
    )
