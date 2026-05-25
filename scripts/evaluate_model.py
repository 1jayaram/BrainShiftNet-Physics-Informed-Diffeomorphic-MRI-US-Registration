"""
scripts/evaluate_model.py

Comprehensive model evaluation with detailed metrics display.
Outputs metrics for:
  - Target Registration Error (TRE) in mm
  - Jacobian determinant statistics
  - Registration quality metrics

Usage:
  python scripts/evaluate_model.py \
    --checkpoint outputs/run_02_resume/best_model.pth \
    --data_dir data/processed2 \
    --split test \
    --output_dir results/evaluation
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tabulate import tabulate
import numpy as np
import torch

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import (
    compute_tre, compute_dice, jacobian_stats, compute_hausdorff
)


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    split: str = "test",
    output_dir: str = "results/evaluation",
    device: str = "cuda",
):
    """Run comprehensive evaluation and display metrics."""
    import yaml
    from data.dataset import BrainShiftDataset
    from models.transmorph_brain_shift import build_model

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Load checkpoint ──
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint or use defaults
    if "config" in ckpt:
        cfg = ckpt["config"]
    else:
        # Try to load from config file
        config_path = Path(checkpoint_path).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            print("[WARN] No config found, using defaults")
            cfg = {}

    # ── Build model ──
    target_shape = tuple(cfg.get("target_shape", [128, 128, 128]))
    in_ch = int(cfg.get("use_t1", True)) + int(cfg.get("use_flair", True)) + 1
    
    print(f"[INFO] Building model: {cfg.get('model_type', 'diffeomorphic')}")
    model = build_model({
        "model_type":    cfg.get("model_type", "diffeomorphic"),
        "in_channels":   in_ch,
        "base_features": cfg.get("base_features", 8),
        "img_size":      list(target_shape),
        "use_attention": cfg.get("use_attention", False),
    }).to(device)

    # Load weights
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load dataset ──
    print(f"\n[INFO] Loading {split} dataset from {data_dir}")
    dataset = BrainShiftDataset(
        data_dir=data_dir,
        split=split,
        target_shape=target_shape,
        use_t1=cfg.get("use_t1", True),
        use_flair=cfg.get("use_flair", True),
        augment=False,
        us_stage=cfg.get("us_stage", "before"),
        segmentation_dir=cfg.get("segmentation_dir"),
        landmark_dir=cfg.get("landmark_dir"),
    )

    # ── Create output directory ──
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Evaluate ──
    print(f"\n[INFO] Evaluating on {len(dataset)} cases...")
    print("─" * 80)

    results = []
    tre_all = []
    dice_all = []
    hausdorff_all = []
    
    table_data = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            case_id = sample["case_id"]

            mri = sample["mri"].unsqueeze(0).to(device)
            us = sample["us"].unsqueeze(0).to(device)
            lm = sample.get("landmarks")
            seg_tumor = sample.get("tumor_seg")
            
            if lm is not None:
                lm = lm.unsqueeze(0).to(device)

            # Forward pass
            flow, mri_warped = model(mri, us)

            case_result = {
                "case_id": case_id,
                "flow_shape": list(flow.shape),
            }

            # TRE metric
            tre_val = None
            if lm is not None and lm.shape[1] > 0:
                tre_val = compute_tre(flow, lm, target_shape)
                case_result["TRE_mm"] = float(tre_val)
                tre_all.append(tre_val)

            # Jacobian stats
            jac = jacobian_stats(flow)
            case_result.update(jac)

            # Dice for tumor segmentation if available
            dice_val = None
            if seg_tumor is not None:
                # Warp segmentation
                from models.transmorph_brain_shift import SpatialTransformer
                stn = SpatialTransformer(target_shape, mode="nearest").to(device)
                seg_warped = stn(seg_tumor.unsqueeze(0).float().to(device), flow)
                dice_val = compute_dice(seg_warped, seg_tumor.unsqueeze(0).to(device))
                case_result["Dice"] = float(dice_val)
                dice_all.append(dice_val)

            results.append(case_result)

            # Table row
            row = [case_id]
            if tre_val is not None:
                row.append(f"{tre_val:.2f}")
            else:
                row.append("N/A")
            if dice_val is not None:
                row.append(f"{dice_val:.4f}")
            else:
                row.append("N/A")
            row.extend([
                f"{jac['jac_det_mean']:.3f}",
                f"{jac['pct_neg_jac']:.1f}%"
            ])
            table_data.append(row)

    # ── Print results table ──
    headers = ["Case ID", "TRE (mm)", "Dice", "Jac Mean", "% Neg Jac"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("─" * 80)

    # ── Print summary statistics ──
    print("\n📊 SUMMARY STATISTICS")
    print("─" * 80)

    if tre_all:
        tre_array = np.array(tre_all)
        print(f"\n🎯 Target Registration Error (TRE):")
        print(f"   Mean:        {float(np.mean(tre_array)):8.3f} mm")
        print(f"   Std Dev:     {float(np.std(tre_array)):8.3f} mm")
        print(f"   Median:      {float(np.median(tre_array)):8.3f} mm")
        print(f"   Min:         {float(np.min(tre_array)):8.3f} mm")
        print(f"   Max:         {float(np.max(tre_array)):8.3f} mm")
        print(f"   95th %ile:   {float(np.percentile(tre_array, 95)):8.3f} mm")

    if dice_all:
        dice_array = np.array(dice_all)
        print(f"\n🎲 Dice Similarity Coefficient:")
        print(f"   Mean:        {float(np.mean(dice_array)):8.4f}")
        print(f"   Std Dev:     {float(np.std(dice_array)):8.4f}")
        print(f"   Median:      {float(np.median(dice_array)):8.4f}")
        print(f"   Min:         {float(np.min(dice_array)):8.4f}")
        print(f"   Max:         {float(np.max(dice_array)):8.4f}")

    # Jacobian stats across all cases
    if results:
        jac_means = [r.get("jac_det_mean", 0) for r in results]
        neg_jacs = [r.get("pct_neg_jac", 0) for r in results]
        print(f"\n📈 Jacobian Determinant Statistics:")
        print(f"   Mean (across cases): {float(np.mean(jac_means)):8.3f}")
        print(f"   Max % Negative:      {float(np.max(neg_jacs)):8.1f}%")

    print("─" * 80)

    # ── Save results ──
    summary = {
        "evaluation_split": split,
        "n_cases": len(results),
        "model_checkpoint": str(checkpoint_path),
        "target_shape": list(target_shape),
        "metrics": {
            "TRE": {
                "mean_mm": float(np.mean(tre_all)) if tre_all else None,
                "std_mm": float(np.std(tre_all)) if tre_all else None,
                "median_mm": float(np.median(tre_all)) if tre_all else None,
                "min_mm": float(np.min(tre_all)) if tre_all else None,
                "max_mm": float(np.max(tre_all)) if tre_all else None,
                "percentile_95_mm": float(np.percentile(tre_all, 95)) if tre_all else None,
            } if tre_all else None,
            "Dice": {
                "mean": float(np.mean(dice_all)) if dice_all else None,
                "std": float(np.std(dice_all)) if dice_all else None,
                "median": float(np.median(dice_all)) if dice_all else None,
                "min": float(np.min(dice_all)) if dice_all else None,
                "max": float(np.max(dice_all)) if dice_all else None,
            } if dice_all else None,
        },
        "per_case_results": results,
    }

    # Save JSON
    json_path = out_path / f"evaluation_{split}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")

    # Save markdown report
    md_path = out_path / f"evaluation_{split}.md"
    with open(md_path, "w") as f:
        f.write(f"# Brain Shift Model Evaluation Report\n\n")
        f.write(f"**Split:** {split}\n")
        f.write(f"**Number of Cases:** {len(results)}\n")
        f.write(f"**Checkpoint:** {checkpoint_path}\n\n")

        f.write(f"## Summary Metrics\n\n")
        if tre_all:
            f.write(f"### Target Registration Error (TRE)\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Mean | {float(np.mean(tre_all)):.3f} mm |\n")
            f.write(f"| Std Dev | {float(np.std(tre_all)):.3f} mm |\n")
            f.write(f"| Median | {float(np.median(tre_all)):.3f} mm |\n")
            f.write(f"| 95th Percentile | {float(np.percentile(tre_all, 95)):.3f} mm |\n\n")

        if dice_all:
            f.write(f"### Dice Similarity Coefficient\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Mean | {float(np.mean(dice_all)):.4f} |\n")
            f.write(f"| Std Dev | {float(np.std(dice_all)):.4f} |\n")
            f.write(f"| Median | {float(np.median(dice_all)):.4f} |\n\n")

        f.write(f"## Per-Case Results\n\n")
        f.write(f"| Case ID | TRE (mm) | Dice | Jacobian Mean | % Negative Jac |\n")
        f.write(f"|---------|----------|------|---------------|----------------|\n")
        for row in table_data:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n")

    print(f"✅ Markdown report saved to: {md_path}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--checkpoint", default="outputs/run_02_resume/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="data/processed2",
                        help="Processed data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output_dir", default="results/evaluation",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
    )
