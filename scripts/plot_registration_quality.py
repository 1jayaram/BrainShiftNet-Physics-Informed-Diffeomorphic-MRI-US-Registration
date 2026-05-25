"""
scripts/plot_registration_quality.py

Registration quality visualisation for the brain shift prediction project.
Produces four figures:

  Fig 1 — TRE per case  (box + strip plot, sorted by median)
  Fig 2 — Dice score distribution  (histogram + KDE + per-case bar)
  Fig 3 — Before vs After registration  (image grid from NIfTI outputs)
  Fig 4 — Summary dashboard  (TRE improvement scatter, Jac det, CDF)

Usage
-----
  # Minimal — evaluation JSON only (Figs 1, 2, 4):
  python scripts/plot_registration_quality.py \
      --eval_json results/evaluation/evaluation_test.json

  # Full — also renders image comparison grid (Fig 3):
  python scripts/plot_registration_quality.py \
      --eval_json results/evaluation/evaluation_test.json \
      --infer_dir results/patient_outputs \
      --data_dir  data/processed2 \
      --outdir    results/figures

  # LOOCV aggregate (pass multiple JSON files):
  python scripts/plot_registration_quality.py \
      --eval_json results/loocv/fold*/evaluation_test.json \
      --outdir    results/figures/loocv

Dependencies
------------
  pip install matplotlib seaborn numpy nibabel scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

# Optional soft dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[WARN] seaborn not installed — fallback to plain matplotlib")

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[WARN] nibabel not installed — image comparison panel skipped")

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


# ─── STYLE ─────────────────────────────────────────────────────────────────

PALETTE = {
    "purple":   "#7F77DD",
    "teal":     "#1D9E75",
    "coral":    "#D85A30",
    "amber":    "#EF9F27",
    "blue":     "#378ADD",
    "gray":     "#888780",
    "red":      "#E24B4A",
    "green":    "#639922",
    "pink":     "#D4537E",
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    True,
        "axes.spines.bottom":  True,
        "axes.linewidth":      0.6,
        "axes.labelsize":      10,
        "axes.titlesize":      11,
        "axes.titleweight":    "medium",
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "legend.frameon":      False,
        "grid.color":          "#e0e0e0",
        "grid.linewidth":      0.5,
        "font.family":         "DejaVu Sans",
        "figure.dpi":          150,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.15,
    })


# ─── DATA LOADING ──────────────────────────────────────────────────────────

def load_eval_jsons(paths: list[str]) -> list[dict]:
    """Load one or more evaluation JSON files and merge per-case results."""
    records = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for r in data.get("per_case_results", []):
            r.setdefault("_source_file", str(p))
            records.append(r)
    return records


def records_to_arrays(records: list[dict]) -> dict:
    """Extract metric arrays from per-case records dict."""
    case_ids   = [r["case_id"]      for r in records]
    tre_vals   = [r.get("TRE_mm")   for r in records]
    dice_vals  = [r.get("Dice")     for r in records]
    jac_means  = [r.get("jac_det_mean",  float("nan")) for r in records]
    pct_neg    = [r.get("pct_neg_jac",   float("nan")) for r in records]

    return dict(
        case_ids  = case_ids,
        tre       = np.array([v if v is not None else np.nan for v in tre_vals]),
        dice      = np.array([v if v is not None else np.nan for v in dice_vals]),
        jac_mean  = np.array(jac_means),
        pct_neg   = np.array(pct_neg),
    )


# ─── FIG 1 — TRE PER CASE ──────────────────────────────────────────────────

def fig_tre_per_case(
    data: dict,
    initial_tre: Optional[np.ndarray] = None,
    outdir: Path = Path("."),
    show: bool = False,
) -> Path:
    """
    Sorted per-case TRE bar chart with individual point overlay.
    If initial_tre is provided, also renders before/after arrows.
    """
    apply_style()

    case_ids = data["case_ids"]
    tre = data["tre"]

    valid = ~np.isnan(tre)
    if not valid.any():
        print("[SKIP] No TRE values found — skipping Fig 1")
        return None

    ids_v  = np.array(case_ids)[valid]
    tre_v  = tre[valid]

    # Sort by TRE ascending
    order   = np.argsort(tre_v)
    ids_s   = ids_v[order]
    tre_s   = tre_v[order]
    x       = np.arange(len(ids_s))

    # Figure
    fig, ax = plt.subplots(figsize=(max(8, len(ids_s) * 0.55), 4.5))

    # If we have initial TRE, draw grey "before" bars underneath
    if initial_tre is not None:
        init_v = np.array(initial_tre)[valid][order]
        ax.bar(x, init_v, width=0.65, color=PALETTE["gray"], alpha=0.25,
               label="Before registration", zorder=1)

    # Colour bars by TRE magnitude
    cmap     = plt.get_cmap("RdYlGn_r")
    tre_norm = Normalize(vmin=0, vmax=max(tre_s.max(), 8))
    colors   = [cmap(tre_norm(v)) for v in tre_s]

    bars = ax.bar(x, tre_s, width=0.65, color=colors,
                  edgecolor="white", linewidth=0.4,
                  label="After registration (TRE)", zorder=2)

    # Value labels on top of each bar
    for bar, val in zip(bars, tre_s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7, color="#444")

    # Mean & median reference lines
    mean_tre   = float(np.nanmean(tre_s))
    median_tre = float(np.nanmedian(tre_s))
    ax.axhline(mean_tre,   color=PALETTE["blue"],  ls="--", lw=1.2,
               label=f"Mean {mean_tre:.2f} mm")
    ax.axhline(median_tre, color=PALETTE["teal"],  ls=":",  lw=1.2,
               label=f"Median {median_tre:.2f} mm")

    # Clinical threshold bands
    ax.axhspan(0, 2,   alpha=0.06, color=PALETTE["green"], zorder=0)
    ax.axhspan(2, 4,   alpha=0.06, color=PALETTE["amber"], zorder=0)
    ax.axhspan(4, 100, alpha=0.06, color=PALETTE["red"],   zorder=0)

    # Axis
    ax.set_xticks(x)
    ax.set_xticklabels(ids_s, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Target registration error (mm)")
    ax.set_title("TRE per patient — sorted ascending")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, tre_s.max() * 1.2 + 0.5)
    ax.yaxis.grid(True, zorder=0)

    # Colourbar
    sm = ScalarMappable(cmap=cmap, norm=tre_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.8)
    cb.set_label("TRE (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Threshold labels in right margin
    ax.text(len(x) - 0.1, 1.0,  "< 2 mm  ✓",  ha="right", va="center",
            fontsize=7.5, color=PALETTE["green"], style="italic")
    ax.text(len(x) - 0.1, 3.0,  "2–4 mm  ~",  ha="right", va="center",
            fontsize=7.5, color=PALETTE["amber"], style="italic")
    ax.text(len(x) - 0.1, 5.5,  "> 4 mm  ✗",  ha="right", va="center",
            fontsize=7.5, color=PALETTE["red"],   style="italic")

    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out = outdir / "fig1_tre_per_case.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 2 — DICE SCORE DISTRIBUTION ──────────────────────────────────────

def fig_dice_distribution(
    data: dict,
    outdir: Path = Path("."),
    show: bool = False,
) -> Path:
    """
    Two-panel Dice figure:
      Left  — histogram + KDE of Dice scores across all cases
      Right — per-case Dice bar chart, sorted descending
    """
    apply_style()

    dice   = data["dice"]
    ids    = data["case_ids"]
    valid  = ~np.isnan(dice)

    if not valid.any():
        print("[SKIP] No Dice values found — skipping Fig 2")
        return None

    dice_v = dice[valid]
    ids_v  = np.array(ids)[valid]

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5),
                                           gridspec_kw={"width_ratios": [1, 1.6]})

    # ── Left: histogram + KDE ──
    bins = np.linspace(0, 1, 21)
    ax_hist.hist(dice_v, bins=bins, color=PALETTE["teal"], alpha=0.55,
                 edgecolor="white", linewidth=0.5, density=True, label="Histogram")

    if HAS_SCIPY and len(dice_v) >= 3:
        kde = gaussian_kde(dice_v, bw_method=0.25)
        xs  = np.linspace(0, 1, 200)
        ax_hist.plot(xs, kde(xs), color=PALETTE["teal"], lw=2, label="KDE")

    ax_hist.axvline(np.mean(dice_v),   color=PALETTE["coral"], ls="--", lw=1.4,
                    label=f"Mean {np.mean(dice_v):.3f}")
    ax_hist.axvline(np.median(dice_v), color=PALETTE["purple"], ls=":",  lw=1.4,
                    label=f"Median {np.median(dice_v):.3f}")

    ax_hist.set_xlabel("Dice similarity coefficient")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Dice distribution across patients")
    ax_hist.set_xlim(0, 1)
    ax_hist.legend(fontsize=8)
    ax_hist.yaxis.grid(True)

    # ── Right: per-case bar, sorted descending ──
    order  = np.argsort(dice_v)[::-1]
    ids_s  = ids_v[order]
    dice_s = dice_v[order]
    x      = np.arange(len(ids_s))

    cmap   = plt.get_cmap("RdYlGn")
    norm   = Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(v)) for v in dice_s]

    bars = ax_bar.bar(x, dice_s, width=0.7, color=colors,
                      edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, dice_s):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color="#444")

    ax_bar.axhline(np.mean(dice_v),   color=PALETTE["coral"],  ls="--", lw=1.2)
    ax_bar.axhline(np.median(dice_v), color=PALETTE["purple"], ls=":",  lw=1.2)

    # Clinical reference at Dice = 0.7
    ax_bar.axhline(0.7, color=PALETTE["gray"], ls="-.", lw=0.8, alpha=0.6)
    ax_bar.text(len(x) - 0.5, 0.71, "0.70 threshold",
                ha="right", va="bottom", fontsize=7.5, color=PALETTE["gray"])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ids_s, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Dice")
    ax_bar.set_title("Dice per patient — sorted descending")
    ax_bar.set_ylim(0, 1.08)
    ax_bar.yaxis.grid(True)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_bar, pad=0.01, shrink=0.85)
    cb.set_label("Dice", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Dice Similarity Coefficient — Tumour Mask Registration",
                 fontsize=11, fontweight="medium", y=1.01)
    fig.tight_layout()

    out = outdir / "fig2_dice_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 3 — BEFORE / AFTER IMAGE COMPARISON ───────────────────────────────

def _load_mid_slice(path: Path, axis: int = 0) -> Optional[np.ndarray]:
    """Load NIfTI and return the central slice along given axis."""
    if not HAS_NIBABEL:
        return None
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        idx = arr.shape[axis] // 2
        sl  = np.take(arr, idx, axis=axis)
        # Normalise [1,99] percentile
        lo  = np.percentile(arr, 1);  hi = np.percentile(arr, 99)
        sl  = np.clip(sl, lo, hi)
        sl  = (sl - lo) / (hi - lo + 1e-8)
        return sl
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


def _case_aliases(case_id: str) -> list[str]:
    """Return common folder spellings for a case id, e.g. Case5 and Case05."""
    aliases = [case_id]
    prefix = "".join(ch for ch in case_id if not ch.isdigit())
    digits = "".join(ch for ch in case_id if ch.isdigit())
    if digits:
        aliases.append(f"{prefix}{int(digits):02d}")
        aliases.append(f"{prefix}{int(digits)}")
    return list(dict.fromkeys(aliases))


def _find_case_dir(root: Path, case_id: str) -> Path:
    for alias in _case_aliases(case_id):
        candidate = root / alias
        if candidate.exists():
            return candidate
    return root / case_id


def _load_first_mid_slice(paths: list[Path]) -> Optional[np.ndarray]:
    for path in paths:
        sl = _load_mid_slice(path)
        if sl is not None:
            return sl
    return None


def fig_before_after_grid(
    case_ids: list[str],
    infer_dir: str,
    data_dir: Optional[str] = None,
    outdir: Path = Path("."),
    max_cases: int = 6,
    show: bool = False,
) -> Optional[Path]:
    """
    Grid of axial slices: US (fixed) | MRI original | MRI warped | |difference|

    Expects infer_dir/<case_id>/ to contain:
      mri_warped.nii.gz     — registered MRI
      flow_magnitude.nii.gz — displacement magnitude

    And optionally data_dir/<case_id>/us_before.nii.gz etc.
    """
    if not HAS_NIBABEL:
        print("[SKIP] nibabel not installed — Fig 3 skipped")
        return None

    infer_root = Path(infer_dir)
    n_show     = min(len(case_ids), max_cases)
    n_cols     = 4      # US | MRI orig | MRI warped | |Δ|
    col_titles = ["US (fixed)", "MRI (original)", "MRI (warped)", "|Difference|"]
    cmaps_row  = ["hot", "gray", "gray", "RdYlGn_r"]

    apply_style()
    fig, axes = plt.subplots(n_show, n_cols,
                              figsize=(n_cols * 2.8, n_show * 2.8 + 0.6))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row_i, case_id in enumerate(case_ids[:n_show]):
        case_dir = _find_case_dir(infer_root, case_id)
        data_case_dir = _find_case_dir(Path(data_dir), case_id) if data_dir else None

        # Try to locate volumes
        us_sl = _load_first_mid_slice([
            case_dir / "us_fixed.nii.gz",
            case_dir / "us_before.nii.gz",
            case_dir / "US_before_processed.nii.gz",
        ])
        mri_orig = _load_first_mid_slice([
            case_dir / "mri_original.nii.gz",
            case_dir / "T1_processed.nii.gz",
        ])
        mri_warped = _load_mid_slice(case_dir / "mri_warped.nii.gz")
        flow_mag = _load_mid_slice(case_dir / "flow_magnitude.nii.gz")

        # Fallback: read fixed/original inputs from processed data.
        if us_sl is None and data_case_dir:
            us_sl = _load_first_mid_slice([
                data_case_dir / "US_before_processed.nii.gz",
                data_case_dir / "us_before.nii.gz",
                data_case_dir / "us_fixed.nii.gz",
            ])
        if mri_orig is None and data_dir:
            # Try reading from data_dir structure
            for candidate in ["T1.nii.gz", "T1_processed.nii.gz",
                               "mri_T1.nii.gz", "preop_T1.nii.gz"]:
                mri_orig = _load_mid_slice(data_case_dir / candidate)
                if mri_orig is not None:
                    break

        if mri_warped is None:
            print(f"  [WARN] Missing warped MRI for {case_id}: expected {case_dir / 'mri_warped.nii.gz'}")

        slices = [us_sl, mri_orig, mri_warped, None]

        # Compute difference slice
        if mri_warped is not None and us_sl is not None:
            # Resize if shapes differ slightly
            if mri_warped.shape != us_sl.shape:
                from scipy.ndimage import zoom
                f = [us_sl.shape[i] / mri_warped.shape[i] for i in range(2)]
                mri_warped_rs = zoom(mri_warped, f, order=1)
            else:
                mri_warped_rs = mri_warped
            slices[3] = np.abs(mri_warped_rs - us_sl)
        elif flow_mag is not None:
            slices[3] = flow_mag      # fallback: show shift magnitude

        for col_i, (sl, cmap) in enumerate(zip(slices, cmaps_row)):
            ax = axes[row_i, col_i]
            if sl is not None:
                ax.imshow(sl.T, cmap=cmap, origin="lower", aspect="equal",
                          interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa")
                ax.set_facecolor("#f5f5f5")
            ax.axis("off")

            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9)
            if col_i == 0:
                ax.set_ylabel(case_id, fontsize=8, rotation=0,
                              labelpad=50, va="center")

    fig.suptitle("Before / after registration — central axial slice per patient",
                 fontsize=11, fontweight="medium", y=1.01)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)

    out = outdir / "fig3_before_after_grid.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 4 — SUMMARY DASHBOARD ─────────────────────────────────────────────

def fig_summary_dashboard(
    data: dict,
    initial_tre: Optional[np.ndarray] = None,
    outdir: Path = Path("."),
    show: bool = False,
) -> Path:
    """
    Four-panel summary dashboard:
      A — TRE improvement scatter (initial vs final TRE)
      B — Jacobian determinant distribution
      C — Cumulative TRE CDF
      D — Metric summary table
    """
    apply_style()

    tre      = data["tre"]
    dice     = data["dice"]
    jac_mean = data["jac_mean"]
    pct_neg  = data["pct_neg"]
    case_ids = data["case_ids"]

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.40, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── A: TRE improvement scatter ──────────────────────────────────────────
    valid_t = ~np.isnan(tre)
    if valid_t.any() and initial_tre is not None:
        init_v = np.array(initial_tre)[valid_t]
        fin_v  = tre[valid_t]
        ax_A.scatter(init_v, fin_v,
                     c=PALETTE["purple"], alpha=0.75, s=60,
                     edgecolors="white", linewidths=0.5, zorder=3)
        # Identity line
        lim = max(init_v.max(), fin_v.max()) * 1.1
        ax_A.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="No change")
        # Regression line
        m, b = np.polyfit(init_v, fin_v, 1)
        xs   = np.linspace(0, lim, 100)
        ax_A.plot(xs, m * xs + b, color=PALETTE["teal"], lw=1.5, label="Linear fit")
        ax_A.set_xlim(0, lim); ax_A.set_ylim(0, lim)
        ax_A.set_xlabel("TRE before registration (mm)")
        ax_A.set_ylabel("TRE after registration (mm)")
        ax_A.set_title("A — Registration improvement")
        ax_A.legend(fontsize=8)
        ax_A.yaxis.grid(True); ax_A.xaxis.grid(True)
        # Label percentage improved
        n_improved = int((fin_v < init_v).sum())
        ax_A.text(0.05, 0.93,
                  f"{n_improved}/{valid_t.sum()} cases improved",
                  transform=ax_A.transAxes, fontsize=8.5,
                  color=PALETTE["teal"], fontweight="medium")
    elif valid_t.any():
        # No initial TRE: show simple violin / strip
        parts = ax_A.violinplot(tre[valid_t].tolist(),
                                positions=[0], showmedians=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE["purple"]); pc.set_alpha(0.55)
        parts["cmedians"].set_color(PALETTE["coral"]); parts["cmedians"].set_linewidth(2)
        # Jitter strip
        jitter = np.random.uniform(-0.05, 0.05, valid_t.sum())
        ax_A.scatter(jitter, tre[valid_t],
                     color=PALETTE["purple"], alpha=0.6, s=30, zorder=3)
        ax_A.set_xticks([0]); ax_A.set_xticklabels(["All cases"])
        ax_A.set_ylabel("TRE (mm)")
        ax_A.set_title("A — TRE distribution")
        ax_A.yaxis.grid(True)
    else:
        ax_A.text(0.5, 0.5, "No TRE data", ha="center", va="center",
                  transform=ax_A.transAxes, color="#aaa")
        ax_A.set_title("A — Registration improvement")

    # ── B: Jacobian determinant distribution ────────────────────────────────
    valid_j = ~np.isnan(jac_mean)
    if valid_j.any():
        jac_v = jac_mean[valid_j]
        pn_v  = pct_neg[valid_j]

        # Jac mean histogram
        ax_B.bar(range(len(jac_v)),
                 np.sort(jac_v)[::-1],
                 color=PALETTE["blue"], alpha=0.65,
                 edgecolor="white", linewidth=0.4, label="Jac det mean")
        ax_B.axhline(1.0, color="k",         ls="--", lw=0.8, alpha=0.5, label="det=1 (rigid)")
        ax_B.axhline(0.0, color=PALETTE["red"], ls="--", lw=0.8, alpha=0.7, label="det=0 (folding)")

        # Second y-axis for % negative
        ax_B2 = ax_B.twinx()
        ax_B2.plot(range(len(pn_v)),
                   np.array(pn_v)[np.argsort(jac_v)[::-1]],
                   "o-", color=PALETTE["coral"], ms=4, lw=1.2,
                   alpha=0.8, label="% neg Jac")
        ax_B2.set_ylabel("% negative Jac det", color=PALETTE["coral"], fontsize=9)
        ax_B2.tick_params(axis="y", labelcolor=PALETTE["coral"], labelsize=8)
        ax_B2.set_ylim(0, max(pn_v.max() * 1.5, 1))

        ax_B.set_xlabel("Patient (sorted by Jac det)")
        ax_B.set_ylabel("Jacobian det mean")
        ax_B.set_title("B — Deformation field regularity")

        lines1, labels1 = ax_B.get_legend_handles_labels()
        lines2, labels2 = ax_B2.get_legend_handles_labels()
        ax_B.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
                    loc="upper right")
        ax_B.yaxis.grid(True, alpha=0.5)
    else:
        ax_B.text(0.5, 0.5, "No Jacobian data", ha="center", va="center",
                  transform=ax_B.transAxes, color="#aaa")
        ax_B.set_title("B — Jacobian determinant")

    # ── C: Cumulative TRE CDF ───────────────────────────────────────────────
    if valid_t.any():
        tre_v = np.sort(tre[valid_t])
        cdf   = np.arange(1, len(tre_v) + 1) / len(tre_v)

        ax_C.plot(tre_v, cdf * 100,
                  color=PALETTE["purple"], lw=2.2, drawstyle="steps-post",
                  label="Registered")
        ax_C.fill_between(tre_v, cdf * 100, step="post",
                          color=PALETTE["purple"], alpha=0.10)

        # Clinical threshold dotted reference lines
        for thr, col in [(2, PALETTE["green"]), (4, PALETTE["amber"]), (6, PALETTE["red"])]:
            pct = float((tre_v <= thr).sum()) / len(tre_v) * 100
            ax_C.axvline(thr, color=col, ls="--", lw=0.9, alpha=0.7)
            ax_C.text(thr + 0.05, 8, f"{pct:.0f}%\n≤{thr}mm",
                      fontsize=7.5, color=col, va="bottom")

        ax_C.set_xlabel("TRE threshold (mm)")
        ax_C.set_ylabel("% of patients")
        ax_C.set_title("C — Cumulative distribution (CDF)")
        ax_C.set_xlim(0, max(tre_v.max() + 1, 7))
        ax_C.set_ylim(0, 105)
        ax_C.yaxis.grid(True)
        ax_C.legend(fontsize=8)
    else:
        ax_C.text(0.5, 0.5, "No TRE data", ha="center", va="center",
                  transform=ax_C.transAxes, color="#aaa")
        ax_C.set_title("C — CDF")

    # ── D: Summary statistics table ─────────────────────────────────────────
    ax_D.axis("off")
    rows = []
    if valid_t.any():
        t = tre[valid_t]
        rows += [
            ["TRE mean (mm)",   f"{np.mean(t):.2f} ± {np.std(t):.2f}"],
            ["TRE median (mm)", f"{np.median(t):.2f}"],
            ["TRE 95th %-ile",  f"{np.percentile(t, 95):.2f}"],
            ["TRE min / max",   f"{np.min(t):.2f} / {np.max(t):.2f}"],
            ["N patients (TRE)",f"{valid_t.sum()}"],
        ]
    dice_valid = ~np.isnan(dice)
    if dice_valid.any():
        d = dice[dice_valid]
        rows += [
            ["Dice mean",       f"{np.mean(d):.3f} ± {np.std(d):.3f}"],
            ["Dice median",     f"{np.median(d):.3f}"],
            ["Dice min / max",  f"{np.min(d):.3f} / {np.max(d):.3f}"],
            ["N patients (Dice)",f"{dice_valid.sum()}"],
        ]
    if valid_j.any():
        rows += [
            ["Jac det mean",    f"{np.mean(jac_mean[valid_j]):.3f}"],
            ["% neg Jac (max)", f"{np.max(pct_neg[valid_j]):.2f}%"],
        ]

    if rows:
        table = ax_D.table(
            cellText  = rows,
            colLabels = ["Metric", "Value"],
            cellLoc   = "center",
            loc       = "center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.55)

        # Style header
        for j in range(2):
            cell = table[0, j]
            cell.set_facecolor("#EEEDFE")
            cell.set_text_props(color="#3C3489", fontweight="medium")

        # Alternating row shading
        for i in range(1, len(rows) + 1):
            for j in range(2):
                cell = table[i, j]
                cell.set_facecolor("#F9F9FB" if i % 2 == 0 else "white")
                cell.set_edgecolor("#e8e8e8")

    ax_D.set_title("D — Summary statistics", fontsize=11, fontweight="medium")

    fig.suptitle("Registration Quality — Summary Dashboard",
                 fontsize=13, fontweight="medium", y=1.01)

    out = outdir / "fig4_summary_dashboard.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── OPTIONAL: SYNTHETIC DEMO DATA ─────────────────────────────────────────

def make_demo_data(n_cases: int = 23, seed: int = 42) -> tuple:
    """
    Generate plausible synthetic evaluation data to preview figures
    without a real checkpoint (reproduces RESECT-like statistics).
    """
    rng = np.random.default_rng(seed)
    case_ids   = [f"Case{str(i+1).zfill(2)}" for i in range(n_cases)]

    initial_tre = rng.normal(loc=5.2, scale=1.8, size=n_cases).clip(1.5, 12)
    improvement = rng.beta(a=3, b=2, size=n_cases) * 4.5    # 0–4.5 mm improvement
    final_tre   = np.clip(initial_tre - improvement + rng.normal(0, 0.3, n_cases), 0.3, None)

    dice        = rng.beta(a=7, b=3, size=n_cases).clip(0.3, 1.0)
    jac_mean    = rng.normal(loc=0.98, scale=0.04, size=n_cases).clip(0.7, 1.15)
    pct_neg     = np.abs(rng.normal(loc=0.3, scale=0.2, size=n_cases)).clip(0, 3)

    records = [
        {
            "case_id":      case_ids[i],
            "TRE_mm":       float(final_tre[i]),
            "Dice":         float(dice[i]),
            "jac_det_mean": float(jac_mean[i]),
            "pct_neg_jac":  float(pct_neg[i]),
        }
        for i in range(n_cases)
    ]
    return records, initial_tre


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot registration quality metrics (TRE, Dice, before/after)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use real evaluation outputs:
  python scripts/plot_registration_quality.py \\
      --eval_json results/evaluation/evaluation_test.json

  # Also render image grid (requires NIfTI inference outputs):
  python scripts/plot_registration_quality.py \\
      --eval_json results/evaluation/evaluation_test.json \\
      --infer_dir results/patient_outputs \\
      --data_dir  data/processed2

  # Preview with synthetic data (no data required):
  python scripts/plot_registration_quality.py --demo
        """
    )
    parser.add_argument("--eval_json",   nargs="*", default=None,
                        help="Path(s) to evaluation JSON(s)")
    parser.add_argument("--initial_tre", nargs="*", type=float, default=None,
                        help="Initial (pre-registration) TRE per case (same order as JSON)")
    parser.add_argument("--infer_dir",   default=None,
                        help="Directory containing per-case inference output folders")
    parser.add_argument("--data_dir",    default=None,
                        help="Processed data directory (for original MRI volumes)")
    parser.add_argument("--outdir",      default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--demo",        action="store_true",
                        help="Generate all figures using synthetic demo data")
    parser.add_argument("--max_cases",   type=int, default=6,
                        help="Max patients to show in image grid (Fig 3)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Output directory: {outdir}\n")

    # ── Load or generate data ──
    if args.demo:
        print("[INFO] Using synthetic demo data (--demo flag set)")
        records, initial_tre = make_demo_data(n_cases=23)
    elif args.eval_json:
        records = load_eval_jsons(args.eval_json)
        initial_tre = np.array(args.initial_tre) if args.initial_tre else None
    else:
        print("[ERROR] Provide --eval_json or --demo. Run with --help for usage.")
        return

    data = records_to_arrays(records)
    print(f"[INFO] Loaded {len(records)} patient records\n")

    # ── Generate figures ──
    print("Generating Fig 1 — TRE per case...")
    fig_tre_per_case(data, initial_tre, outdir)

    print("Generating Fig 2 — Dice distribution...")
    fig_dice_distribution(data, outdir)

    if args.infer_dir:
        print("Generating Fig 3 — Before/after image grid...")
        fig_before_after_grid(
            data["case_ids"], args.infer_dir,
            data_dir=args.data_dir, outdir=outdir,
            max_cases=args.max_cases,
        )
    else:
        print("[SKIP] Fig 3 — pass --infer_dir to enable image comparison grid")

    print("Generating Fig 4 — Summary dashboard...")
    fig_summary_dashboard(data, initial_tre, outdir)

    print(f"\n✅ All figures saved to: {outdir}/")
    print("   fig1_tre_per_case.png")
    print("   fig2_dice_distribution.png")
    print("   fig3_before_after_grid.png  (if --infer_dir provided)")
    print("   fig4_summary_dashboard.png\n")


if __name__ == "__main__":
    main()
