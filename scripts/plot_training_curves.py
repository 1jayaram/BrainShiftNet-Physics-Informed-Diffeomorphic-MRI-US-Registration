"""
scripts/plot_training_curves.py

Training diagnostics visualiser for the brain shift registration project.
Reads outputs/run_XX/training_history.json and produces four figures:

  Fig 1 — Total loss    train vs val, annotated with best epoch + early stop
  Fig 2 — Component loss grid    sim / reg / jac / landmark  (train + val)
  Fig 3 — Learning rate & overfitting monitor  (LR decay | train-val gap)
  Fig 4 — TRE convergence + proportional loss breakdown

History JSON schema (written by training/train.py):
  [
    {
      "epoch": int,
      "train": { "total": f, "sim": f, "reg": f, "jac": f, "landmark": f },
      "val":   { "total": f, "sim": f, "reg": f, "jac": f, "landmark": f,
                 "TRE_mean": f?,  "TRE_std": f? }
    }, ...
  ]

Usage
-----
  # Real run:
  python scripts/plot_training_curves.py \\
      --history outputs/run_01/training_history.json

  # Compare two runs on the same axes:
  python scripts/plot_training_curves.py \\
      --history outputs/run_01/training_history.json \\
               outputs/run_02/training_history.json \\
      --labels  "λ_lm=5  (default)" "λ_lm=2  (ablation)"

  # Synthetic demo — no files needed:
  python scripts/plot_training_curves.py --demo

Dependencies
------------
  pip install matplotlib numpy scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


# ─── COLOUR / STYLE ────────────────────────────────────────────────────────

# One palette per run (supports up to 3 simultaneous runs)
RUN_PALETTES = [
    {"train": "#7F77DD", "val": "#D85A30"},   # purple / coral   — run 1
    {"train": "#1D9E75", "val": "#EF9F27"},   # teal   / amber   — run 2
    {"train": "#378ADD", "val": "#D4537E"},   # blue   / pink    — run 3
]

# Colours for individual loss components
COMPONENT_COLORS = {
    "sim":      "#7F77DD",   # purple
    "reg":      "#1D9E75",   # teal
    "jac":      "#EF9F27",   # amber
    "landmark": "#D85A30",   # coral
}

COMPONENT_LABELS = {
    "sim":      "Similarity  (MI / NCC)",
    "reg":      "Regularisation  (bending)",
    "jac":      "Jacobian penalty",
    "landmark": "Landmark supervision",
}

# λ weights from DEFAULT_CONFIG — used to show weighted component breakdown
LAMBDA_WEIGHTS = {
    "sim":      1.0,
    "reg":      2.0,
    "jac":      0.5,
    "landmark": 5.0,
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "axes.titleweight":   "medium",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    8.5,
        "legend.frameon":     False,
        "grid.color":         "#e4e4e4",
        "grid.linewidth":     0.5,
        "font.family":        "DejaVu Sans",
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "lines.linewidth":    1.6,
    })


# ─── DATA HELPERS ──────────────────────────────────────────────────────────

def load_history(path: Path) -> list[dict]:
    with path.open() as f:
        history = json.load(f)
    if not history:
        raise ValueError(
            f"Training history is empty: {path}\n"
            "This usually means the resume run did not execute any epochs. "
            "Resume with --epochs greater than the checkpoint epoch, or use a populated history file."
        )
    return history


def resolve_history_paths(paths: list[str]) -> list[Path]:
    """Validate requested history files and show nearby alternatives."""
    resolved = [Path(p) for p in paths]
    missing = [p for p in resolved if not p.exists()]
    if not missing:
        return resolved

    available = sorted(Path("outputs").glob("**/training_history.json"))
    msg = ["Missing training history file(s):"]
    msg.extend(f"  - {p}" for p in missing)
    if available:
        msg.append("\nAvailable history file(s):")
        msg.extend(f"  - {p}" for p in available)
    else:
        msg.append("\nNo training_history.json files were found under outputs/.")
    msg.append("\nRun the missing experiment first, or remove the missing path from --history.")
    raise FileNotFoundError("\n".join(msg))


def merge_histories(histories: list[list[dict]]) -> list[dict]:
    """Merge continuation histories by epoch, keeping the latest record per epoch."""
    by_epoch = {}
    for history in histories:
        for record in history:
            by_epoch[int(record["epoch"])] = record
    return [by_epoch[e] for e in sorted(by_epoch)]


def extract(history: list[dict], split: str, key: str) -> np.ndarray:
    """Pull one metric series; returns NaN where the key is absent."""
    return np.array([r[split].get(key, np.nan) for r in history], dtype=float)


def epochs(history: list[dict]) -> np.ndarray:
    return np.array([r["epoch"] for r in history])


def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal running mean — does not peek ahead."""
    if not HAS_SCIPY or len(arr) < window:
        return arr
    out = uniform_filter1d(arr, size=window, mode="nearest")
    # Keep first `window` epochs unsmoothed so the curve starts at the true value
    out[:window] = arr[:window]
    return out


def reconstruct_lr(
    n_epochs: int,
    lr_init: float = 1e-4,
    gamma: float   = 0.95,
) -> np.ndarray:
    """Reproduce ExponentialLR(gamma) decay used in train.py."""
    return np.array([lr_init * (gamma ** e) for e in range(n_epochs)])


def reconstruct_lr_for_epochs(
    ep: np.ndarray,
    lr_init: float = 1e-4,
    gamma: float = 0.95,
) -> np.ndarray:
    """Reproduce ExponentialLR by absolute epoch number, including resumed runs."""
    return np.array([lr_init * (gamma ** max(int(e) - 1, 0)) for e in ep])


def best_epoch(history: list[dict]) -> int:
    """Epoch with minimum val total loss."""
    vals = extract(history, "val", "total")
    valid = ~np.isnan(vals)
    if not valid.any():
        return -1
    return int(epochs(history)[valid][np.argmin(vals[valid])])


# ─── FIG 1 — TOTAL LOSS ────────────────────────────────────────────────────

def fig_total_loss(
    runs: list[list[dict]],
    labels: list[str],
    outdir: Path,
    yscale: str = "auto",
    epoch_range: tuple[float, float] | None = None,
) -> Path:
    """
    Train + val total loss for one or more runs on the same axes.
    Annotations: best-epoch star, early-stop dashed vertical, smoothed overlay.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for run_i, (hist, label, pal) in enumerate(zip(runs, labels, RUN_PALETTES)):
        ep   = epochs(hist)
        tr   = extract(hist, "train", "total")
        va   = extract(hist, "val",   "total")

        suffix = f" — {label}" if len(runs) > 1 else ""

        # Raw curves (faint)
        ax.plot(ep, tr, color=pal["train"], alpha=0.25, lw=1.0)
        ax.plot(ep, va, color=pal["val"],   alpha=0.25, lw=1.0)

        # Smoothed curves (bold)
        ax.plot(ep, smooth(tr), color=pal["train"], lw=2.0,
                label=f"Train{suffix}")
        ax.plot(ep, smooth(va), color=pal["val"],   lw=2.0,
                ls="--", label=f"Val{suffix}")

        # Best epoch star
        be = best_epoch(hist)
        if be > 0:
            be_idx = list(ep).index(be) if be in ep else -1
            if be_idx >= 0:
                best_val = va[be_idx]
                ax.scatter(be, best_val, marker="*", s=180,
                           color=pal["val"], zorder=5,
                           label=f"Best epoch {be} ({best_val:.4f})")

        # Early-stop marker — last epoch if training ended before max
        last_ep = int(ep[-1])
        max_ep  = last_ep  # can't know config here; annotate if < 200
        if last_ep < 195 and run_i == 0:
            ax.axvline(last_ep, color="#888", ls=":", lw=1.0, alpha=0.7)
            ax.text(last_ep + 0.5, ax.get_ylim()[1] * 0.95,
                    f"Early stop\nepoch {last_ep}",
                    fontsize=7.5, color="#666", va="top")

    all_totals = []
    for hist in runs:
        all_totals.extend(extract(hist, "train", "total"))
        all_totals.extend(extract(hist, "val", "total"))
    all_totals = np.array(all_totals, dtype=float)
    valid_totals = all_totals[np.isfinite(all_totals) & (all_totals > 0)]
    use_log = (
        yscale == "log"
        or (
            yscale == "auto"
            and len(valid_totals) > 0
            and valid_totals.max() / max(valid_totals.min(), 1e-12) > 100
        )
    )

    ax.set_xlabel("Epoch")
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel("Total loss (log scale)")
        print("  [INFO] Fig 1 using log y-scale because runs have very different loss ranges.")
    else:
        ax.set_ylabel("Total loss")
    ax.set_title("Training and validation loss — total")
    if epoch_range:
        ax.set_xlim(*epoch_range)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True, alpha=0.4)
    ax.legend(loc="upper right")

    # Raw-vs-smooth legend entry
    raw_patch = Line2D([0], [0], color="#aaa", lw=1.0, alpha=0.6, label="Raw (unsmoothed)")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + [raw_patch], lbls + ["Raw (unsmoothed)"],
              loc="upper right", fontsize=8)

    out = outdir / "fig1_total_loss.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 2 — COMPONENT LOSS GRID ───────────────────────────────────────────

def fig_component_grid(
    runs: list[list[dict]],
    labels: list[str],
    outdir: Path,
    epoch_range: tuple[float, float] | None = None,
) -> Path:
    """
    2 × 2 grid — one panel per loss component.
    Each panel shows train (solid) + val (dashed) for every run.
    Smoothed overlay + shaded train/val gap.
    """
    apply_style()
    components = ["sim", "reg", "jac", "landmark"]
    fig, axes  = plt.subplots(2, 2, figsize=(12, 7))
    axes_flat  = axes.flatten()

    for ax, comp in zip(axes_flat, components):
        color = COMPONENT_COLORS[comp]

        for run_i, (hist, label, pal) in enumerate(zip(runs, labels, RUN_PALETTES)):
            ep = epochs(hist)
            tr = extract(hist, "train", comp)
            va = extract(hist, "val",   comp)

            # Use component colour for single-run; run palette colour for multi-run
            c_tr = color     if len(runs) == 1 else pal["train"]
            c_va = color     if len(runs) == 1 else pal["val"]

            suffix = f" ({label})" if len(runs) > 1 else ""

            # Shaded gap only for first (or only) run
            if run_i == 0:
                tr_sm = smooth(tr)
                va_sm = smooth(va)
                ax.fill_between(ep, tr_sm, va_sm,
                                where=(va_sm > tr_sm),
                                alpha=0.08, color=c_va,
                                label="Val > Train gap")

            ax.plot(ep, tr, color=c_tr, alpha=0.20, lw=0.9)
            ax.plot(ep, va, color=c_va, alpha=0.20, lw=0.9)
            ax.plot(ep, smooth(tr), color=c_tr, lw=2.0,
                    label=f"Train{suffix}")
            ax.plot(ep, smooth(va), color=c_va, lw=2.0,
                    ls="--", label=f"Val{suffix}")

            # Best epoch marker
            be = best_epoch(hist)
            if be > 0 and be in ep:
                bi = list(ep).index(be)
                ax.axvline(be, color="#bbb", ls=":", lw=0.9, zorder=0)

        lam = LAMBDA_WEIGHTS.get(comp, 1.0)
        ax.set_title(f"{COMPONENT_LABELS[comp]}   (λ = {lam})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if epoch_range:
            ax.set_xlim(*epoch_range)
        ax.yaxis.grid(True)
        ax.legend(fontsize=7.5)

    fig.suptitle("Per-component loss — train vs validation",
                 fontsize=12, fontweight="medium", y=1.01)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)

    out = outdir / "fig2_component_loss_grid.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 3 — LR SCHEDULE + OVERFITTING MONITOR ─────────────────────────────

def fig_lr_and_gap(
    runs: list[list[dict]],
    labels: list[str],
    outdir: Path,
    lr_init: float = 1e-4,
    gamma:   float = 0.95,
    epoch_range: tuple[float, float] | None = None,
) -> Path:
    """
    Left  — Reconstructed ExponentialLR schedule (log scale)
    Right — Train / val gap per epoch (overfitting monitor)
             Gap = val_total − train_total  (positive = overfitting)
    """
    apply_style()
    fig, (ax_lr, ax_gap) = plt.subplots(1, 2, figsize=(12, 4.5))

    for run_i, (hist, label, pal) in enumerate(zip(runs, labels, RUN_PALETTES)):
        ep  = epochs(hist)
        lrs = reconstruct_lr_for_epochs(ep, lr_init, gamma)

        suffix = f" — {label}" if len(runs) > 1 else ""

        # ── LR panel ──
        ax_lr.plot(ep, lrs, color=pal["train"], lw=2.0,
                   label=f"LR{suffix}")

        # Mark where best epoch falls on the LR curve
        be = best_epoch(hist)
        if be > 0 and be in ep:
            bi     = list(ep).index(be)
            lr_be  = lrs[bi]
            ax_lr.scatter(be, lr_be, marker="*", s=160,
                          color=pal["train"], zorder=5,
                          label=f"Best epoch {be}  (LR={lr_be:.2e})")

        # ── Gap panel ──
        tr  = extract(hist, "train", "total")
        va  = extract(hist, "val",   "total")
        gap = va - tr

        ax_gap.plot(ep, gap, color=pal["train"], alpha=0.25, lw=0.9)
        gap_sm = smooth(gap, window=7)
        ax_gap.plot(ep, gap_sm, color=pal["train"], lw=2.0,
                    label=f"Val − Train{suffix}")

    # ── LR panel styling ──
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate  (log scale)")
    ax_lr.set_title(f"LR schedule — ExponentialLR(γ={gamma})")
    ax_lr.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax_lr.yaxis.grid(True, which="both", alpha=0.4)
    ax_lr.xaxis.grid(True, alpha=0.4)
    ax_lr.legend()
    if epoch_range:
        ax_lr.set_xlim(*epoch_range)

    # Annotate 10× decay points
    for decade in [1e-4, 1e-5, 1e-6]:
        if decade < lr_init:
            ep_decade = np.log(decade / lr_init) / np.log(gamma)
            ax_lr.axvline(ep_decade, color="#ccc", ls=":", lw=0.8)
            ax_lr.text(ep_decade + 0.5, decade * 1.3,
                       f"{decade:.0e}", fontsize=7, color="#999")

    # ── Gap panel styling ──
    ax_gap.axhline(0, color="#aaa", ls="-", lw=0.8, alpha=0.6)

    # Shade regions: positive = overfitting, negative = underfitting
    ep_ref = epochs(runs[0])
    gap_ref = smooth(extract(runs[0], "val", "total") -
                     extract(runs[0], "train", "total"), window=7)
    ax_gap.fill_between(ep_ref, gap_ref, 0,
                        where=(gap_ref > 0),
                        alpha=0.10, color="#D85A30",
                        label="Overfitting region  (val > train)")
    ax_gap.fill_between(ep_ref, gap_ref, 0,
                        where=(gap_ref < 0),
                        alpha=0.10, color="#1D9E75",
                        label="Underfitting region  (train > val)")

    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Val loss − Train loss")
    ax_gap.set_title("Overfitting monitor  (generalisation gap)")
    ax_gap.yaxis.grid(True)
    ax_gap.xaxis.grid(True, alpha=0.4)
    ax_gap.legend(fontsize=8)
    if epoch_range:
        ax_gap.set_xlim(*epoch_range)

    fig.suptitle("Learning rate schedule and overfitting monitor",
                 fontsize=12, fontweight="medium", y=1.01)
    fig.tight_layout()

    out = outdir / "fig3_lr_and_gap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── FIG 4 — TRE CONVERGENCE + PROPORTIONAL BREAKDOWN ─────────────────────

def fig_tre_and_breakdown(
    runs: list[list[dict]],
    labels: list[str],
    outdir: Path,
    tre_runs: list[list[dict]] | None = None,
    tre_labels: list[str] | None = None,
    epoch_range: tuple[float, float] | None = None,
) -> Path:
    """
    Left  — Val TRE mean ± std over epochs  (if available in history)
    Right — Proportional (stacked area) breakdown of val loss components
             showing how much each term contributes to the total over time
    """
    apply_style()
    fig, (ax_tre, ax_stack) = plt.subplots(1, 2, figsize=(12, 4.5))
    tre_runs = tre_runs if tre_runs is not None else runs
    tre_labels = tre_labels if tre_labels is not None else labels

    # ── TRE panel ──
    any_tre = False
    for run_i, (hist, label, pal) in enumerate(zip(tre_runs, tre_labels, RUN_PALETTES)):
        ep      = epochs(hist)
        tre_mu  = extract(hist, "val", "TRE_mean")
        tre_std = extract(hist, "val", "TRE_std")
        valid   = ~np.isnan(tre_mu)

        if not valid.any():
            continue
        any_tre = True

        suffix  = f" — {label}" if len(runs) > 1 else ""
        ep_v    = ep[valid]
        mu_v    = tre_mu[valid]
        std_v   = tre_std[valid] if not np.isnan(tre_std[valid]).all() else np.zeros_like(mu_v)

        ax_tre.fill_between(ep_v,
                             mu_v - std_v, mu_v + std_v,
                             alpha=0.15, color=pal["val"])
        ax_tre.plot(ep_v, mu_v, color=pal["val"], lw=1.0, alpha=0.3)
        ax_tre.plot(ep_v, smooth(mu_v, window=5),
                    color=pal["val"], lw=2.2,
                    label=f"Val TRE{suffix}")

        # Best TRE epoch
        best_i = int(np.argmin(mu_v))
        ax_tre.scatter(ep_v[best_i], mu_v[best_i],
                       marker="*", s=180, color=pal["val"], zorder=5,
                       label=f"Best TRE {mu_v[best_i]:.2f} mm  (ep {ep_v[best_i]})")

        # Clinical target lines
        for thr, col, lbl in [(2.0, "#1D9E75", "< 2 mm target"),
                               (4.0, "#EF9F27", "4 mm threshold")]:
            if run_i == 0:
                ax_tre.axhline(thr, color=col, ls="--", lw=0.9, alpha=0.7,
                               label=lbl)

    if any_tre:
        ax_tre.set_xlabel("Epoch")
        ax_tre.set_ylabel("TRE (mm)")
        ax_tre.set_title("Val TRE convergence  (mean ± std)")
        ax_tre.set_ylim(bottom=0)
        ax_tre.yaxis.grid(True)
        ax_tre.xaxis.grid(True, alpha=0.4)
        ax_tre.legend(fontsize=8)
        if epoch_range:
            ax_tre.set_xlim(*epoch_range)
    else:
        ax_tre.text(0.5, 0.5,
                    "TRE_mean not found in history.\n"
                    "Train/resume with landmarks loaded\n"
                    "to record per-epoch validation TRE.",
                    ha="center", va="center",
                    transform=ax_tre.transAxes,
                    color="#888", fontsize=9, style="italic",
                    multialignment="center")
        ax_tre.set_title("Val TRE convergence  (not available)")
        ax_tre.set_xlabel("Epoch")
        if epoch_range:
            ax_tre.set_xlim(*epoch_range)

    # ── Proportional stacked-area panel — use first run only ──
    hist0 = runs[0]
    ep0   = epochs(hist0)
    comps = ["sim", "reg", "jac", "landmark"]

    # Weighted raw values  (λ_i * component_i) to match the actual total
    weighted = {}
    for c in comps:
        raw = extract(hist0, "val", c)
        weighted[c] = np.abs(raw) * LAMBDA_WEIGHTS[c]   # abs: avoid neg MI values

    # Stack: fill gaps where a component is all-NaN with zeros
    stack_arrays = []
    for c in comps:
        arr = weighted[c].copy()
        arr[np.isnan(arr)] = 0.0
        stack_arrays.append(smooth(arr, window=5))

    stack_np = np.vstack(stack_arrays)             # (4, T)
    row_sum  = stack_np.sum(axis=0, keepdims=True) + 1e-12
    stack_pct = stack_np / row_sum * 100           # proportional 0–100 %

    cumulative = np.zeros(len(ep0))
    for c, arr_pct in zip(comps, stack_pct):
        ax_stack.fill_between(ep0,
                              cumulative,
                              cumulative + arr_pct,
                              color=COMPONENT_COLORS[c],
                              alpha=0.75,
                              label=f"{COMPONENT_LABELS[c]}  (λ={LAMBDA_WEIGHTS[c]})")
        cumulative += arr_pct

    ax_stack.set_ylim(0, 100)
    ax_stack.set_xlabel("Epoch")
    ax_stack.set_ylabel("% of weighted val loss")
    ax_stack.set_title("Val loss composition  (proportional, weighted by λ)")
    ax_stack.yaxis.set_major_formatter(mticker.PercentFormatter())
    if epoch_range:
        ax_stack.set_xlim(*epoch_range)
    ax_stack.yaxis.grid(True, alpha=0.4, color="white")
    ax_stack.legend(loc="upper right", fontsize=7.5,
                    frameon=True, framealpha=0.85,
                    edgecolor="#ddd", fancybox=False)

    fig.suptitle("TRE convergence and loss composition",
                 fontsize=12, fontweight="medium", y=1.01)
    fig.tight_layout()

    out = outdir / "fig4_tre_and_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─── DEMO DATA ─────────────────────────────────────────────────────────────

def make_demo_history(
    n_epochs: int  = 160,
    seed: int      = 0,
    lr_init: float = 1e-4,
    gamma: float   = 0.95,
    has_tre: bool  = True,
) -> list[dict]:
    """
    Synthesise a plausible training_history.json for the RESECT brain-shift
    registration task.  Models the typical training dynamics:
      - MI similarity loss starts negative (−0.3), becomes more negative (better)
      - Bending energy and Jacobian losses shrink as deformation regularises
      - Landmark loss falls steeply once the network learns rough alignment
      - Val TRE: starts ~4.5 mm, converges to ~1.8 mm after ~80 epochs
      - ExponentialLR(γ=0.95) decays learning rate; loss plateaus match LR drops
    """
    rng = np.random.default_rng(seed)

    def decay(start, end, n, noise=0.02, curve=2.0):
        """Power-law decay with Gaussian noise."""
        t   = np.linspace(0, 1, n) ** curve
        sig = start + (end - start) * t
        sig += rng.normal(0, noise * np.abs(end - start), n)
        return sig

    ep = np.arange(1, n_epochs + 1)

    # --- Component raw values (before λ scaling) ---
    # sim: NMI starts near −0.3, falls toward −0.65 (more negative = better)
    sim_tr = decay(-0.32, -0.67, n_epochs, noise=0.012, curve=1.8)
    sim_va = sim_tr + rng.normal(0.02, 0.010, n_epochs)

    # reg (bending energy): starts 0.25, decays to 0.06
    reg_tr = decay(0.25, 0.06, n_epochs, noise=0.006, curve=1.5)
    reg_va = reg_tr + rng.normal(0.008, 0.005, n_epochs)

    # jac penalty: starts 0.05, near-zero after warm-up
    jac_tr = decay(0.050, 0.002, n_epochs, noise=0.002, curve=2.5)
    jac_va = jac_tr + rng.normal(0.002, 0.002, n_epochs)

    # landmark: starts 4.8 mm equivalent, falls to 1.6 mm
    lm_tr  = decay(4.8, 1.6, n_epochs, noise=0.06, curve=2.2)
    lm_va  = lm_tr  + rng.normal(0.12, 0.08, n_epochs)

    # Total = λ_sim * sim + λ_reg * reg + λ_jac * jac + λ_lm * lm
    # (sim is negative so λ=1 reduces total)
    def total(sim, reg, jac, lm):
        return (1.0 * sim + 2.0 * reg + 0.5 * jac + 5.0 * lm)

    tot_tr = total(sim_tr, reg_tr, jac_tr, lm_tr)
    tot_va = total(sim_va, reg_va, jac_va, lm_va)

    # TRE: starts ~4.5 mm, converges to ~1.8 mm
    tre_mu  = decay(4.5, 1.78, n_epochs, noise=0.04, curve=2.0) if has_tre else None
    tre_std = decay(0.8, 0.22, n_epochs, noise=0.015, curve=1.5) if has_tre else None

    history = []
    for i, e in enumerate(ep):
        record = {
            "epoch": int(e),
            "train": {
                "total":    float(tot_tr[i]),
                "sim":      float(sim_tr[i]),
                "reg":      float(reg_tr[i]),
                "jac":      float(jac_tr[i]),
                "landmark": float(lm_tr[i]),
            },
            "val": {
                "total":    float(tot_va[i]),
                "sim":      float(sim_va[i]),
                "reg":      float(reg_va[i]),
                "jac":      float(jac_va[i]),
                "landmark": float(lm_va[i]),
            },
        }
        if has_tre:
            record["val"]["TRE_mean"] = float(np.clip(tre_mu[i], 0, None))
            record["val"]["TRE_std"]  = float(np.clip(tre_std[i], 0, None))
        history.append(record)

    return history


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from training_history.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_training_curves.py \\
      --history outputs/run_01/training_history.json

  python scripts/plot_training_curves.py \\
      --history outputs/run_01/training_history.json \\
               outputs/run_02/training_history.json \\
      --labels  "default  (λ_lm=5)" "ablation  (λ_lm=2)"

  python scripts/plot_training_curves.py --demo
        """
    )
    parser.add_argument("--history", nargs="*", default=None,
                        help="Path(s) to training_history.json (one per run)")
    parser.add_argument("--labels",  nargs="*", default=None,
                        help="Display label per run (must match --history count)")
    parser.add_argument("--merge_histories", action="store_true",
                        help="Merge --history files into one continuation run sorted by epoch")
    parser.add_argument("--tre_history", nargs="*", default=None,
                        help="Optional history file(s) used only for the TRE convergence panel")
    parser.add_argument("--tre_labels", nargs="*", default=None,
                        help="Labels for --tre_history (defaults to --labels)")
    parser.add_argument("--epoch_range", nargs=2, type=float, default=None,
                        metavar=("START", "END"),
                        help="Force x-axis epoch range for all figures, e.g. --epoch_range 0 130")
    parser.add_argument("--lr",      type=float, default=1e-4,
                        help="Initial learning rate (default 1e-4)")
    parser.add_argument("--gamma",   type=float, default=0.95,
                        help="ExponentialLR gamma (default 0.95)")
    parser.add_argument("--outdir",  default="results/figures",
                        help="Output directory (default: results/figures)")
    parser.add_argument("--total_yscale", choices=["auto", "linear", "log"],
                        default="auto",
                        help="Y scale for Fig 1 total loss (default: auto)")
    parser.add_argument("--demo",    action="store_true",
                        help="Generate figures from synthetic demo data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Output directory: {outdir}\n")

    # ── Load or synthesise histories ──
    if args.demo:
        print("[INFO] Synthesising two demo runs for comparison view...")
        runs = [
            make_demo_history(n_epochs=160, seed=0, has_tre=True),   # λ_lm=5
            make_demo_history(n_epochs=140, seed=7, has_tre=True),   # λ_lm=2
        ]
        labels = ["λ_lm = 5.0  (default)", "λ_lm = 2.0  (ablation)"]
    elif args.history:
        try:
            history_paths = resolve_history_paths(args.history)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        try:
            runs = [load_history(p) for p in history_paths]
        except ValueError as exc:
            parser.error(str(exc))
        if args.labels:
            expected_labels = 1 if args.merge_histories else len(history_paths)
            if len(args.labels) != expected_labels:
                parser.error(
                    f"--labels count ({len(args.labels)}) must match --history count "
                    f"({expected_labels})"
                )
            labels = args.labels
        else:
            labels = [p.parent.name for p in history_paths]
        if args.merge_histories:
            runs = [merge_histories(runs)]
            labels = labels[:1] if args.labels else [" + ".join(labels)]
    else:
        parser.error("Provide --history path(s) or --demo")

    tre_runs = None
    tre_labels = None
    if args.tre_history:
        try:
            tre_paths = resolve_history_paths(args.tre_history)
            tre_runs = [load_history(p) for p in tre_paths]
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        if args.tre_labels:
            if len(args.tre_labels) != len(tre_runs):
                parser.error(
                    f"--tre_labels count ({len(args.tre_labels)}) must match "
                    f"--tre_history count ({len(tre_runs)})"
                )
            tre_labels = args.tre_labels
        else:
            tre_labels = args.tre_labels or [p.parent.name for p in tre_paths]

    epoch_range = tuple(args.epoch_range) if args.epoch_range else None

    # Trim palette list to number of runs
    palettes = RUN_PALETTES[: len(runs)]

    print(f"[INFO] Loaded {len(runs)} run(s):  {labels}\n")

    print("Generating Fig 1 — total loss curves...")
    fig_total_loss(runs, labels, outdir, yscale=args.total_yscale,
                   epoch_range=epoch_range)

    print("Generating Fig 2 — component loss grid...")
    fig_component_grid(runs, labels, outdir, epoch_range=epoch_range)

    print("Generating Fig 3 — LR schedule + overfitting monitor...")
    fig_lr_and_gap(runs, labels, outdir, lr_init=args.lr, gamma=args.gamma,
                   epoch_range=epoch_range)

    print("Generating Fig 4 — TRE convergence + loss breakdown...")
    fig_tre_and_breakdown(runs, labels, outdir, tre_runs=tre_runs,
                          tre_labels=tre_labels, epoch_range=epoch_range)

    print(f"\n✅  All figures saved to: {outdir}/")
    for name in ["fig1_total_loss.png", "fig2_component_loss_grid.png",
                 "fig3_lr_and_gap.png", "fig4_tre_and_breakdown.png"]:
        print(f"   {name}")
    print()


if __name__ == "__main__":
    main()
