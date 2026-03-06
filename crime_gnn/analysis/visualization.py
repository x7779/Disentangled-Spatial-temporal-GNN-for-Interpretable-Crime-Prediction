# -*- coding: utf-8 -*-
"""
Visualization utilities for training diagnostics and graph inspection.

Provides:
  - plot_training_history     Loss, task metrics, and channel weights over epochs.
  - plot_adjacency_comparison Three-panel heatmap: A_sp, A_ctx, A_lat.
  - plot_edge_alpha_heatmaps  Per-edge gating weights (non-edges masked).
  - _add_phase_bands          Helper to shade Phase-1 / Phase-2 regions.
"""

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _add_phase_bands(ax: plt.Axes, phases: List[int]) -> None:
    """Shade Phase 1 (red) and Phase 2 (green) bands on an axes.

    Args:
        ax:     Matplotlib Axes to annotate.
        phases: List of phase values (1 or 2), one per epoch.
    """
    if not phases:
        return
    colors = {1: "#FFE0E0", 2: "#E0FFE0"}
    labels = {1: "Phase 1", 2: "Phase 2"}
    prev = phases[0]
    start = 0
    labeled: set = set()
    for i, p in enumerate(phases + [None]):
        if p != prev:
            lbl = labels.get(prev, "") if prev not in labeled else ""
            ax.axvspan(start, i, alpha=0.15, color=colors.get(prev, "#F0F0F0"),
                       label=lbl)
            labeled.add(prev)
            start = i
            prev = p


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def plot_training_history(
    history: Dict,
    task_type: str = "regression",
    save_path: Optional[str] = None,
) -> None:
    """Plot training-loss curves, task-specific metrics, and channel weights.

    For regression: plots MAE (top-right) and RMSE+R² (bottom-left).
    For classification: plots AUC-ROC (top-right) and F1 (bottom-left).

    Args:
        history:    History dict returned by ``CurriculumTrainer.train``.
        task_type:  ``"regression"`` or ``"classification"``.
        save_path:  If given, save the figure to this path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    phases = history.get("phase", [])

    # --- Top-left: Loss ---
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", alpha=0.8)
    ax.plot(history["val_loss"], label="Val", alpha=0.8)
    _add_phase_bands(ax, phases)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.set_title("Training Loss")

    # --- Top-right: primary metric ---
    ax = axes[0, 1]
    if task_type == "regression":
        vals = [m.get("mae", 0) for m in history["val_metrics"]]
        ax.plot(vals, color="green")
        ax.set_ylabel("MAE")
        ax.set_title("Validation MAE")
    else:
        vals = [m.get("auc_roc", 0) for m in history["val_metrics"]]
        ax.plot(vals, color="purple")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Validation AUC-ROC")
    _add_phase_bands(ax, phases)
    ax.set_xlabel("Epoch")

    # --- Bottom-left: secondary metric ---
    ax = axes[1, 0]
    if task_type == "regression":
        rmse_vals = [m.get("rmse", 0) for m in history["val_metrics"]]
        r2_vals = [m.get("r2", 0) for m in history["val_metrics"]]
        ax.plot(rmse_vals, label="RMSE", color="red")
        ax2 = ax.twinx()
        ax2.plot(r2_vals, label="R²", color="blue", alpha=0.7)
        ax2.set_ylabel("R²", color="blue")
        _add_phase_bands(ax, phases)
        ax.set_ylabel("RMSE", color="red")
        ax.set_title("RMSE & R²")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        f1_vals = [m.get("f1", 0) for m in history["val_metrics"]]
        prec_vals = [m.get("precision", 0) for m in history["val_metrics"]]
        rec_vals = [m.get("recall", 0) for m in history["val_metrics"]]
        ax.plot(f1_vals, label="F1", color="blue")
        ax.plot(prec_vals, label="Precision", color="orange", alpha=0.7)
        ax.plot(rec_vals, label="Recall", color="green", alpha=0.7)
        _add_phase_bands(ax, phases)
        ax.legend()
        ax.set_ylabel("Score")
        ax.set_title("F1 / Precision / Recall")
    ax.set_xlabel("Epoch")

    # --- Bottom-right: channel weights ---
    ax = axes[1, 1]
    ax.plot(history["alpha_sp"], label="w_sp", color="blue")
    ax.plot(history["alpha_ctx"], label="w_ctx", color="green")
    ax.plot(history["alpha_lat"], label="w_lat", color="orange")
    _add_phase_bands(ax, phases)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean weight")
    ax.set_title("Channel Weight Evolution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_adjacency_comparison(
    A_sp: np.ndarray,
    A_ctx: np.ndarray,
    A_lat: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel heatmap of spatial, contextual, and latent adjacency matrices.

    Non-edges (value < 0.01) are masked (shown as white).

    Args:
        A_sp:   Spatial adjacency [N, N].
        A_ctx:  Contextual adjacency [N, N].
        A_lat:  Latent adjacency [N, N] (typically mean over test batches).
        save_path: Optional save path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    configs = [
        (A_sp,  "A_sp (Spatial)",     "Blues"),
        (A_ctx, "A_ctx (Functional)", "Greens"),
        (A_lat, "A_lat (Latent)",     "Oranges"),
    ]
    for ax, (A, title, cmap) in zip(axes, configs):
        masked = np.ma.masked_where(A < 0.01, A)
        im = ax.imshow(masked, cmap=cmap, vmin=0)
        ax.set_title(f"{title}\nedges={(A > 0.01).sum():.0f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Adjacency Matrices (non-edges masked)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_edge_alpha_heatmaps(
    alpha_sp: np.ndarray,
    alpha_ctx: np.ndarray,
    alpha_lat: np.ndarray,
    A_sp: Optional[np.ndarray] = None,
    A_ctx: Optional[np.ndarray] = None,
    A_lat_mean: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel heatmap of per-edge gating weights with non-edges masked.

    Args:
        alpha_sp/ctx/lat: [N, N] per-edge averaged gating weights.
        A_sp, A_ctx:      Adjacency arrays used to compute the union mask.
        A_lat_mean:       Mean latent adjacency used in the union mask.
        save_path:        Optional save path.
    """
    if A_sp is not None and A_ctx is not None:
        union = (A_sp > 0.01) | (A_ctx > 0.01)
        if A_lat_mean is not None:
            union = union | (A_lat_mean > 0.01)
        hide = ~union
    else:
        hide = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    configs = [
        (alpha_sp,  "a_sp (per edge)",  "Blues"),
        (alpha_ctx, "a_ctx (per edge)", "Greens"),
        (alpha_lat, "a_lat (per edge)", "Oranges"),
    ]
    for ax, (d, title, cmap) in zip(axes, configs):
        if hide is not None:
            masked = np.ma.masked_where(hide, d)
            edge_mean = d[~hide].mean() if (~hide).any() else 0.0
        else:
            masked = d
            edge_mean = d.mean()
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title}\nmean={edge_mean:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Per-Edge Gating Weights (non-edges masked)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
