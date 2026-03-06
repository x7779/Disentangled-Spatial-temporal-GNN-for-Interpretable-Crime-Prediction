#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — CLI entry point for the Interpretable Spatio-Temporal GNN.

Usage examples
--------------
# Full regression run (default):
python train.py --crime_csv_path data/crimes.csv --shapefile_path data/beats.shp

# Classification run:
python train.py --task_type classification --label_col is_violent \
    --crime_csv_path data/crimes.csv --shapefile_path data/beats.shp

# Ablation: disable latent graph:
python train.py --disable_latent_graph

# Multi-horizon regression (predict 1, 2, and 4 weeks ahead simultaneously):
python train.py --multi_horizon 1 2 4

# Run two experiments and compare:
python train.py --run_ablation
"""

import os
import random
import sys
import warnings
from typing import Dict

import numpy as np
import torch

from crime_gnn.config import build_argparser, build_config_from_args
from crime_gnn.data.dataset import prepare_data
from crime_gnn.models.stgnn import InterpretableSTGNN
from crime_gnn.training.losses import CrimePredictionLoss
from crime_gnn.training.trainer import CurriculumTrainer
from crime_gnn.analysis.attribution import (
    OrthogonalAttributionAnalyzer,
    DisentanglementReporter,
    build_pairwise_edge_table,
    summarise_edge_table,
)
from crime_gnn.analysis.visualization import (
    plot_training_history,
    plot_adjacency_comparison,
    plot_edge_alpha_heatmaps,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _count_params(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config, label: str = "full_model") -> tuple:
    """Run a single experiment with the given config.

    Returns:
        (history dict, test_results dict)
    """
    _seed_everything(config.seed)

    # Auto-set num_horizons from multi_horizon list
    if config.data.multi_horizon is not None:
        config.model.num_horizons = len(config.data.multi_horizon)
    else:
        config.model.num_horizons = 1

    print(f"\n{'#' * 70}")
    print(f"# Experiment: {label}")
    print(f"{'#' * 70}")

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    dataloaders, A_sp, A_ctx_pure, data_info = prepare_data(config)
    config.model.num_nodes = data_info["num_nodes"]
    config.model.num_contextual_features = data_info["num_contextual_features"]

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    X_ctx_tensor = torch.tensor(data_info["X_ctx"], dtype=torch.float32)
    model = InterpretableSTGNN(config, A_sp, A_ctx_pure, X_ctx_tensor)

    total_params, trainable = _count_params(model)
    print(f"\nModel: {total_params:,} params ({trainable:,} trainable)")

    # ------------------------------------------------------------------
    # 3. Loss
    # ------------------------------------------------------------------
    loss_fn = CrimePredictionLoss(config)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    experiment_dir = os.path.join(config.output_dir, label)
    trainer = CurriculumTrainer(
        model, loss_fn, config, A_sp, A_ctx_pure,
        dataloaders,
        target_info=data_info["target_info"],
        output_dir=experiment_dir,
    )
    history, test_results = trainer.train()

    # ------------------------------------------------------------------
    # 5. Collect per-edge alphas and mean A_lat
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Collecting Per-Edge Attribution")
    print("=" * 70)
    a_sp_e, a_ctx_e, a_lat_e, a_lat_adj, persistence = trainer.collect_edge_alphas(
        split="test"
    )

    # ------------------------------------------------------------------
    # 6. Interpretability analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Interpretability Analysis")
    print("=" * 70)

    analyzer = OrthogonalAttributionAnalyzer(
        model=model,
        A_sp_np=data_info["A_sp_norm"],
        A_ctx_np=data_info["A_ctx_pure"],
        idx_to_region=data_info["idx_to_region"],
    )
    alpha_stats = analyzer.analyze_edges(
        dataloaders["test"], device=config.training.device
    )

    top_lat = analyzer.get_top_latent_edges(num_edges=20, A_lat_mean=a_lat_adj)
    print("\nTop Latent Graph Edges (strongest in mean A_lat):")
    if len(top_lat) > 0:
        print(top_lat.to_string(index=False))
    else:
        print("  No strong latent edges found.")

    # ------------------------------------------------------------------
    # 7. Disentanglement report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Graph Disentanglement Report")
    print("=" * 70)

    reporter = DisentanglementReporter(
        data_info["A_sp_norm"],
        data_info["A_ctx_pure"],
    )
    report_text = reporter.summary(
        alpha_stats, persistence=persistence, A_lat_mean=a_lat_adj
    )
    print("\n" + report_text)

    report_path = os.path.join(experiment_dir, "disentanglement_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # ------------------------------------------------------------------
    # 8. Pairwise edge attribution table
    # ------------------------------------------------------------------
    edge_table = build_pairwise_edge_table(
        a_sp_e, a_ctx_e, a_lat_e,
        A_sp=data_info["A_sp_norm"],
        A_ctx=data_info["A_ctx_pure"],
        A_lat_mean=a_lat_adj,
        idx_to_region=data_info["idx_to_region"],
        persistence=persistence,
    )
    print(summarise_edge_table(edge_table))

    edge_csv = os.path.join(experiment_dir, "pairwise_edge_attribution.csv")
    edge_table.to_csv(edge_csv, index=False)
    print(f"\nEdge table saved → {edge_csv}")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    task_type = getattr(config.data, "task_type", "regression")
    try:
        plot_training_history(
            history,
            task_type=task_type,
            save_path=os.path.join(experiment_dir, "training_history.png"),
        )
        plot_adjacency_comparison(
            data_info["A_sp_norm"], data_info["A_ctx_pure"], a_lat_adj,
            save_path=os.path.join(experiment_dir, "adjacency_comparison.png"),
        )
        plot_edge_alpha_heatmaps(
            a_sp_e, a_ctx_e, a_lat_e,
            A_sp=data_info["A_sp_norm"],
            A_ctx=data_info["A_ctx_pure"],
            A_lat_mean=a_lat_adj,
            save_path=os.path.join(experiment_dir, "edge_alpha_heatmaps.png"),
        )
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    return history, test_results


# ---------------------------------------------------------------------------
# Ablation comparison helper
# ---------------------------------------------------------------------------

def print_ablation_comparison(results: Dict[str, Dict]):
    """Pretty-print a side-by-side comparison of ablation study results."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPARISON")
    print("=" * 70)

    labels = list(results.keys())
    all_keys: set = set()
    for r in results.values():
        all_keys.update(r.keys())

    # Prioritise regression metrics, then per-horizon, then channel weights
    priority = ["mae", "rmse", "mse", "mape", "r2", "composite_mae_rmse",
                "auc_roc", "f1", "precision", "recall"]
    per_h_keys = sorted([k for k in all_keys if k.startswith("mae_h") or k.startswith("rmse_h")])
    priority += per_h_keys + ["alpha_sp", "alpha_ctx", "alpha_lat"]
    ordered_keys = [k for k in priority if k in all_keys]
    ordered_keys += sorted(all_keys - set(ordered_keys))

    col_w = max((len(l) for l in labels), default=12) + 2
    col_w = max(col_w, 12)
    header = f"{'Metric':<24s}" + "".join(f"{l:>{col_w}s}" for l in labels)
    print(header)
    print("-" * len(header))

    lower_is_better = {"mae", "rmse", "mse", "mape", "composite_mae_rmse"}
    lower_is_better.update(
        k for k in all_keys if k.startswith("mae_h") or k.startswith("rmse_h")
    )

    for key in ordered_keys:
        row_str = f"{key:<24s}"
        vals = [results[lab].get(key, float("nan")) for lab in labels]
        for v in vals:
            row_str += f"{v:>{col_w}.4f}"
        if key not in ("alpha_sp", "alpha_ctx", "alpha_lat"):
            valid = [v for v in vals if not np.isnan(v)]
            if len(valid) > 1:
                best = min(valid) if key in lower_is_better else max(valid)
                best_idx = [i for i, v in enumerate(vals) if v == best]
                if len(best_idx) == 1:
                    row_str += f"  <- {labels[best_idx[0]]}"
        print(row_str)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_argparser()
    parser.add_argument(
        "--run_ablation",
        action="store_true",
        default=False,
        help="Run full model + no-latent + no-gating ablations and compare.",
    )
    args = parser.parse_args()
    config = build_config_from_args(args)

    if args.run_ablation:
        results: Dict[str, Dict] = {}

        # Full model
        _, test_full = run_experiment(config, label="full_model")
        results["full_model"] = test_full

        # No latent graph
        from copy import deepcopy
        cfg_nolat = deepcopy(config)
        cfg_nolat.ablation.disable_latent_graph = True
        _, test_nolat = run_experiment(cfg_nolat, label="no_latent")
        results["no_latent"] = test_nolat

        # No gating
        cfg_nogate = deepcopy(config)
        cfg_nogate.ablation.disable_gating = True
        _, test_nogate = run_experiment(cfg_nogate, label="no_gating")
        results["no_gating"] = test_nogate

        print_ablation_comparison(results)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
