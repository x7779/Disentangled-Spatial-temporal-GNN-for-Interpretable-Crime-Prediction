# -*- coding: utf-8 -*-
"""
Interpretability and attribution analysis.

Provides:
  - OrthogonalAttributionAnalyzer  Per-node channel-weight analysis.
  - DisentanglementReporter        Textual disentanglement summary.
  - build_pairwise_edge_table      Per-edge attribution DataFrame.
  - summarise_edge_table           Human-readable edge-table summary.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


class OrthogonalAttributionAnalyzer:
    """Edge-level and node-level attribution via union-mask-weighted alphas.

    Args:
        model:          Trained ``InterpretableSTGNN``.
        A_sp_np:        Spatial adjacency [N, N] as numpy array.
        A_ctx_np:       Contextual adjacency [N, N] as numpy array.
        idx_to_region:  Mapping {node_index → region_name}.
    """

    def __init__(self, model, A_sp_np: np.ndarray, A_ctx_np: np.ndarray,
                 idx_to_region: Dict):
        self.model = model
        self.A_sp = A_sp_np
        self.A_ctx = A_ctx_np
        self.idx_to_region = idx_to_region

    @torch.no_grad()
    def analyze_edges(self, dataloader, device: str = "cpu") -> Dict[str, np.ndarray]:
        """Batch-size-weighted per-node channel weights over a dataloader.

        Returns:
            Dict with keys:
              ``node_w_sp``, ``node_w_ctx``, ``node_w_lat`` — [N] arrays.
              ``global_w_sp``, ``global_w_ctx``, ``global_w_lat`` — scalars.
        """
        self.model.eval()
        self.model.to(device)
        N = self.A_sp.shape[0]

        w_sp_acc = np.zeros(N)
        w_ctx_acc = np.zeros(N)
        w_lat_acc = np.zeros(N)
        total_samples = 0

        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = self.model(batch)
            bs = out["alpha_sp"].size(0)
            w_sp_acc += out["mon_w_sp"].sum(0).cpu().numpy()
            w_ctx_acc += out["mon_w_ctx"].sum(0).cpu().numpy()
            w_lat_acc += out["mon_w_lat"].sum(0).cpu().numpy()
            total_samples += bs

        d = max(total_samples, 1)
        w_sp_mean = w_sp_acc / d
        w_ctx_mean = w_ctx_acc / d
        w_lat_mean = w_lat_acc / d

        return {
            "node_w_sp": w_sp_mean,
            "node_w_ctx": w_ctx_mean,
            "node_w_lat": w_lat_mean,
            "global_w_sp": float(w_sp_mean.mean()),
            "global_w_ctx": float(w_ctx_mean.mean()),
            "global_w_lat": float(w_lat_mean.mean()),
        }

    def get_top_latent_edges(
        self,
        num_edges: int = 20,
        A_lat_mean: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Return the top-N strongest latent edges with graph-membership flags.

        Args:
            num_edges:  Maximum number of edges to return.
            A_lat_mean: Mean A_lat array [N, N].  Falls back to the model's
                        static latent graph if None.

        Returns:
            DataFrame with columns: source, target, latent_weight,
            in_spatial, in_contextual, unique_to_latent.
        """
        if A_lat_mean is not None:
            A_lat = A_lat_mean
        else:
            A_lat = self.model.get_static_latent_graph().cpu().numpy()

        N = A_lat.shape[0]
        rows = []
        for i in range(N):
            for j in range(i + 1, N):
                lat_w = max(A_lat[i, j], A_lat[j, i])
                if lat_w > 0.01:
                    in_sp = self.A_sp[i, j] > 0.01
                    in_ctx = self.A_ctx[i, j] > 0.01
                    rows.append({
                        "source": self.idx_to_region.get(i, i),
                        "target": self.idx_to_region.get(j, j),
                        "latent_weight": lat_w,
                        "in_spatial": in_sp,
                        "in_contextual": in_ctx,
                        "unique_to_latent": not in_sp and not in_ctx,
                    })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("latent_weight", ascending=False).head(num_edges)
        return df


class DisentanglementReporter:
    """Generates a textual summary of graph disentanglement quality.

    Args:
        A_sp:  Spatial adjacency numpy array.
        A_ctx: Contextual adjacency numpy array.
    """

    def __init__(self, A_sp: np.ndarray, A_ctx: np.ndarray):
        self.A_sp = A_sp
        self.A_ctx = A_ctx

    def summary(
        self,
        alpha_stats: Dict,
        persistence: Optional[np.ndarray] = None,
        A_lat_mean: Optional[np.ndarray] = None,
    ) -> str:
        """Render a human-readable disentanglement report.

        Args:
            alpha_stats:  Output of ``OrthogonalAttributionAnalyzer.analyze_edges``.
            persistence:  Latent-edge persistence array [N, N] (fraction of
                          samples where each latent edge was active).
            A_lat_mean:   Mean A_lat array [N, N].

        Returns:
            Multi-line string report.
        """
        N = self.A_sp.shape[0]
        n_sp = (self.A_sp > 0.01).sum()
        n_ctx = (self.A_ctx > 0.01).sum()
        overlap = ((self.A_sp > 0.01) & (self.A_ctx > 0.01)).sum()

        lines = [
            "=" * 60,
            "Graph Disentanglement Report",
            "=" * 60,
            f"Number of nodes:                  {N}",
            f"Spatial edges (A_sp > 0.01):      {int(n_sp)}",
            f"Contextual edges (A_ctx > 0.01):  {int(n_ctx)}",
            f"Overlapping edges (both > 0.01):  {int(overlap)} "
            f"({overlap / max(n_ctx, 1) * 100:.1f}% of contextual)",
            "",
            "Per-node channel weights (union-mask mean across nodes):",
            f"  w_sp  = {alpha_stats['global_w_sp']:.4f}",
            f"  w_ctx = {alpha_stats['global_w_ctx']:.4f}",
            f"  w_lat = {alpha_stats['global_w_lat']:.4f}",
            "",
        ]

        if persistence is not None and A_lat_mean is not None:
            n_lat_total = n_persistent = n_transient = 0
            for i in range(N):
                for j in range(i + 1, N):
                    p = max(persistence[i, j], persistence[j, i])
                    a = max(A_lat_mean[i, j], A_lat_mean[j, i])
                    if a > 0.01:
                        n_lat_total += 1
                        if p >= 0.5:
                            n_persistent += 1
                        else:
                            n_transient += 1

            lines.append("Latent edge statistics:")
            lines.append(
                f"  Total latent edges (A_lat_mean > 0.01):  {n_lat_total}"
            )
            lines.append(
                f"  Persistent (active ≥ 50% of samples):   {n_persistent}"
            )
            lines.append(
                f"  Transient  (active < 50% of samples):   {n_transient}"
            )
            if n_lat_total > 0:
                lines.append(
                    f"  Persistence ratio: "
                    f"{n_persistent / n_lat_total * 100:.1f}% persistent, "
                    f"{n_transient / n_lat_total * 100:.1f}% transient"
                )
            lines.append("")

        w_lat = alpha_stats["global_w_lat"]
        if w_lat < 0.1:
            lines.append(
                "Latent contribution is low — A_sp + A_ctx capture most "
                "predictive relationships."
            )
        elif w_lat < 0.25:
            lines.append(
                f"Latent w_lat = {w_lat:.4f}: A_lat provides moderate "
                "complementary signal beyond spatial and contextual graphs."
            )
        else:
            lines.append(
                f"Latent w_lat = {w_lat:.4f}: A_lat contributes substantially, "
                "suggesting hidden crime-diffusion patterns beyond spatial "
                "proximity and functional similarity."
            )

        return "\n".join(lines)


def build_pairwise_edge_table(
    alpha_sp_edges: np.ndarray,
    alpha_ctx_edges: np.ndarray,
    alpha_lat_edges: np.ndarray,
    A_sp: np.ndarray,
    A_ctx: np.ndarray,
    A_lat_mean: np.ndarray,
    idx_to_region: Dict,
    persistence: Optional[np.ndarray] = None,
    min_edge_strength: float = 0.01,
) -> pd.DataFrame:
    """Build a per-edge attribution DataFrame.

    Each row represents a symmetric node pair (i < j) that appears in at
    least one of the three graph channels.  Rows include per-channel
    adjacency weights, gating alphas, effective contributions, percentage
    breakdowns, dominant source, persistence (for latent edges), and a
    categorical ``edge_type`` label.

    Args:
        alpha_sp/ctx/lat_edges: [N, N] averaged per-edge gating weights.
        A_sp, A_ctx, A_lat_mean: [N, N] adjacency arrays.
        idx_to_region:  {node_index → region_name}.
        persistence:    [N, N] latent-edge persistence fractions.
        min_edge_strength: Minimum adjacency value to count an edge as present.

    Returns:
        DataFrame sorted by dominant_source, then descending alpha.
    """
    N = A_sp.shape[0]
    rows = []

    for i in range(N):
        for j in range(i + 1, N):
            has_sp = A_sp[i, j] > min_edge_strength
            has_ctx = A_ctx[i, j] > min_edge_strength
            has_lat = max(A_lat_mean[i, j], A_lat_mean[j, i]) > min_edge_strength

            if not (has_sp or has_ctx or has_lat):
                continue

            # Symmetric average of directed alphas
            a_sp = (alpha_sp_edges[i, j] + alpha_sp_edges[j, i]) / 2
            a_ctx = (alpha_ctx_edges[i, j] + alpha_ctx_edges[j, i]) / 2
            a_lat = (alpha_lat_edges[i, j] + alpha_lat_edges[j, i]) / 2
            total = a_sp + a_ctx + a_lat + 1e-8

            # Effective contribution: alpha × adjacency
            eff_sp = a_sp * A_sp[i, j]
            eff_ctx = a_ctx * A_ctx[i, j]
            eff_lat = a_lat * max(A_lat_mean[i, j], A_lat_mean[j, i])

            contributions: Dict[str, float] = {}
            if has_sp:
                contributions["spatial"] = eff_sp
            if has_ctx:
                contributions["contextual"] = eff_ctx
            if has_lat:
                contributions["latent"] = eff_lat
            dominant = max(contributions, key=contributions.get)

            edge_persist = (
                max(persistence[i, j], persistence[j, i])
                if persistence is not None
                else (1.0 if has_lat else 0.0)
            )

            # Categorical edge type
            if has_lat and has_sp and has_ctx:
                edge_type = "all_three"
            elif has_lat and has_sp:
                edge_type = "spatial+latent"
            elif has_lat and has_ctx:
                edge_type = "contextual+latent"
            elif has_lat:
                edge_type = "persistent_latent" if edge_persist >= 0.5 else "transient_latent"
            elif has_sp and has_ctx:
                edge_type = "spatial+contextual"
            elif has_sp:
                edge_type = "spatial_only"
            else:
                edge_type = "contextual_only"

            rows.append({
                "source": idx_to_region.get(i, str(i)),
                "target": idx_to_region.get(j, str(j)),
                "src_idx": i,
                "tgt_idx": j,
                "A_sp": A_sp[i, j],
                "A_ctx": A_ctx[i, j],
                "A_lat": max(A_lat_mean[i, j], A_lat_mean[j, i]),
                "alpha_sp": a_sp,
                "alpha_ctx": a_ctx,
                "alpha_lat": a_lat,
                "eff_sp": eff_sp,
                "eff_ctx": eff_ctx,
                "eff_lat": eff_lat,
                "pct_sp": a_sp / total * 100,
                "pct_ctx": a_ctx / total * 100,
                "pct_lat": a_lat / total * 100,
                "dominant_source": dominant,
                "persistence": edge_persist,
                "edge_type": edge_type,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            ["dominant_source", "alpha_sp"], ascending=[True, False]
        )
    return df


def summarise_edge_table(df: pd.DataFrame) -> str:
    """Render a human-readable summary of a pairwise edge-attribution table.

    Args:
        df: Output of ``build_pairwise_edge_table``.

    Returns:
        Multi-line string report.
    """
    if len(df) == 0:
        return "No edges to summarise."

    col_map = {"spatial": "alpha_sp", "contextual": "alpha_ctx", "latent": "alpha_lat"}
    lines = [f"Total edges analysed: {len(df)}", ""]

    dom_counts = df["dominant_source"].value_counts()
    lines.append("Edges by dominant source:")
    for src, cnt in dom_counts.items():
        col = col_map.get(src, "")
        mean_a = df[df["dominant_source"] == src][col].mean() if col else 0.0
        lines.append(
            f"  {src:12s}: {cnt:4d} edges ({cnt / len(df) * 100:5.1f}%), "
            f"mean alpha={mean_a:.3f}"
        )
    lines.append("")

    if "edge_type" in df.columns:
        type_counts = df["edge_type"].value_counts()
        lines.append("Edges by graph membership:")
        for etype, cnt in type_counts.items():
            lines.append(
                f"  {etype:24s}: {cnt:4d} edges ({cnt / len(df) * 100:5.1f}%)"
            )
        lines.append("")

    if "persistence" in df.columns:
        lat_edges = df[df["A_lat"] > 0.01]
        if len(lat_edges) > 0:
            n_persistent = (lat_edges["persistence"] >= 0.5).sum()
            n_transient = (lat_edges["persistence"] < 0.5).sum()
            mean_p = lat_edges["persistence"].mean()
            lines.append(
                f"Latent edge persistence (among {len(lat_edges)} latent edges):"
            )
            lines.append(
                f"  Persistent (≥ 50%): {n_persistent}  |  "
                f"Transient (< 50%): {n_transient}"
            )
            lines.append(f"  Mean persistence: {mean_p:.3f}")
            lines.append("")

    adjacency_col = {"spatial": "A_sp", "contextual": "A_ctx", "latent": "A_lat"}
    for src in ["spatial", "contextual", "latent"]:
        col = col_map.get(src)
        adj_col = adjacency_col.get(src)
        if col and col in df.columns:
            subset = df[df[adj_col] > 0.01] if adj_col and adj_col in df.columns else df
            top = subset.nlargest(5, col)
            lines.append(f"Top 5 edges by {src} attribution:")
            for _, row in top.iterrows():
                extra = ""
                if src == "latent" and "persistence" in row:
                    extra = f" persist={row['persistence']:.2f}"
                lines.append(
                    f"  {str(row['source']):>8s} - {str(row['target']):<8s}: "
                    f"a_sp={row['alpha_sp']:.3f} "
                    f"a_ctx={row['alpha_ctx']:.3f} "
                    f"a_lat={row['alpha_lat']:.3f} "
                    f"[A_sp={row['A_sp']:.3f}, A_ctx={row['A_ctx']:.3f}, "
                    f"A_lat={row['A_lat']:.3f}]{extra}"
                )
            lines.append("")

    return "\n".join(lines)
