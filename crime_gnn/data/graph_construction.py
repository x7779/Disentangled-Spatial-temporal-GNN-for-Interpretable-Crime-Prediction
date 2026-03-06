# -*- coding: utf-8 -*-
"""
Graph construction utilities.

Provides:
  - compute_binary_spatial_adjacency
  - compute_continuous_spatial_adjacency
  - normalize_graph_scales
  - orthogonalize_graphs
  - graphs_to_torch
"""

from typing import Dict, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Spatial adjacency
# ---------------------------------------------------------------------------

def compute_binary_spatial_adjacency(
    gdf,
    region_to_idx: dict,
    method: str = "queen",
    knn_k: int = 5,
) -> np.ndarray:
    """Binary spatial adjacency from shapefile contiguity.

    Args:
        gdf: GeoDataFrame with polygons (already dissolved and sorted by
             region_idx).
        region_to_idx: mapping from region name → integer index.
        method: contiguity method — "queen" | "rook" | "knn".
        knn_k: number of nearest neighbours (only used when method="knn").

    Returns:
        A_sp: symmetric binary adjacency matrix [N, N].
    """
    from libpysal.weights import Queen, Rook, KNN

    num_nodes = len(region_to_idx)
    A_sp = np.zeros((num_nodes, num_nodes))

    if method == "queen":
        w = Queen.from_dataframe(gdf)
    elif method == "rook":
        w = Rook.from_dataframe(gdf)
    elif method == "knn":
        w = KNN.from_dataframe(gdf, k=knn_k)
    else:
        w = Queen.from_dataframe(gdf)

    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            A_sp[i, j] = 1.0

    A_sp = np.maximum(A_sp, A_sp.T)
    np.fill_diagonal(A_sp, 0)
    return A_sp


def compute_continuous_spatial_adjacency(
    gdf,
    region_to_idx: dict,
    method: str = "queen",
    sigma: float = 0.0,
    keep_topk: int = 15,
) -> np.ndarray:
    """Continuous spatial adjacency with distance-decay weighting.

    A_sp[i,j] = exp(-d² / 2σ²) for contiguous neighbours.
    Non-neighbours receive 0.  The matrix is symmetrised and normalised
    so that the maximum entry equals 1.

    Args:
        gdf: GeoDataFrame (dissolved, sorted).
        region_to_idx: region name → index.
        method: contiguity method.
        sigma: decay bandwidth; 0 = auto-calibrate from median neighbour
               distance.
        keep_topk: after distance-decay, retain only the top-k neighbours
                   per row.  0 disables.

    Returns:
        A_sp: continuous adjacency [N, N] in [0, 1].
    """
    from libpysal.weights import Queen, Rook, KNN
    from scipy.spatial.distance import cdist

    num_nodes = len(region_to_idx)

    if method == "queen":
        w = Queen.from_dataframe(gdf)
    elif method == "rook":
        w = Rook.from_dataframe(gdf)
    elif method == "knn":
        w = KNN.from_dataframe(gdf, k=keep_topk)
    else:
        w = Queen.from_dataframe(gdf)

    centroids = gdf.geometry.centroid
    coords = np.array([[c.x, c.y] for c in centroids])
    dist_matrix = cdist(coords, coords, metric="euclidean")

    if sigma <= 0:
        neighbor_dists = [
            dist_matrix[i, j]
            for i, neighbors in w.neighbors.items()
            for j in neighbors
        ]
        sigma = np.median(neighbor_dists) if neighbor_dists else 1.0
        print(f"  Auto-calibrated spatial decay sigma = {sigma:.4f}")

    A_sp = np.zeros((num_nodes, num_nodes))
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            A_sp[i, j] = np.exp(-dist_matrix[i, j] ** 2 / (2 * sigma ** 2))

    if keep_topk > 0:
        for i in range(num_nodes):
            row = A_sp[i].copy()
            if np.count_nonzero(row) > keep_topk:
                threshold = np.sort(row)[-keep_topk]
                A_sp[i, row < threshold] = 0.0

    A_sp = np.maximum(A_sp, A_sp.T)
    np.fill_diagonal(A_sp, 0)
    if A_sp.max() > 0:
        A_sp /= A_sp.max()

    return A_sp


# ---------------------------------------------------------------------------
# Scale normalisation
# ---------------------------------------------------------------------------

def normalize_graph_scales(
    A_sp: np.ndarray,
    A_ctx: np.ndarray,
    method: str = "frobenius",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Normalise two adjacency matrices to comparable scales.

    Args:
        A_sp: spatial adjacency [N, N].
        A_ctx: contextual adjacency [N, N].
        method: "frobenius" (geometric-mean target) | "max" | "mean_edge" |
                "none".

    Returns:
        A_sp_n, A_ctx_n: normalised matrices (clipped to [0, 1]).
        scale_info: dict with normalisation metadata.
    """
    scale_info: Dict[str, float] = {}

    if method == "frobenius":
        frob_sp = np.linalg.norm(A_sp, "fro")
        frob_ctx = np.linalg.norm(A_ctx, "fro")
        target = np.sqrt(frob_sp * frob_ctx)
        A_sp_n = A_sp * (target / max(frob_sp, 1e-10))
        A_ctx_n = A_ctx * (target / max(frob_ctx, 1e-10))
        scale_info = {
            "method": "frobenius",
            "sp_original_frob": frob_sp,
            "ctx_original_frob": frob_ctx,
            "target_norm": target,
        }
    elif method == "max":
        A_sp_n = A_sp / max(A_sp.max(), 1e-10)
        A_ctx_n = A_ctx / max(A_ctx.max(), 1e-10)
        scale_info = {"method": "max"}
    elif method == "mean_edge":
        m_sp = A_sp[A_sp > 0].mean() if (A_sp > 0).any() else 1.0
        m_ctx = A_ctx[A_ctx > 0].mean() if (A_ctx > 0).any() else 1.0
        target = np.sqrt(m_sp * m_ctx)
        A_sp_n = A_sp * (target / m_sp)
        A_ctx_n = A_ctx * (target / m_ctx)
        scale_info = {"method": "mean_edge"}
    else:
        A_sp_n, A_ctx_n = A_sp.copy(), A_ctx.copy()
        scale_info = {"method": "none"}

    A_sp_n = np.clip(A_sp_n, 0, 1)
    A_ctx_n = np.clip(A_ctx_n, 0, 1)

    print(f"  Scale Normalisation ({method}):")
    if (A_sp_n > 0).any():
        print(f"    A_sp : mean_edge {A_sp[A_sp > 0].mean():.4f}"
              f" -> {A_sp_n[A_sp_n > 0].mean():.4f}")
    if (A_ctx_n > 0).any():
        print(f"    A_ctx: mean_edge {A_ctx[A_ctx > 0].mean():.4f}"
              f" -> {A_ctx_n[A_ctx_n > 0].mean():.4f}")

    return A_sp_n, A_ctx_n, scale_info


# ---------------------------------------------------------------------------
# Orthogonalisation
# ---------------------------------------------------------------------------

def orthogonalize_graphs(
    A_sp: np.ndarray,
    A_ctx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose A_ctx into components orthogonal and parallel to A_sp.

    Both inputs should already be on the same scale (call
    ``normalize_graph_scales`` first).

    Returns:
        A_ctx_pure: the component of A_ctx orthogonal to A_sp.
        A_ctx_overlap: the component of A_ctx parallel to A_sp.
        overlap_mask: boolean mask where both graphs are active.
    """
    N = A_sp.shape[0]
    a_sp = A_sp.flatten().astype(np.float64)
    a_ctx = A_ctx.flatten().astype(np.float64)

    sp_norm_sq = np.dot(a_sp, a_sp)
    if sp_norm_sq < 1e-10:
        return A_ctx.copy(), np.zeros_like(A_ctx), np.zeros_like(A_ctx)

    proj_coeff = np.dot(a_ctx, a_sp) / sp_norm_sq
    a_ctx_parallel = proj_coeff * a_sp
    A_ctx_overlap = np.clip(a_ctx_parallel.reshape(N, N), 0, 1)

    a_ctx_pure = a_ctx - a_ctx_parallel
    A_ctx_pure = np.clip(a_ctx_pure.reshape(N, N), 0, 1)

    overlap_mask = ((A_sp > 0.1) & (A_ctx > 0.1)).astype(np.float32)

    total_ctx = (A_ctx > 0.01).sum()
    overlap_cnt = overlap_mask.sum()
    pure_cnt = ((A_ctx_pure > 0.01) & (overlap_mask < 0.5)).sum()
    print(f"  Graph Orthogonalisation:")
    print(f"    Total A_ctx edges:     {int(total_ctx)}")
    print(f"    Overlapping with A_sp: {int(overlap_cnt)} "
          f"({overlap_cnt / max(total_ctx, 1) * 100:.1f}%)")
    print(f"    Pure functional edges: {int(pure_cnt)}")
    print(f"    Projection coefficient: {proj_coeff:.4f}")

    return A_ctx_pure, A_ctx_overlap, overlap_mask


# ---------------------------------------------------------------------------
# Torch conversion
# ---------------------------------------------------------------------------

def graphs_to_torch(
    A_sp: np.ndarray,
    A_ctx: np.ndarray,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy adjacency matrices to float32 tensors."""
    A_sp_t = torch.tensor(A_sp, dtype=torch.float32, device=device)
    A_ctx_t = torch.tensor(A_ctx, dtype=torch.float32, device=device)
    return A_sp_t, A_ctx_t
