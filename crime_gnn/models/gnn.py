# -*- coding: utf-8 -*-
"""
Disentangled GNN layers:
  - DisentangledConv  (single MessagePassing layer, 3-channel W_sp / W_ctx / W_lat)
  - DisentangledGNN   (multi-layer stack with LayerNorm + ReLU + dropout)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class DisentangledConv(MessagePassing):
    """Single-layer disentangled message passing with per-channel normalisation.

    Three separate weight matrices (W_sp, W_ctx, W_lat) transform neighbour
    features independently.  Per-edge gating weights (alpha) and adjacency
    weights (w) control each channel's contribution.  Symmetric degree
    normalisation is applied *per channel* to prevent entanglement across
    graphs.

    A learnable ``deg_power`` parameter (initialised to −0.5 for all
    channels, i.e. symmetric normalisation) allows the model to discover the
    optimal normalisation exponent for each channel during training.

    Args:
        in_dim:  Number of input features per node.
        out_dim: Number of output features per node.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add", node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_sp = nn.Linear(in_dim, out_dim, bias=False)
        self.W_ctx = nn.Linear(in_dim, out_dim, bias=False)
        self.W_lat = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Self-loop projection (residual connection inside the conv)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable degree-normalisation power per channel (one scalar each).
        # Initialised to -0.5 → symmetric D^{-0.5} A D^{-0.5} normalisation.
        self.deg_power = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        alpha_sp: torch.Tensor,
        alpha_ctx: torch.Tensor,
        alpha_lat: torch.Tensor,
        w_sp: torch.Tensor,
        w_ctx: torch.Tensor,
        w_lat: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features [num_nodes_total, in_dim].
            edge_index: COO edge index [2, num_edges].
            alpha_sp:   Per-edge spatial gating weight [num_edges].
            alpha_ctx:  Per-edge contextual gating weight [num_edges].
            alpha_lat:  Per-edge latent gating weight [num_edges].
            w_sp:       Per-edge spatial adjacency weight [num_edges].
            w_ctx:      Per-edge contextual adjacency weight [num_edges].
            w_lat:      Per-edge latent adjacency weight [num_edges].
            num_nodes:  Total node count (needed when batched).

        Returns:
            Updated node features [num_nodes_total, out_dim].
        """
        N = num_nodes or x.size(0)
        src, tgt = edge_index[0], edge_index[1]

        # Effective edge weights: gate × adjacency
        eff_sp = alpha_sp * w_sp
        eff_ctx = alpha_ctx * w_ctx
        eff_lat = alpha_lat * w_lat

        # Per-channel weighted degree (target side only; symmetric via message)
        deg_sp = torch.ones(N, device=x.device)
        deg_ctx = torch.ones(N, device=x.device)
        deg_lat = torch.ones(N, device=x.device)
        deg_sp.scatter_add_(0, tgt, eff_sp)
        deg_ctx.scatter_add_(0, tgt, eff_ctx)
        deg_lat.scatter_add_(0, tgt, eff_lat)

        deg_sp_isqrt = deg_sp.clamp(min=1e-6).pow(self.deg_power[0])
        deg_ctx_isqrt = deg_ctx.clamp(min=1e-6).pow(self.deg_power[1])
        deg_lat_isqrt = deg_lat.clamp(min=1e-6).pow(self.deg_power[2])

        out = self.propagate(
            edge_index,
            x=x,
            alpha_sp=alpha_sp, alpha_ctx=alpha_ctx, alpha_lat=alpha_lat,
            w_sp=w_sp, w_ctx=w_ctx, w_lat=w_lat,
            deg_sp_isqrt=deg_sp_isqrt,
            deg_ctx_isqrt=deg_ctx_isqrt,
            deg_lat_isqrt=deg_lat_isqrt,
        )

        out = out + self.self_loop(x)
        return out + self.bias

    def message(
        self,
        x_j: torch.Tensor,
        alpha_sp: torch.Tensor,
        alpha_ctx: torch.Tensor,
        alpha_lat: torch.Tensor,
        w_sp: torch.Tensor,
        w_ctx: torch.Tensor,
        w_lat: torch.Tensor,
        deg_sp_isqrt_i: torch.Tensor,
        deg_ctx_isqrt_i: torch.Tensor,
        deg_lat_isqrt_i: torch.Tensor,
        deg_sp_isqrt_j: torch.Tensor,
        deg_ctx_isqrt_j: torch.Tensor,
        deg_lat_isqrt_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted, disentangled message from neighbour j to node i."""
        # Symmetric degree normalisation: D_i^p * w_ij * D_j^p
        norm_sp = (alpha_sp * w_sp * deg_sp_isqrt_i * deg_sp_isqrt_j).unsqueeze(-1)
        norm_ctx = (alpha_ctx * w_ctx * deg_ctx_isqrt_i * deg_ctx_isqrt_j).unsqueeze(-1)
        norm_lat = (alpha_lat * w_lat * deg_lat_isqrt_i * deg_lat_isqrt_j).unsqueeze(-1)

        return (
            norm_sp * self.W_sp(x_j)
            + norm_ctx * self.W_ctx(x_j)
            + norm_lat * self.W_lat(x_j)
        )


class DisentangledGNN(nn.Module):
    """Multi-layer disentangled GNN.

    Stacks ``num_layers`` DisentangledConv layers with LayerNorm, ReLU, and
    dropout between intermediate layers (not after the final layer).

    Args:
        in_dim:     Input feature dimension.
        hidden_dim: Hidden dimension for intermediate layers.
        out_dim:    Output feature dimension.
        num_layers: Number of DisentangledConv layers (default 2).
        dropout:    Dropout probability between layers (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(DisentangledConv(in_ch, out_ch))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_ch))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        alpha_sp: torch.Tensor,
        alpha_ctx: torch.Tensor,
        alpha_lat: torch.Tensor,
        w_sp: torch.Tensor,
        w_ctx: torch.Tensor,
        w_lat: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features [num_nodes_total, in_dim].
            edge_index: COO edge index [2, num_edges].
            alpha_sp/ctx/lat: Per-edge gating weights [num_edges].
            w_sp/ctx/lat:     Per-edge adjacency weights [num_edges].
            num_nodes:  Total node count.

        Returns:
            Updated node features [num_nodes_total, out_dim].
        """
        for i, conv in enumerate(self.convs):
            x = conv(
                x, edge_index,
                alpha_sp, alpha_ctx, alpha_lat,
                w_sp, w_ctx, w_lat,
                num_nodes=num_nodes,
            )
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
