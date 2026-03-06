# -*- coding: utf-8 -*-
"""
Reusable model components:
  - TemporalEncoder
  - ResidualLatentGraphGenerator
  - LearnableGraphScaling
  - ScaleAwareGatingNetwork
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temporal Encoder
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Per-node temporal encoder with optional temporal self-attention.

    Input:  [B*N, T, D_in]
    Output: (all_outputs [B*N, T, D_hidden], final_hidden [B*N, D_hidden])

    When ``use_temporal_attention=True``, the last timestep's hidden state
    queries all timestep outputs via multi-head attention, producing a richer
    summary that captures *which weeks* mattered most.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        encoder_type: str = "GRU",
        dropout: float = 0.1,
        use_temporal_attention: bool = False,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.use_temporal_attention = use_temporal_attention
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if encoder_type == "GRU":
            self.rnn = nn.GRU(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "LSTM":
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "Transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True,
            )
            self.rnn = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type!r}")

        if use_temporal_attention:
            self.attn_query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.temporal_attn = nn.MultiheadAttention(
                hidden_dim, num_attn_heads,
                dropout=dropout, batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)

        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.input_proj(x)

        if self.encoder_type == "Transformer":
            out = self.rnn(x)
            h_final = out[:, -1, :]
        else:
            out, _ = self.rnn(x, hidden)
            h_final = out[:, -1, :]

        if self.use_temporal_attention:
            query = self.attn_query_proj(h_final).unsqueeze(1)  # [B*N, 1, H]
            attn_out, self._attn_weights = self.temporal_attn(query, out, out)
            h_final = self.attn_norm(query + attn_out).squeeze(1)

        return out, h_final


# ---------------------------------------------------------------------------
# Residual Latent Graph Generator
# ---------------------------------------------------------------------------

class ResidualLatentGraphGenerator(nn.Module):
    """Generates A_lat capturing relationships NOT explained by A_sp + A_ctx.

    A_lat = clamp(A_static + A_dynamic, 0, 1)

    where:
      A_static  = σ(E₁ · E₂ᵀ / τ)       — time-invariant node-pair affinities
      A_dynamic = σ(proj(H) · proj(H)ᵀ / τ) — driven by current node states H

    Top-k masking uses a correct straight-through estimator so that gradients
    flow to all entries while only the top-k contribute in the forward pass.

    A residual mask (derived from Phase-1 prediction errors) focuses latent
    edges on high-residual pairs not already covered by A_sp or A_ctx.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        topk_per_node: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.topk_per_node = topk_per_node

        self.E1 = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.1)

        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_mask: Optional[torch.Tensor] = None

    def set_residual_mask(
        self,
        residual_scores: torch.Tensor,
        A_sp: Optional[torch.Tensor] = None,
        A_ctx: Optional[torch.Tensor] = None,
        sparsity_target: float = 0.15,
        temp: float = 1.0,
    ):
        """Focus latent edges on high-residual node pairs not covered by known
        graphs.

        Uses a quantile-based soft threshold so that only the top
        ``sparsity_target`` fraction of high-residual pairs are active.
        """
        threshold = torch.quantile(residual_scores.flatten(), 1.0 - sparsity_target)
        mask = torch.sigmoid((residual_scores - threshold) / max(temp, 0.1))
        mask = mask * (1 - torch.eye(self.num_nodes, device=mask.device))

        if A_sp is not None and A_ctx is not None:
            covered = ((A_sp > 0.1) | (A_ctx > 0.1)).float()
            mask = mask * (1.0 - 0.5 * covered)

        self.residual_mask = mask.detach()

    def clear_residual_mask(self):
        self.residual_mask = None

    def _apply_topk(self, A: torch.Tensor) -> torch.Tensor:
        """Keep only top-k edges per row with a straight-through estimator."""
        k = min(self.topk_per_node, A.size(-1) - 1)
        _, topk_idx = A.topk(k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, topk_idx, 1.0)
        # STE: forward = masked, backward = identity
        return A + (A * mask - A).detach()

    def forward(
        self,
        H: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            H: node embeddings [B, N, hidden_dim] for the dynamic component.
               Pass None to use only the static component.

        Returns:
            A_lat:     combined latent graph [B, N, N]
            A_static:  static component      [N, N]
            A_dynamic: dynamic component     [B, N, N]
        """
        tau = self.temperature.clamp(min=0.1)

        A_static = torch.sigmoid(self.E1 @ self.E2.t() / tau)
        A_static = A_static * (1 - torch.eye(self.num_nodes, device=A_static.device))

        if H is None:
            A_lat = A_static
            if self.residual_mask is not None:
                A_lat = A_lat * self.residual_mask.to(A_lat.device)
            A_lat = self._apply_topk(A_lat)
            return A_lat, A_static, torch.zeros_like(A_static)

        H_proj = self.dynamic_proj(H)                          # [B, N, emb_dim]
        A_dynamic = torch.bmm(H_proj, H_proj.transpose(1, 2))
        A_dynamic = torch.sigmoid(A_dynamic / tau)
        eye = torch.eye(self.num_nodes, device=A_dynamic.device).unsqueeze(0)
        A_dynamic = A_dynamic * (1 - eye)

        A_lat = torch.clamp(A_static.unsqueeze(0) + A_dynamic, 0, 1)

        if self.residual_mask is not None:
            A_lat = A_lat * self.residual_mask.unsqueeze(0).to(A_lat.device)

        A_lat = self._apply_topk(A_lat)

        return A_lat, A_static, A_dynamic


# ---------------------------------------------------------------------------
# Learnable Graph Scaling
# ---------------------------------------------------------------------------

class LearnableGraphScaling(nn.Module):
    """Per-graph multiplicative scaling: A' = exp(log_s) · A."""

    def __init__(self, num_graphs: int = 3):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_graphs))

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self.log_scale)


# ---------------------------------------------------------------------------
# Scale-Aware Gating Network
# ---------------------------------------------------------------------------

class ScaleAwareGatingNetwork(nn.Module):
    """Per-edge softmax gating: α_sp, α_ctx, α_lat.

    Produces per-edge gating weights for interpretability and for direct use
    in the disentangled message-passing layers.

    A MASK_BIAS is applied to all channels where adjacency is zero, so edges
    with no graph support produce near-zero alpha rather than a misleading
    uniform 1/3.

    The latent channel is initialised with a negative bias so that it must
    earn its weight through predictive value before it receives attention.
    """

    MASK_BIAS = -10.0

    def __init__(
        self,
        hidden_dim: int,
        gate_hidden_dim: int,
        num_nodes: int,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        self.node_transform = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
        )

        # Edge feature: [h_i ‖ h_j ‖ a_sp_ij ‖ a_ctx_ij ‖ a_lat_ij]
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * gate_hidden_dim + 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 3),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._sp_scale: float = 1.0
        self._ctx_scale: float = 1.0
        self._sp_mask: Optional[torch.Tensor] = None
        self._ctx_mask: Optional[torch.Tensor] = None

        self._init_balanced()

    def _init_balanced(self):
        """Initialise: spatial & contextual channels start equal;
        latent channel starts near-zero (bias = -3 → softmax ≈ 5%)."""
        with torch.no_grad():
            last = self.edge_gate[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias[2] = -3.0   # push latent logit down

    def cache_edge_scales(self, A_sp: torch.Tensor, A_ctx: torch.Tensor):
        """Precompute and cache normalisation scales and zero-adjacency masks."""
        self._sp_scale = (float(A_sp[A_sp > 0].mean())
                          if (A_sp > 0).any() else 1.0)
        self._ctx_scale = (float(A_ctx[A_ctx > 0].mean())
                           if (A_ctx > 0).any() else 1.0)
        self._sp_mask = (A_sp > 0)
        self._ctx_mask = (A_ctx > 0)

    def forward(
        self,
        H: torch.Tensor,
        A_sp: torch.Tensor,
        A_ctx: torch.Tensor,
        A_lat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = H.size(0)
        N = self.num_nodes
        device = H.device
        temp = self.temperature.clamp(min=0.3)

        H_t = self.node_transform(H)                            # [B, N, g_hid]
        H_i = H_t.unsqueeze(2).expand(-1, -1, N, -1)
        H_j = H_t.unsqueeze(1).expand(-1, N, -1, -1)

        a_sp_n = (A_sp / self._sp_scale).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
        a_ctx_n = (A_ctx / self._ctx_scale).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)

        if A_lat is not None and A_lat.abs().sum() > 0:
            with torch.no_grad():
                lat_scale = (float(A_lat.detach()[A_lat.detach() > 0].mean())
                             if (A_lat > 0).any() else 1.0)
            a_lat_n = A_lat / max(lat_scale, 1e-6)
            if a_lat_n.dim() == 2:
                a_lat_n = a_lat_n.unsqueeze(0).expand(B, -1, -1)
            a_lat_n = a_lat_n.unsqueeze(-1)                     # [B, N, N, 1]
        else:
            a_lat_n = torch.zeros(B, N, N, 1, device=device)

        edge_feat = torch.cat([H_i, H_j, a_sp_n, a_ctx_n, a_lat_n], dim=-1)
        logits = self.edge_gate(edge_feat) / temp               # [B, N, N, 3]

        # Mask-bias: penalise channels with no adjacency
        if self._sp_mask is not None:
            sp_off = (~self._sp_mask.to(device)).float().unsqueeze(0).unsqueeze(-1)
            logits[..., 0:1] = logits[..., 0:1] + sp_off * self.MASK_BIAS
        if self._ctx_mask is not None:
            ctx_off = (~self._ctx_mask.to(device)).float().unsqueeze(0).unsqueeze(-1)
            logits[..., 1:2] = logits[..., 1:2] + ctx_off * self.MASK_BIAS

        if A_lat is not None:
            if A_lat.dim() == 3:
                lat_off = (A_lat < 1e-6).float().unsqueeze(-1)
            else:
                lat_off = (A_lat < 1e-6).float().unsqueeze(0).unsqueeze(-1)
            logits[..., 2:3] = logits[..., 2:3] + lat_off * self.MASK_BIAS

        gates = F.softmax(logits, dim=-1)
        return gates[..., 0], gates[..., 1], gates[..., 2]
