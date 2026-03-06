# -*- coding: utf-8 -*-
"""
InterpretableSTGNN — unified model for both regression and classification tasks.

Regression-authoritative design:
  • latent_temporal_proj: A_lat is driven by crime temporal dynamics only,
    avoiding contextual leakage from the fused representation.
  • use_separate_gating_input: optional decoupled gating projection.
  • _latent_warmup_factor / _alpha_lat_cap: prevent latent from disrupting
    learned spatial+contextual representations when Phase 2 begins.
  • Multi-horizon prediction head (num_horizons > 1).
  • Classification uses sigmoid on top of the same pred_head.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch as PyGBatch

from crime_gnn.models.components import (
    TemporalEncoder,
    ResidualLatentGraphGenerator,
    LearnableGraphScaling,
    ScaleAwareGatingNetwork,
)
from crime_gnn.models.gnn import DisentangledGNN


# Minimum latent-edge weight to include in the union graph
LATENT_EDGE_THRESHOLD = 0.05


class InterpretableSTGNN(nn.Module):
    """Interpretable Spatial-Temporal GNN with late-fusion architecture.

    Pipeline
    --------
    1a. Crime encoder + temporal encoder → h_temporal     [B, N, D_temporal]
    1b. Context encoder                 → h_context       [B, N, D_ctx]
    1c. Late fusion: concat → project   → H               [B, N, hidden]
    2.  Latent graph generator           → A_lat           [B, N, N]
    3.  Per-edge gating                  → α_sp, α_ctx, α_lat
    4.  Build union-graph PyG Batch with per-edge channel weights
    5.  DisentangledGNN                  → Z               [B, N, hidden]
    6.  Residual Z = Z + H
    7.  Prediction head:
         • regression: raw output [B, N] or [B, H, N]
         • classification: sigmoid(output) [B, N]

    Args:
        config:  Global ``Config`` object.
        A_sp:    Spatial adjacency tensor [N, N].
        A_ctx:   Contextual adjacency tensor [N, N].
        X_ctx:   Static context feature matrix [N, C].
    """

    def __init__(
        self,
        config,
        A_sp: torch.Tensor,
        A_ctx: torch.Tensor,
        X_ctx: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        N = config.model.num_nodes
        self.num_nodes = N
        hidden = config.model.hidden_dim
        self.num_horizons = config.model.num_horizons
        self.task_type = getattr(config.data, "task_type", "regression")

        # Ablation flags
        self.disable_latent_graph = getattr(config.ablation, "disable_latent_graph", False)
        self.disable_gating = getattr(config.ablation, "disable_gating", False)

        self.register_buffer("A_sp", A_sp)
        self.register_buffer("A_ctx", A_ctx)
        self.register_buffer("X_ctx", X_ctx)

        # Fixed-graph union mask (A_sp ∪ A_ctx), reused every forward pass
        fixed_union = (A_sp > 1e-6) | (A_ctx > 1e-6)
        fixed_union.fill_diagonal_(False)
        self.register_buffer("fixed_union_mask", fixed_union)

        crime_dim = 1
        ctx_dim = config.model.num_contextual_features

        # ------------------------------------------------------------------
        # 1. Node encoding
        # ------------------------------------------------------------------
        temporal_input_dim = getattr(config.model, "crime_encoder_dim", hidden // 2)
        self.crime_encoder = nn.Sequential(
            nn.Linear(crime_dim, temporal_input_dim),
            nn.ReLU(),
            nn.Linear(temporal_input_dim, temporal_input_dim),
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=config.model.temporal_hidden_dim,
            num_layers=config.model.num_temporal_layers,
            encoder_type=config.model.temporal_encoder,
            dropout=config.model.dropout,
            use_temporal_attention=config.model.use_temporal_attention,
            num_attn_heads=config.model.num_attn_heads,
        )
        temporal_out_dim = config.model.temporal_hidden_dim

        ctx_hidden = getattr(config.model, "ctx_encoder_dim", hidden // 2)
        self.context_encoder = nn.Sequential(
            nn.Linear(ctx_dim, ctx_hidden),
            nn.ReLU(),
            nn.Linear(ctx_hidden, ctx_hidden),
        )
        context_out_dim = ctx_hidden

        # Late fusion
        fusion_input_dim = temporal_out_dim + context_out_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden),
            nn.ReLU(),
        )
        self.use_fusion_norm = getattr(config.model, "use_fusion_layernorm", True)
        if self.use_fusion_norm:
            self.fusion_norm = nn.LayerNorm(hidden)

        # ------------------------------------------------------------------
        # Regression-authoritative: dedicated temporal-only projection for
        # latent graph input.  Avoids contextual-feature leakage: the latent
        # graph discovers hidden crime-diffusion relationships from temporal
        # dynamics, not from socio-economic features already captured by A_ctx.
        # ------------------------------------------------------------------
        self.latent_temporal_proj = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden),
            nn.ReLU(),
        )

        # Optional separate gating input: [h_temporal ‖ h_context] → hidden.
        # Decouples the gating network's learned view from the main fusion.
        self.use_separate_gating_input = getattr(
            config.model, "use_separate_gating_input", False
        )
        if self.use_separate_gating_input:
            self.gating_input_proj = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden),
                nn.ReLU(),
            )

        # ------------------------------------------------------------------
        # 2. Latent graph generator
        # ------------------------------------------------------------------
        self.latent_graph_gen = ResidualLatentGraphGenerator(
            num_nodes=N,
            embedding_dim=config.graph.latent_embedding_dim,
            hidden_dim=hidden,
            topk_per_node=config.graph.latent_topk_per_node,
        )

        # ------------------------------------------------------------------
        # 3. Learnable graph scaling
        # ------------------------------------------------------------------
        self.use_scaling = config.model.use_learnable_graph_scaling
        if self.use_scaling:
            self.graph_scaling = LearnableGraphScaling(num_graphs=3)

        # ------------------------------------------------------------------
        # 4. Per-edge gating
        # ------------------------------------------------------------------
        self.gating = ScaleAwareGatingNetwork(
            hidden_dim=hidden,
            gate_hidden_dim=config.model.gate_hidden_dim,
            num_nodes=N,
        )
        self.gating.cache_edge_scales(A_sp, A_ctx)

        # ------------------------------------------------------------------
        # 5. Disentangled GNN
        # ------------------------------------------------------------------
        self.disentangled_gnn = DisentangledGNN(
            in_dim=hidden,
            hidden_dim=config.model.gnn_hidden_dim,
            out_dim=hidden,
            num_layers=config.model.num_gnn_layers,
            dropout=config.model.dropout,
        )

        # ------------------------------------------------------------------
        # 6. Prediction head
        # ------------------------------------------------------------------
        self.pred_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden // 2, self.num_horizons),
        )

        # ------------------------------------------------------------------
        # Training-phase state (managed by CurriculumTrainer)
        # ------------------------------------------------------------------
        self._training_phase = 2
        # Latent warm-up ramp: linearly ramped 0→1 by the trainer over
        # ``latent_warmup_epochs`` epochs at the start of Phase 2.
        self._latent_warmup_factor = 0.0
        # Hard cap on alpha_lat to prevent it cannibalising known channels.
        self._alpha_lat_cap = getattr(config.training, "alpha_lat_cap", 0.25)

        self._init_weights()

    # ------------------------------------------------------------------
    # Phase control (called by CurriculumTrainer)
    # ------------------------------------------------------------------

    def set_training_phase(self, phase: int):
        """Switch between Phase 1 (base graphs only) and Phase 2 (+ latent).

        In Phase 1 the latent graph generator is frozen and cleared.
        In Phase 2 it is unfrozen and free to accumulate residual signal.
        """
        if phase == 1:
            for p in self.latent_graph_gen.parameters():
                p.requires_grad = False
            self.latent_graph_gen.clear_residual_mask()
        elif phase == 2:
            for p in self.latent_graph_gen.parameters():
                p.requires_grad = True
        self._training_phase = phase

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _edge_to_node_weight(
        self,
        alpha: torch.Tensor,
        A_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average per-edge alpha over the masked edges for each node.

        Args:
            alpha:  [B, N, N] per-edge gating weights.
            A_mask: [N, N] or [B, N, N] boolean mask.

        Returns:
            [B, N] per-node mean weight.
        """
        if A_mask.dim() == 2:
            mask_f = A_mask.float().unsqueeze(0)
        else:
            mask_f = A_mask.float()
        weighted = (alpha * mask_f).sum(dim=-1)
        count = mask_f.sum(dim=-1).clamp(min=1.0)
        return weighted / count

    def _build_disentangled_batch(
        self,
        H: torch.Tensor,
        alpha_sp: torch.Tensor,
        alpha_ctx: torch.Tensor,
        alpha_lat: torch.Tensor,
        A_lat: torch.Tensor,
        B: int,
        N: int,
    ) -> PyGBatch:
        """Build a PyG Batch with per-edge channel attributes for DisentangledConv.

        Each item in the batch is one graph sample.  Edges from all three
        channels are unioned; per-edge channel weights are stored as edge
        attributes.
        """
        device = H.device

        if self.use_scaling:
            scales = self.graph_scaling.get_scales()
            w_sp_base = self.A_sp * scales[0]
            w_ctx_base = self.A_ctx * scales[1]
            lat_scale = scales[2]
        else:
            w_sp_base = self.A_sp
            w_ctx_base = self.A_ctx
            lat_scale = 1.0

        data_list = []
        for b in range(B):
            A_lat_b = A_lat[b] if A_lat.dim() == 3 else A_lat

            union_mask = self.fixed_union_mask | (A_lat_b > LATENT_EDGE_THRESHOLD)
            union_mask.fill_diagonal_(False)

            edge_index = union_mask.nonzero(as_tuple=False).T
            src, tgt = edge_index[0], edge_index[1]

            data_list.append(Data(
                x=H[b],
                edge_index=edge_index,
                e_alpha_sp=alpha_sp[b, src, tgt],
                e_alpha_ctx=alpha_ctx[b, src, tgt],
                e_alpha_lat=alpha_lat[b, src, tgt],
                e_w_sp=w_sp_base[src, tgt],
                e_w_ctx=w_ctx_base[src, tgt],
                e_w_lat=A_lat_b[src, tgt] * lat_scale,
                num_nodes=N,
            ))

        return PyGBatch.from_data_list(data_list)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys:
                ``crime_seq``    [B, T, N] — weekly crime counts.
                ``ctx_features`` [B, N, C] or [N, C] — static context.

        Returns:
            dict with predictions, adjacency matrices, gating weights, and
            node embeddings.
        """
        crime_seq = batch["crime_seq"]        # [B, T, N]
        ctx_features = batch["ctx_features"]  # [B, N, C] or [N, C]
        B, T, N = crime_seq.shape

        if ctx_features.dim() == 3:
            ctx_features = ctx_features[0]   # static: [N, C]

        # ------------------------------------------------------------------
        # 1. Late-fusion node encoding
        # ------------------------------------------------------------------
        crime_exp = crime_seq.unsqueeze(-1)                           # [B, T, N, 1]
        crime_encoded = self.crime_encoder(crime_exp)                 # [B, T, N, D_crime]
        crime_flat = crime_encoded.permute(0, 2, 1, 3) \
                                  .reshape(B * N, T, -1)              # [B*N, T, D_crime]

        _, h_temporal = self.temporal_encoder(crime_flat)             # [B*N, D_temporal]
        h_temporal = h_temporal.view(B, N, -1)                        # [B, N, D_temporal]

        h_context = self.context_encoder(ctx_features)                # [N, D_ctx]
        h_context = h_context.unsqueeze(0).expand(B, -1, -1)         # [B, N, D_ctx]

        H = self.fusion_proj(torch.cat([h_temporal, h_context], dim=-1))
        if self.use_fusion_norm:
            H = self.fusion_norm(H)

        # Dedicated temporal-only representation for latent graph generation.
        # Keeps the latent graph discovering crime-dynamic relationships,
        # not contextual-feature similarity already encoded by A_ctx.
        h_for_latent = self.latent_temporal_proj(h_temporal)          # [B, N, hidden]

        # Optional separate gating projection (decoupled from main fusion)
        if self.use_separate_gating_input:
            H_gate = self.gating_input_proj(
                torch.cat([h_temporal, h_context], dim=-1)
            )
        else:
            H_gate = H

        # ------------------------------------------------------------------
        # 2. Latent graph
        # ------------------------------------------------------------------
        if self._training_phase == 1 or self.disable_latent_graph:
            A_lat = torch.zeros(B, N, N, device=H.device)
            A_lat_static = torch.zeros(N, N, device=H.device)
            A_lat_dynamic = torch.zeros(B, N, N, device=H.device)
        else:
            A_lat, A_lat_static, A_lat_dynamic = self.latent_graph_gen(h_for_latent)

        # ------------------------------------------------------------------
        # 3. Per-edge gating
        # ------------------------------------------------------------------
        if self.disable_gating:
            sp_mask = (self.A_sp > 1e-6).float().unsqueeze(0).expand(B, -1, -1)
            ctx_mask = (self.A_ctx > 1e-6).float().unsqueeze(0).expand(B, -1, -1)
            if self.disable_latent_graph:
                active = (sp_mask + ctx_mask).clamp(min=1e-8)
                alpha_sp = sp_mask / active
                alpha_ctx = ctx_mask / active
                alpha_lat = torch.zeros(B, N, N, device=H.device)
            else:
                lat_mask = (A_lat > LATENT_EDGE_THRESHOLD).float()
                active = (sp_mask + ctx_mask + lat_mask).clamp(min=1e-8)
                alpha_sp = sp_mask / active
                alpha_ctx = ctx_mask / active
                alpha_lat = lat_mask / active
        else:
            alpha_sp, alpha_ctx, alpha_lat = self.gating(
                H_gate, self.A_sp, self.A_ctx, A_lat
            )

        # ------------------------------------------------------------------
        # Latent warm-up ramp: scale α_lat by [0, 1] at Phase 2 start.
        # Prevents the latent channel from disrupting learned representations
        # at the phase transition.
        # ------------------------------------------------------------------
        if self._latent_warmup_factor < 1.0 and not self.disable_latent_graph:
            alpha_lat = alpha_lat * self._latent_warmup_factor
            alpha_sum = (alpha_sp + alpha_ctx + alpha_lat).clamp(min=1e-8)
            alpha_sp = alpha_sp / alpha_sum
            alpha_ctx = alpha_ctx / alpha_sum
            alpha_lat = alpha_lat / alpha_sum

        # Hard cap on α_lat — prevents cannibalisation of known channels.
        if self._alpha_lat_cap < 1.0 and not self.disable_latent_graph:
            alpha_lat = alpha_lat.clamp(max=self._alpha_lat_cap)
            alpha_sum = (alpha_sp + alpha_ctx + alpha_lat).clamp(min=1e-8)
            alpha_sp = alpha_sp / alpha_sum
            alpha_ctx = alpha_ctx / alpha_sum
            alpha_lat = alpha_lat / alpha_sum

        # In Phase 1 zero out latent and re-normalise sp + ctx to sum to 1
        if self._training_phase == 1:
            alpha_sum = alpha_sp + alpha_ctx + 1e-8
            alpha_sp = alpha_sp / alpha_sum
            alpha_ctx = alpha_ctx / alpha_sum
            alpha_lat = torch.zeros_like(alpha_lat)

        # ------------------------------------------------------------------
        # 4. Build union-graph batch and run DisentangledGNN
        # ------------------------------------------------------------------
        pyg_batch = self._build_disentangled_batch(
            H, alpha_sp, alpha_ctx, alpha_lat, A_lat, B, N,
        )
        Z_flat = self.disentangled_gnn(
            pyg_batch.x,
            pyg_batch.edge_index,
            pyg_batch.e_alpha_sp,
            pyg_batch.e_alpha_ctx,
            pyg_batch.e_alpha_lat,
            pyg_batch.e_w_sp,
            pyg_batch.e_w_ctx,
            pyg_batch.e_w_lat,
            num_nodes=pyg_batch.num_nodes,
        )
        Z = Z_flat.view(B, N, -1)

        # ------------------------------------------------------------------
        # 5. Residual connection
        # ------------------------------------------------------------------
        Z = Z + H

        # ------------------------------------------------------------------
        # 6. Prediction head
        # ------------------------------------------------------------------
        raw = self.pred_head(Z)   # [B, N, num_horizons]

        if self.task_type == "classification":
            # Classification: apply sigmoid and squeeze horizon dim
            logits = raw.squeeze(-1)           # [B, N]
            predictions = torch.sigmoid(logits)
        else:
            # Regression: no activation
            if self.num_horizons == 1:
                logits = raw.squeeze(-1)       # [B, N]
            else:
                logits = raw.permute(0, 2, 1)  # [B, H, N]
            predictions = logits

        # ------------------------------------------------------------------
        # 7. Compute per-node attribution weights for loss and monitoring
        # ------------------------------------------------------------------
        sp_mask = (self.A_sp > 1e-6)
        ctx_mask = (self.A_ctx > 1e-6)
        if A_lat.dim() == 3:
            lat_mask = A_lat.detach() > LATENT_EDGE_THRESHOLD
        else:
            lat_mask = A_lat.detach() > LATENT_EDGE_THRESHOLD

        w_sp = self._edge_to_node_weight(alpha_sp, sp_mask)
        w_ctx = self._edge_to_node_weight(alpha_ctx, ctx_mask)
        w_lat = self._edge_to_node_weight(alpha_lat, lat_mask)
        total = (w_sp + w_ctx + w_lat).clamp(min=1e-8)
        w_sp = w_sp / total
        w_ctx = w_ctx / total
        w_lat = w_lat / total

        # Union-mask weights (for monitoring / disentanglement report)
        if A_lat.dim() == 3:
            union_mask = (
                sp_mask.unsqueeze(0)
                | ctx_mask.unsqueeze(0)
                | (A_lat.detach() > LATENT_EDGE_THRESHOLD)
            )
        else:
            union_mask = sp_mask | ctx_mask | (A_lat.detach() > LATENT_EDGE_THRESHOLD)

        mon_sp = self._edge_to_node_weight(alpha_sp, union_mask)
        mon_ctx = self._edge_to_node_weight(alpha_ctx, union_mask)
        mon_lat = self._edge_to_node_weight(alpha_lat, union_mask)
        mon_total = (mon_sp + mon_ctx + mon_lat).clamp(min=1e-8)
        mon_sp = mon_sp / mon_total
        mon_ctx = mon_ctx / mon_total
        mon_lat = mon_lat / mon_total

        return {
            "logits": logits,
            "predictions": predictions,
            "A_lat": A_lat,
            "A_lat_static": A_lat_static,
            "A_lat_dynamic": A_lat_dynamic,
            "alpha_sp": alpha_sp,
            "alpha_ctx": alpha_ctx,
            "alpha_lat": alpha_lat,
            "node_w_sp": w_sp,
            "node_w_ctx": w_ctx,
            "node_w_lat": w_lat,
            "mon_w_sp": mon_sp,
            "mon_w_ctx": mon_ctx,
            "mon_w_lat": mon_lat,
            "node_embeddings": Z,
            "H": H,
            "h_temporal": h_temporal,
        }

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def get_static_latent_graph(self) -> torch.Tensor:
        """Return the learned static latent graph with top-k applied."""
        with torch.no_grad():
            tau = self.latent_graph_gen.temperature.clamp(min=0.1)
            A = torch.sigmoid(
                self.latent_graph_gen.E1 @ self.latent_graph_gen.E2.t() / tau
            )
            A = A * (1 - torch.eye(self.num_nodes, device=A.device))
            A = self.latent_graph_gen._apply_topk(A)
        return A

    def compute_residual_scores(
        self,
        dataloader,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Compute per-node-pair residual error scores under Phase 1.

        These scores guide the residual mask in Phase 2: node pairs with
        high Phase-1 prediction error are prioritised for latent edges.

        Returns:
            scores [N, N]: symmetric residual error matrix.
        """
        was_training = self.training
        self.eval()
        old_phase = self._training_phase
        self.set_training_phase(1)

        node_errors = torch.zeros(self.num_nodes, device=device)
        node_counts = torch.zeros(self.num_nodes, device=device)

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = self.forward(batch)
                preds = out["predictions"]
                target = batch["target"]
                # For multi-horizon: average error across horizons per node
                if preds.dim() == 3:
                    err = torch.abs(preds - target).mean(dim=1)   # [B, N]
                else:
                    err = torch.abs(preds - target)               # [B, N]
                node_errors += err.sum(dim=0)
                node_counts += err.size(0)

        node_errors = node_errors / node_counts.clamp(min=1)
        scores = node_errors.unsqueeze(0) + node_errors.unsqueeze(1)

        self.set_training_phase(old_phase)
        if was_training:
            self.train()
        return scores
