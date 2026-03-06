# -*- coding: utf-8 -*-
"""
CrimePredictionLoss — unified multi-objective loss for regression and
classification tasks.

Components
----------
1. Prediction loss
   • Regression:     Huber / Smooth-L1 / MSE (configurable)
   • Classification: Focal BCE (or plain BCE with optional pos_weight)
2. Predictive-uniqueness regularisation
   • Regression:     multiplicative form — penalises A_lat only where both
                     A_sp AND A_ctx already cover an edge (true double-
                     coverage).  Avoids penalising edges that complement only
                     one known graph.
   • Classification: additive form — penalises A_lat proportionally to the
                     overlap with either A_sp or A_ctx.
3. Sparsity regularisation  |mean(A_lat) − target|
4. Channel-orthogonality regularisation  Cov(w_sp, w_ctx) + Cov(w_sp, w_lat) + …
5. Floor regularisation
   • Regression:     sp + ctx only — the latent channel must earn its weight.
   • Classification: sp + ctx + lat.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrimePredictionLoss(nn.Module):
    """Multi-objective loss for interpretable crime prediction.

    Args:
        config: Global ``Config`` object.  Reads ``training`` and ``graph``
                sub-configs.
    """

    def __init__(self, config):
        super().__init__()
        tr = config.training
        self.task_type = getattr(config.data, "task_type", "regression")

        self.lambda_pred_unique = tr.lambda_pred_unique
        self.lambda_sparse = tr.lambda_sparse
        self.lambda_ortho = tr.lambda_ortho
        self.lambda_floor = tr.lambda_floor
        self.alpha_min = tr.alpha_min
        self.sparsity_target = config.graph.latent_sparsity_target

        # Regression-specific
        self.pred_loss_type = getattr(tr, "pred_loss", "huber")
        self.huber_delta = getattr(tr, "huber_delta", 1.0)

        # Classification-specific
        self.use_focal_loss = getattr(tr, "use_focal_loss", True)
        self.focal_gamma = getattr(tr, "focal_gamma", 2.0)
        self.pos_weight_val = getattr(tr, "pos_weight", 1.0)

    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        A_sp: torch.Tensor,
        A_ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            outputs: Model output dict (must contain ``logits``, ``A_lat``,
                     ``node_w_sp``, ``node_w_ctx``, ``node_w_lat``).
            targets: Ground-truth tensor, same shape as ``logits``.
            A_sp:    Spatial adjacency [N, N].
            A_ctx:   Contextual adjacency [N, N].

        Returns:
            Dict of scalar tensors with keys: total, pred, pred_unique,
            sparse, ortho, floor.
        """
        logits = outputs["logits"]
        A_lat = outputs["A_lat"]
        node_w_sp = outputs["node_w_sp"]
        node_w_ctx = outputs["node_w_ctx"]
        node_w_lat = outputs["node_w_lat"]

        # ------------------------------------------------------------------
        # 1. Prediction loss
        # ------------------------------------------------------------------
        if self.task_type == "regression":
            if self.pred_loss_type == "huber":
                l_pred = F.huber_loss(logits, targets, delta=self.huber_delta)
            elif self.pred_loss_type == "smooth_l1":
                l_pred = F.smooth_l1_loss(logits, targets)
            else:  # mse
                l_pred = F.mse_loss(logits, targets)
        else:
            # Classification
            if self.use_focal_loss:
                l_pred = self._focal_bce_loss(logits, targets)
            else:
                pos_weight = torch.tensor(
                    self.pos_weight_val, device=logits.device
                )
                l_pred = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight
                )

        # ------------------------------------------------------------------
        # 2. Predictive-uniqueness regularisation
        # ------------------------------------------------------------------
        A_sp_exp = A_sp.unsqueeze(0).expand_as(A_lat)
        A_ctx_exp = A_ctx.unsqueeze(0).expand_as(A_lat)

        if self.task_type == "regression":
            # Multiplicative: penalise only where BOTH A_sp AND A_ctx exist.
            # Does not punish latent edges complementing only one known graph.
            redundancy = A_sp_exp.clamp(max=1) * A_ctx_exp.clamp(max=1)
        else:
            # Additive average: proportional overlap with either known graph.
            redundancy = (A_sp_exp.clamp(max=1) + A_ctx_exp.clamp(max=1)) / 2.0

        l_pred_unique = (A_lat * redundancy).mean()

        # ------------------------------------------------------------------
        # 3. Sparsity regularisation
        # ------------------------------------------------------------------
        l_sparse = torch.abs(A_lat.mean() - self.sparsity_target)

        # ------------------------------------------------------------------
        # 4. Channel-orthogonality regularisation
        # ------------------------------------------------------------------
        l_ortho = self._channel_orthogonality_loss(node_w_sp, node_w_ctx, node_w_lat)

        # ------------------------------------------------------------------
        # 5. Floor regularisation
        # ------------------------------------------------------------------
        if self.task_type == "regression":
            # Regression: floor on sp + ctx only — the latent channel must
            # earn its weight; propping it up with regularisation would let
            # it cannibalism the informative channels with noise.
            l_floor = (
                F.relu(self.alpha_min - node_w_sp.mean())
                + F.relu(self.alpha_min - node_w_ctx.mean())
            )
        else:
            # Classification: floor on all three channels.
            l_floor = (
                F.relu(self.alpha_min - node_w_sp.mean())
                + F.relu(self.alpha_min - node_w_ctx.mean())
                + F.relu(self.alpha_min - node_w_lat.mean())
            )

        # ------------------------------------------------------------------
        # Total
        # ------------------------------------------------------------------
        l_total = (
            l_pred
            + self.lambda_pred_unique * l_pred_unique
            + self.lambda_sparse * l_sparse
            + self.lambda_ortho * l_ortho
            + self.lambda_floor * l_floor
        )

        return {
            "total": l_total,
            "pred": l_pred,
            "pred_unique": l_pred_unique,
            "sparse": l_sparse,
            "ortho": l_ortho,
            "floor": l_floor,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _focal_bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Focal binary-cross-entropy loss.

        FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t),   γ = focal_gamma.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce).mean()

    @staticmethod
    def _channel_orthogonality_loss(
        w_sp: torch.Tensor,
        w_ctx: torch.Tensor,
        w_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Decorrelate per-node channel weights across the batch.

        Minimising pairwise covariances pushes the three channels to
        contribute complementary information.
        """
        def cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            xc = x - x.mean(dim=-1, keepdim=True)
            yc = y - y.mean(dim=-1, keepdim=True)
            return (xc * yc).mean()

        return (
            torch.abs(cov(w_sp, w_ctx))
            + torch.abs(cov(w_sp, w_lat))
            + torch.abs(cov(w_ctx, w_lat))
        )
