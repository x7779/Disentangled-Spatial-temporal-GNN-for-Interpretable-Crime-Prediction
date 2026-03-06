# -*- coding: utf-8 -*-
"""
CurriculumTrainer — two-phase curriculum training for the InterpretableSTGNN.

Phase 1  Train base model (spatial + contextual graphs, latent frozen).
Phase 2  Introduce residual-guided latent graph; periodic mask refresh.

Regression-authoritative design
--------------------------------
• Separate param-group for latent-graph parameters at ``latent_lr_multiplier``
  of the base learning rate.  This slows down latent-graph adaptation so it
  cannot outrun the spatial/contextual channels.
• Phase-2 readiness guard: requires val R² ≥ ``phase2_min_r2`` before
  transitioning, preventing the latent graph from modelling residual noise
  when the base model is still weak.
• Inverse-transform of predictions/targets to original count scale before
  computing MAE / RMSE / R² / MAPE metrics.
• Composite early-stopping metric: w·MAE + (1−w)·RMSE.
• Latent warm-up ramp: ``_latent_warmup_factor`` linearly 0→1 over
  ``latent_warmup_epochs`` epochs at Phase 2 start.

Classification specifics (when config.data.task_type == "classification")
• Metrics: AUC-ROC, F1, precision, recall.
• Early-stopping metric: auc_roc (max mode).
• No inverse transform.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


class CurriculumTrainer:
    """Two-phase curriculum trainer for the InterpretableSTGNN.

    Args:
        model:        InterpretableSTGNN instance.
        loss_fn:      CrimePredictionLoss instance.
        config:       Global Config object.
        A_sp:         Spatial adjacency tensor (device-agnostic; moved internally).
        A_ctx:        Contextual adjacency tensor.
        dataloaders:  Dict with keys ``"train"``, ``"val"``, ``"test"``.
        target_info:  Dict returned by ``prepare_data``; used for inverse transform.
        output_dir:   Directory for checkpoints and reports.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config,
        A_sp: torch.Tensor,
        A_ctx: torch.Tensor,
        dataloaders: Dict[str, DataLoader],
        target_info: Dict,
        output_dir: str = "./output",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = config.training.device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.task_type = getattr(config.data, "task_type", "regression")
        self.A_sp = A_sp.to(self.device)
        self.A_ctx = A_ctx.to(self.device)
        self.dataloaders = dataloaders
        self.target_info = target_info

        # ------------------------------------------------------------------
        # Separate param groups: latent parameters get a reduced LR so they
        # cannot cannibalise the spatial/contextual channels.
        # ------------------------------------------------------------------
        latent_ids = set(id(p) for p in model.latent_graph_gen.parameters())
        base_params = [p for p in model.parameters() if id(p) not in latent_ids]
        latent_params = list(model.latent_graph_gen.parameters())
        lat_lr_mult = getattr(config.training, "latent_lr_multiplier", 0.3)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": base_params},
                {"params": latent_params,
                 "lr": config.training.learning_rate * lat_lr_mult},
            ],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = None

        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [], "val_metrics": [],
            "alpha_sp": [], "alpha_ctx": [], "alpha_lat": [],
            "phase": [],
        }

        self.best_metric = (
            -float("inf")
            if config.training.early_stopping_mode == "max"
            else float("inf")
        )
        self.best_epoch = 0
        self.patience_counter = 0
        self._current_phase = 0
        self._global_best_metric = self.best_metric
        self._global_best_epoch = 0

    # ------------------------------------------------------------------
    # Scheduler management
    # ------------------------------------------------------------------

    def _create_scheduler(self, num_epochs: int, start_lr: Optional[float] = None):
        cfg = self.config.training
        lr = start_lr if start_lr is not None else cfg.learning_rate
        lat_lr_mult = getattr(cfg, "latent_lr_multiplier", 0.3)

        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = lr * lat_lr_mult if i == len(self.optimizer.param_groups) - 1 else lr

        if cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(num_epochs, 1), eta_min=cfg.min_lr,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.early_stopping_mode,
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
                min_lr=cfg.min_lr,
            )

    def _reset_optimizer_momentum(self):
        """Zero optimizer state for latent-graph params at phase transition."""
        lat_ids = set(id(p) for p in self.model.latent_graph_gen.parameters())
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in lat_ids:
                    state = self.optimizer.state.get(p, {})
                    for key in list(state.keys()):
                        if isinstance(state[key], torch.Tensor):
                            state[key].zero_()

    # ------------------------------------------------------------------
    # Target inverse-transform (regression only)
    # ------------------------------------------------------------------

    def _inverse_transform_targets(self, values: np.ndarray) -> np.ndarray:
        """Inverse-transform model outputs back to original crime-count scale."""
        ti = self.target_info
        out = values.copy()
        if "zscore" in ti.get("transform", ""):
            out = out * ti["target_std"] + ti["target_mean"]
        if ti.get("applied_log1p", False):
            out = np.expm1(out)
        return np.clip(out, 0, None)

    # ------------------------------------------------------------------
    # Phase-2 readiness guard (regression)
    # ------------------------------------------------------------------

    def _check_phase2_readiness(self) -> bool:
        """Return True iff the base model is strong enough for Phase 2."""
        min_r2 = self.config.training.phase2_min_r2
        if min_r2 <= 0:
            return True
        _, metrics, _, _ = self._validate(split="val")
        r2 = metrics.get("r2", -1.0)
        if r2 < min_r2:
            print(f"  Phase 2 guard: val R²={r2:.4f} < threshold {min_r2:.4f}")
            print(f"  Staying in Phase 1 (latent graph would model noise)")
            return False
        print(f"  Phase 2 guard: val R²={r2:.4f} >= threshold {min_r2:.4f} — proceeding")
        return True

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Tuple[Dict, Dict]:
        """Run full curriculum training and evaluate on test split.

        Returns:
            (history dict, test_results dict)
        """
        cfg = self.config.training
        ablation = self.config.ablation
        self.model.to(self.device)

        p1 = cfg.phase1_epochs if cfg.use_curriculum else 0
        p2 = cfg.phase2_epochs
        total_epochs = p1 + p2

        print("=" * 70)
        print("Curriculum Training Plan")
        print("=" * 70)
        if ablation.disable_latent_graph or ablation.disable_gating:
            print("  *** ABLATION MODE ***")
            if ablation.disable_latent_graph:
                print("    - Latent graph: DISABLED")
            if ablation.disable_gating:
                print("    - Learned gating: DISABLED (uniform channel weights)")
        if cfg.use_curriculum:
            print(f"  Phase 1 (base only):       epochs 1–{p1}")
            print(f"  Phase 2 (residual A_lat):  epochs {p1 + 1}–{total_epochs}")
        else:
            print(f"  Single-phase training: {total_epochs} epochs")
        print("=" * 70)

        self._create_scheduler(p1 if cfg.use_curriculum else p2)

        skip_latent_setup = ablation.disable_latent_graph

        for epoch in range(1, total_epochs + 1):
            # ----------------------------------------------------------
            # Determine phase
            # ----------------------------------------------------------
            if cfg.use_curriculum:
                phase = 1 if epoch <= p1 else 2
                if phase == 2 and not skip_latent_setup:
                    if epoch == p1 + 1:
                        if not self._check_phase2_readiness():
                            skip_latent_setup = True
                        else:
                            print("\n--- Phase 2: Computing residual guidance ---")
                            scores = self.model.compute_residual_scores(
                                self.dataloaders["train"], self.device
                            )
                            self.model.latent_graph_gen.set_residual_mask(
                                scores, A_sp=self.A_sp, A_ctx=self.A_ctx,
                            )
                            self._log_residual_mask_stats()
                    elif (epoch - p1) % cfg.residual_mask_refresh_interval == 0:
                        print(f"\n--- Phase 2: Refreshing residual mask (epoch {epoch}) ---")
                        scores = self.model.compute_residual_scores(
                            self.dataloaders["train"], self.device
                        )
                        self.model.latent_graph_gen.set_residual_mask(
                            scores, A_sp=self.A_sp, A_ctx=self.A_ctx,
                        )
                        self._log_residual_mask_stats()
            else:
                phase = 2

            # ----------------------------------------------------------
            # Phase transition bookkeeping
            # ----------------------------------------------------------
            if phase != self._current_phase:
                self._current_phase = phase
                self.patience_counter = 0
                self.best_metric = (
                    -float("inf")
                    if cfg.early_stopping_mode == "max"
                    else float("inf")
                )
                if phase == 1:
                    self._create_scheduler(p1)
                elif phase == 2:
                    self._reset_optimizer_momentum()
                    self._create_scheduler(p2)

            self.model.set_training_phase(phase)

            # Latent warm-up ramp
            if phase == 2 and cfg.use_curriculum and not skip_latent_setup:
                warmup = getattr(cfg, "latent_warmup_epochs", 15)
                ramp = min(1.0, (epoch - p1) / max(warmup, 1))
                self.model._latent_warmup_factor = ramp
            else:
                self.model._latent_warmup_factor = 1.0 if phase == 2 else 0.0

            # ----------------------------------------------------------
            # Train / validate
            # ----------------------------------------------------------
            train_losses = self._train_epoch()
            val_losses, val_metrics, alpha_stats, pred_stats = self._validate()

            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_losses["total"])
            self.history["val_metrics"].append(val_metrics)
            self.history["alpha_sp"].append(alpha_stats["alpha_sp"])
            self.history["alpha_ctx"].append(alpha_stats["alpha_ctx"])
            self.history["alpha_lat"].append(alpha_stats["alpha_lat"])
            self.history["phase"].append(phase)

            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(
                    val_metrics.get(cfg.early_stopping_metric, val_losses["total"])
                )
            else:
                self.scheduler.step()

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if epoch % cfg.log_interval == 0 or epoch <= 3:
                lr = self.optimizer.param_groups[0]["lr"]
                ramp = getattr(self.model, "_latent_warmup_factor", 1.0)
                if self.task_type == "regression":
                    mae = val_metrics.get("mae", float("nan"))
                    rmse = val_metrics.get("rmse", float("nan"))
                    comp = val_metrics.get("composite_mae_rmse", float("nan"))
                    print(
                        f"Epoch {epoch:3d} [P{phase}] | "
                        f"Train {train_losses['total']:.4f} | "
                        f"Val {val_losses['total']:.4f} | "
                        f"MAE {mae:.4f} | RMSE {rmse:.4f} | "
                        f"Comp {comp:.4f} | "
                        f"w_sp={alpha_stats['alpha_sp']:.3f} "
                        f"w_ctx={alpha_stats['alpha_ctx']:.3f} "
                        f"w_lat={alpha_stats['alpha_lat']:.3f} | "
                        f"pred[{pred_stats['min']:.3f},{pred_stats['mean']:.3f},"
                        f"{pred_stats['max']:.3f}] "
                        f"std={pred_stats['std']:.4f} | "
                        f"ramp={ramp:.2f} | lr={lr:.2e}"
                    )
                else:
                    auc = val_metrics.get("auc_roc", float("nan"))
                    f1 = val_metrics.get("f1", float("nan"))
                    print(
                        f"Epoch {epoch:3d} [P{phase}] | "
                        f"Train {train_losses['total']:.4f} | "
                        f"Val {val_losses['total']:.4f} | "
                        f"AUC {auc:.4f} | F1 {f1:.4f} | "
                        f"w_sp={alpha_stats['alpha_sp']:.3f} "
                        f"w_ctx={alpha_stats['alpha_ctx']:.3f} "
                        f"w_lat={alpha_stats['alpha_lat']:.3f} | "
                        f"ramp={ramp:.2f} | lr={lr:.2e}"
                    )

            # ----------------------------------------------------------
            # Early stopping
            # ----------------------------------------------------------
            metric_val = val_metrics.get(
                cfg.early_stopping_metric, -val_losses["total"]
            )
            improved = (
                metric_val > self.best_metric
                if cfg.early_stopping_mode == "max"
                else metric_val < self.best_metric
            )
            if improved:
                self.best_metric = metric_val
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            global_improved = (
                metric_val > self._global_best_metric
                if cfg.early_stopping_mode == "max"
                else metric_val < self._global_best_metric
            )
            if global_improved:
                self._global_best_metric = metric_val
                self._global_best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            if self.patience_counter >= cfg.early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(phase best: epoch {self.best_epoch}, "
                    f"global best: epoch {self._global_best_epoch}, "
                    f"{cfg.early_stopping_metric}={self._global_best_metric:.4f})"
                )
                break

            if epoch % cfg.save_interval == 0:
                self._save_checkpoint(epoch)

        # ------------------------------------------------------------------
        # Final evaluation on test split
        # ------------------------------------------------------------------
        self._load_best_checkpoint()
        test_losses, test_metrics, test_alphas, test_pred = self._validate(split="test")

        print("\n" + "=" * 70)
        print("Test Results")
        print("=" * 70)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(
            f"  Prediction stats: min={test_pred['min']:.4f}, "
            f"max={test_pred['max']:.4f}, mean={test_pred['mean']:.4f}, "
            f"std={test_pred['std']:.4f}"
        )
        print(
            f"  Node channel weights: sp={test_alphas['alpha_sp']:.3f}, "
            f"ctx={test_alphas['alpha_ctx']:.3f}, "
            f"lat={test_alphas['alpha_lat']:.3f}"
        )

        return self.history, {**test_metrics, **test_alphas}

    # ------------------------------------------------------------------
    # Internal epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        accum: Dict[str, float] = {}
        n = 0
        for batch in self.dataloaders["train"]:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch["target"], self.A_sp, self.A_ctx)

            if torch.isnan(losses["total"]):
                continue

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.max_grad_norm
            )
            self.optimizer.step()

            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + v.item()
            n += 1

        return {k: v / max(n, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self, split: str = "val"):
        self.model.eval()
        accum: Dict[str, float] = {}
        all_preds, all_targets = [], []
        w_sp_sum = w_ctx_sum = w_lat_sum = 0.0
        total_samples = 0

        for batch in self.dataloaders[split]:
            batch = self._to_device(batch)
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch["target"], self.A_sp, self.A_ctx)

            bs = outputs["predictions"].size(0)
            for k, v in losses.items():
                val = v.item()
                if not np.isnan(val):
                    accum[k] = accum.get(k, 0.0) + val * bs

            all_preds.append(outputs["predictions"].cpu())
            all_targets.append(batch["target"].cpu())
            w_sp_sum += outputs["mon_w_sp"].mean().item() * bs
            w_ctx_sum += outputs["mon_w_ctx"].mean().item() * bs
            w_lat_sum += outputs["mon_w_lat"].mean().item() * bs
            total_samples += bs

        avg_loss = {k: v / max(total_samples, 1) for k, v in accum.items()}

        preds_raw = torch.cat(all_preds).numpy()
        targets_raw = torch.cat(all_targets).numpy()
        preds_flat = preds_raw.reshape(-1)
        targets_flat = targets_raw.reshape(-1)

        pred_stats = {
            "min": float(preds_flat.min()),
            "max": float(preds_flat.max()),
            "mean": float(preds_flat.mean()),
            "std": float(preds_flat.std()),
        }

        if self.task_type == "regression":
            preds_orig = self._inverse_transform_targets(preds_flat)
            targets_orig = self._inverse_transform_targets(targets_flat)
            metrics = self._compute_regression_metrics(preds_orig, targets_orig)

            # Per-horizon metrics
            if self.config.model.num_horizons > 1:
                horizons = self.config.data.multi_horizon
                for h_idx, h in enumerate(horizons):
                    p_h = self._inverse_transform_targets(
                        preds_raw[:, h_idx, :].reshape(-1)
                    )
                    t_h = self._inverse_transform_targets(
                        targets_raw[:, h_idx, :].reshape(-1)
                    )
                    metrics[f"mae_h{h}w"] = float(mean_absolute_error(t_h, p_h))
                    metrics[f"rmse_h{h}w"] = float(
                        np.sqrt(mean_squared_error(t_h, p_h))
                    )
        else:
            metrics = self._compute_classification_metrics(preds_flat, targets_flat)

        alphas = {
            "alpha_sp": w_sp_sum / max(total_samples, 1),
            "alpha_ctx": w_ctx_sum / max(total_samples, 1),
            "alpha_lat": w_lat_sum / max(total_samples, 1),
        }
        return avg_loss, metrics, alphas, pred_stats

    def _compute_regression_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict:
        metrics: Dict[str, float] = {}
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            preds = preds.copy()
            preds[nan_mask] = 0.0
        try:
            metrics["mae"] = float(mean_absolute_error(targets, preds))
        except Exception:
            pass
        try:
            mse = mean_squared_error(targets, preds)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mse"] = float(mse)
        except Exception:
            pass
        try:
            nonzero = targets != 0
            if nonzero.sum() > 0:
                metrics["mape"] = float(
                    np.mean(np.abs((targets[nonzero] - preds[nonzero]) / targets[nonzero])) * 100
                )
        except Exception:
            pass
        try:
            if targets.std() > 1e-8:
                metrics["r2"] = float(r2_score(targets, preds))
        except Exception:
            pass

        w = getattr(self.config.training, "composite_mae_weight", 0.5)
        if "mae" in metrics and "rmse" in metrics:
            metrics["composite_mae_rmse"] = (
                w * metrics["mae"] + (1 - w) * metrics["rmse"]
            )
        return metrics

    def _compute_classification_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict:
        metrics: Dict[str, float] = {}
        try:
            metrics["auc_roc"] = float(roc_auc_score(targets, preds))
        except Exception:
            pass
        binary_preds = (preds >= 0.5).astype(int)
        try:
            metrics["f1"] = float(f1_score(targets, binary_preds, zero_division=0))
        except Exception:
            pass
        try:
            metrics["precision"] = float(
                precision_score(targets, binary_preds, zero_division=0)
            )
        except Exception:
            pass
        try:
            metrics["recall"] = float(
                recall_score(targets, binary_preds, zero_division=0)
            )
        except Exception:
            pass
        return metrics

    # ------------------------------------------------------------------
    # Per-edge alpha collection (for analysis after training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_edge_alphas(
        self,
        split: str = "test",
        lat_presence_threshold: float = 0.05,
    ):
        """Collect batch-averaged per-edge alphas, mean A_lat, and persistence.

        Returns:
            (alpha_sp_edges, alpha_ctx_edges, alpha_lat_edges,
             A_lat_mean, persistence)  — all as [N, N] numpy arrays.
        """
        from crime_gnn.models.stgnn import LATENT_EDGE_THRESHOLD

        self.model.eval()
        N = self.model.num_nodes
        device = self.device

        sp_mask = (self.model.A_sp > 1e-6)
        ctx_mask = (self.model.A_ctx > 1e-6)
        fixed_union = sp_mask | ctx_mask

        a_sp = torch.zeros(N, N)
        a_ctx = torch.zeros(N, N)
        a_lat = torch.zeros(N, N)
        a_lat_adj = torch.zeros(N, N)
        lat_presence = torch.zeros(N, N)
        edge_counts = torch.zeros(N, N)
        total_samples = 0

        for batch in self.dataloaders[split]:
            batch = self._to_device(batch)
            out = self.model(batch)
            bs = out["alpha_sp"].size(0)
            A_lat_b = out["A_lat"]

            if A_lat_b.dim() == 3:
                lat_active = A_lat_b > LATENT_EDGE_THRESHOLD
                union_b = fixed_union.unsqueeze(0) | lat_active
            else:
                lat_active = A_lat_b > LATENT_EDGE_THRESHOLD
                union_b = (fixed_union | lat_active).unsqueeze(0).expand(bs, -1, -1)

            union_f = union_b.float()
            a_sp += (out["alpha_sp"] * union_f).sum(0).cpu()
            a_ctx += (out["alpha_ctx"] * union_f).sum(0).cpu()
            a_lat += (out["alpha_lat"] * union_f).sum(0).cpu()
            edge_counts += union_f.sum(0).cpu()

            if A_lat_b.dim() == 3:
                a_lat_adj += A_lat_b.sum(0).cpu()
                lat_presence += (A_lat_b > lat_presence_threshold).float().sum(0).cpu()
            else:
                a_lat_adj += A_lat_b.cpu() * bs
                lat_presence += (A_lat_b > lat_presence_threshold).float().cpu() * bs
            total_samples += bs

        d = max(total_samples, 1)
        safe_counts = edge_counts.clamp(min=1.0)
        return (
            (a_sp / safe_counts).numpy(),
            (a_ctx / safe_counts).numpy(),
            (a_lat / safe_counts).numpy(),
            (a_lat_adj / d).numpy(),
            (lat_presence / d).numpy(),
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _log_residual_mask_stats(self):
        mask = self.model.latent_graph_gen.residual_mask
        if mask is None:
            print("  Residual mask: None")
            return
        total = mask.numel()
        active = (mask > 0.5).sum().item()
        sp_overlap = (
            (mask > 0.5) & (self.A_sp > 1e-6)
        ).sum().item()
        ctx_overlap = (
            (mask > 0.5) & (self.A_ctx > 1e-6)
        ).sum().item()
        print(
            f"  Residual mask: {active}/{total} active ({active / total * 100:.1f}%), "
            f"mean={mask.mean():.4f}, std={mask.std():.4f}, "
            f"A_sp overlap={sp_overlap}, A_ctx overlap={ctx_overlap}"
        )

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        fname = "best_model.pt" if is_best else f"checkpoint_e{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "history": self.history,
            },
            os.path.join(self.output_dir, fname),
        )

    def _load_best_checkpoint(self):
        path = os.path.join(self.output_dir, "best_model.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"Loaded best model from epoch {ckpt['epoch']}")
