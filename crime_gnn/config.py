# -*- coding: utf-8 -*-
"""
Configuration dataclasses for the Crime Spatio-Temporal GNN.

All hyper-parameters live here.  Use `build_config_from_args` to
construct a `Config` from a parsed `argparse.Namespace`.
"""

import argparse
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    crime_csv_path: str = "data/crime.csv"
    shapefile_path: str = "data/police_beats/police_beats.shp"
    region_col: str = "BEAT"
    target_col: str = "cnt"
    # Classification only: column that already contains binary labels (0/1).
    # For regression this field is ignored.
    label_col: str = "label"
    contextual_cols: List[str] = field(default_factory=lambda: [
        "white_ratio", "black_ratio", "poor_family_ratio", "unemployment_ratio",
        "BlightVal_min", "BlightVal_25_percentile", "BlightVal_75_percentile",
        "BlightVal_max", "BlightVal_average",
        "building_Commercial_sum_percent", "building_Industrial_sum_percent",
        "building_MF_Residential_sum_percent", "building_Mixed_Res/Com_sum_percent",
        "building_Other_sum_percent", "building_SF_Residential_sum_percent",
        "risk_percentage",
    ])
    lookback_weeks: int = 4
    forecast_horizon: int = 1
    train_years: List[int] = field(default_factory=lambda: list(range(2001, 2017)))
    val_years: List[int] = field(default_factory=lambda: [2017, 2018])
    test_years: List[int] = field(default_factory=lambda: [2019, 2020])

    # Task type: "regression" or "classification"
    task_type: str = "regression"

    # Regression only — multi-horizon prediction (e.g. [1, 2, 4]).
    # When set, the model predicts all horizons simultaneously.
    multi_horizon: Optional[List[int]] = None

    # Regression only — target transform applied before training.
    # "log1p" applies log(1+y); "zscore" normalises; "log1p+zscore" does both.
    # Predictions are inverse-transformed before metric computation.
    target_transform: str = "log1p+zscore"


@dataclass
class GraphConfig:
    spatial_method: str = "queen"         # "queen" | "rook" | "knn"
    knn_k: int = 5
    use_continuous_spatial: bool = True   # continuous decay vs binary adjacency
    spatial_decay_sigma: float = 0.0      # 0 = auto-calibrate
    spatial_keep_topk: int = 15
    contextual_similarity: str = "cosine" # "cosine" | "rbf" | "correlation"
    contextual_threshold: float = 0.5     # used when contextual_knn <= 0
    contextual_knn: int = 10
    latent_embedding_dim: int = 32
    latent_sparsity_target: float = 0.03
    latent_topk_per_node: int = 5
    use_orthogonalization: bool = True
    normalize_graph_scales: bool = True


@dataclass
class ModelConfig:
    num_nodes: int = 121                  # auto-set from data
    num_contextual_features: int = 16     # auto-set from data
    hidden_dim: int = 128
    gnn_hidden_dim: int = 128
    temporal_hidden_dim: int = 128
    num_gnn_layers: int = 3
    temporal_encoder: str = "GRU"         # "GRU" | "LSTM" | "Transformer"
    num_temporal_layers: int = 2
    dropout: float = 0.2
    gate_hidden_dim: int = 32
    use_learnable_graph_scaling: bool = True
    use_residual_latent: bool = True
    use_temporal_attention: bool = True
    num_attn_heads: int = 4
    # Number of prediction horizons (auto-set from data.multi_horizon).
    num_horizons: int = 1
    # Regression only: give the gating network its own separate projection of
    # [h_temporal || h_context], decoupled from the main fusion pathway.
    use_separate_gating_input: bool = True
    # Internal dims (auto-set to hidden_dim // 2 if 0)
    crime_encoder_dim: int = 0
    ctx_encoder_dim: int = 0
    use_fusion_layernorm: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # --- Loss weights ---
    lambda_pred_unique: float = 0.1
    lambda_sparse: float = 0.05
    lambda_ortho: float = 0.05
    lambda_floor: float = 0.5
    alpha_min: float = 0.05

    # Regression prediction loss: "mse" | "huber" | "smooth_l1"
    pred_loss: str = "huber"
    huber_delta: float = 1.0

    # Classification prediction loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    pos_weight: float = 3.0

    # --- Scheduler ---
    scheduler: str = "plateau"            # "cosine" | "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # --- Early stopping ---
    # Regression: "composite_mae_rmse" (mode="min")
    # Classification: "auc_roc" (mode="max")
    early_stopping_patience: int = 30
    early_stopping_metric: str = "composite_mae_rmse"
    early_stopping_mode: str = "min"

    # Regression: weight for MAE in composite metric = w*MAE + (1-w)*RMSE
    composite_mae_weight: float = 0.5

    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 10

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Curriculum ---
    use_curriculum: bool = True
    phase1_epochs: int = 20
    phase2_epochs: int = 130

    # Residual mask refresh interval (epochs within Phase 2)
    residual_mask_refresh_interval: int = 10

    # Regression only: skip phase 2 if phase-1 val R² is below this threshold.
    # Set to <= 0 to disable the guard.
    phase2_min_r2: float = 0.0

    # Phase-2 latent warm-up: ramp alpha_lat from 0→1 over this many epochs.
    latent_warmup_epochs: int = 15

    # LR multiplier for latent-graph parameters (regression).
    latent_lr_multiplier: float = 0.3

    # Maximum fraction of attention the latent channel may absorb (regression).
    alpha_lat_cap: float = 0.40


@dataclass
class AblationConfig:
    """Flags for ablation studies."""
    disable_latent_graph: bool = False
    disable_gating: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    seed: int = 42
    output_dir: str = "./output"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return a fully-specified ArgumentParser covering all Config fields."""
    p = argparse.ArgumentParser(
        description="Train an Interpretable Spatio-Temporal GNN for crime prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Task ----
    p.add_argument("--task", choices=["regression", "classification"],
                   default="regression",
                   help="Prediction task type.")

    # ---- Paths ----
    g = p.add_argument_group("Paths")
    g.add_argument("--crime-csv", default=DataConfig.crime_csv_path,
                   help="Path to crime CSV file.")
    g.add_argument("--shapefile", default=DataConfig.shapefile_path,
                   help="Path to police-beats shapefile (.shp).")
    g.add_argument("--output-dir", default="./output",
                   help="Root directory for checkpoints and results.")

    # ---- Data ----
    g = p.add_argument_group("Data")
    g.add_argument("--region-col", default="BEAT")
    g.add_argument("--target-col", default="cnt")
    g.add_argument("--label-col", default="label",
                   help="Column for pre-computed binary labels (classification only).")
    g.add_argument("--lookback-weeks", type=int, default=4)
    g.add_argument("--forecast-horizon", type=int, default=1)
    g.add_argument("--train-years", nargs="+", type=int,
                   default=list(range(2001, 2017)))
    g.add_argument("--val-years", nargs="+", type=int, default=[2017, 2018])
    g.add_argument("--test-years", nargs="+", type=int, default=[2019, 2020])
    g.add_argument("--multi-horizon", nargs="+", type=int, default=None,
                   help="Regression only: list of forecast horizons (e.g. 1 2 4).")
    g.add_argument("--target-transform", default="log1p+zscore",
                   choices=["none", "log1p", "zscore", "log1p+zscore"],
                   help="Regression only: target transform pipeline.")

    # ---- Graph ----
    g = p.add_argument_group("Graph construction")
    g.add_argument("--spatial-method", default="queen",
                   choices=["queen", "rook", "knn"])
    g.add_argument("--knn-k", type=int, default=5)
    g.add_argument("--no-continuous-spatial", action="store_true",
                   help="Use binary (0/1) instead of distance-decay spatial adjacency.")
    g.add_argument("--spatial-sigma", type=float, default=0.0,
                   help="Distance-decay sigma; 0 = auto-calibrate.")
    g.add_argument("--spatial-topk", type=int, default=15)
    g.add_argument("--ctx-similarity", default="cosine",
                   choices=["cosine", "rbf", "correlation"])
    g.add_argument("--ctx-knn", type=int, default=10)
    g.add_argument("--latent-embed-dim", type=int, default=32)
    g.add_argument("--latent-sparsity", type=float, default=0.03)
    g.add_argument("--latent-topk", type=int, default=5)
    g.add_argument("--no-orthogonalization", action="store_true")
    g.add_argument("--no-graph-scale-norm", action="store_true")

    # ---- Model ----
    g = p.add_argument_group("Model architecture")
    g.add_argument("--hidden-dim", type=int, default=128)
    g.add_argument("--gnn-hidden-dim", type=int, default=128)
    g.add_argument("--temporal-hidden-dim", type=int, default=128)
    g.add_argument("--gnn-layers", type=int, default=3)
    g.add_argument("--temporal-encoder", default="GRU",
                   choices=["GRU", "LSTM", "Transformer"])
    g.add_argument("--temporal-layers", type=int, default=2)
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--gate-hidden-dim", type=int, default=32)
    g.add_argument("--no-learnable-scaling", action="store_true")
    g.add_argument("--no-temporal-attention", action="store_true")
    g.add_argument("--attn-heads", type=int, default=4)
    g.add_argument("--no-separate-gating-input", action="store_true",
                   help="Regression only: disable separate gating projection.")

    # ---- Training ----
    g = p.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--lambda-unique", type=float, default=0.1)
    g.add_argument("--lambda-sparse", type=float, default=0.05)
    g.add_argument("--lambda-ortho", type=float, default=0.05)
    g.add_argument("--lambda-floor", type=float, default=0.5)
    g.add_argument("--alpha-min", type=float, default=0.05)
    g.add_argument("--pred-loss", default="huber",
                   choices=["mse", "huber", "smooth_l1"],
                   help="Regression prediction loss type.")
    g.add_argument("--huber-delta", type=float, default=1.0)
    g.add_argument("--no-focal-loss", action="store_true",
                   help="Classification: use standard BCE instead of Focal loss.")
    g.add_argument("--focal-gamma", type=float, default=2.0)
    g.add_argument("--pos-weight", type=float, default=3.0,
                   help="Classification: positive-class weight for BCE.")
    g.add_argument("--scheduler", default="plateau", choices=["cosine", "plateau"])
    g.add_argument("--scheduler-patience", type=int, default=10)
    g.add_argument("--scheduler-factor", type=float, default=0.5)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--es-patience", type=int, default=30,
                   help="Early-stopping patience (epochs).")
    g.add_argument("--es-metric", default=None,
                   help="Metric for early stopping. Defaults to composite_mae_rmse "
                        "(regression) or auc_roc (classification).")
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--log-interval", type=int, default=10)
    g.add_argument("--save-interval", type=int, default=10)
    g.add_argument("--device", default=None,
                   help="Device to use (default: cuda if available else cpu).")
    g.add_argument("--no-curriculum", action="store_true",
                   help="Skip two-phase curriculum; run all epochs as phase 2.")
    g.add_argument("--phase1-epochs", type=int, default=20)
    g.add_argument("--phase2-epochs", type=int, default=130)
    g.add_argument("--mask-refresh-interval", type=int, default=10)
    g.add_argument("--phase2-min-r2", type=float, default=0.0,
                   help="Regression: skip phase 2 if phase-1 R² < this value.")
    g.add_argument("--latent-warmup-epochs", type=int, default=15)
    g.add_argument("--latent-lr-mult", type=float, default=0.3)
    g.add_argument("--alpha-lat-cap", type=float, default=0.40)
    g.add_argument("--composite-mae-weight", type=float, default=0.5,
                   help="Regression: weight w in composite = w*MAE + (1-w)*RMSE.")

    # ---- Ablation ----
    g = p.add_argument_group("Ablation")
    g.add_argument("--disable-latent", action="store_true",
                   help="Ablation: disable latent graph learning entirely.")
    g.add_argument("--disable-gating", action="store_true",
                   help="Ablation: use uniform channel gating instead of learned.")

    # ---- Misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="full_model",
                   help="Experiment label (used as sub-directory name).")

    return p


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Construct a Config object from parsed CLI arguments."""
    cfg = Config()

    # Data
    cfg.data.task_type = args.task
    cfg.data.crime_csv_path = args.crime_csv
    cfg.data.shapefile_path = args.shapefile
    cfg.data.region_col = args.region_col
    cfg.data.target_col = args.target_col
    cfg.data.label_col = args.label_col
    cfg.data.lookback_weeks = args.lookback_weeks
    cfg.data.forecast_horizon = args.forecast_horizon
    cfg.data.train_years = args.train_years
    cfg.data.val_years = args.val_years
    cfg.data.test_years = args.test_years
    cfg.data.multi_horizon = args.multi_horizon
    cfg.data.target_transform = args.target_transform

    # Graph
    cfg.graph.spatial_method = args.spatial_method
    cfg.graph.knn_k = args.knn_k
    cfg.graph.use_continuous_spatial = not args.no_continuous_spatial
    cfg.graph.spatial_decay_sigma = args.spatial_sigma
    cfg.graph.spatial_keep_topk = args.spatial_topk
    cfg.graph.contextual_similarity = args.ctx_similarity
    cfg.graph.contextual_knn = args.ctx_knn
    cfg.graph.latent_embedding_dim = args.latent_embed_dim
    cfg.graph.latent_sparsity_target = args.latent_sparsity
    cfg.graph.latent_topk_per_node = args.latent_topk
    cfg.graph.use_orthogonalization = not args.no_orthogonalization
    cfg.graph.normalize_graph_scales = not args.no_graph_scale_norm

    # Model
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.gnn_hidden_dim = args.gnn_hidden_dim
    cfg.model.temporal_hidden_dim = args.temporal_hidden_dim
    cfg.model.num_gnn_layers = args.gnn_layers
    cfg.model.temporal_encoder = args.temporal_encoder
    cfg.model.num_temporal_layers = args.temporal_layers
    cfg.model.dropout = args.dropout
    cfg.model.gate_hidden_dim = args.gate_hidden_dim
    cfg.model.use_learnable_graph_scaling = not args.no_learnable_scaling
    cfg.model.use_temporal_attention = not args.no_temporal_attention
    cfg.model.num_attn_heads = args.attn_heads
    cfg.model.use_separate_gating_input = not args.no_separate_gating_input

    # Training
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.lr
    cfg.training.weight_decay = args.weight_decay
    cfg.training.lambda_pred_unique = args.lambda_unique
    cfg.training.lambda_sparse = args.lambda_sparse
    cfg.training.lambda_ortho = args.lambda_ortho
    cfg.training.lambda_floor = args.lambda_floor
    cfg.training.alpha_min = args.alpha_min
    cfg.training.pred_loss = args.pred_loss
    cfg.training.huber_delta = args.huber_delta
    cfg.training.use_focal_loss = not args.no_focal_loss
    cfg.training.focal_gamma = args.focal_gamma
    cfg.training.pos_weight = args.pos_weight
    cfg.training.scheduler = args.scheduler
    cfg.training.scheduler_patience = args.scheduler_patience
    cfg.training.scheduler_factor = args.scheduler_factor
    cfg.training.min_lr = args.min_lr
    cfg.training.early_stopping_patience = args.es_patience
    cfg.training.max_grad_norm = args.max_grad_norm
    cfg.training.log_interval = args.log_interval
    cfg.training.save_interval = args.save_interval
    cfg.training.use_curriculum = not args.no_curriculum
    cfg.training.phase1_epochs = args.phase1_epochs
    cfg.training.phase2_epochs = args.phase2_epochs
    cfg.training.residual_mask_refresh_interval = args.mask_refresh_interval
    cfg.training.phase2_min_r2 = args.phase2_min_r2
    cfg.training.latent_warmup_epochs = args.latent_warmup_epochs
    cfg.training.latent_lr_multiplier = args.latent_lr_mult
    cfg.training.alpha_lat_cap = args.alpha_lat_cap
    cfg.training.composite_mae_weight = args.composite_mae_weight

    if args.device is not None:
        cfg.training.device = args.device

    # Set task-appropriate early-stopping defaults
    if args.es_metric is not None:
        cfg.training.early_stopping_metric = args.es_metric
    elif args.task == "regression":
        cfg.training.early_stopping_metric = "composite_mae_rmse"
        cfg.training.early_stopping_mode = "min"
    else:
        cfg.training.early_stopping_metric = "auc_roc"
        cfg.training.early_stopping_mode = "max"

    # Ablation
    cfg.ablation.disable_latent_graph = args.disable_latent
    cfg.ablation.disable_gating = args.disable_gating

    # Misc
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir

    return cfg
