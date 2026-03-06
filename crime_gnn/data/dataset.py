# -*- coding: utf-8 -*-
"""
Data loading, sequence creation, and DataLoader construction.

The pipeline follows:
  1. Load CSV → aggregate to (region, week) panel.
  2. Build spatial adjacency from shapefile.
  3. Build contextual similarity graph from node features.
  4. (Optionally) normalise graph scales, then orthogonalise.
  5. Create sliding-window sequences.
  6. Year-based train / val / test split with target transform (regression).
  7. Wrap in DataLoaders with year-balanced sampling for training.
"""

import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from crime_gnn.data.graph_construction import (
    compute_binary_spatial_adjacency,
    compute_continuous_spatial_adjacency,
    graphs_to_torch,
    normalize_graph_scales,
    orthogonalize_graphs,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


# ---------------------------------------------------------------------------
# Data processor
# ---------------------------------------------------------------------------

class CrimeDataProcessor:
    """Loads and preprocesses crime data, builds graphs."""

    def __init__(self, config):
        self.config = config
        self.scaler_crime = StandardScaler()
        self.scaler_context = StandardScaler()
        self.region_to_idx: Dict[str, int] = {}
        self.idx_to_region: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load CSV, assign region / time indices, aggregate to weekly panel."""
        df = pd.read_csv(self.config.data.crime_csv_path)

        regions = sorted(df[self.config.data.region_col].unique())
        self.region_to_idx = {r: i for i, r in enumerate(regions)}
        self.idx_to_region = {i: r for r, i in self.region_to_idx.items()}
        df["region_idx"] = df[self.config.data.region_col].map(self.region_to_idx)

        time_keys = (
            df[["year", "week"]]
            .drop_duplicates()
            .sort_values(["year", "week"])
        )
        time_keys["time_idx"] = range(len(time_keys))
        df = df.merge(time_keys, on=["year", "week"], how="left")

        agg_dict = {self.config.data.target_col: "sum"}
        for col in self.config.data.contextual_cols:
            if col in df.columns:
                agg_dict[col] = "mean"
        # Classification: also aggregate the pre-computed binary label.
        if (self.config.data.task_type == "classification"
                and self.config.data.label_col in df.columns):
            agg_dict[self.config.data.label_col] = "max"

        df = (
            df.groupby(
                [self.config.data.region_col, "region_idx",
                 "year", "week", "time_idx"]
            )
            .agg(agg_dict)
            .reset_index()
        )
        df = df.sort_values(["region_idx", "time_idx"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Contextual features
    # ------------------------------------------------------------------

    def _get_raw_contextual_features(
        self, df: pd.DataFrame, years: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Raw contextual features per region; optionally restrict to years."""
        ctx_cols = self.config.data.contextual_cols
        subset = df[df["year"].isin(years)] if years is not None else df
        region_features = subset.groupby("region_idx")[ctx_cols].mean().sort_index()
        features = np.zeros((len(self.region_to_idx), len(ctx_cols)))
        for idx in region_features.index:
            features[idx] = region_features.loc[idx].values
        return features

    def fit_context_scaler(self, df: pd.DataFrame):
        """Fit scaler on training-year features only (no data leakage)."""
        train_features = self._get_raw_contextual_features(
            df, years=self.config.data.train_years
        )
        self.scaler_context.fit(train_features)
        print(f"  Context scaler fit on training years: "
              f"{self.config.data.train_years[0]}"
              f"-{self.config.data.train_years[-1]}")

    def get_scaled_contextual_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return contextual features scaled with the train-fitted scaler."""
        raw = self._get_raw_contextual_features(df)
        return self.scaler_context.transform(raw)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def compute_spatial_adjacency(self, df: pd.DataFrame) -> np.ndarray:
        """Build spatial adjacency from the shapefile."""
        try:
            import geopandas as gpd

            gdf = gpd.read_file(self.config.data.shapefile_path)
            if "BEAT" not in gdf.columns:
                raise ValueError("BEAT column not found in shapefile")

            print(f"  Raw shapefile: {len(gdf)} polygons, dissolving by BEAT...")
            gdf = gdf.dissolve(by="BEAT").reset_index()
            gdf = gdf[gdf["BEAT"].isin(self.region_to_idx.keys())]
            gdf["region_idx"] = gdf["BEAT"].map(self.region_to_idx)
            gdf = gdf.sort_values("region_idx").reset_index(drop=True)
            print(f"  After dissolve & filter: {len(gdf)} regions")

            if self.config.graph.use_continuous_spatial:
                A_sp = compute_continuous_spatial_adjacency(
                    gdf, self.region_to_idx,
                    method=self.config.graph.spatial_method,
                    sigma=self.config.graph.spatial_decay_sigma,
                    keep_topk=self.config.graph.spatial_keep_topk,
                )
                print(f"  Continuous spatial adjacency: "
                      f"mean_edge={A_sp[A_sp > 0].mean():.4f}, "
                      f"edges={int((A_sp > 0).sum())}")
            else:
                A_sp = compute_binary_spatial_adjacency(
                    gdf, self.region_to_idx,
                    method=self.config.graph.spatial_method,
                    knn_k=self.config.graph.knn_k,
                )
                print(f"  Binary spatial adjacency: edges={int(A_sp.sum())}")

        except Exception as e:
            print(f"  CRITICAL ERROR: Shapefile processing failed ({e}).")
            print(f"  Please ensure the shapefile exists at: "
                  f"{self.config.data.shapefile_path}")
            raise RuntimeError(
                "Pipeline halted: valid spatial shapefile is required."
            ) from e

        return A_sp

    def compute_contextual_similarity(self, df: pd.DataFrame) -> np.ndarray:
        """Build contextual graph using a local (graph-only) scaler.

        Note: a *separate* StandardScaler is used here so that the
        contextual-graph topology is not contaminated by the model's
        training-data scaler.
        """
        num_nodes = len(self.region_to_idx)
        raw = self._get_raw_contextual_features(df)
        ctx_features = StandardScaler().fit_transform(raw)

        method = self.config.graph.contextual_similarity
        if method == "cosine":
            sim = cosine_similarity(ctx_features)
        elif method == "rbf":
            sim = rbf_kernel(ctx_features)
        elif method == "correlation":
            sim = np.corrcoef(ctx_features)
        else:
            sim = cosine_similarity(ctx_features)

        A_ctx = np.zeros((num_nodes, num_nodes))
        knn = self.config.graph.contextual_knn
        if knn > 0:
            for i in range(num_nodes):
                s = sim[i].copy()
                s[i] = -np.inf
                top_k = np.argsort(s)[-knn:]
                A_ctx[i, top_k] = sim[i, top_k]
        else:
            thresh = self.config.graph.contextual_threshold
            A_ctx = (sim > thresh).astype(float) * sim

        A_ctx = np.maximum(A_ctx, A_ctx.T)
        np.fill_diagonal(A_ctx, 0)
        if A_ctx.max() > 0:
            A_ctx /= A_ctx.max()

        print(f"  Contextual similarity ({method}): edges={int((A_ctx > 0).sum())}")
        return A_ctx

    # ------------------------------------------------------------------
    # Sequences
    # ------------------------------------------------------------------

    def create_sequences(self, df: pd.DataFrame):
        """Create sliding-window (X_crime, y, time_info) arrays.

        For regression:
          - Supports multi-horizon prediction when config.data.multi_horizon
            is set: y shape is [T, H, N].
          - Otherwise y shape is [T, N].
        For classification:
          - y is binarised: 1 if crime_count > 0, else 0.
        """
        lookback = self.config.data.lookback_weeks
        horizon = self.config.data.forecast_horizon
        task_type = self.config.data.task_type
        multi_horizon = self.config.data.multi_horizon  # regression only

        crime_pivot = (
            df.pivot(
                index="time_idx", columns="region_idx",
                values=self.config.data.target_col,
            )
            .fillna(0)
            .sort_index()
        )

        time_indices = crime_pivot.index.values
        X_crime, y, time_info = [], [], []

        if task_type == "regression" and multi_horizon is not None:
            max_h = max(multi_horizon)
            for t in range(lookback, len(time_indices) - max_h + 1):
                X_crime.append(crime_pivot.iloc[t - lookback: t].values)
                targets_h = [crime_pivot.iloc[t + h - 1].values for h in multi_horizon]
                y.append(np.stack(targets_h, axis=0))  # [H, N]
                time_info.append(time_indices[t])
            print(f"  Multi-horizon regression: horizons={multi_horizon}")
        else:
            for t in range(lookback, len(time_indices) - horizon + 1):
                X_crime.append(crime_pivot.iloc[t - lookback: t].values)
                y.append(crime_pivot.iloc[t + horizon - 1].values)
                time_info.append(time_indices[t])

        X_crime = np.array(X_crime)
        y = np.array(y, dtype=np.float32)

        if task_type == "classification":
            y = (y > 0).astype(np.float32)
            print(f"  Binary classification, positive rate: {y.mean():.2%}")
        else:
            target_shape = y.shape[1:] if y.ndim > 2 else f"({y.shape[1]},)"
            print(f"  Sequences: {len(X_crime)}, input: {X_crime.shape}, "
                  f"target per sample: {target_shape}")

        return X_crime, y, np.array(time_info)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    def split_data(self, X_crime, y, time_info, df):
        """Year-based split with input normalisation and (regression) target
        transform.

        Returns a dict with keys:
          "train", "val", "test": each {"X_crime": ..., "y": ...}
          "X_ctx": contextual features [N, C]
          "crime_mean", "crime_std": input normalisation stats
          "target_info": (regression only) dict with transform parameters
          "train_years": year label per training sample
        """
        time_to_year = df.groupby("time_idx")["year"].first().to_dict()
        years = np.array([time_to_year.get(t, 2001) for t in time_info])

        masks = {
            "train": np.isin(years, self.config.data.train_years),
            "val": np.isin(years, self.config.data.val_years),
            "test": np.isin(years, self.config.data.test_years),
        }

        # Input normalisation (fit on train)
        mean = X_crime[masks["train"]].mean()
        std = X_crime[masks["train"]].std() + 1e-6
        X_norm = (X_crime - mean) / std

        splits = {}

        if self.config.data.task_type == "regression":
            target_tf = self.config.data.target_transform
            y_transformed = y.copy()
            target_info = {"transform": target_tf}

            if "log1p" in target_tf:
                y_transformed = np.log1p(y_transformed)
                target_info["applied_log1p"] = True
                print(f"  Target log1p: range [{y.min():.1f}, {y.max():.1f}]"
                      f" -> [{y_transformed.min():.3f}, {y_transformed.max():.3f}]")
            else:
                target_info["applied_log1p"] = False

            if "zscore" in target_tf:
                y_train = y_transformed[masks["train"]]
                target_mean = float(y_train.mean())
                target_std = float(y_train.std()) + 1e-8
                y_transformed = (y_transformed - target_mean) / target_std
                target_info["target_mean"] = target_mean
                target_info["target_std"] = target_std
                print(f"  Target z-score: mean={target_mean:.4f}, "
                      f"std={target_std:.4f}")
            else:
                target_info["target_mean"] = 0.0
                target_info["target_std"] = 1.0

            for name, mask in masks.items():
                splits[name] = {"X_crime": X_norm[mask], "y": y_transformed[mask]}
                print(f"  {name}: {mask.sum()} samples")

            splits["target_info"] = target_info

        else:  # classification
            for name, mask in masks.items():
                splits[name] = {"X_crime": X_norm[mask], "y": y[mask]}
                print(f"  {name}: {mask.sum()} samples")
            splits["target_info"] = {"transform": "none",
                                     "applied_log1p": False,
                                     "target_mean": 0.0,
                                     "target_std": 1.0}

        # Fit context scaler on train years, transform all
        self.fit_context_scaler(df)
        X_ctx = self.get_scaled_contextual_features(df)

        splits["X_ctx"] = X_ctx
        splits["crime_mean"] = mean
        splits["crime_std"] = std
        splits["train_years"] = years[masks["train"]]

        return splits


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class CrimeDataset(Dataset):
    def __init__(self, X_crime: np.ndarray, X_ctx: np.ndarray, y: np.ndarray):
        self.X_crime = torch.FloatTensor(X_crime)
        self.X_ctx = torch.FloatTensor(X_ctx)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_crime)

    def __getitem__(self, idx):
        return {
            "crime_seq": self.X_crime[idx],
            "ctx_features": self.X_ctx,   # static: same for every sample
            "target": self.y[idx],
        }


def create_dataloaders(splits: dict, config) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders.

    Training uses year-balanced WeightedRandomSampler so that each calendar
    year contributes equally regardless of its number of weeks.
    """
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = CrimeDataset(
            splits[name]["X_crime"], splits["X_ctx"], splits[name]["y"]
        )

        if name == "train" and "train_years" in splits:
            train_years = splits["train_years"]
            year_counts = Counter(train_years)
            weights = np.array([1.0 / year_counts[yr] for yr in train_years])
            weights /= weights.sum()
            sampler = WeightedRandomSampler(
                torch.DoubleTensor(weights), len(weights), replacement=True,
            )
            loaders[name] = DataLoader(
                ds,
                batch_size=config.training.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=(config.training.device == "cuda"),
            )
            print(f"  Year-balanced sampler: {len(year_counts)} years, "
                  f"weights range [{weights.min():.6f}, {weights.max():.6f}]")
        else:
            loaders[name] = DataLoader(
                ds,
                batch_size=config.training.batch_size,
                shuffle=(name == "train"),
                num_workers=0,
                pin_memory=(config.training.device == "cuda"),
            )
    return loaders


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_data(config):
    """Run the complete data-preparation pipeline.

    Order of operations:
      1. Load CSV.
      2. Compute spatial adjacency.
      3. Compute contextual similarity.
      4. Normalise graph scales (Frobenius).
      5. Orthogonalise A_ctx w.r.t. A_sp.
      6. Create sliding-window sequences.
      7. Train/val/test split + target transform.
      8. Build DataLoaders.

    Returns:
        dataloaders: dict with "train", "val", "test" DataLoaders.
        A_sp_t:     spatial adjacency tensor (normalised).
        A_ctx_t:    contextual adjacency tensor (orthogonalised).
        data_info:  metadata dict (num_nodes, idx_to_region, target_info, …).
    """
    print("=" * 60)
    print("Data Preprocessing")
    print("=" * 60)

    proc = CrimeDataProcessor(config)

    print("\n[1/7] Loading data...")
    df = proc.load_data()

    print("\n[2/7] Computing spatial adjacency...")
    A_sp = proc.compute_spatial_adjacency(df)

    print("\n[3/7] Computing contextual similarity...")
    A_ctx = proc.compute_contextual_similarity(df)

    if config.graph.normalize_graph_scales:
        print("\n[4/7] Normalising graph scales...")
        A_sp_norm, A_ctx_norm, scale_info = normalize_graph_scales(
            A_sp, A_ctx, method="frobenius"
        )
    else:
        A_sp_norm, A_ctx_norm = A_sp.copy(), A_ctx.copy()
        scale_info = {}

    if config.graph.use_orthogonalization:
        print("\n[5/7] Orthogonalising contextual graph...")
        A_ctx_pure, A_ctx_overlap, overlap_mask = orthogonalize_graphs(
            A_sp_norm, A_ctx_norm,
        )
    else:
        A_ctx_pure = A_ctx_norm.copy()

    print("\n[6/7] Creating sequences...")
    X_crime, y, time_info = proc.create_sequences(df)

    print("\n[7/7] Splitting data (train-only feature scaling)...")
    splits = proc.split_data(X_crime, y, time_info, df)

    dataloaders = create_dataloaders(splits, config)

    A_sp_t, A_ctx_pure_t = graphs_to_torch(A_sp_norm, A_ctx_pure, device="cpu")

    data_info = {
        "num_nodes": len(proc.region_to_idx),
        "num_contextual_features": len(config.data.contextual_cols),
        "region_to_idx": proc.region_to_idx,
        "idx_to_region": proc.idx_to_region,
        "X_ctx": splits["X_ctx"],
        "crime_mean": splits["crime_mean"],
        "crime_std": splits["crime_std"],
        "target_info": splits["target_info"],
        "A_sp_raw": A_sp,
        "A_ctx_raw": A_ctx,
        "A_sp_norm": A_sp_norm,
        "A_ctx_norm": A_ctx_norm,
        "A_ctx_pure": A_ctx_pure,
        "scale_info": scale_info,
    }

    print("\n" + "=" * 60)
    print("Data preprocessing complete!")
    print("=" * 60)

    return dataloaders, A_sp_t, A_ctx_pure_t, data_info
