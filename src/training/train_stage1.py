from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

# Make both `python src/training/train_stage1.py` and `python -m src.training.train_stage1` work.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
_ROOT = _THIS.parents[2]
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.data.preprocess import (
    build_feature_transformer,
    load_dataframe,
    resolve_feature_columns,
    split_indices,              # legacy, not used in main flow but kept for reference
    split_indices_three_way,    # new function for 3-way splits with different strategies
    transform_features,
)
from src.data.shot_taxonomy import (  # noqa: E402
    BucketSpec,
    ShotTaxonomyConfig,
    assign_bins,
    build_shot_taxonomy,
    summarize_shot_labels,
)
from src.models.dir_modules import (  # noqa: E402
    MLPRegressor,
    prepare_sample_weights,
    weighted_mse_loss,
)
from src.utils.config_loader import (  # noqa: E402
    build_run_metadata,
    get_target_column,
    load_yaml_config,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def safe_gmean_abs_error(abs_err: np.ndarray) -> float:
    vals = np.maximum(np.asarray(abs_err, dtype=np.float64), 1e-12)
    return float(np.exp(np.mean(np.log(vals))))


class TabularDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method_buckets: np.ndarray,
        weights: Optional[np.ndarray] = None,
        cam_ids: Optional[Sequence] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.method_buckets = torch.tensor(method_buckets, dtype=torch.long)
        self.weights = torch.tensor(
            np.ones(len(y), dtype=np.float32) if weights is None else weights,
            dtype=torch.float32,
        ).view(-1, 1)
        self.cam_ids = np.asarray(cam_ids) if cam_ids is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.weights[idx], self.method_buckets[idx]


@dataclass
class RunArtifacts:
    run_dir: Path
    best_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None


METHODS = {"plain", "inverse", "sqrt_inv", "lds", "fds", "lds_fds"}


def parse_method(method: str) -> Tuple[str, bool, bool]:
    m = method.lower()
    if m not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if m == "plain":
        return "none", False, False
    if m == "inverse":
        return "inverse", False, False
    if m == "sqrt_inv":
        return "sqrt_inv", False, False
    if m == "lds":
        return "sqrt_inv", True, False
    if m == "fds":
        return "none", False, True
    return "sqrt_inv", True, True  # lds_fds



def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_buckets: np.ndarray,
    shot_map: Dict[int, str],
    train_eval_bucket_counts: np.ndarray,
    cam_ids: Optional[Sequence] = None,
    cold_threshold: float = 0.0,
    hot_threshold: float = 30.0,
    prefix: str = "",
) -> Dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    metrics: Dict[str, float] = {
        f"{prefix}overall_mae": float(mae),
        f"{prefix}overall_rmse": float(rmse),
        f"{prefix}overall_mse": float(mse),
        f"{prefix}gmean_abs_error": safe_gmean_abs_error(abs_err),
    }

    group_indices = {"many": [], "medium": [], "few": [], "empty": []}
    for i, b in enumerate(eval_buckets):
        group_indices[shot_map.get(int(b), "empty")].append(i)

    for g in ("many", "medium", "few"):
        idxs = group_indices[g]
        if len(idxs) == 0:
            metrics[f"{prefix}{g}_mae"] = float("nan")
            metrics[f"{prefix}{g}_rmse"] = float("nan")
        else:
            yt = y_true[idxs]
            yp = y_pred[idxs]
            metrics[f"{prefix}{g}_mae"] = float(mean_absolute_error(yt, yp))
            metrics[f"{prefix}{g}_rmse"] = float(math.sqrt(mean_squared_error(yt, yp)))

    low_thr = np.percentile(y_true, 10)
    high_thr = np.percentile(y_true, 90)
    low_mask = y_true <= low_thr
    high_mask = y_true >= high_thr
    metrics[f"{prefix}low10_mae"] = float(mean_absolute_error(y_true[low_mask], y_pred[low_mask])) if np.any(low_mask) else float("nan")
    metrics[f"{prefix}high10_mae"] = float(mean_absolute_error(y_true[high_mask], y_pred[high_mask])) if np.any(high_mask) else float("nan")

    cold_mask = y_true <= cold_threshold
    hot_mask = y_true >= hot_threshold
    metrics[f"{prefix}cold_tail_mae"] = float(mean_absolute_error(y_true[cold_mask], y_pred[cold_mask])) if np.any(cold_mask) else float("nan")
    metrics[f"{prefix}hot_tail_mae"] = float(mean_absolute_error(y_true[hot_mask], y_pred[hot_mask])) if np.any(hot_mask) else float("nan")

    if cam_ids is not None:
        cam_df = pd.DataFrame({"cam": cam_ids, "y": y_true, "p": y_pred})
        cam_maes = [mean_absolute_error(g["y"].values, g["p"].values) for _, g in cam_df.groupby("cam") if len(g) > 0]
        metrics[f"{prefix}camera_mae_mean"] = float(np.mean(cam_maes)) if cam_maes else float("nan")
        metrics[f"{prefix}camera_mae_std"] = float(np.std(cam_maes)) if cam_maes else float("nan")
    else:
        metrics[f"{prefix}camera_mae_mean"] = float("nan")
        metrics[f"{prefix}camera_mae_std"] = float("nan")

    shot_counts = summarize_shot_labels(eval_buckets, shot_map)
    metrics.update({f"{prefix}{k}_count": int(v) for k, v in shot_counts.items()})
    metrics[f"{prefix}train_eval_bucket_nonempty"] = int(np.sum(train_eval_bucket_counts > 0))
    metrics[f"{prefix}train_eval_bucket_empty"] = int(np.sum(train_eval_bucket_counts == 0))
    return metrics



def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, labels, buckets = [], [], []
    with torch.no_grad():
        for xb, yb, _wb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            pred, _ = model(xb, mb, epoch=0)
            preds.append(pred.cpu().numpy().reshape(-1))
            labels.append(yb.numpy().reshape(-1))
            buckets.append(mb.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(buckets)



def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")



def save_split(df: pd.DataFrame, idx: np.ndarray, path: Path) -> None:
    df.iloc[idx].to_csv(path, index=False)



def make_run_dir(base_dir: str, feature_set: str, method: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{stamp}_{feature_set}_{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir



def _load_camera_json(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        cams = obj.get("cameras")
        if cams is None:
            raise ValueError(f"Camera JSON '{path}' must contain key 'cameras' or be a plain list.")
    elif isinstance(obj, list):
        cams = obj
    else:
        raise ValueError(f"Unsupported camera JSON format in '{path}'.")
    return [str(x) for x in cams]



def _load_camera_manifest(path: str) -> Tuple[List[str], List[str], List[str]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("camera_split_manifest must be a JSON object.")
    for key in ("train_cameras", "val_cameras"):
        if key not in obj:
            raise ValueError(f"camera_split_manifest missing required key: {key}")
    train_cams = [str(x) for x in obj["train_cameras"]]
    val_cams = [str(x) for x in obj["val_cameras"]]
    test_cams = [str(x) for x in obj.get("test_cameras", [])]
    return train_cams, val_cams, test_cams



def _indices_from_camera_sets(
    df: pd.DataFrame,
    group_col: str,
    train_cams: Sequence[str],
    val_cams: Sequence[str],
    test_cams: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found for fixed_camera_json split")

    train_set = set(map(str, train_cams))
    val_set = set(map(str, val_cams))
    test_set = set(map(str, test_cams or []))

    if train_set & val_set:
        raise ValueError("Train and validation camera sets overlap.")
    if train_set & test_set:
        raise ValueError("Train and test camera sets overlap.")
    if val_set & test_set:
        raise ValueError("Validation and test camera sets overlap.")

    cam_series = df[group_col].astype(str)
    train_mask = cam_series.isin(train_set).values
    val_mask = cam_series.isin(val_set).values
    test_mask = cam_series.isin(test_set).values if test_set else np.zeros(len(df), dtype=bool)

    covered = train_mask | val_mask | test_mask
    if not np.all(covered):
        uncovered = sorted(cam_series.loc[~covered].unique().tolist())
        raise ValueError(
            "Fixed camera split does not cover all cameras present in dataframe after target filtering. "
            f"Uncovered cameras: {uncovered}"
        )

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0] if test_set else None

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Fixed camera split produced empty train or validation set.")
    if test_set and (test_idx is None or len(test_idx) == 0):
        raise ValueError("Fixed camera split produced empty test set.")

    return train_idx, val_idx, test_idx



def _random_3way_indices(
    n: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    idx = np.arange(n)
    if test_ratio <= 0:
        tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)
        return np.asarray(tr), np.asarray(va), None

    trainval_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=seed, shuffle=True)
    inner_val_ratio = val_ratio / (1.0 - test_ratio)
    tr, va = train_test_split(trainval_idx, test_size=inner_val_ratio, random_state=seed, shuffle=True)
    return np.asarray(tr), np.asarray(va), np.asarray(test_idx)



def _cross_camera_3way_indices(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found for cross_camera split")

    idx = np.arange(len(df))
    groups = df[group_col].values

    if test_ratio <= 0:
        tr, va = base_split_indices(df, "cross_camera", val_ratio, seed, group_col=group_col)
        return np.asarray(tr), np.asarray(va), None

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_rel, test_idx = next(gss_outer.split(idx, groups=groups))

    trainval_idx = idx[trainval_rel]
    trainval_groups = groups[trainval_rel]
    inner_val_ratio = val_ratio / (1.0 - test_ratio)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=seed)
    tr_rel, va_rel = next(gss_inner.split(trainval_idx, groups=trainval_groups))

    tr = trainval_idx[tr_rel]
    va = trainval_idx[va_rel]
    return np.asarray(tr), np.asarray(va), np.asarray(test_idx)



def resolve_split_indices(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    split_meta: Dict[str, object] = {"split_mode": args.split_mode}

    if args.split_mode == "fixed_camera_json":
        if args.camera_split_manifest:
            train_cams, val_cams, test_cams = _load_camera_manifest(args.camera_split_manifest)
            split_meta["camera_split_manifest"] = args.camera_split_manifest
        else:
            train_cams = _load_camera_json(args.train_cameras_json)
            val_cams = _load_camera_json(args.val_cameras_json)
            test_cams = _load_camera_json(args.test_cameras_json) or []
            if not train_cams or not val_cams:
                raise ValueError(
                    "For split_mode=fixed_camera_json, provide either --camera_split_manifest or both "
                    "--train_cameras_json and --val_cameras_json."
                )
        tr, va, te = _indices_from_camera_sets(df, args.group_col, train_cams, val_cams, test_cams)
        split_meta.update({
            "train_cameras": list(train_cams),
            "val_cameras": list(val_cams),
            "test_cameras": list(test_cams),
        })
        return tr, va, te, split_meta

    if args.split_mode == "random":
        tr, va, te = _random_3way_indices(len(df), args.val_ratio, args.test_ratio, args.seed)
        split_meta.update({"val_ratio": args.val_ratio, "test_ratio": args.test_ratio})
        return tr, va, te, split_meta

    if args.split_mode == "cross_camera":
        tr, va, te = _cross_camera_3way_indices(df, args.group_col, args.val_ratio, args.test_ratio, args.seed)
        split_meta.update({"group_col": args.group_col, "val_ratio": args.val_ratio, "test_ratio": args.test_ratio})
        return tr, va, te, split_meta

    raise ValueError(f"Unsupported split_mode: {args.split_mode}")



def train(args: argparse.Namespace) -> RunArtifacts:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    config = load_yaml_config(args.feature_config)
    df = load_dataframe(args.csv_path)
    target_col = args.target_col or get_target_column(config)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    df = df[df[target_col].notna()].copy().reset_index(drop=True)
    feature_df, _, _ = resolve_feature_columns(config, args.feature_set, df, args.cyclical_time)

    train_idx, val_idx, test_idx, split_meta = split_indices_three_way(
        df=df,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_col=args.group_col,
        camera_split_manifest=args.camera_split_manifest,
        train_cameras_json=args.train_cameras_json,
        val_cameras_json=args.val_cameras_json,
        test_cameras_json=args.test_cameras_json,
    )
    train_feat = feature_df.iloc[train_idx].copy()
    val_feat = feature_df.iloc[val_idx].copy()
    test_feat = feature_df.iloc[test_idx].copy() if test_idx is not None else None

    y_train = df.iloc[train_idx][target_col].astype(np.float32).values
    y_val = df.iloc[val_idx][target_col].astype(np.float32).values
    y_test = df.iloc[test_idx][target_col].astype(np.float32).values if test_idx is not None else None

    train_cam = df.iloc[train_idx][args.group_col].values if args.group_col in df.columns else None
    val_cam = df.iloc[val_idx][args.group_col].values if args.group_col in df.columns else None
    test_cam = df.iloc[test_idx][args.group_col].values if (test_idx is not None and args.group_col in df.columns) else None

    # 1) Fixed evaluation shot taxonomy (independent of LDS/FDS bucket settings)
    eval_bucket_spec = BucketSpec(
        mode=args.eval_bucket_mode,
        min_value=args.eval_min,
        max_value=args.eval_max,
        bin_width=args.eval_bin_width,
        bucket_num=args.eval_bucket_num,
    )
    shot_cfg = ShotTaxonomyConfig(
        strategy=args.shot_strategy,
        few_quantile=args.few_quantile,
        many_quantile=args.many_quantile,
        few_threshold=args.few_shot_thr,
        many_threshold=args.many_shot_thr,
    )
    eval_tax = build_shot_taxonomy(y_train, eval_bucket_spec, shot_cfg)
    train_eval_buckets = assign_bins(y_train, eval_tax.bucket_edges)
    val_eval_buckets = assign_bins(y_val, eval_tax.bucket_edges)
    test_eval_buckets = assign_bins(y_test, eval_tax.bucket_edges) if y_test is not None else None

    # 2) Method buckets for LDS/FDS (can be decoupled)
    method_bucket_spec = BucketSpec(
        mode=args.method_bucket_mode,
        min_value=args.method_min,
        max_value=args.method_max,
        bin_width=args.method_bin_width,
        bucket_num=args.method_bucket_num,
    )
    method_edges = method_bucket_spec.build_edges(y_train if method_bucket_spec.mode == "equal_width" else None)
    train_method_buckets = assign_bins(y_train, method_edges)
    val_method_buckets = assign_bins(y_val, method_edges)
    test_method_buckets = assign_bins(y_test, method_edges) if y_test is not None else None
    method_bucket_num = len(method_edges) - 1

    # 3) Feature transform (supports numeric + categorical)
    feat_art = build_feature_transformer(train_feat)
    if len(test_idx) > 0:
        X_train, X_val, X_test = transform_features(train_feat, val_feat, feat_art, test_feat=test_feat)
    else:
        X_train, X_val = transform_features(train_feat, val_feat, feat_art)
        X_test = None
    X_test = feat_art.transformer.transform(test_feat).astype(np.float32) if test_feat is not None else None

    reweight, use_lds, use_fds = parse_method(args.method)
    sample_weights, effective_counts = prepare_sample_weights(
        bucket_idx=train_method_buckets,
        bucket_num=method_bucket_num,
        reweight=reweight,
        lds=use_lds,
        lds_kernel=args.lds_kernel,
        lds_ks=args.lds_ks,
        lds_sigma=args.lds_sigma,
    )

    train_ds = TabularDataset(X_train, y_train, train_method_buckets, weights=sample_weights, cam_ids=train_cam)
    val_ds = TabularDataset(X_val, y_val, val_method_buckets, weights=None, cam_ids=val_cam)
    test_ds = TabularDataset(X_test, y_test, test_method_buckets, weights=None, cam_ids=test_cam) if X_test is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_ds is not None else None

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    model = MLPRegressor(
        in_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        use_fds=use_fds,
        fds_kwargs={
            "bucket_num": method_bucket_num,
            "bucket_start": args.fds_bucket_start,
            "start_update": args.fds_start_update,
            "start_smooth": args.fds_start_smooth,
            "kernel": args.fds_kernel,
            "ks": args.fds_ks,
            "sigma": args.fds_sigma,
            "momentum": None if args.fds_momentum < 0 else args.fds_momentum,
        },
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = make_run_dir(args.run_dir, args.feature_set, args.method, args.seed)
    shutil.copy2(args.feature_config, run_dir / "feature_config_snapshot.yaml")
    save_split(df, train_idx, run_dir / "split_train.csv")
    save_split(df, val_idx, run_dir / "split_val.csv")
    if test_idx is not None:
        save_split(df, test_idx, run_dir / "split_test.csv")

    feature_meta = {
        "resolved_feature_set": args.feature_set,
        "input_columns_after_resolution": list(train_feat.columns),
        "transformed_feature_names": feat_art.output_feature_names,
        "numeric_cols": feat_art.numeric_cols,
        "categorical_cols": feat_art.categorical_cols,
    }
    save_json(run_dir / "feature_columns.json", feature_meta)

    pd.DataFrame({
        "eval_bucket": np.arange(len(eval_tax.bucket_counts)),
        "train_count": eval_tax.bucket_counts,
        "shot": [eval_tax.shot_map[i] for i in range(len(eval_tax.bucket_counts))],
        "left": eval_tax.bucket_edges[:-1],
        "right": eval_tax.bucket_edges[1:],
    }).to_csv(run_dir / "eval_bucket_info.csv", index=False)

    pd.DataFrame({
        "method_bucket": np.arange(method_bucket_num),
        "effective_count": effective_counts,
        "left": method_edges[:-1],
        "right": method_edges[1:],
    }).to_csv(run_dir / "method_bucket_info.csv", index=False)

    run_meta = build_run_metadata(config, args.feature_set, args.method, args.split_mode)
    run_meta.update({
        "target_column": target_col,
        "group_col": args.group_col,
        "seed": args.seed,
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)) if test_idx is not None else 0,
        "run_dir": str(run_dir),
        "eval_bucket_spec": vars(eval_bucket_spec),
        "shot_taxonomy": vars(shot_cfg),
        "method_bucket_spec": vars(method_bucket_spec),
        "hidden_dims": hidden_dims,
        "use_lds": bool(use_lds),
        "use_fds": bool(use_fds),
        "split_meta": split_meta,
    })
    save_json(run_dir / "run_meta.json", run_meta)

    history: List[Dict[str, float]] = []
    best_epoch = -1
    best_val_mae = float("inf")
    best_metrics: Dict[str, float] = {}
    best_preds_df: Optional[pd.DataFrame] = None

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        cached_feat = []
        cached_buckets = []

        for xb, yb, wb, mb in train_loader:
            xb, yb, wb, mb = xb.to(device), yb.to(device), wb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred, raw_feat = model(xb, mb, epoch=epoch)
            loss = weighted_mse_loss(pred, yb, wb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if use_fds:
                cached_feat.append(raw_feat.detach().cpu())
                cached_buckets.append(mb.detach().cpu())

        if use_fds:
            all_feat = torch.cat(cached_feat, dim=0)
            all_buckets = torch.cat(cached_buckets, dim=0)
            model.fds.update_last_epoch_stats(epoch)
            model.fds.update_running_stats(all_feat, all_buckets, epoch)

        y_pred, y_true, _ = run_inference(model, val_loader, device)
        metrics = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            eval_buckets=val_eval_buckets,
            shot_map=eval_tax.shot_map,
            train_eval_bucket_counts=eval_tax.bucket_counts,
            cam_ids=val_cam,
            cold_threshold=args.cold_threshold,
            hot_threshold=args.hot_threshold,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(metrics)

        print(
            f"Epoch {epoch+1:03d} | train_loss={metrics['train_loss']:.4f} | "
            f"val_mae={metrics['overall_mae']:.4f} | val_rmse={metrics['overall_rmse']:.4f} | "
            f"few_mae={metrics['few_mae']:.4f} | low10_mae={metrics['low10_mae']:.4f}"
        )

        if metrics["overall_mae"] < best_val_mae:
            best_val_mae = metrics["overall_mae"]
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            best_preds_df = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
                "eval_bucket": val_eval_buckets,
                "method_bucket": val_method_buckets,
                "shot": [eval_tax.shot_map.get(int(b), "empty") for b in val_eval_buckets],
                "cam_id": val_cam if val_cam is not None else np.repeat("NA", len(y_true)),
            })
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "metrics_history.csv", index=False)
    best_metrics["best_epoch"] = int(best_epoch)
    save_json(run_dir / "best_metrics.json", best_metrics)
    if best_preds_df is not None:
        best_preds_df.to_csv(run_dir / "val_predictions_best.csv", index=False)

    test_metrics: Optional[Dict[str, float]] = None
    if test_loader is not None:
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
        y_pred_t, y_true_t, _ = run_inference(model, test_loader, device)
        test_metrics = evaluate_predictions(
            y_true=y_true_t,
            y_pred=y_pred_t,
            eval_buckets=test_eval_buckets,
            shot_map=eval_tax.shot_map,
            train_eval_bucket_counts=eval_tax.bucket_counts,
            cam_ids=test_cam,
            cold_threshold=args.cold_threshold,
            hot_threshold=args.hot_threshold,
        )
        test_metrics["selected_by_val_best_epoch"] = int(best_epoch)
        save_json(run_dir / "test_metrics_best.json", test_metrics)
        pd.DataFrame({
            "y_true": y_true_t,
            "y_pred": y_pred_t,
            "eval_bucket": test_eval_buckets,
            "method_bucket": test_method_buckets,
            "shot": [eval_tax.shot_map.get(int(b), "empty") for b in test_eval_buckets],
            "cam_id": test_cam if test_cam is not None else np.repeat("NA", len(y_true_t)),
        }).to_csv(run_dir / "test_predictions_best.csv", index=False)

    return RunArtifacts(run_dir=run_dir, best_metrics=best_metrics, test_metrics=test_metrics)



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 DIR training for SkyFinder structured features")

    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--feature_config", type=str, required=True)
    p.add_argument("--feature_set", type=str, required=True)
    p.add_argument("--target_col", type=str, default=None)

    p.add_argument("--run_dir", type=str, default="runs/stage1")
    p.add_argument("--split_mode", type=str, default="cross_camera", choices=["cross_camera", "random", "fixed_camera_json"])
    p.add_argument("--group_col", type=str, default="CamId")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.0)
    p.add_argument("--camera_split_manifest", type=str, default=None)
    p.add_argument("--train_cameras_json", type=str, default=None)
    p.add_argument("--val_cameras_json", type=str, default=None)
    p.add_argument("--test_cameras_json", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cyclical_time", action="store_true")

    # Eval shot taxonomy: fixed, stable, and independent from LDS/FDS buckets.
    p.add_argument("--eval_bucket_mode", type=str, default="fixed_width", choices=["fixed_width", "equal_width"])
    p.add_argument("--eval_min", type=float, default=-28.0)
    p.add_argument("--eval_max", type=float, default=50.0)
    p.add_argument("--eval_bin_width", type=float, default=1.0)
    p.add_argument("--eval_bucket_num", type=int, default=80)
    p.add_argument("--shot_strategy", type=str, default="quantile", choices=["quantile", "absolute"])
    p.add_argument("--few_quantile", type=float, default=0.2)
    p.add_argument("--many_quantile", type=float, default=0.8)
    p.add_argument("--few_shot_thr", type=int, default=100)
    p.add_argument("--many_shot_thr", type=int, default=1000)

    # Method buckets for LDS/FDS.
    p.add_argument("--method_bucket_mode", type=str, default="fixed_width", choices=["fixed_width", "equal_width"])
    p.add_argument("--method_min", type=float, default=-28.0)
    p.add_argument("--method_max", type=float, default=50.0)
    p.add_argument("--method_bin_width", type=float, default=1.0)
    p.add_argument("--method_bucket_num", type=int, default=80)

    p.add_argument("--method", type=str, default="plain", choices=sorted(METHODS))
    p.add_argument("--lds_kernel", type=str, default="gaussian", choices=["gaussian", "triang", "laplace"])
    p.add_argument("--lds_ks", type=int, default=5)
    p.add_argument("--lds_sigma", type=float, default=2.0)

    p.add_argument("--fds_kernel", type=str, default="gaussian", choices=["gaussian", "triang", "laplace"])
    p.add_argument("--fds_ks", type=int, default=5)
    p.add_argument("--fds_sigma", type=float, default=2.0)
    p.add_argument("--fds_bucket_start", type=int, default=0)
    p.add_argument("--fds_start_update", type=int, default=0)
    p.add_argument("--fds_start_smooth", type=int, default=1)
    p.add_argument("--fds_momentum", type=float, default=0.9, help="Set < 0 for cumulative averaging")

    p.add_argument("--hidden_dims", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    p.add_argument("--cold_threshold", type=float, default=0.0)
    p.add_argument("--hot_threshold", type=float, default=30.0)

    p.add_argument("--cpu", action="store_true")
    return p



def main() -> None:
    args = build_argparser().parse_args()
    artifacts = train(args)
    print("\n=== Best Validation Metrics ===")
    for k, v in artifacts.best_metrics.items():
        print(f"{k}: {v}")
    if artifacts.test_metrics is not None:
        print("\n=== Test Metrics (best checkpoint selected by val) ===")
        for k, v in artifacts.test_metrics.items():
            print(f"{k}: {v}")
    print(f"\nSaved run artifacts to: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
