from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


from src.utils.config_loader import expand_feature_set, get_feature_groups


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.replace([-9999, -9999.0], np.nan)


CYCLICAL_PERIODS = {
    "Month": 12.0,
    "Hour": 24.0,
    "Min": 60.0,
}


def maybe_cycle_encode(
    df: pd.DataFrame,
    feature_groups: Dict[str, Dict],
    selected_groups: List[str],
    enable: bool,
) -> pd.DataFrame:
    if not enable:
        return df

    cyclical_candidates = []
    for g in selected_groups:
        group_cfg = feature_groups.get(g, {})
        cyc = group_cfg.get("preprocessing", {}).get("cyclical_candidates", [])
        if isinstance(cyc, list):
            cyclical_candidates.extend(cyc)

    cyclical_candidates = [
        c for c in dict.fromkeys(cyclical_candidates)
        if c in df.columns and c in CYCLICAL_PERIODS
    ]
    out = df.copy()
    for col in cyclical_candidates:
        period = CYCLICAL_PERIODS[col]
        angle = 2.0 * np.pi * out[col].astype(float) / period
        out[f"{col}_sin"] = np.sin(angle)
        out[f"{col}_cos"] = np.cos(angle)
        out.drop(columns=[col], inplace=True)
    return out


def resolve_feature_columns(
    config: dict,
    feature_set: str,
    df: pd.DataFrame,
    cyclical_time: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feature_sets = config["feature_sets"]
    include_groups = list(feature_sets[feature_set]["include_groups"])
    base_cols = expand_feature_set(config, feature_set)
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from dataframe for feature_set={feature_set}: {missing}")

    feat_df = df[base_cols].copy()
    feat_df = maybe_cycle_encode(feat_df, get_feature_groups(config), include_groups, cyclical_time)
    return feat_df, list(feat_df.columns), include_groups


def split_indices(
    df: pd.DataFrame,
    split_mode: str,
    val_ratio: float,
    seed: int,
    group_col: str = "CamId",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible 2-way split.
    """
    idx = np.arange(len(df))
    if split_mode == "random":
        tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)
        return tr, va
    if split_mode == "cross_camera":
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found for cross_camera split")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        groups = df[group_col].values
        tr, va = next(gss.split(idx, groups=groups))
        return tr, va
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _normalize_camera_values(values: Sequence) -> List[str]:
    return [str(v) for v in values]


def _load_camera_json(path: str | Path) -> List[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "cameras" not in obj:
            raise ValueError(f"Camera JSON {path} must be either a list or a dict with key 'cameras'")
        cams = obj["cameras"]
    elif isinstance(obj, list):
        cams = obj
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")
    return _normalize_camera_values(cams)


def load_fixed_camera_split(
    camera_split_manifest: Optional[str] = None,
    train_cameras_json: Optional[str] = None,
    val_cameras_json: Optional[str] = None,
    test_cameras_json: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Supports:
    1) a manifest JSON containing train_cameras / val_cameras / test_cameras
    2) three separate JSON files (each either a list or {"cameras": [...]})
    """
    meta: dict = {}
    if camera_split_manifest:
        meta = json.loads(Path(camera_split_manifest).read_text(encoding="utf-8"))
        missing = [k for k in ("train_cameras", "val_cameras", "test_cameras") if k not in meta]
        if missing:
            raise ValueError(
                f"camera_split_manifest missing keys: {missing}. "
                f"Expected train_cameras / val_cameras / test_cameras."
            )
        train_cams = _normalize_camera_values(meta["train_cameras"])
        val_cams = _normalize_camera_values(meta["val_cameras"])
        test_cams = _normalize_camera_values(meta["test_cameras"])
        return train_cams, val_cams, test_cams, meta

    if not train_cameras_json or not val_cameras_json:
        raise ValueError(
            "For split_mode='fixed_camera_json', provide either "
            "--camera_split_manifest OR both --train_cameras_json and --val_cameras_json "
            "(optionally --test_cameras_json)."
        )
    train_cams = _load_camera_json(train_cameras_json)
    val_cams = _load_camera_json(val_cameras_json)
    test_cams = _load_camera_json(test_cameras_json) if test_cameras_json else []
    return train_cams, val_cams, test_cams, meta


def split_indices_three_way(
    df: pd.DataFrame,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_col: str = "CamId",
    camera_split_manifest: Optional[str] = None,
    train_cameras_json: Optional[str] = None,
    val_cameras_json: Optional[str] = None,
    test_cameras_json: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    3-way split used by stage-1 training.

    Returns:
        train_idx, val_idx, test_idx, split_meta
    """
    idx = np.arange(len(df))
    split_meta: dict = {"split_mode": split_mode}

    if split_mode == "random":
        if test_ratio < 0 or val_ratio <= 0 or (test_ratio + val_ratio) >= 1.0:
            raise ValueError("For random split, require val_ratio>0 and val_ratio+test_ratio<1.")
        if test_ratio > 0:
            trainval_idx, test_idx = train_test_split(
                idx, test_size=test_ratio, random_state=seed, shuffle=True
            )
        else:
            trainval_idx, test_idx = idx, np.asarray([], dtype=int)

        inner_val_ratio = val_ratio / (1.0 - test_ratio) if test_ratio > 0 else val_ratio
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=inner_val_ratio, random_state=seed, shuffle=True
        )
        split_meta["counts"] = {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        }
        return train_idx, val_idx, test_idx, split_meta

    if split_mode == "cross_camera":
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found for cross_camera split")
        groups = df[group_col].astype(str).values

        if test_ratio > 0:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            trainval_idx, test_idx = next(gss_test.split(idx, groups=groups))
        else:
            trainval_idx, test_idx = idx, np.asarray([], dtype=int)

        trainval_groups = groups[trainval_idx]
        inner_val_ratio = val_ratio / (1.0 - test_ratio) if test_ratio > 0 else val_ratio
        gss_val = GroupShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=seed)
        rel_train_idx, rel_val_idx = next(gss_val.split(np.arange(len(trainval_idx)), groups=trainval_groups))
        train_idx = trainval_idx[rel_train_idx]
        val_idx = trainval_idx[rel_val_idx]

        split_meta["train_cameras"] = sorted(pd.unique(groups[train_idx]).tolist())
        split_meta["val_cameras"] = sorted(pd.unique(groups[val_idx]).tolist())
        split_meta["test_cameras"] = sorted(pd.unique(groups[test_idx]).tolist()) if len(test_idx) else []
        return train_idx, val_idx, test_idx, split_meta

    if split_mode == "fixed_camera_json":
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found for fixed_camera_json split")

        train_cams, val_cams, test_cams, manifest_meta = load_fixed_camera_split(
            camera_split_manifest=camera_split_manifest,
            train_cameras_json=train_cameras_json,
            val_cameras_json=val_cameras_json,
            test_cameras_json=test_cameras_json,
        )
        groups = df[group_col].astype(str).values

        train_set, val_set, test_set = set(train_cams), set(val_cams), set(test_cams)
        if train_set & val_set:
            raise ValueError(f"Train/val cameras overlap: {sorted(train_set & val_set)}")
        if train_set & test_set:
            raise ValueError(f"Train/test cameras overlap: {sorted(train_set & test_set)}")
        if val_set & test_set:
            raise ValueError(f"Val/test cameras overlap: {sorted(val_set & test_set)}")

        train_mask = np.isin(groups, list(train_set))
        val_mask = np.isin(groups, list(val_set))
        test_mask = np.isin(groups, list(test_set)) if test_set else np.zeros(len(df), dtype=bool)

        covered = train_mask | val_mask | test_mask
        if not np.all(covered):
            missing_cams = sorted(pd.unique(groups[~covered]).tolist())
            raise ValueError(
                "Some dataframe cameras are not covered by the provided fixed split JSON/manifest: "
                f"{missing_cams}"
            )

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        split_meta.update(manifest_meta)
        split_meta["train_cameras"] = sorted(train_set)
        split_meta["val_cameras"] = sorted(val_set)
        split_meta["test_cameras"] = sorted(test_set)
        split_meta["counts"] = {
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "train_cameras": int(len(train_set)),
            "val_cameras": int(len(val_set)),
            "test_cameras": int(len(test_set)),
        }
        return train_idx, val_idx, test_idx, split_meta

    raise ValueError(f"Unsupported split_mode: {split_mode}")


@dataclass
class FeatureTransformerArtifacts:
    transformer: ColumnTransformer
    output_feature_names: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]


def build_feature_transformer(train_feat: pd.DataFrame) -> FeatureTransformerArtifacts:
    numeric_cols = list(train_feat.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_cols = [c for c in train_feat.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot_encoder()),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    transformer.fit(train_feat)

    feature_names: List[str] = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        cat_encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(cat_encoder.get_feature_names_out(categorical_cols)))

    return FeatureTransformerArtifacts(
        transformer=transformer,
        output_feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def transform_features(
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    artifacts: FeatureTransformerArtifacts,
    test_feat: Optional[pd.DataFrame] = None,
):
    X_train = artifacts.transformer.transform(train_feat).astype(np.float32)
    X_val = artifacts.transformer.transform(val_feat).astype(np.float32)
    if test_feat is None:
        return X_train, X_val
    X_test = artifacts.transformer.transform(test_feat).astype(np.float32)
    return X_train, X_val, X_test
