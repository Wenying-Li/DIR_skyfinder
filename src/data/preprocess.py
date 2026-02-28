from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def maybe_cycle_encode(df: pd.DataFrame, feature_groups: Dict[str, Dict], selected_groups: List[str], enable: bool) -> pd.DataFrame:
    if not enable:
        return df

    cyclical_candidates = []
    for g in selected_groups:
        group_cfg = feature_groups.get(g, {})
        cyc = group_cfg.get("preprocessing", {}).get("cyclical_candidates", [])
        if isinstance(cyc, list):
            cyclical_candidates.extend(cyc)

    cyclical_candidates = [c for c in dict.fromkeys(cyclical_candidates) if c in df.columns and c in CYCLICAL_PERIODS]
    out = df.copy()
    for col in cyclical_candidates:
        period = CYCLICAL_PERIODS[col]
        angle = 2.0 * np.pi * out[col].astype(float) / period
        out[f"{col}_sin"] = np.sin(angle)
        out[f"{col}_cos"] = np.cos(angle)
        out.drop(columns=[col], inplace=True)
    return out


def resolve_feature_columns(config: dict, feature_set: str, df: pd.DataFrame, cyclical_time: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feature_sets = config["feature_sets"]
    include_groups = list(feature_sets[feature_set]["include_groups"])
    base_cols = expand_feature_set(config, feature_set)
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from dataframe for feature_set={feature_set}: {missing}")

    feat_df = df[base_cols].copy()
    feat_df = maybe_cycle_encode(feat_df, get_feature_groups(config), include_groups, cyclical_time)
    return feat_df, list(feat_df.columns), include_groups


def split_indices(df: pd.DataFrame, split_mode: str, val_ratio: float, seed: int, group_col: str = "CamId") -> Tuple[np.ndarray, np.ndarray]:
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


def transform_features(train_feat: pd.DataFrame, val_feat: pd.DataFrame, artifacts: FeatureTransformerArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    X_train = artifacts.transformer.transform(train_feat).astype(np.float32)
    X_val = artifacts.transformer.transform(val_feat).astype(np.float32)
    return X_train, X_val
