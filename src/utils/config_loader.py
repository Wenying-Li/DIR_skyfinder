# src/utils/config_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
import pandas as pd


class ConfigError(Exception):
    """Raised when the YAML experiment config is invalid."""
    pass


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file into a Python dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError("Top-level YAML content must be a dictionary.")

    return config


def get_non_input_columns(config: Dict[str, Any]) -> List[str]:
    """
    Return the dataset.non_input_columns list.
    """
    try:
        non_input = config["dataset"]["non_input_columns"]
    except KeyError as e:
        raise ConfigError(f"Missing required config key: {e}")

    if not isinstance(non_input, list):
        raise ConfigError("dataset.non_input_columns must be a list.")

    return non_input


def get_feature_groups(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return the feature_groups mapping.
    """
    if "feature_groups" not in config:
        raise ConfigError("Missing 'feature_groups' in config.")
    feature_groups = config["feature_groups"]
    if not isinstance(feature_groups, dict):
        raise ConfigError("'feature_groups' must be a dictionary.")
    return feature_groups


def get_feature_sets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return the feature_sets mapping.
    """
    if "feature_sets" not in config:
        raise ConfigError("Missing 'feature_sets' in config.")
    feature_sets = config["feature_sets"]
    if not isinstance(feature_sets, dict):
        raise ConfigError("'feature_sets' must be a dictionary.")
    return feature_sets


def get_target_column(config: Dict[str, Any]) -> str:
    """
    Return the dataset.target_column.
    """
    try:
        target_col = config["dataset"]["target_column"]
    except KeyError as e:
        raise ConfigError(f"Missing required config key: {e}")

    if not isinstance(target_col, str):
        raise ConfigError("dataset.target_column must be a string.")

    return target_col


def expand_feature_set(
    config: Dict[str, Any],
    feature_set_name: str,
    remove_non_input: bool = True,
    deduplicate: bool = True
) -> List[str]:
    """
    Expand a named feature set into the final list of columns.

    Steps:
      1. Look up feature_sets[feature_set_name]
      2. Read include_groups
      3. Concatenate all columns from referenced feature groups
      4. Optionally remove dataset.non_input_columns
      5. Optionally deduplicate while preserving order

    Returns:
      List[str]: final feature columns
    """
    feature_sets = get_feature_sets(config)
    feature_groups = get_feature_groups(config)
    non_input_columns = get_non_input_columns(config)

    if feature_set_name not in feature_sets:
        raise ConfigError(f"Unknown feature set: {feature_set_name}")

    fs = feature_sets[feature_set_name]
    if "include_groups" not in fs:
        raise ConfigError(f"Feature set '{feature_set_name}' missing 'include_groups'.")

    include_groups = fs["include_groups"]
    if not isinstance(include_groups, list):
        raise ConfigError(f"feature_sets.{feature_set_name}.include_groups must be a list.")

    columns: List[str] = []

    for group_name in include_groups:
        if group_name not in feature_groups:
            raise ConfigError(
                f"Feature set '{feature_set_name}' references unknown feature group '{group_name}'."
            )

        group_cfg = feature_groups[group_name]
        if "columns" not in group_cfg:
            raise ConfigError(f"Feature group '{group_name}' missing 'columns'.")

        group_cols = group_cfg["columns"]
        if not isinstance(group_cols, list):
            raise ConfigError(f"feature_groups.{group_name}.columns must be a list.")

        columns.extend(group_cols)

    if remove_non_input:
        columns = [c for c in columns if c not in non_input_columns]

    if deduplicate:
        seen = set()
        deduped = []
        for c in columns:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        columns = deduped

    return columns


def validate_feature_set_against_dataframe(
    df: pd.DataFrame,
    config: Dict[str, Any],
    feature_set_name: str
) -> Tuple[List[str], List[str]]:
    """
    Compare the expanded feature set against actual dataframe columns.

    Returns:
      existing_cols: columns present in df
      missing_cols: columns absent from df
    """
    desired_cols = expand_feature_set(config, feature_set_name)

    existing_cols = [c for c in desired_cols if c in df.columns]
    missing_cols = [c for c in desired_cols if c not in df.columns]

    return existing_cols, missing_cols


def build_run_metadata(
    config: Dict[str, Any],
    feature_set_name: str,
    method_name: str,
    split_name: str
) -> Dict[str, Any]:
    """
    Build a lightweight run metadata dictionary for logging.
    """
    target_col = get_target_column(config)
    feature_cols = expand_feature_set(config, feature_set_name)

    metadata = {
        "project_name": config.get("project", {}).get("name", "unknown_project"),
        "task": config.get("project", {}).get("task", "unknown_task"),
        "target_column": target_col,
        "feature_set": feature_set_name,
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "method": method_name,
        "split": split_name,
    }
    return metadata


if __name__ == "__main__":
    # ===== Example Usage =====
    # Adjust paths as needed
    CONFIG_PATH = "configs/skyfinder_feature_config.yaml"
    CSV_PATH = "data/processed/skyfinder_cleaned.csv"

    config = load_yaml_config(CONFIG_PATH)
    df = pd.read_csv(CSV_PATH)

    feature_set_name = "S2_A_plus_B_core"

    feature_cols, missing_cols = validate_feature_set_against_dataframe(
        df=df,
        config=config,
        feature_set_name=feature_set_name
    )

    print("=" * 60)
    print(f"Feature set: {feature_set_name}")
    print(f"Found columns ({len(feature_cols)}):")
    print(feature_cols)
    print("-" * 60)
    print(f"Missing columns ({len(missing_cols)}):")
    print(missing_cols)
    print("=" * 60)

    run_meta = build_run_metadata(
        config=config,
        feature_set_name=feature_set_name,
        method_name="LDS",
        split_name="cross_camera"
    )
    print("Run metadata:")
    print(run_meta)