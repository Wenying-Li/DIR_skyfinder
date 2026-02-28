# cross_camera_missing_analysis.py

import os
import argparse
import numpy as np
import pandas as pd


def compute_camera_overall_missing_stats(df: pd.DataFrame, camera_col: str = "CamId") -> pd.DataFrame:
    """
    按 camera 统计整体缺失情况。
    整体缺失率定义为：
        该 camera 下所有单元格中缺失值所占比例
    """
    rows = []

    feature_cols = [c for c in df.columns if c != camera_col]

    for cam_id, group in df.groupby(camera_col):
        n_rows = len(group)
        n_features = len(feature_cols)

        missing_count = group[feature_cols].isna().sum().sum()
        total_cells = n_rows * n_features
        overall_missing_rate = missing_count / total_cells if total_cells > 0 else np.nan

        rows.append({
            "CamId": cam_id,
            "num_samples": n_rows,
            "num_features": n_features,
            "missing_count_total": int(missing_count),
            "total_cells": int(total_cells),
            "overall_missing_rate": overall_missing_rate,
        })

    out = pd.DataFrame(rows).sort_values(
        by=["overall_missing_rate", "num_samples"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return out


def compute_camera_feature_missing_long(df: pd.DataFrame, camera_col: str = "CamId") -> pd.DataFrame:
    """
    输出长表：
        CamId | feature | num_samples | missing_count | missing_rate
    """
    rows = []
    feature_cols = [c for c in df.columns if c != camera_col]

    for cam_id, group in df.groupby(camera_col):
        n_rows = len(group)

        for feature in feature_cols:
            missing_count = group[feature].isna().sum()
            missing_rate = missing_count / n_rows if n_rows > 0 else np.nan

            rows.append({
                "CamId": cam_id,
                "feature": feature,
                "num_samples": n_rows,
                "missing_count": int(missing_count),
                "missing_rate": missing_rate,
                "dtype": str(df[feature].dtype),
            })

    out = pd.DataFrame(rows).sort_values(
        by=["feature", "missing_rate", "CamId"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    return out


def pivot_camera_feature_missing(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    将长表转为宽表：
        index = CamId
        columns = feature
        values = missing_rate
    """
    matrix_df = long_df.pivot(index="CamId", columns="feature", values="missing_rate")
    matrix_df = matrix_df.sort_index()
    return matrix_df


def compute_feature_missing_variability(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    分析每个特征在不同 camera 间缺失率的波动程度。
    这一步有助于识别“camera-sensitive”特征。

    输出列包括：
    - mean_missing_rate
    - std_missing_rate
    - min_missing_rate
    - max_missing_rate
    - range_missing_rate
    - num_cameras_all_present  (缺失率=0 的 camera 数)
    - num_cameras_all_missing  (缺失率=1 的 camera 数)
    """
    rows = []

    for feature in matrix_df.columns:
        vals = matrix_df[feature].dropna().values

        if len(vals) == 0:
            rows.append({
                "feature": feature,
                "mean_missing_rate": np.nan,
                "std_missing_rate": np.nan,
                "min_missing_rate": np.nan,
                "max_missing_rate": np.nan,
                "range_missing_rate": np.nan,
                "num_cameras_all_present": 0,
                "num_cameras_all_missing": 0,
                "num_cameras_observed": 0,
            })
            continue

        rows.append({
            "feature": feature,
            "mean_missing_rate": float(np.mean(vals)),
            "std_missing_rate": float(np.std(vals)),
            "min_missing_rate": float(np.min(vals)),
            "max_missing_rate": float(np.max(vals)),
            "range_missing_rate": float(np.max(vals) - np.min(vals)),
            "num_cameras_all_present": int(np.sum(vals == 0.0)),
            "num_cameras_all_missing": int(np.sum(vals == 1.0)),
            "num_cameras_observed": int(len(vals)),
        })

    out = pd.DataFrame(rows).sort_values(
        by=["std_missing_rate", "range_missing_rate", "mean_missing_rate"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return out


def main():
    parser = argparse.ArgumentParser(description="Cross-camera missing-rate analysis for cleaned SkyFinder CSV")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to cleaned CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output CSV files")
    parser.add_argument("--camera_col", type=str, default="CamId", help="Camera ID column name (default: CamId)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to print")

    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir
    camera_col = args.camera_col
    top_k = args.top_k

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    if camera_col not in df.columns:
        raise ValueError(f"Camera column '{camera_col}' not found in input CSV.")

    # 1) camera-level overall missing
    camera_overall_df = compute_camera_overall_missing_stats(df, camera_col=camera_col)

    # 2) camera x feature long table
    camera_feature_long_df = compute_camera_feature_missing_long(df, camera_col=camera_col)

    # 3) matrix
    camera_feature_matrix_df = pivot_camera_feature_missing(camera_feature_long_df)

    # 4) feature variability across cameras
    feature_variability_df = compute_feature_missing_variability(camera_feature_matrix_df)

    # Save
    camera_overall_path = os.path.join(output_dir, "camera_overall_missing_stats.csv")
    camera_feature_long_path = os.path.join(output_dir, "camera_feature_missing_rates.csv")
    camera_feature_matrix_path = os.path.join(output_dir, "camera_feature_missing_matrix.csv")
    feature_variability_path = os.path.join(output_dir, "feature_missing_variability_across_cameras.csv")

    camera_overall_df.to_csv(camera_overall_path, index=False)
    camera_feature_long_df.to_csv(camera_feature_long_path, index=False)
    camera_feature_matrix_df.to_csv(camera_feature_matrix_path)
    feature_variability_df.to_csv(feature_variability_path, index=False)

    # Print summary
    print("=" * 80)
    print("Cross-Camera Missing Analysis Completed")
    print("=" * 80)
    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print("-" * 80)
    print(f"Number of cameras: {df[camera_col].nunique()}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")
    print("-" * 80)

    print(f"[Top {top_k}] Cameras with highest overall missing rate:")
    print(camera_overall_df.head(top_k).to_string(index=False))
    print("-" * 80)

    print(f"[Top {top_k}] Features with largest cross-camera missing-rate variability:")
    print(feature_variability_df.head(top_k).to_string(index=False))
    print("-" * 80)

    # 额外给出几类可直接用于后续决策的提示
    high_var_features = feature_variability_df[
        feature_variability_df["std_missing_rate"] >= 0.30
    ]["feature"].tolist()

    all_missing_some_cameras = feature_variability_df[
        feature_variability_df["num_cameras_all_missing"] > 0
    ]["feature"].tolist()

    print("Features with very high cross-camera missing variability (std >= 0.30):")
    print(high_var_features[:top_k] if len(high_var_features) > 0 else "None")
    print("-" * 80)

    print("Features that are completely missing in at least one camera:")
    print(all_missing_some_cameras[:top_k] if len(all_missing_some_cameras) > 0 else "None")
    print("=" * 80)

    print("Saved files:")
    print(f"1. {camera_overall_path}")
    print(f"2. {camera_feature_long_path}")
    print(f"3. {camera_feature_matrix_path}")
    print(f"4. {feature_variability_path}")


if __name__ == "__main__":
    main()