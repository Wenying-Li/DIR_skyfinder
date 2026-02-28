# select_mi_features.py

import os
import argparse
import pandas as pd
import numpy as np


MISSING_SENTINELS = [-9999, -9999.0]


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    读取 CSV，并将 -9999 统一替换为 NaN。
    这里不做额外复杂清洗，假设输入已是清洗后或可直接分析的表。
    """
    df = pd.read_csv(csv_path)
    df = df.replace(MISSING_SENTINELS, np.nan)
    return df


def find_mi_pairs(columns):
    """
    找到所有以 M/I 结尾的成对特征。
    例如：
      PressureM / PressureI -> base = Pressure
      TempM / TempI -> base = Temp
    返回:
      pairs = {base_name: {"M": col_m, "I": col_i}}
    """
    pairs = {}

    col_set = set(columns)
    for col in columns:
        if len(col) < 2:
            continue
        suffix = col[-1]
        if suffix not in {"M", "I"}:
            continue

        base = col[:-1]
        if not base:
            continue

        m_col = f"{base}M"
        i_col = f"{base}I"

        if m_col in col_set and i_col in col_set:
            pairs[base] = {"M": m_col, "I": i_col}

    return pairs


def overall_missing_rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].isna().mean())


def cross_camera_missing_stats(df: pd.DataFrame, col: str, cam_col: str = "CamId"):
    """
    计算 cross-camera 缺失情况：
    1) mean_cam_missing_rate: 各 camera 缺失率的平均值
    2) max_cam_missing_rate: 各 camera 缺失率的最大值
    3) fully_missing_camera_count: 完全缺失（该 camera 下全是 NaN）的 camera 数
    """
    if cam_col not in df.columns:
        # 如果没有 CamId，则返回 NaN，后续只依赖全局缺失率
        return {
            "mean_cam_missing_rate": np.nan,
            "max_cam_missing_rate": np.nan,
            "fully_missing_camera_count": np.nan,
        }

    grouped = df.groupby(cam_col)[col].apply(lambda s: s.isna().mean())

    mean_cam_missing_rate = float(grouped.mean())
    max_cam_missing_rate = float(grouped.max())
    fully_missing_camera_count = int((grouped == 1.0).sum())

    return {
        "mean_cam_missing_rate": mean_cam_missing_rate,
        "max_cam_missing_rate": max_cam_missing_rate,
        "fully_missing_camera_count": fully_missing_camera_count,
    }


def choose_between_mi(
    df: pd.DataFrame,
    col_m: str,
    col_i: str,
    base_name: str,
    target_col: str,
    cam_col: str = "CamId",
    eps: float = 1e-12,
):
    """
    对一对 M/I 特征做二选一。

    规则：
    - 如果是 Temp，对应目标列优先保留，另一列直接删除
    - 其他情况：
      1) 全局缺失率更低者优先
      2) 若相同，比较平均 camera 缺失率
      3) 若相同，比较最大 camera 缺失率
      4) 若相同，比较完全缺失的 camera 数
      5) 若仍相同，默认保留 M
    """
    # 温度列特判：目标列必须保留，另一列去掉
    if base_name == "Temp":
        if target_col == col_m:
            keep_col, drop_col = col_m, col_i
            reason = "target temperature column is fixed"
        elif target_col == col_i:
            keep_col, drop_col = col_i, col_m
            reason = "target temperature column is fixed"
        else:
            # 理论上不应出现，但若 target 不是这两者之一，则退化为普通规则
            pass

        if base_name == "Temp" and target_col in {col_m, col_i}:
            m_overall = overall_missing_rate(df, col_m)
            i_overall = overall_missing_rate(df, col_i)
            m_cam = cross_camera_missing_stats(df, col_m, cam_col=cam_col)
            i_cam = cross_camera_missing_stats(df, col_i, cam_col=cam_col)

            return {
                "base_name": base_name,
                "m_col": col_m,
                "i_col": col_i,
                "keep_col": keep_col,
                "drop_col": drop_col,
                "reason": reason,
                "m_overall_missing_rate": m_overall,
                "i_overall_missing_rate": i_overall,
                "m_mean_cam_missing_rate": m_cam["mean_cam_missing_rate"],
                "i_mean_cam_missing_rate": i_cam["mean_cam_missing_rate"],
                "m_max_cam_missing_rate": m_cam["max_cam_missing_rate"],
                "i_max_cam_missing_rate": i_cam["max_cam_missing_rate"],
                "m_fully_missing_camera_count": m_cam["fully_missing_camera_count"],
                "i_fully_missing_camera_count": i_cam["fully_missing_camera_count"],
            }

    # 普通 M/I 对
    m_overall = overall_missing_rate(df, col_m)
    i_overall = overall_missing_rate(df, col_i)

    m_cam = cross_camera_missing_stats(df, col_m, cam_col=cam_col)
    i_cam = cross_camera_missing_stats(df, col_i, cam_col=cam_col)

    # Step 1: 全局缺失率
    if m_overall < i_overall - eps:
        keep_col, drop_col = col_m, col_i
        reason = "lower overall missing rate"
    elif i_overall < m_overall - eps:
        keep_col, drop_col = col_i, col_m
        reason = "lower overall missing rate"
    else:
        # Step 2: 平均 camera 缺失率
        m_mean_cam = m_cam["mean_cam_missing_rate"]
        i_mean_cam = i_cam["mean_cam_missing_rate"]

        if not np.isnan(m_mean_cam) and not np.isnan(i_mean_cam):
            if m_mean_cam < i_mean_cam - eps:
                keep_col, drop_col = col_m, col_i
                reason = "tie on overall missing; lower mean cross-camera missing rate"
            elif i_mean_cam < m_mean_cam - eps:
                keep_col, drop_col = col_i, col_m
                reason = "tie on overall missing; lower mean cross-camera missing rate"
            else:
                # Step 3: 最大 camera 缺失率
                m_max_cam = m_cam["max_cam_missing_rate"]
                i_max_cam = i_cam["max_cam_missing_rate"]

                if m_max_cam < i_max_cam - eps:
                    keep_col, drop_col = col_m, col_i
                    reason = "tie on overall/mean-camera missing; lower max cross-camera missing rate"
                elif i_max_cam < m_max_cam - eps:
                    keep_col, drop_col = col_i, col_m
                    reason = "tie on overall/mean-camera missing; lower max cross-camera missing rate"
                else:
                    # Step 4: 完全缺失 camera 数量
                    m_full = m_cam["fully_missing_camera_count"]
                    i_full = i_cam["fully_missing_camera_count"]

                    if m_full < i_full:
                        keep_col, drop_col = col_m, col_i
                        reason = "tie on missing rates; fewer fully-missing cameras"
                    elif i_full < m_full:
                        keep_col, drop_col = col_i, col_m
                        reason = "tie on missing rates; fewer fully-missing cameras"
                    else:
                        # Step 5: 默认优先 M
                        keep_col, drop_col = col_m, col_i
                        reason = "all tie; default prefer M"
        else:
            # 如果没有 CamId，则直接默认优先 M
            keep_col, drop_col = col_m, col_i
            reason = "tie on overall missing and no CamId available; default prefer M"

    return {
        "base_name": base_name,
        "m_col": col_m,
        "i_col": col_i,
        "keep_col": keep_col,
        "drop_col": drop_col,
        "reason": reason,
        "m_overall_missing_rate": m_overall,
        "i_overall_missing_rate": i_overall,
        "m_mean_cam_missing_rate": m_cam["mean_cam_missing_rate"],
        "i_mean_cam_missing_rate": i_cam["mean_cam_missing_rate"],
        "m_max_cam_missing_rate": m_cam["max_cam_missing_rate"],
        "i_max_cam_missing_rate": i_cam["max_cam_missing_rate"],
        "m_fully_missing_camera_count": m_cam["fully_missing_camera_count"],
        "i_fully_missing_camera_count": i_cam["fully_missing_camera_count"],
    }


def select_mi_columns(
    df: pd.DataFrame,
    target_col: str = "TempM",
    cam_col: str = "CamId"
):
    """
    对所有 M/I 特征做二选一，返回：
    - reduced_df: 筛选后的数据
    - selection_report_df: 每对特征的选择报告
    """
    pairs = find_mi_pairs(df.columns)
    report_rows = []
    cols_to_drop = set()

    for base_name, pair in sorted(pairs.items()):
        col_m = pair["M"]
        col_i = pair["I"]

        decision = choose_between_mi(
            df=df,
            col_m=col_m,
            col_i=col_i,
            base_name=base_name,
            target_col=target_col,
            cam_col=cam_col,
        )

        report_rows.append(decision)
        cols_to_drop.add(decision["drop_col"])

    reduced_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    selection_report_df = pd.DataFrame(report_rows)

    return reduced_df, selection_report_df


def main():
    parser = argparse.ArgumentParser(
        description="Select one feature from each M/I pair based on missingness and cross-camera missingness."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save reduced CSV")
    parser.add_argument(
        "--report_csv",
        type=str,
        default=None,
        help="Path to save M/I selection report; default = <output_csv stem>_mi_selection_report.csv"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="TempM",
        help="Target temperature column to keep; the other temp column will be dropped automatically"
    )
    parser.add_argument(
        "--cam_col",
        type=str,
        default="CamId",
        help="Camera ID column used for cross-camera missingness comparison"
    )

    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    target_col = args.target_col
    cam_col = args.cam_col

    if args.report_csv is None:
        base, ext = os.path.splitext(output_csv)
        report_csv = f"{base}_mi_selection_report.csv"
    else:
        report_csv = args.report_csv

    df = load_dataframe(input_csv)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in input CSV.")

    reduced_df, report_df = select_mi_columns(
        df=df,
        target_col=target_col,
        cam_col=cam_col
    )

    reduced_df.to_csv(output_csv, index=False)
    report_df.to_csv(report_csv, index=False)

    print("=" * 80)
    print("M/I Feature Selection Completed")
    print("=" * 80)
    print(f"Input CSV:   {input_csv}")
    print(f"Output CSV:  {output_csv}")
    print(f"Report CSV:  {report_csv}")
    print(f"Target kept: {target_col}")
    print("-" * 80)
    print(f"Original columns: {len(df.columns)}")
    print(f"Reduced columns:  {len(reduced_df.columns)}")
    print(f"Dropped columns:  {len(df.columns) - len(reduced_df.columns)}")
    print("-" * 80)

    if len(report_df) == 0:
        print("No M/I pairs found.")
    else:
        print("Selection summary:")
        display_cols = [
            "base_name", "m_col", "i_col",
            "keep_col", "drop_col", "reason"
        ]
        print(report_df[display_cols].to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()