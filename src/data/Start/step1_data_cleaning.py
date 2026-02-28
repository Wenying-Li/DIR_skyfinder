# data_cleaning.py

import os
import argparse
import pandas as pd
import numpy as np


# 统一视为缺失值的哨兵值
MISSING_SENTINELS = [-9999, -9999.0]

# 明确不做 numeric 转换的常见非数值列
# 这里按你当前任务需要，保留文本语义，避免误转
NON_NUMERIC_COLUMNS = {
    "Date",       # 日期字符串/序列化日期文本（当前不强制解析）
    "Timezone",   # 时区
    "WDirE",      # 风向文本，如 ESE
    "Conds",      # 天气描述，如 Clear
    "Icon",       # 天气图标标签
    "Metar",      # 原始气象文本
}

# 对于 object/string 列，若不在 NON_NUMERIC_COLUMNS 中，则尝试转 numeric
# 这样可以处理“本应是数值但被读成字符串”的情况


def clean_dataframe(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按当前实验要求清洗数据：
    1. -9999 -> NaN
    2. 非明确文本列，尝试转 numeric
    3. 删除目标变量缺失样本
    4. 输出缺失率统计表

    Returns:
        cleaned_df: 清洗后的 DataFrame
        missing_rate_df: 各特征缺失率统计表
    """

    # 1) 统一替换缺失哨兵值
    cleaned_df = df.replace(MISSING_SENTINELS, np.nan).copy()

    # 2) 仅对“非明确文本列”尝试转 numeric
    # 注意：不对已经是数值 dtype 的列做额外处理；
    # 对 object/string 且不在 NON_NUMERIC_COLUMNS 中的列，尝试转 numeric
    for col in cleaned_df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue

        col_dtype = cleaned_df[col].dtype

        # 只对 object/string 列尝试转换
        if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
            converted = pd.to_numeric(cleaned_df[col], errors="ignore")
            cleaned_df[col] = converted

    # 3) 检查目标列是否存在
    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input CSV.")

    # 4) 去除目标变量缺失样本
    cleaned_df = cleaned_df[cleaned_df[target_col].notna()].copy()

    # 5) 计算各特征缺失率统计表（基于清洗后数据）
    missing_count = cleaned_df.isna().sum()
    total_rows = len(cleaned_df)

    missing_rate_df = pd.DataFrame({
        "feature": cleaned_df.columns,
        "missing_count": [missing_count[col] for col in cleaned_df.columns],
        "missing_rate": [
            (missing_count[col] / total_rows) if total_rows > 0 else np.nan
            for col in cleaned_df.columns
        ],
        "dtype": [str(cleaned_df[col].dtype) for col in cleaned_df.columns],
    })

    missing_rate_df = missing_rate_df.sort_values(
        by=["missing_rate", "missing_count", "feature"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return cleaned_df, missing_rate_df


def main():
    parser = argparse.ArgumentParser(description="Clean SkyFinder-style CSV for DIR experiments.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save cleaned CSV")
    parser.add_argument(
        "--missing_stats_csv",
        type=str,
        default=None,
        help="Path to save missing-rate statistics CSV; default = <output_csv stem>_missing_stats.csv"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="TempM",
        help="Target column used to drop missing-label rows (default: TempM)"
    )

    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    target_col = args.target_col

    if args.missing_stats_csv is None:
        base, ext = os.path.splitext(output_csv)
        missing_stats_csv = f"{base}_missing_stats.csv"
    else:
        missing_stats_csv = args.missing_stats_csv

    # 读取原始数据
    df = pd.read_csv(input_csv)

    # 清洗
    cleaned_df, missing_rate_df = clean_dataframe(df, target_col=target_col)

    # 保存清洗后的数据
    cleaned_df.to_csv(output_csv, index=False)

    # 保存缺失率统计表
    missing_rate_df.to_csv(missing_stats_csv, index=False)

    # 输出结果
    print("=" * 60)
    print("Data Cleaning Completed")
    print("=" * 60)
    print(f"Input file:  {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"Missing stats: {missing_stats_csv}")
    print(f"Target column: {target_col}")
    print("-" * 60)
    print(f"Cleaned sample count: {len(cleaned_df)}")
    print("-" * 60)
    print("Top 20 features by missing rate:")
    print(missing_rate_df.head(20).to_string(index=False))
    print("=" * 60)

    # 额外说明：CamId 保留在 cleaned_df 中，用于 split / 分析
    if "CamId" in cleaned_df.columns:
        print("Note: 'CamId' is preserved in the cleaned CSV for split/analysis.")
    else:
        print("Warning: 'CamId' column not found in the cleaned CSV.")


if __name__ == "__main__":
    main()