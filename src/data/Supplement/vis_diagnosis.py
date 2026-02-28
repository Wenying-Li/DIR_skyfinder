# vis_diagnosis.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_missing_by_group(df: pd.DataFrame, feature_col: str, group_col: str = "CamId") -> pd.DataFrame:
    """
    按 group（默认 CamId）统计某个特征的缺失情况
    """
    grouped = df.groupby(group_col).agg(
        total_samples=(feature_col, "size"),
        missing_count=(feature_col, lambda x: x.isna().sum()),
        non_missing_count=(feature_col, lambda x: x.notna().sum()),
    ).reset_index()

    grouped["missing_rate"] = grouped["missing_count"] / grouped["total_samples"]
    grouped = grouped.sort_values(["missing_rate", "missing_count"], ascending=[False, False]).reset_index(drop=True)
    return grouped


def compare_target_by_missing(df: pd.DataFrame, feature_col: str, target_col: str = "TempM") -> pd.DataFrame:
    """
    比较 feature 缺失 vs 非缺失时，目标变量的分布差异
    """
    tmp = df[[feature_col, target_col]].copy()
    tmp["is_missing"] = tmp[feature_col].isna()

    stats = tmp.groupby("is_missing")[target_col].agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        p10=lambda x: np.percentile(x.dropna(), 10) if len(x.dropna()) > 0 else np.nan,
        median="median",
        p90=lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan,
        max="max",
    ).reset_index()

    stats["group"] = stats["is_missing"].map({True: "feature_missing", False: "feature_observed"})
    cols = ["group", "count", "mean", "std", "min", "p10", "median", "p90", "max"]
    return stats[cols]


def correlation_with_target(df: pd.DataFrame, feature_col: str, target_col: str = "TempM") -> pd.DataFrame:
    """
    计算 feature 与 target 的 Pearson / Spearman 相关（仅使用非缺失样本）
    """
    sub = df[[feature_col, target_col]].dropna().copy()

    if len(sub) < 3:
        return pd.DataFrame([{
            "feature": feature_col,
            "n_valid": len(sub),
            "pearson": np.nan,
            "spearman": np.nan
        }])

    pearson = sub[feature_col].corr(sub[target_col], method="pearson")
    spearman = sub[feature_col].corr(sub[target_col], method="spearman")

    return pd.DataFrame([{
        "feature": feature_col,
        "n_valid": len(sub),
        "pearson": pearson,
        "spearman": spearman
    }])


def feature_variation_summary(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """
    查看 feature 的有效值分布与变异性
    """
    x = df[feature_col].dropna().astype(float)

    if len(x) == 0:
        return pd.DataFrame([{
            "feature": feature_col,
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
            "n_unique": 0,
            "top_value_ratio": np.nan
        }])

    value_counts = x.value_counts(dropna=True)
    top_value_ratio = value_counts.iloc[0] / len(x)

    return pd.DataFrame([{
        "feature": feature_col,
        "n_valid": len(x),
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "p10": np.percentile(x, 10),
        "p25": np.percentile(x, 25),
        "median": np.percentile(x, 50),
        "p75": np.percentile(x, 75),
        "p90": np.percentile(x, 90),
        "max": x.max(),
        "n_unique": x.nunique(),
        "top_value_ratio": top_value_ratio
    }])


def binned_target_curve(df: pd.DataFrame, feature_col: str, target_col: str = "TempM", q: int = 10) -> pd.DataFrame:
    """
    将 feature 按分位数分桶，查看各桶 target 均值，帮助判断是否存在非线性趋势
    """
    sub = df[[feature_col, target_col]].dropna().copy()

    if len(sub) < q:
        return pd.DataFrame(columns=[
            "bin", "count", "feature_min", "feature_max", "feature_mean", "target_mean", "target_std"
        ])

    # 避免重复值过多导致 qcut 失败
    try:
        sub["bin"] = pd.qcut(sub[feature_col], q=q, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=[
            "bin", "count", "feature_min", "feature_max", "feature_mean", "target_mean", "target_std"
        ])

    grouped = sub.groupby("bin").agg(
        count=(feature_col, "size"),
        feature_min=(feature_col, "min"),
        feature_max=(feature_col, "max"),
        feature_mean=(feature_col, "mean"),
        target_mean=(target_col, "mean"),
        target_std=(target_col, "std"),
    ).reset_index()

    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def plot_missing_rate_by_cam(missing_df: pd.DataFrame, feature_col: str, save_path: str, top_k: int = 20):
    """
    绘制缺失率最高的前 top_k 个 camera
    """
    plot_df = missing_df.head(top_k).copy()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["CamId"].astype(str), plot_df["missing_rate"])
    plt.xticks(rotation=90)
    plt.ylabel("Missing Rate")
    plt.xlabel("CamId")
    plt.title(f"{feature_col} Missing Rate by CamId (Top {top_k})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_histogram(df: pd.DataFrame, feature_col: str, save_path: str, bins: int = 40):
    """
    feature 的直方图（仅非缺失）
    """
    x = df[feature_col].dropna().astype(float)

    if len(x) == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=bins)
    plt.xlabel(feature_col)
    plt.ylabel("Count")
    plt.title(f"Histogram of {feature_col}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_binned_curve(curve_df: pd.DataFrame, feature_col: str, target_col: str, save_path: str):
    """
    绘制按 feature 分桶后的 target 均值曲线
    """
    if len(curve_df) == 0:
        return

    x = np.arange(len(curve_df))
    y = curve_df["target_mean"].values

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    plt.xticks(x, [f"B{i+1}" for i in range(len(curve_df))], rotation=0)
    plt.xlabel(f"{feature_col} quantile bins")
    plt.ylabel(f"Mean {target_col}")
    plt.title(f"Binned relationship: {feature_col} vs {target_col}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_diagnosis(df: pd.DataFrame, feature_col: str, target_col: str, output_dir: str):
    """
    对单个 feature 做完整诊断
    """
    print("=" * 70)
    print(f"Diagnosing {feature_col}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1) 缺失模式（按 CamId）
    if "CamId" in df.columns:
        missing_by_cam = summarize_missing_by_group(df, feature_col, group_col="CamId")
        missing_by_cam.to_csv(os.path.join(output_dir, f"{feature_col}_missing_by_cam.csv"), index=False)
        print("\nTop 10 cameras by missing rate:")
        print(missing_by_cam.head(10).to_string(index=False))

        plot_missing_rate_by_cam(
            missing_by_cam,
            feature_col=feature_col,
            save_path=os.path.join(output_dir, f"{feature_col}_missing_by_cam_top20.png"),
            top_k=20
        )
    else:
        print("\n[Warning] CamId not found; skipping missing-by-camera analysis.")

    # 2) feature 缺失 vs 非缺失的目标分布
    target_compare = compare_target_by_missing(df, feature_col, target_col=target_col)
    target_compare.to_csv(os.path.join(output_dir, f"{feature_col}_target_by_missing.csv"), index=False)
    print("\nTarget distribution when feature is missing vs observed:")
    print(target_compare.to_string(index=False))

    # 3) 相关性
    corr_df = correlation_with_target(df, feature_col, target_col=target_col)
    corr_df.to_csv(os.path.join(output_dir, f"{feature_col}_correlation.csv"), index=False)
    print("\nCorrelation with target:")
    print(corr_df.to_string(index=False))

    # 4) 分布与变异性
    var_df = feature_variation_summary(df, feature_col)
    var_df.to_csv(os.path.join(output_dir, f"{feature_col}_variation.csv"), index=False)
    print("\nFeature variation summary:")
    print(var_df.to_string(index=False))

    # 5) 分桶趋势
    curve_df = binned_target_curve(df, feature_col, target_col=target_col, q=10)
    curve_df.to_csv(os.path.join(output_dir, f"{feature_col}_binned_curve.csv"), index=False)
    print("\nBinned target curve (quantile bins):")
    if len(curve_df) > 0:
        print(curve_df.to_string(index=False))
    else:
        print("[Empty or insufficient valid data]")

    # 6) 图
    plot_histogram(
        df,
        feature_col=feature_col,
        save_path=os.path.join(output_dir, f"{feature_col}_hist.png"),
        bins=40
    )
    plot_binned_curve(
        curve_df,
        feature_col=feature_col,
        target_col=target_col,
        save_path=os.path.join(output_dir, f"{feature_col}_binned_curve.png")
    )

    print(f"\nSaved outputs to: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose VisM / VisI usefulness before modeling.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to cleaned CSV")
    parser.add_argument("--target_col", type=str, default="TempM", help="Target column (default: TempM)")
    parser.add_argument("--output_dir", type=str, default="vis_diagnosis_outputs", help="Directory to save outputs")
    parser.add_argument(
        "--features",
        type=str,
        default="VisM,VisI",
        help="Comma-separated feature names to diagnose, default: VisM,VisI"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    features = [x.strip() for x in args.features.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    for feature_col in features:
        if feature_col not in df.columns:
            print(f"[Warning] {feature_col} not found in CSV, skipping.")
            continue

        feature_out_dir = os.path.join(args.output_dir, feature_col)
        run_diagnosis(df, feature_col=feature_col, target_col=args.target_col, output_dir=feature_out_dir)


if __name__ == "__main__":
    main()