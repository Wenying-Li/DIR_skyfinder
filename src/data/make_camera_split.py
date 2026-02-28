from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Make both `python src/data/make_camera_split.py` and `python -m src.data.make_camera_split` work.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] if len(_THIS.parents) > 1 else _THIS.parent
_ROOT = _THIS.parents[2] if len(_THIS.parents) > 2 else _THIS.parent
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.data.preprocess import load_dataframe  # noqa: E402
from src.utils.config_loader import get_target_column, load_yaml_config  # noqa: E402


@dataclass
class SplitPlan:
    train: List[str]
    val: List[str]
    test: List[str]


def _safe_qbin(s: pd.Series, q: int = 3) -> pd.Series:
    """Quantile bins with graceful fallback when unique values are limited."""
    x = pd.Series(s).astype(float)
    valid = x.dropna()
    if valid.empty:
        return pd.Series([0] * len(x), index=x.index, dtype=int)

    n_unique = valid.nunique()
    bins = int(max(1, min(q, n_unique)))
    if bins == 1:
        return pd.Series([0] * len(x), index=x.index, dtype=int)

    binned = pd.qcut(valid, q=bins, labels=False, duplicates="drop")
    out = pd.Series([0] * len(x), index=x.index, dtype=float)
    out.loc[valid.index] = binned.astype(float)
    out = out.fillna(0).astype(int)
    return out


def build_camera_profile(df: pd.DataFrame, cam_col: str, target_col: str) -> pd.DataFrame:
    if cam_col not in df.columns:
        raise ValueError(f"Camera column '{cam_col}' not found in dataframe")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    grouped = df.groupby(cam_col, dropna=False)
    profile = grouped[target_col].agg(["count", "mean", "std", "min", "max"]).rename(
        columns={
            "count": "n_samples",
            "mean": "temp_mean",
            "std": "temp_std",
            "min": "temp_min",
            "max": "temp_max",
        }
    )

    profile["cold_ratio"] = grouped[target_col].apply(lambda x: float(np.mean(x <= 0.0)))
    profile["hot_ratio"] = grouped[target_col].apply(lambda x: float(np.mean(x >= 30.0)))

    for c in ["Latitude", "Longitude", "Year"]:
        if c in df.columns:
            profile[c] = grouped[c].median()

    if "Month" in df.columns:
        profile["month_min"] = grouped["Month"].min()
        profile["month_max"] = grouped["Month"].max()
        profile["month_span"] = profile["month_max"] - profile["month_min"] + 1

    profile = profile.reset_index().rename(columns={cam_col: "CamId"})

    # Coarse stratification bins.
    profile["temp_bin"] = _safe_qbin(profile["temp_mean"], q=3)
    profile["size_bin"] = _safe_qbin(profile["n_samples"], q=3)
    if "Latitude" in profile.columns:
        profile["lat_bin"] = _safe_qbin(profile["Latitude"], q=3)
    else:
        profile["lat_bin"] = 0

    profile["stratum"] = (
        profile["temp_bin"].astype(str)
        + "_"
        + profile["size_bin"].astype(str)
        + "_"
        + profile["lat_bin"].astype(str)
    )

    return profile


def _allocate_group_counts(stratum_size: int, remaining: Dict[str, int], total_remaining: int) -> Dict[str, int]:
    """Allocate this stratum across splits using dynamic largest-remainder."""
    if stratum_size > total_remaining:
        raise ValueError("Stratum size exceeds total remaining cameras")

    groups = ["train", "val", "test"]
    raw = {g: (remaining[g] / total_remaining) * stratum_size for g in groups}
    alloc = {g: int(np.floor(raw[g])) for g in groups}

    # Respect capacity.
    for g in groups:
        alloc[g] = min(alloc[g], remaining[g])

    assigned = sum(alloc.values())
    leftovers = stratum_size - assigned

    if leftovers > 0:
        order = sorted(
            groups,
            key=lambda g: (raw[g] - np.floor(raw[g]), remaining[g] - alloc[g]),
            reverse=True,
        )
        idx = 0
        while leftovers > 0:
            g = order[idx % len(order)]
            if alloc[g] < remaining[g]:
                alloc[g] += 1
                leftovers -= 1
            idx += 1
            if idx > 1000:
                raise RuntimeError("Allocation loop did not converge")

    return alloc


def stratified_camera_split(
    profile: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
) -> SplitPlan:
    total = len(profile)
    if n_train + n_val + n_test != total:
        raise ValueError(
            f"Requested split sizes ({n_train}, {n_val}, {n_test}) do not sum to number of cameras ({total})"
        )

    rng = np.random.default_rng(seed)
    remaining = {"train": n_train, "val": n_val, "test": n_test}
    plan = {"train": [], "val": [], "test": []}

    strata = []
    for stratum, sub in profile.groupby("stratum", sort=False):
        cam_ids = sub["CamId"].astype(str).tolist()
        rng.shuffle(cam_ids)
        strata.append((stratum, cam_ids))

    # Large strata first stabilizes the split.
    strata.sort(key=lambda x: len(x[1]), reverse=True)

    total_remaining = total
    for _stratum, cam_ids in strata:
        alloc = _allocate_group_counts(len(cam_ids), remaining, total_remaining)
        start = 0
        for g in ["train", "val", "test"]:
            k = alloc[g]
            chosen = cam_ids[start : start + k]
            plan[g].extend(chosen)
            remaining[g] -= k
            start += k
        total_remaining -= len(cam_ids)

    if any(v != 0 for v in remaining.values()):
        raise RuntimeError(f"Split allocation failed, remaining quotas: {remaining}")

    # Sort for stable JSON outputs.
    return SplitPlan(
        train=sorted(plan["train"]),
        val=sorted(plan["val"]),
        test=sorted(plan["test"]),
    )


def summarize_split(profile: pd.DataFrame, cams: Sequence[str]) -> Dict[str, float]:
    sub = profile[profile["CamId"].astype(str).isin(list(cams))].copy()
    out: Dict[str, float] = {
        "num_cameras": int(len(sub)),
        "num_samples": int(sub["n_samples"].sum()),
        "temp_mean_mean": float(sub["temp_mean"].mean()),
        "temp_mean_std": float(sub["temp_mean"].std(ddof=0)),
        "temp_min_min": float(sub["temp_min"].min()),
        "temp_max_max": float(sub["temp_max"].max()),
        "cold_ratio_mean": float(sub["cold_ratio"].mean()),
        "hot_ratio_mean": float(sub["hot_ratio"].mean()),
        "n_samples_mean": float(sub["n_samples"].mean()),
        "n_samples_std": float(sub["n_samples"].std(ddof=0)),
    }
    if "Latitude" in sub.columns:
        out["latitude_mean"] = float(sub["Latitude"].mean())
        out["latitude_std"] = float(sub["Latitude"].std(ddof=0))
    if "Longitude" in sub.columns:
        out["longitude_mean"] = float(sub["Longitude"].mean())
        out["longitude_std"] = float(sub["Longitude"].std(ddof=0))
    if "month_span" in sub.columns:
        out["month_span_mean"] = float(sub["month_span"].mean())
    return out


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_outputs(
    out_dir: Path,
    profile: pd.DataFrame,
    split: SplitPlan,
    seed: int,
    source_csv: str,
    feature_config: str,
    target_col: str,
    cam_col: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    assign_df = pd.DataFrame(
        {
            "CamId": split.train + split.val + split.test,
            "split": ["train"] * len(split.train) + ["val"] * len(split.val) + ["test"] * len(split.test),
        }
    )
    assign_df.to_csv(out_dir / "camera_split_assignment.csv", index=False)

    profile_merge = profile.copy()
    profile_merge["CamId"] = profile_merge["CamId"].astype(str)
    profile_merge = profile_merge.merge(assign_df, on="CamId", how="left")
    profile_merge.to_csv(out_dir / "camera_profiles_with_split.csv", index=False)

    train_json = {
        "split": "train",
        "num_cameras": len(split.train),
        "cameras": split.train,
    }
    val_json = {
        "split": "val",
        "num_cameras": len(split.val),
        "cameras": split.val,
    }
    test_json = {
        "split": "test",
        "num_cameras": len(split.test),
        "cameras": split.test,
    }
    save_json(out_dir / "train_cameras.json", train_json)
    save_json(out_dir / "val_cameras.json", val_json)
    save_json(out_dir / "test_cameras.json", test_json)

    combined = {
        "seed": seed,
        "source_csv": source_csv,
        "feature_config": feature_config,
        "target_column": target_col,
        "group_column": cam_col,
        "split_type": "camera_disjoint_stratified_37_8_8",
        "counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "train_cameras": split.train,
        "val_cameras": split.val,
        "test_cameras": split.test,
        "summary": {
            "overall": summarize_split(profile, profile["CamId"].astype(str).tolist()),
            "train": summarize_split(profile, split.train),
            "val": summarize_split(profile, split.val),
            "test": summarize_split(profile, split.test),
        },
        "notes": [
            "Camera-level split with no CamId overlap between train/val/test.",
            "Split built by coarse stratification on camera-level temp_mean, n_samples, and Latitude (if available).",
            "Intended for fixed, repeatable cross-camera generalization experiments.",
        ],
    }
    save_json(out_dir / "camera_split_manifest.json", combined)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a fixed camera-disjoint 37/8/8 split for SkyFinder")
    p.add_argument("--csv_path", type=str, default=None, help="Path to cleaned CSV. If omitted, use dataset.cleaned_csv from feature config.")
    p.add_argument("--feature_config", type=str, required=True, help="Path to skyfinder_feature_config.yaml")
    p.add_argument("--output_dir", type=str, default="configs/splits/camera_split_37_8_8", help="Directory for generated JSON/CSV artifacts")
    p.add_argument("--cam_col", type=str, default="CamId")
    p.add_argument("--train_cameras", type=int, default=37)
    p.add_argument("--val_cameras", type=int, default=8)
    p.add_argument("--test_cameras", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    config = load_yaml_config(args.feature_config)
    csv_path = args.csv_path or config.get("dataset", {}).get("cleaned_csv")
    if not csv_path:
        raise ValueError("csv_path not provided and dataset.cleaned_csv missing in feature config")

    df = load_dataframe(csv_path)
    target_col = get_target_column(config)

    df = df[df[target_col].notna()].copy()
    profile = build_camera_profile(df, cam_col=args.cam_col, target_col=target_col)
    split = stratified_camera_split(
        profile=profile,
        n_train=args.train_cameras,
        n_val=args.val_cameras,
        n_test=args.test_cameras,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    save_outputs(
        out_dir=out_dir,
        profile=profile,
        split=split,
        seed=args.seed,
        source_csv=str(csv_path),
        feature_config=str(args.feature_config),
        target_col=target_col,
        cam_col=args.cam_col,
    )

    print("Created fixed camera split:")
    print(f"  train ({len(split.train)}): {split.train}")
    print(f"  val   ({len(split.val)}): {split.val}")
    print(f"  test  ({len(split.test)}): {split.test}")
    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
