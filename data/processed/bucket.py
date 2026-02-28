import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ShotTaxonomyConfig:
    eval_min: float = -28.0
    eval_max: float = 50.0
    bin_width: float = 1.0
    low_quantile: float = 0.2   # bottom 20% buckets => few
    high_quantile: float = 0.8  # top 20% buckets => many
    min_nonempty_bins_required: int = 5


@dataclass
class ShotTaxonomyResult:
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    nonempty_bin_indices: np.ndarray
    few_threshold: int
    many_threshold: int
    bin_to_shot: Dict[int, str]
    summary: Dict[str, float]


def build_fixed_bin_edges(eval_min: float, eval_max: float, bin_width: float) -> np.ndarray:
    """
    Fixed-width bin edges, inclusive of eval_min and covering eval_max.
    """
    n_bins = int(np.ceil((eval_max - eval_min) / bin_width))
    edges = eval_min + np.arange(n_bins + 1) * bin_width
    # Make sure last edge covers eval_max
    if edges[-1] < eval_max:
        edges = np.append(edges, edges[-1] + bin_width)
    return edges


def assign_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Map values to bin indices.
    Values outside range are clamped to boundary bins.
    """
    idx = np.digitize(values, bin_edges) - 1
    idx = np.clip(idx, 0, len(bin_edges) - 2)
    return idx


def fit_shot_taxonomy_from_train(
    y_train: np.ndarray,
    config: ShotTaxonomyConfig
) -> ShotTaxonomyResult:
    """
    Fit shot taxonomy from training labels only.
    1) Use fixed-width bins
    2) Count train samples per bin
    3) Determine few/many thresholds by bucket-count quantiles over NON-EMPTY bins
    """
    y_train = np.asarray(y_train, dtype=float)
    bin_edges = build_fixed_bin_edges(config.eval_min, config.eval_max, config.bin_width)
    train_bin_idx = assign_bins(y_train, bin_edges)

    n_bins = len(bin_edges) - 1
    bin_counts = np.bincount(train_bin_idx, minlength=n_bins)

    nonempty_bin_indices = np.where(bin_counts > 0)[0]
    nonempty_counts = bin_counts[nonempty_bin_indices]

    if len(nonempty_counts) < config.min_nonempty_bins_required:
        raise ValueError(
            f"Too few non-empty bins ({len(nonempty_counts)}). "
            f"Check eval_min/eval_max/bin_width."
        )

    # Quantile thresholds over non-empty bucket counts
    few_thr = int(np.floor(np.quantile(nonempty_counts, config.low_quantile)))
    many_thr = int(np.ceil(np.quantile(nonempty_counts, config.high_quantile)))

    # Avoid degenerate overlap
    if many_thr <= few_thr:
        many_thr = few_thr + 1

    bin_to_shot = {}
    for b in range(n_bins):
        c = int(bin_counts[b])
        if c == 0:
            # Empty bins are not used to define train taxonomy, but keep label for completeness
            bin_to_shot[b] = "empty"
        elif c <= few_thr:
            bin_to_shot[b] = "few"
        elif c >= many_thr:
            bin_to_shot[b] = "many"
        else:
            bin_to_shot[b] = "medium"

    summary = {
        "num_bins": int(n_bins),
        "num_nonempty_bins": int(len(nonempty_bin_indices)),
        "few_threshold": int(few_thr),
        "many_threshold": int(many_thr),
        "low_quantile": float(config.low_quantile),
        "high_quantile": float(config.high_quantile),
        "eval_min": float(config.eval_min),
        "eval_max": float(config.eval_max),
        "bin_width": float(config.bin_width),
    }

    return ShotTaxonomyResult(
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        nonempty_bin_indices=nonempty_bin_indices,
        few_threshold=few_thr,
        many_threshold=many_thr,
        bin_to_shot=bin_to_shot,
        summary=summary,
    )


def assign_sample_shots(
    y: np.ndarray,
    shot_result: ShotTaxonomyResult
) -> np.ndarray:
    """
    Assign each sample into few/medium/many/empty according to the TRAIN-fitted taxonomy.
    """
    y = np.asarray(y, dtype=float)
    bin_idx = assign_bins(y, shot_result.bin_edges)
    shot_labels = np.array([shot_result.bin_to_shot[int(b)] for b in bin_idx], dtype=object)
    return shot_labels


def summarize_sample_shots(shot_labels: np.ndarray) -> Dict[str, int]:
    """
    Count samples in each shot group.
    """
    shot_labels = np.asarray(shot_labels, dtype=object)
    uniq, counts = np.unique(shot_labels, return_counts=True)
    out = {str(k): int(v) for k, v in zip(uniq, counts)}
    for key in ["few", "medium", "many", "empty"]:
        out.setdefault(key, 0)
    return out


def export_bin_table(shot_result: ShotTaxonomyResult) -> pd.DataFrame:
    """
    Export per-bin info for inspection / logging.
    """
    rows = []
    edges = shot_result.bin_edges
    for b in range(len(edges) - 1):
        rows.append({
            "bin_index": b,
            "left": edges[b],
            "right": edges[b + 1],
            "train_count": int(shot_result.bin_counts[b]),
            "shot": shot_result.bin_to_shot[b],
        })
    return pd.DataFrame(rows)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example: replace this with your actual y_train / y_val
    rng = np.random.default_rng(42)
    y_train = rng.normal(loc=16, scale=10, size=70000)
    y_train = np.clip(y_train, -27.2, 50.0)

    y_val = rng.normal(loc=16, scale=10, size=20000)
    y_val = np.clip(y_val, -27.2, 50.0)

    cfg = ShotTaxonomyConfig(
        eval_min=-28.0,
        eval_max=50.0,
        bin_width=1.0,
        low_quantile=0.2,
        high_quantile=0.8,
    )

    shot_result = fit_shot_taxonomy_from_train(y_train, cfg)

    print("=== Taxonomy Summary ===")
    for k, v in shot_result.summary.items():
        print(f"{k}: {v}")

    val_shots = assign_sample_shots(y_val, shot_result)
    print("\n=== Val Sample Shot Counts ===")
    print(summarize_sample_shots(val_shots))

    bin_table = export_bin_table(shot_result)
    print("\n=== First 10 bins ===")
    print(bin_table.head(10))