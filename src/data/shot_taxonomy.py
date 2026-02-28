from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import numpy as np


BucketMode = Literal["fixed_width", "equal_width"]
ShotStrategy = Literal["quantile", "absolute"]


@dataclass
class BucketSpec:
    mode: BucketMode = "fixed_width"
    min_value: float = -28.0
    max_value: float = 50.0
    bin_width: float = 1.0
    bucket_num: int = 80

    def build_edges(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mode == "fixed_width":
            n_bins = int(np.ceil((self.max_value - self.min_value) / self.bin_width))
            edges = self.min_value + np.arange(n_bins + 1, dtype=np.float32) * self.bin_width
            if edges[-1] < self.max_value:
                edges = np.append(edges, np.float32(edges[-1] + self.bin_width))
            return edges.astype(np.float32)

        if self.mode == "equal_width":
            if values is None or len(values) == 0:
                raise ValueError("values are required when mode='equal_width'")
            y = np.asarray(values, dtype=np.float32)
            y_min = np.float32(np.nanmin(y) if self.min_value is None else self.min_value)
            y_max = np.float32(np.nanmax(y) if self.max_value is None else self.max_value)
            if y_max <= y_min:
                y_max = np.float32(y_min + 1.0)
            return np.linspace(y_min, y_max, int(self.bucket_num) + 1, dtype=np.float32)

        raise ValueError(f"Unsupported BucketMode: {self.mode}")


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    idx = np.digitize(vals, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(np.int64)


@dataclass
class ShotTaxonomyConfig:
    strategy: ShotStrategy = "quantile"
    few_quantile: float = 0.2
    many_quantile: float = 0.8
    few_threshold: int = 100
    many_threshold: int = 1000
    min_nonempty_bins_required: int = 5


@dataclass
class ShotTaxonomyResult:
    bucket_edges: np.ndarray
    bucket_counts: np.ndarray
    shot_map: Dict[int, str]
    few_threshold: int
    many_threshold: int


def build_shot_taxonomy(
    y_train: np.ndarray,
    bucket_spec: BucketSpec,
    taxonomy_cfg: ShotTaxonomyConfig,
) -> ShotTaxonomyResult:
    edges = bucket_spec.build_edges(y_train if bucket_spec.mode == "equal_width" else None)
    idx = assign_bins(y_train, edges)
    bucket_counts = np.bincount(idx, minlength=len(edges) - 1).astype(np.int64)

    nonempty = bucket_counts[bucket_counts > 0]
    if len(nonempty) < taxonomy_cfg.min_nonempty_bins_required:
        raise ValueError(
            f"Too few non-empty eval buckets ({len(nonempty)}). Adjust eval bucket specification."
        )

    if taxonomy_cfg.strategy == "quantile":
        few_thr = int(np.floor(np.quantile(nonempty, taxonomy_cfg.few_quantile)))
        many_thr = int(np.ceil(np.quantile(nonempty, taxonomy_cfg.many_quantile)))
        if many_thr <= few_thr:
            many_thr = few_thr + 1
    elif taxonomy_cfg.strategy == "absolute":
        few_thr = int(taxonomy_cfg.few_threshold)
        many_thr = int(taxonomy_cfg.many_threshold)
        if many_thr <= few_thr:
            raise ValueError("many_threshold must be > few_threshold for absolute shot taxonomy.")
    else:
        raise ValueError(f"Unsupported ShotStrategy: {taxonomy_cfg.strategy}")

    shot_map: Dict[int, str] = {}
    for b, c in enumerate(bucket_counts):
        if c == 0:
            shot_map[b] = "empty"
        elif c <= few_thr:
            shot_map[b] = "few"
        elif c >= many_thr:
            shot_map[b] = "many"
        else:
            shot_map[b] = "medium"

    return ShotTaxonomyResult(
        bucket_edges=edges,
        bucket_counts=bucket_counts,
        shot_map=shot_map,
        few_threshold=few_thr,
        many_threshold=many_thr,
    )


def summarize_shot_labels(bin_indices: Iterable[int], shot_map: Dict[int, str]) -> Dict[str, int]:
    out = {"few": 0, "medium": 0, "many": 0, "empty": 0}
    for b in bin_indices:
        out[shot_map.get(int(b), "empty")] += 1
    return out
