from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# LDS
# -----------------------------


def get_lds_kernel_window(kernel: str = "gaussian", ks: int = 5, sigma: float = 2.0) -> np.ndarray:
    if ks % 2 == 0:
        raise ValueError("ks must be odd")
    half = ks // 2
    x = np.arange(-half, half + 1, dtype=np.float32)

    kernel = kernel.lower()
    if kernel == "gaussian":
        win = np.exp(-(x ** 2) / (2 * (sigma ** 2))).astype(np.float32)
    elif kernel == "triang":
        win = (half + 1 - np.abs(x)).astype(np.float32)
    elif kernel == "laplace":
        win = np.exp(-np.abs(x) / sigma).astype(np.float32)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    win = win / np.max(win)
    return win.astype(np.float32)


def _transform_counts_for_reweight(raw_counts: np.ndarray, reweight: str) -> np.ndarray:
    reweight = reweight.lower()
    if reweight == "none":
        return raw_counts.astype(np.float32)
    if reweight == "sqrt_inv":
        return np.sqrt(np.clip(raw_counts, 1.0, None)).astype(np.float32)
    if reweight == "inverse":
        return np.clip(raw_counts, 5.0, 1000.0).astype(np.float32)
    raise ValueError(f"Unsupported reweight: {reweight}")


def prepare_sample_weights(
    bucket_idx: np.ndarray,
    bucket_num: int,
    reweight: str,
    lds: bool,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 2.0,
):
    reweight = reweight.lower()
    if reweight == "none":
        return np.ones(len(bucket_idx), dtype=np.float32), np.zeros(bucket_num, dtype=np.float32)
    if lds and reweight == "none":
        raise ValueError("LDS requires reweight != 'none'")

    raw_counts = np.bincount(bucket_idx, minlength=bucket_num).astype(np.float32)
    transformed = _transform_counts_for_reweight(raw_counts, reweight)
    effective = transformed.copy()

    if lds:
        kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        pad = len(kernel) // 2
        padded = np.pad(transformed, (pad, pad), mode="constant", constant_values=0.0)
        smoothed = np.zeros_like(transformed)
        for i in range(bucket_num):
            smoothed[i] = np.sum(padded[i : i + len(kernel)] * kernel)
        effective = smoothed.astype(np.float32)

    per_sample = np.clip(effective[bucket_idx], 1e-8, None)
    weights = (1.0 / per_sample).astype(np.float32)
    weights *= len(weights) / np.sum(weights)
    return weights, effective.astype(np.float32)


# -----------------------------
# FDS
# -----------------------------


def calibrate_mean_var(
    matrix: torch.Tensor,
    m1: torch.Tensor,
    v1: torch.Tensor,
    m2: torch.Tensor,
    v2: torch.Tensor,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> torch.Tensor:
    v1_safe = torch.where(v1 > 0, v1, torch.ones_like(v1))
    factor = torch.clamp(v2 / v1_safe, clip_min, clip_max)
    out = (matrix - m1) * torch.sqrt(factor) + m2
    zero_mask = (v1 <= 0)
    if zero_mask.any():
        out = torch.where(zero_mask.unsqueeze(0), matrix, out)
    return out


class FDS(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        bucket_num: int,
        bucket_start: int = 0,
        start_update: int = 0,
        start_smooth: int = 1,
        kernel: str = "gaussian",
        ks: int = 5,
        sigma: float = 2.0,
        momentum: Optional[float] = 0.9,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.bucket_num = int(bucket_num)
        self.bucket_start = int(bucket_start)
        self.start_update = int(start_update)
        self.start_smooth = int(start_smooth)
        self.momentum = momentum
        self.ks = int(ks)

        n_buckets = self.bucket_num - self.bucket_start
        if n_buckets <= 0:
            raise ValueError("bucket_num must be > bucket_start")

        self.register_buffer("running_mean", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("running_var", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("running_mean_last_epoch", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("running_var_last_epoch", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("smoothed_mean_last_epoch", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("smoothed_var_last_epoch", torch.zeros(n_buckets, feature_dim))
        self.register_buffer("num_samples_tracked", torch.zeros(n_buckets))
        self.register_buffer("epoch", torch.tensor([-1], dtype=torch.long))

        kw = get_lds_kernel_window(kernel=kernel, ks=ks, sigma=sigma)
        kw = kw / np.sum(kw)
        self.register_buffer("kernel_window", torch.tensor(kw, dtype=torch.float32).view(1, 1, -1))

    def update_last_epoch_stats(self, epoch: int) -> None:
        if epoch != int(self.epoch.item()) + 1:
            return
        self.running_mean_last_epoch.copy_(self.running_mean)
        self.running_var_last_epoch.copy_(self.running_var)
        if self.running_mean_last_epoch.shape[0] <= 1:
            self.smoothed_mean_last_epoch.copy_(self.running_mean_last_epoch)
            self.smoothed_var_last_epoch.copy_(self.running_var_last_epoch)
            self.epoch.fill_(epoch)
            return

        pad = self.ks // 2
        mean_in = self.running_mean_last_epoch.t().unsqueeze(0)
        var_in = self.running_var_last_epoch.t().unsqueeze(0)
        mean_pad = F.pad(mean_in, (pad, pad), mode="reflect")
        var_pad = F.pad(var_in, (pad, pad), mode="reflect")
        weight = self.kernel_window.expand(self.feature_dim, 1, self.ks)
        mean_sm = F.conv1d(mean_pad, weight, groups=self.feature_dim)
        var_sm = F.conv1d(var_pad, weight, groups=self.feature_dim)
        self.smoothed_mean_last_epoch.copy_(mean_sm.squeeze(0).t())
        self.smoothed_var_last_epoch.copy_(var_sm.squeeze(0).t())
        self.epoch.fill_(epoch)

    @torch.no_grad()
    def update_running_stats(self, features: torch.Tensor, buckets: torch.Tensor, epoch: int) -> None:
        if epoch < self.start_update:
            return
        for bucket in buckets.unique(sorted=True):
            b = int(bucket.item())
            if b < self.bucket_start or b >= self.bucket_num:
                continue
            rel = b - self.bucket_start
            mask = buckets == b
            feat_b = features[mask]
            if feat_b.numel() == 0:
                continue
            mean_b = feat_b.mean(dim=0)
            var_b = feat_b.var(dim=0, unbiased=False)
            curr_n = float(feat_b.shape[0])

            if epoch == self.start_update:
                factor = 0.0
            elif self.momentum is None:
                total = float(self.num_samples_tracked[rel].item() + curr_n)
                factor = 1.0 - curr_n / max(total, 1.0)
            else:
                factor = float(self.momentum)

            self.running_mean[rel] = factor * self.running_mean[rel] + (1.0 - factor) * mean_b
            self.running_var[rel] = factor * self.running_var[rel] + (1.0 - factor) * var_b
            self.num_samples_tracked[rel] += curr_n

    def smooth(self, features: torch.Tensor, buckets: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch < self.start_smooth:
            return features
        out = features.clone()
        for bucket in buckets.unique(sorted=True):
            b = int(bucket.item())
            if b < self.bucket_start or b >= self.bucket_num:
                continue
            rel = b - self.bucket_start
            mask = buckets == b
            if not mask.any():
                continue
            out[mask] = calibrate_mean_var(
                out[mask],
                self.running_mean_last_epoch[rel],
                self.running_var_last_epoch[rel],
                self.smoothed_mean_last_epoch[rel],
                self.smoothed_var_last_epoch[rel],
            )
        return out


# -----------------------------
# Model
# -----------------------------


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, use_fds: bool, fds_kwargs: dict):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.use_fds = bool(use_fds)
        self.fds = FDS(feature_dim=prev, **fds_kwargs) if self.use_fds else None

    def forward(self, x: torch.Tensor, buckets: Optional[torch.Tensor] = None, epoch: int = 0):
        feat = self.backbone(x)
        raw_feat = feat
        if self.use_fds and self.training and buckets is not None:
            feat = self.fds.smooth(feat, buckets, epoch)
        pred = self.head(feat)
        return pred, raw_feat


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(weight * (pred - target) ** 2)
