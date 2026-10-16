"""Shared evaluation helpers: predictions, per-machine metrics, bootstrap CIs, latency."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from src import config
from src.metrics import MAX_FPR, evaluate, threshold_for_fpr, tpr_at_fpr


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_logits(model, dataset, device, batch_size: int = 64) -> np.ndarray:
    """Raw logits for every clip (full 10-s clips, no augmentation)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for x, m, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(x.to(device, non_blocking=True), m.to(device))
        out.append(logit.float().cpu().numpy())
    return np.concatenate(out)


def predict_scores(model, dataset, device, batch_size: int = 64) -> np.ndarray:
    """Fault scores (sigmoid of the logit) for every clip."""
    return expit(predict_logits(model, dataset, device, batch_size))


def metrics_by_type(y: np.ndarray, s: np.ndarray, mtype: np.ndarray) -> dict:
    """Overall + per machine type metrics. mtype: array of 'pump'/'fan' strings."""
    res = {"all": evaluate(y, s)}
    for t in ["pump", "fan"]:
        k = mtype == t
        if k.any() and len(np.unique(y[k])) == 2:
            res[t] = evaluate(y[k], s[k])
    return res


def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn=tpr_at_fpr, n: int = 1000,
                 seed: int = config.SEED, stratify: np.ndarray | None = None) -> tuple[float, float]:
    """95 % percentile bootstrap CI, resampling clips within each label (and machine type)."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(list(zip(y, stratify if stratify is not None else np.zeros_like(y))))
    idx_by_group = [np.where(groups == g)[0] for g in groups.unique()]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in idx_by_group])
        vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def operating_point(y_val, s_val, y_test, s_test, max_fpr: float = MAX_FPR) -> dict:
    """Threshold chosen on VALIDATION, then applied unchanged to TEST (what the app will do)."""
    thr = threshold_for_fpr(y_val, s_val, max_fpr)
    pred = s_test >= thr
    tp = int(((pred == 1) & (y_test == 1)).sum()); fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum()); tn = int(((pred == 0) & (y_test == 0)).sum())
    return {"threshold": float(thr), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "test_TPR_at_val_threshold": tp / max(tp + fn, 1),
            "test_FPR_at_val_threshold": fp / max(fp + tn, 1)}


def time_it(fn, repeats: int = 20, warmup: int = 3) -> float:
    """Median wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
