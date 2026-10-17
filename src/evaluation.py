"""Shared evaluation helpers: predictions, per-type metrics, bootstrap CIs, operating points, latency."""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from src import config
from src.metrics import MAX_FPR, TYPES, evaluate, macro_tpr_at_fpr, threshold_for_fpr


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
    """'macro' (PRIMARY: mean over machine types, i.e. one threshold per type),
    'all' (pooled ranking, the Q1 definition, secondary) and each type separately."""
    res = {"all": evaluate(y, s)}
    for t in TYPES:
        k = mtype == t
        if k.any() and len(np.unique(y[k])) == 2:
            res[t] = evaluate(y[k], s[k])
    present = [t for t in TYPES if t in res]
    res["macro"] = {key: round(float(np.mean([res[t][key] for t in present])), 4)
                    for key in ["TPR@FPR5", "ROC_AUC", "pAUC_0.1"]}
    return res


def bootstrap_ci(y, s, mtype, fn=macro_tpr_at_fpr, n: int = 1000, seed: int = config.SEED):
    """95 % percentile bootstrap CI; resamples clips within each (label, machine type) cell."""
    rng = np.random.default_rng(seed)
    cells = [np.where((y == lab) & (mtype == t))[0] for lab in (0, 1) for t in np.unique(mtype)]
    cells = [c for c in cells if len(c)]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells])
        vals.append(fn(y[idx], s[idx], mtype[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def type_thresholds(y_val, s_val, t_val, max_fpr: float = MAX_FPR) -> dict:
    """One alarm threshold per machine type, chosen on VALIDATION at 5 % false alarms."""
    return {t: threshold_for_fpr(y_val[t_val == t], s_val[t_val == t], max_fpr) for t in TYPES}


def operating_points(y_val, s_val, t_val, y_test, s_test, t_test, max_fpr: float = MAX_FPR) -> dict:
    """Per-type threshold chosen on VALIDATION, applied unchanged to TEST (exactly what the app does)."""
    thr = type_thresholds(y_val, s_val, t_val, max_fpr)
    out = {}
    for t in TYPES:
        k = t_test == t
        pred, yt = s_test[k] >= thr[t], y_test[k]
        tp = int((pred & (yt == 1)).sum()); fn = int((~pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum()); tn = int((~pred & (yt == 0)).sum())
        out[t] = {"threshold": float(thr[t]), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                  "test_TPR_at_val_threshold": round(tp / max(tp + fn, 1), 4),
                  "test_FPR_at_val_threshold": round(fp / max(fp + tn, 1), 4)}
    return out


def time_it(fn, repeats: int = 20, warmup: int = 3) -> float:
    """Median wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
