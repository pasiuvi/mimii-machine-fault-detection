"""
Evaluation metrics - one implementation used for EVERY model (baseline, ANN, GenAI).

PRIMARY METRIC: recall (true-positive rate) at a 5 % false-alarm rate  -> "TPR@FPR5"
    "If the engineer accepts that 1 in 20 healthy-machine checks is a false alarm,
     what fraction of real faults do we catch?"
Secondary: ROC-AUC (threshold-free ranking quality) and pAUC(FPR <= 0.1).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MAX_FPR = 0.05


def tpr_at_fpr(y_true, scores, max_fpr: float = MAX_FPR) -> float:
    """Highest recall achievable while keeping the false-alarm rate <= max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr[fpr <= max_fpr]))


def threshold_for_fpr(y_val, val_scores, max_fpr: float = MAX_FPR) -> float:
    """Pick the alarm threshold on the VALIDATION set (never on test)."""
    fpr, _, thr = roc_curve(y_val, val_scores)
    ok = np.where(fpr <= max_fpr)[0]
    return float(thr[ok[-1]])


def evaluate(y_true, scores) -> dict:
    return {
        "TPR@FPR5": round(tpr_at_fpr(y_true, scores), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, scores)), 4),
        "pAUC_0.1": round(float(roc_auc_score(y_true, scores, max_fpr=0.1)), 4),
    }
