"""
Evaluation metrics - one implementation used for EVERY model (trivial, baseline, ANN, GenAI).

PRIMARY METRIC (revised in Q3, before any test use): MACRO recall at a 5 % false-alarm rate
    = mean over machine types (pump, fan) of TPR@FPR5 computed within each type.
    The engineer always knows the machine type, so the alarm threshold is set
    PER MACHINE TYPE on the validation machine. Pooling pumps and fans into one
    ranking mixes two populations whose scores can be on different scales; Round 0
    showed this (e.g. pump 0.17 / fan 0.39 but pooled 0.07).
Secondary: pooled TPR@FPR5 (the Q1 definition), ROC-AUC and pAUC(FPR <= 0.1), per type and macro.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MAX_FPR = 0.05
TYPES = ("pump", "fan")


def tpr_at_fpr(y_true, scores, max_fpr: float = MAX_FPR) -> float:
    """Highest recall achievable while keeping the false-alarm rate <= max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr[fpr <= max_fpr]))


def threshold_for_fpr(y_val, val_scores, max_fpr: float = MAX_FPR) -> float:
    """Pick the alarm threshold on the VALIDATION set (never on test)."""
    fpr, _, thr = roc_curve(y_val, val_scores)
    ok = np.where(fpr <= max_fpr)[0]
    return float(min(thr[ok[-1]], 1e9))


def evaluate(y_true, scores) -> dict:
    return {
        "TPR@FPR5": round(tpr_at_fpr(y_true, scores), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, scores)), 4),
        "pAUC_0.1": round(float(roc_auc_score(y_true, scores, max_fpr=0.1)), 4),
    }


def macro_tpr_at_fpr(y, s, mtype, max_fpr: float = MAX_FPR) -> float:
    vals = [tpr_at_fpr(y[mtype == t], s[mtype == t], max_fpr) for t in TYPES
            if (mtype == t).any() and len(np.unique(y[mtype == t])) == 2]
    return float(np.mean(vals))
