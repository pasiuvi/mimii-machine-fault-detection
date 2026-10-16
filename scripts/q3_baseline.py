"""
Question 3a – trivial benchmark + tuned classical baseline (validation only).

  E000  trivial benchmark : logistic regression on clip loudness (RMS dB) only
  B001  classical baseline: MFCC statistics + best of 30 configurations (see src/baseline.py)

Outputs
  experiments/baseline_search.csv      every configuration with validation scores and fit time
  models/baseline.joblib               best baseline (scaler + model) refit on train
  models/trivial.joblib                loudness-only model
  data/processed/baseline_features.npz MFCC features for train/val/test (test used ONLY in final eval)
  rows E000 and B001 in experiments/experiment_log.csv

Run:  python -m scripts.q3_baseline        (about 5-15 minutes, CPU)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.baseline import mfcc_features, scores, search_space  # noqa: E402
from src.eda_features import get_summary  # noqa: E402
from src.evaluation import metrics_by_type  # noqa: E402
from src.train import git_commit, write_log_row  # noqa: E402

FEAT = config.PROCESSED_DIR / "baseline_features.npz"
MODELS = config.ROOT / "models"


def load_features() -> dict:
    if FEAT.exists():
        return dict(np.load(FEAT, allow_pickle=True))
    print("Computing MFCC features for train / val / test (one-off) ...")
    out = {}
    for split in ["train", "val", "test"]:
        X, y, t = mfcc_features(split)
        out[f"X_{split}"], out[f"y_{split}"], out[f"t_{split}"] = X, y, t
        print(f"  {split}: {X.shape}")
    np.savez(FEAT, **out)
    return out


def loudness(split: str) -> np.ndarray:
    feats = get_summary()
    rms = dict(zip(feats["path"], feats["rms_db"]))
    idx = pd.read_csv(config.FEATURE_DIR / f"{split}_index.csv")
    return np.array([rms[p] for p in idx.path], dtype=np.float32)[:, None]


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    d = load_features()
    Xtr, ytr, Xva, yva, tva = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["t_val"]

    # ---------------- trivial benchmark ----------------
    Ltr, Lva = loudness("train"), loudness("val")
    t0 = time.time()
    triv = LogisticRegression().fit(Ltr, ytr)
    tfit = time.time() - t0
    mt = metrics_by_type(yva, triv.predict_proba(Lva)[:, 1], tva)
    joblib.dump(triv, MODELS / "trivial.joblib")
    print(f"Trivial (loudness only): val TPR@FPR5 = {mt['all']['TPR@FPR5']:.3f}, AUC = {mt['all']['ROC_AUC']:.3f}")
    write_log_row({"exp_id": "E000", "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "group": "benchmark",
                   "model_family": "Trivial", "what_changed": "Logistic regression on clip loudness (RMS dB) only",
                   "key_hyperparameters": "feature=rms_db", "seed": config.SEED, "epochs_run": "", "best_epoch": "",
                   "val_TPR@FPR5": round(mt["all"]["TPR@FPR5"], 4), "val_ROC_AUC": round(mt["all"]["ROC_AUC"], 4),
                   "val_pAUC_0.1": round(mt["all"]["pAUC_0.1"], 4),
                   "val_TPR@FPR5_pump": round(mt["pump"]["TPR@FPR5"], 4),
                   "val_TPR@FPR5_fan": round(mt["fan"]["TPR@FPR5"], 4),
                   "train_time_min": round(tfit / 60, 4), "hypothesis": "Reference: how far loudness alone gets",
                   "outcome": "completed", "what_i_learned": "", "git_commit": git_commit()})

    # ---------------- classical baseline search ----------------
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    rows = []
    space = list(search_space())
    print(f"\nSearching {len(space)} classical configurations (selection on VALIDATION) ...")
    for k, (family, params, make) in enumerate(space, 1):
        t0 = time.time()
        m = make().fit(Ztr, ytr)
        fit_s = time.time() - t0
        mv = metrics_by_type(yva, scores(m, Zva), tva)
        rows.append({"config": k, "family": family, "params": json.dumps(params),
                     "val_TPR@FPR5": mv["all"]["TPR@FPR5"], "val_ROC_AUC": mv["all"]["ROC_AUC"],
                     "val_pAUC_0.1": mv["all"]["pAUC_0.1"], "val_TPR@FPR5_pump": mv["pump"]["TPR@FPR5"],
                     "val_TPR@FPR5_fan": mv["fan"]["TPR@FPR5"], "fit_time_s": round(fit_s, 2)})
        print(f"  [{k:2d}/{len(space)}] {family:<12} {json.dumps(params):<60} "
              f"val TPR@5% {mv['all']['TPR@FPR5']:.3f}  AUC {mv['all']['ROC_AUC']:.3f}  ({fit_s:.1f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(config.ROOT / "experiments" / "baseline_search.csv", index=False)

    best = res.sort_values(["val_TPR@FPR5", "val_pAUC_0.1"], ascending=False).iloc[0]
    family, params = best.family, json.loads(best.params)
    make = [mk for f, p, mk in space if f == family and p == params][0]
    t0 = time.time()
    model = make().fit(Ztr, ytr)
    fit_s = time.time() - t0
    joblib.dump({"scaler": scaler, "model": model, "family": family, "params": params}, MODELS / "baseline.joblib")

    per_family = res.groupby("family")["val_TPR@FPR5"].max().round(3).to_dict()
    print(f"\nBest per family (val TPR@FPR5): {per_family}")
    print(f"SELECTED baseline: {family} {params}  val TPR@FPR5 = {best['val_TPR@FPR5']:.3f}")
    write_log_row({"exp_id": "B001", "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "group": "baseline",
                   "model_family": "Classical (MFCC stats)",
                   "what_changed": f"Best of {len(space)} configs over 4 families; best={family}",
                   "key_hyperparameters": json.dumps(params), "seed": config.SEED, "epochs_run": "", "best_epoch": "",
                   "val_TPR@FPR5": round(best["val_TPR@FPR5"], 4), "val_ROC_AUC": round(best["val_ROC_AUC"], 4),
                   "val_pAUC_0.1": round(best["val_pAUC_0.1"], 4),
                   "val_TPR@FPR5_pump": round(best["val_TPR@FPR5_pump"], 4),
                   "val_TPR@FPR5_fan": round(best["val_TPR@FPR5_fan"], 4),
                   "train_time_min": round(fit_s / 60, 3),
                   "hypothesis": "Tuned classical model with comparable search effort (30 configs vs ~22 ANN runs)",
                   "outcome": f"best per family: {per_family}", "what_i_learned": "", "git_commit": git_commit()})
    print("Saved models/baseline.joblib, models/trivial.joblib, experiments/baseline_search.csv")


if __name__ == "__main__":
    main()
