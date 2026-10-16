"""
Question 3a – the ONE-TIME final evaluation on the held-out TEST machines.

Rules (brief section 6):
  * every model and every threshold was chosen on VALIDATION before this runs
  * the test set is used here only, once; a lock file records when it happened
  * re-running needs --force and is written to the experiment log as a re-evaluation

What it does
  1. scores TEST (and VALIDATION, for thresholds) with: random reference, trivial (loudness),
     tuned classical baseline, initial ANN (E001), final ANN (3 seeds)
  2. primary metric with 95 % bootstrap CI, ROC-AUC, pAUC, per machine type
  3. operating point: threshold chosen on validation at 5 % FPR -> real test TPR/FPR
  4. training time and inference latency per 10-s clip (feature extraction + model, CPU and GPU)
  5. error analysis of the deployed final model (by machine type, loudness, score, overlap with baseline)
  6. saves the deployed model for the web app (models/final_model.pt + models/final_model.json)

Outputs: reports/q3_comparison.csv/.md, reports/q3_test_results.json, reports/q3_error_analysis.md,
         reports/q3_hardest_errors.csv, reports/figures/q3_roc_test.png, q3_score_hist.png, q3_errors_loudness.png

Run:  python -m scripts.q3_final_evaluate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.audio_utils import load_audio, rms_db  # noqa: E402
from src.baseline import scores as base_scores  # noqa: E402
from src.dataset import MimiiSpecDataset  # noqa: E402
from src.eda_features import get_summary  # noqa: E402
from src.evaluation import (bootstrap_ci, get_device, metrics_by_type, operating_point,  # noqa: E402
                            predict_scores, time_it)
from src.features import logmel128  # noqa: E402
from src.model import FaultCNN  # noqa: E402
from src.train import RUNS_DIR, git_commit, write_log_row  # noqa: E402

R, FIG, MODELS = config.REPORTS_DIR, config.FIGURES_DIR, config.ROOT / "models"
LOCK = R / ".test_evaluated.json"
FINAL_JSON = config.ROOT / "experiments" / "final_config.json"


def load_cnn(exp_id: str, device) -> FaultCNN:
    ck = torch.load(MODELS / f"{exp_id}.pt", map_location=device, weights_only=False)
    m = FaultCNN(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run the test evaluation (logged)")
    a = ap.parse_args()
    if LOCK.exists() and not a.force:
        info = json.loads(LOCK.read_text())
        raise SystemExit(f"Test set was already evaluated on {info['date']} (commit {info['commit']}).\n"
                         "The brief allows ONE final evaluation. Use --force only if you must, it will be logged.")
    if not FINAL_JSON.exists():
        raise SystemExit("Run first:  python -m scripts.q3_run_experiments --final")

    device = get_device()
    sel = json.loads(FINAL_JSON.read_text())
    fid = sel["selected_from"]
    seeds_ids = [i for i in [fid, f"{fid}_s43", f"{fid}_s44"] if (MODELS / f"{i}.pt").exists()]
    val_ds, test_ds = MimiiSpecDataset("val"), MimiiSpecDataset("test")
    yv, yt = val_ds.labels.astype(int), test_ds.labels.astype(int)
    tv, tt = val_ds.index.machine_type.to_numpy(), test_ds.index.machine_type.to_numpy()

    S = {}  # name -> (val scores, test scores)
    # --- trivial + baseline
    feats = get_summary()
    rms = dict(zip(feats["path"], feats["rms_db"]))
    Lv = np.array([rms[p] for p in val_ds.index.path])[:, None]
    Lt = np.array([rms[p] for p in test_ds.index.path])[:, None]
    triv = joblib.load(MODELS / "trivial.joblib")
    S["Trivial: loudness only"] = (triv.predict_proba(Lv)[:, 1], triv.predict_proba(Lt)[:, 1])
    bl = joblib.load(MODELS / "baseline.joblib")
    bf = np.load(config.PROCESSED_DIR / "baseline_features.npz", allow_pickle=True)
    S[f"Classical baseline ({bl['family']})"] = (base_scores(bl["model"], bl["scaler"].transform(bf["X_val"])),
                                                 base_scores(bl["model"], bl["scaler"].transform(bf["X_test"])))
    # --- ANNs
    S["Initial ANN (E001)"] = tuple(predict_scores(load_cnn("E001", device), ds, device) for ds in (val_ds, test_ds))
    seed_scores = {i: tuple(predict_scores(load_cnn(i, device), ds, device) for ds in (val_ds, test_ds))
                   for i in seeds_ids}

    # deployed final model = seed with the best VALIDATION metric
    val_tpr = {i: metrics_by_type(yv, sv, tv)["all"]["TPR@FPR5"] for i, (sv, _) in seed_scores.items()}
    deployed = max(val_tpr, key=val_tpr.get)
    S[f"Final ANN ({deployed}, deployed)"] = seed_scores[deployed]

    summaries = {p.parent.name: json.loads(p.read_text()) for p in RUNS_DIR.glob("*/summary.json")}
    log = pd.read_csv(config.ROOT / "experiments" / "experiment_log.csv").set_index("exp_id")
    train_min = {"Trivial: loudness only": float(log.loc["E000", "train_time_min"]),
                 f"Classical baseline ({bl['family']})": float(log.loc["B001", "train_time_min"]),
                 "Initial ANN (E001)": summaries["E001"]["train_time_min"],
                 f"Final ANN ({deployed}, deployed)": summaries[deployed]["train_time_min"]}

    # --- latency per 10-s clip (median of repeated runs)
    path = test_ds.index.path.iloc[0]
    t_logmel = time_it(lambda: logmel128(path), repeats=10)
    t_rms = time_it(lambda: rms_db(load_audio(config.ROOT / path)), repeats=10)
    x0, m0, _ = test_ds[0]
    xb, mb = x0.unsqueeze(0), m0.unsqueeze(0)
    cnn_cpu = load_cnn(deployed, torch.device("cpu"))
    t_cnn_cpu = time_it(lambda: cnn_cpu(xb, mb))
    t_cnn_gpu = None
    if device.type == "cuda":
        cnn_gpu = load_cnn(deployed, device); xg, mg = xb.to(device), mb.to(device)

        def _g():
            with torch.no_grad():
                cnn_gpu(xg, mg); torch.cuda.synchronize()
        t_cnn_gpu = time_it(_g)
    zrow = bl["scaler"].transform(bf["X_test"][:1])
    from src.baseline import mfcc_features  # noqa: F401  (feature time ~ logmel + small mfcc step)
    import librosa
    S0 = np.asarray(np.load(config.FEATURE_DIR / "test.npy", mmap_mode="r")[0], dtype=np.float32)
    t_mfcc = time_it(lambda: librosa.feature.delta(librosa.feature.mfcc(S=S0, n_mfcc=20)), repeats=10)
    t_base = time_it(lambda: base_scores(bl["model"], zrow))
    latency = {"Trivial: loudness only": t_rms + time_it(lambda: triv.predict_proba(Lt[:1])),
               f"Classical baseline ({bl['family']})": t_logmel + t_mfcc + t_base,
               "Initial ANN (E001)": t_logmel + t_cnn_cpu,
               f"Final ANN ({deployed}, deployed)": t_logmel + t_cnn_cpu}

    # --- comparison table
    rows = [{"Model": "Random scores (reference)", "Test TPR@FPR5": 0.05, "95% CI": "-", "Test ROC-AUC": 0.5,
             "Test pAUC(0.1)": 0.5, "Pump TPR@FPR5": 0.05, "Fan TPR@FPR5": 0.05,
             "Val TPR@FPR5": 0.05, "Training time (min)": 0, "Latency CPU (ms/clip)": 0}]
    results = {}
    for name, (sv, st) in S.items():
        mt = metrics_by_type(yt, st, tt); mv = metrics_by_type(yv, sv, tv)
        lo, hi = bootstrap_ci(yt, st, stratify=tt)
        op = operating_point(yv, sv, yt, st)
        results[name] = {"test": mt, "val": mv, "ci95": [lo, hi], "operating_point": op}
        rows.append({"Model": name, "Test TPR@FPR5": mt["all"]["TPR@FPR5"], "95% CI": f"{lo:.3f}–{hi:.3f}",
                     "Test ROC-AUC": mt["all"]["ROC_AUC"], "Test pAUC(0.1)": mt["all"]["pAUC_0.1"],
                     "Pump TPR@FPR5": mt["pump"]["TPR@FPR5"], "Fan TPR@FPR5": mt["fan"]["TPR@FPR5"],
                     "Val TPR@FPR5": mv["all"]["TPR@FPR5"], "Training time (min)": round(train_min[name], 2),
                     "Latency CPU (ms/clip)": round(latency[name], 1)})
    seed_test = [metrics_by_type(yt, st, tt)["all"]["TPR@FPR5"] for _, st in seed_scores.values()]
    rows.append({"Model": f"Final ANN, mean ± sd over {len(seed_test)} seeds",
                 "Test TPR@FPR5": float(np.mean(seed_test)), "95% CI": f"sd {np.std(seed_test, ddof=1):.3f}"
                 if len(seed_test) > 1 else "-",
                 "Test ROC-AUC": float(np.mean([metrics_by_type(yt, st, tt)["all"]["ROC_AUC"] for _, st in seed_scores.values()])),
                 "Test pAUC(0.1)": np.nan, "Pump TPR@FPR5": np.nan, "Fan TPR@FPR5": np.nan,
                 "Val TPR@FPR5": float(np.mean(list(val_tpr.values()))), "Training time (min)": np.nan,
                 "Latency CPU (ms/clip)": np.nan})
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(R / "q3_comparison.csv", index=False)
    (R / "q3_comparison.md").write_text(comp.to_markdown(index=False), encoding="utf-8")
    print(comp.to_string(index=False))

    # --- deployable artefact for Q4
    fv, ft = seed_scores[deployed]
    op = operating_point(yv, fv, yt, ft)
    shutil.copy(MODELS / f"{deployed}.pt", MODELS / "final_model.pt")
    (MODELS / "final_model.json").write_text(json.dumps({
        "source_run": deployed, "threshold_val_fpr5": op["threshold"], "cfg": summaries[deployed]["cfg"],
        "norm_stats": "data/processed/norm_stats.npz", "machine_types": config.MACHINE_TYPE_INDEX,
        "note": "score = sigmoid(logit); a ranking score, not a calibrated probability"}, indent=2))

    # --- error analysis of the deployed final model
    error_analysis(yt, ft, tt, test_ds, fv, yv, op, S, rms, bl["family"])

    out = {"date": datetime.now().isoformat(timespec="seconds"), "commit": git_commit(), "deployed": deployed,
           "selected_from": fid, "seed_runs": seeds_ids, "results": results,
           "latency_ms": {"logmel_feature": t_logmel, "cnn_cpu": t_cnn_cpu, "cnn_gpu": t_cnn_gpu,
                          "mfcc_extra": t_mfcc, "baseline_model": t_base}}
    (R / "q3_test_results.json").write_text(json.dumps(out, indent=2, default=float))
    LOCK.write_text(json.dumps({"date": out["date"], "commit": out["commit"], "forced": a.force}))
    write_log_row({"exp_id": "TEST" + ("_RERUN" if a.force else ""), "date": out["date"][:16], "group": "final-test",
                   "model_family": "all", "what_changed": "One-time evaluation on the held-out test machines",
                   "key_hyperparameters": f"deployed={deployed}", "seed": "", "epochs_run": "", "best_epoch": "",
                   "val_TPR@FPR5": "", "val_ROC_AUC": "", "val_pAUC_0.1": "", "val_TPR@FPR5_pump": "",
                   "val_TPR@FPR5_fan": "", "train_time_min": "", "hypothesis": "see reports/Q2_prediction.md",
                   "outcome": f"test TPR@FPR5 final={results[f'Final ANN ({deployed}, deployed)']['test']['all']['TPR@FPR5']:.3f}",
                   "what_i_learned": "", "git_commit": out["commit"]})
    print(f"\nDeployed model: {deployed} -> models/final_model.pt (threshold {op['threshold']:.4f})")
    print("Saved reports/q3_comparison.*, q3_test_results.json, q3_error_analysis.md, figures/q3_*.png")


def error_analysis(yt, st, tt, test_ds, sv, yv, op, S, rms, bl_family):
    thr = op["threshold"]
    pred = (st >= thr).astype(int)
    df = test_ds.index.copy()
    df["score"], df["pred"], df["label"] = st, pred, yt
    df["rms_db"] = [rms[p] for p in df.path]
    df["error"] = np.where((df.pred == 1) & (df.label == 0), "FP", np.where((df.pred == 0) & (df.label == 1), "FN", "ok"))
    base_name = [k for k in S if k.startswith("Classical")][0]
    sv_b, st_b = S[base_name]
    from src.metrics import threshold_for_fpr
    df["baseline_pred"] = (st_b >= threshold_for_fpr(yv, sv_b)).astype(int)

    lines = ["# Error analysis – deployed final ANN on the test machines\n",
             f"Threshold chosen on validation for 5 % FPR: **{thr:.4f}**  ->  on test: "
             f"TPR {op['test_TPR_at_val_threshold']:.3f}, FPR {op['test_FPR_at_val_threshold']:.3f}\n",
             "## 1. Confusion by machine type (at the validation threshold)"]
    rows = []
    for t in ["pump", "fan"]:
        d = df[df.machine_type == t]
        tp = int(((d.pred == 1) & (d.label == 1)).sum()); fn = int(((d.pred == 0) & (d.label == 1)).sum())
        fp = int(((d.pred == 1) & (d.label == 0)).sum()); tn = int(((d.pred == 0) & (d.label == 0)).sum())
        rows.append({"machine": f"{t} (test id_06)", "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                     "recall": round(tp / max(tp + fn, 1), 3), "false-alarm rate": round(fp / max(fp + tn, 1), 3)})
    lines.append(pd.DataFrame(rows).to_markdown(index=False))

    # loudness relative to the machine's normal clips
    lines.append("\n## 2. Errors by loudness (relative to the median normal clip of the same machine)")
    df["rel_db"] = df.rms_db - df.groupby("machine_type").rms_db.transform(
        lambda s: np.median(s[df.loc[s.index, "label"] == 0]))
    ab = df[df.label == 1]
    rows = []
    for t in ["pump", "fan"]:
        for name, k in [("quieter than normal", ab.rel_db < 0), ("louder than normal", ab.rel_db >= 0)]:
            d = ab[(ab.machine_type == t) & k]
            if len(d):
                rows.append({"machine": t, "abnormal clips": name, "n": len(d),
                             "recall (final ANN)": round(d.pred.mean(), 3),
                             "recall (baseline)": round(d.baseline_pred.mean(), 3)})
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    q = pd.qcut(df.rms_db, 4, labels=["Q1 quietest", "Q2", "Q3", "Q4 loudest"])
    tab = df.groupby(q, observed=True).apply(lambda d: pd.Series({
        "n": len(d), "abnormal": int(d.label.sum()),
        "recall": round(d[d.label == 1].pred.mean(), 3) if d.label.sum() else np.nan,
        "false-alarm rate": round(d[d.label == 0].pred.mean(), 3) if (d.label == 0).sum() else np.nan}))
    lines.append("\nBy loudness quartile (all test clips):\n" + tab.to_markdown())

    # overlap with baseline
    lines.append("\n## 3. Overlap of missed faults with the classical baseline")
    ab = df[df.label == 1]
    ov = pd.crosstab(ab.pred.map({1: "ANN detects", 0: "ANN misses"}),
                     ab.baseline_pred.map({1: "baseline detects", 0: "baseline misses"}))
    lines.append(ov.to_markdown())

    # hardest errors
    hard = pd.concat([df[df.error == "FN"].nsmallest(10, "score"), df[df.error == "FP"].nlargest(10, "score")])
    hard[["path", "machine_type", "label", "score", "error", "rms_db", "rel_db"]].to_csv(R / "q3_hardest_errors.csv", index=False)
    lines.append(f"\n## 4. Hardest errors\n{len(hard)} clips saved to reports/q3_hardest_errors.csv "
                 "(lowest-scored missed faults and highest-scored false alarms) for listening and for Q5 (XAI).")
    (R / "q3_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(R / "q3_test_predictions.csv", index=False)

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, t in zip(axes, ["pump", "fan"]):
        k = test_ds.index.machine_type.to_numpy() == t
        for name, (_, s) in S.items():
            fpr, tpr, _ = roc_curve(yt[k], s[k])
            ax.plot(fpr, tpr, label=name, lw=1.3)
        ax.axvline(0.05, color="k", ls="--", lw=0.8); ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
        ax.set_xlim(0, 1); ax.set_title(f"Test ROC – {t} (id_06)"); ax.set_xlabel("false-alarm rate (FPR)")
        ax.set_ylabel("recall (TPR)"); ax.grid(alpha=.3)
    axes[1].legend(fontsize=7.5, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "q3_roc_test.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, t in zip(axes, ["pump", "fan"]):
        d = df[df.machine_type == t]
        ax.hist(d[d.label == 0].score, bins=np.linspace(0, 1, 41), alpha=.6, label="normal", density=True)
        ax.hist(d[d.label == 1].score, bins=np.linspace(0, 1, 41), alpha=.6, label="abnormal", density=True)
        ax.axvline(thr, color="k", ls="--", lw=1, label="threshold (val, 5 % FPR)")
        ax.set_title(f"Final ANN scores on test – {t}"); ax.set_xlabel("fault score"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "q3_score_hist.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, t in zip(axes, ["pump", "fan"]):
        d = df[df.machine_type == t]
        for (lab, err), c, mk in [((0, "ok"), "#1f77b4", "."), ((1, "ok"), "#2ca02c", "."),
                                  ((0, "FP"), "#ff7f0e", "x"), ((1, "FN"), "#d62728", "x")]:
            e = d[(d.label == lab) & (d.error == err)]
            ax.scatter(e.rel_db, e.score, s=10 if mk == "." else 28, c=c, marker=mk,
                       label={(0, "ok"): "normal, correct", (1, "ok"): "abnormal, detected",
                              (0, "FP"): "false alarm", (1, "FN"): "missed fault"}[(lab, err)])
        ax.axhline(thr, color="k", ls="--", lw=.8); ax.axvline(0, color="#999", lw=.8)
        ax.set_xlabel("loudness vs median normal clip (dB)"); ax.set_ylabel("fault score"); ax.set_title(t)
    axes[1].legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(FIG / "q3_errors_loudness.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
