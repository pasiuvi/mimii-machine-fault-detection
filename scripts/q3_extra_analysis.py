"""
Question 3 – extra analysis used in the report. It does NOT evaluate any model again:
it only reads files that already exist after the one-time test evaluation.

Inputs  : reports/q3_test_predictions.csv, reports/q3_comparison.csv, reports/q3_runs_summary.csv
Outputs : reports/q3_extra_analysis.md
          reports/figures/q3_val_vs_test.png     validation vs test for the four compared models
          reports/figures/q3_combination.png     Round 2 figure with readable labels

Run:  python -m scripts.q3_extra_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

R, FIG = config.REPORTS_DIR, config.FIGURES_DIR


def recall_at(y, s, fpr_max):
    f, t, _ = roc_curve(y, s)
    return float(t[f <= fpr_max].max())


def main() -> None:
    d = pd.read_csv(R / "q3_test_predictions.csv")
    lines = ["# Q3 extra analysis (deployed final ANN, test machine id_06)\n",
             "Computed from reports/q3_test_predictions.csv - no model was evaluated again.\n",
             "## Recall at 1 %, 5 % and 10 % false-alarm rate, and ranking quality"]
    rows = []
    for t in ["pump", "fan"]:
        k = d.machine_type == t
        y, s = d.label[k], d.score[k]
        rows.append({"machine": t, "recall@1%FPR": round(recall_at(y, s, 0.01), 3),
                     "recall@5%FPR": round(recall_at(y, s, 0.05), 3), "recall@10%FPR": round(recall_at(y, s, 0.10), 3),
                     "ROC-AUC": round(roc_auc_score(y, s), 3),
                     "median score normal": round(s[y == 0].median(), 3),
                     "median score abnormal": round(s[y == 1].median(), 3)})
    lines.append(pd.DataFrame(rows).to_markdown(index=False))

    lines.append("\n## Loudness relative to the median normal clip of the same machine (data-quality check)")
    d["far_3db"] = d.rel_db.abs() > 3
    g = d.groupby(["machine_type", "label"]).agg(clips=("score", "size"), share_more_than_3dB_away=("far_3db", "mean"),
                                                 median_rel_db=("rel_db", "median")).round(3).reset_index()
    lines.append(g.to_markdown(index=False))
    lines.append("\nDiagnostic only (uses test labels, never used to build or choose a model): ROC-AUC of |level difference| = "
                 + ", ".join(f"{t} {roc_auc_score(d.label[d.machine_type == t], d.rel_db[d.machine_type == t].abs()):.3f}"
                             for t in ["pump", "fan"]))

    ab = d[d.label == 1]
    lines.append("\n## Overlap of detected faults with the classical baseline (at validation thresholds)")
    lines.append(f"both detect {int(((ab.pred == 1) & (ab.baseline_pred == 1)).sum())}, ANN only "
                 f"{int(((ab.pred == 1) & (ab.baseline_pred == 0)).sum())}, baseline only "
                 f"{int(((ab.pred == 0) & (ab.baseline_pred == 1)).sum())}, both miss "
                 f"{int(((ab.pred == 0) & (ab.baseline_pred == 0)).sum())} (of {len(ab)} abnormal clips)")
    (R / "q3_extra_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- figure: validation vs test
    c = pd.read_csv(R / "q3_comparison.csv")
    spec = [("Trivial: loudness only", "Loudness only", "#7f7f7f", 0.0),
            ("Classical baseline", "Classical baseline", "#ff7f0e", 0.018),
            ("Initial ANN", "Initial ANN (E001)", "#2ca02c", 0.0),
            ("deployed", "Final ANN (deployed)", "#d62728", -0.018)]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for key, name, col, dv in spec:
        r = c[c.Model.str.contains(key, regex=False)].iloc[0]
        v, t = r["Val macro TPR@FPR5"], r["Test macro TPR@FPR5"]
        ax.plot([0, 1], [v, t], "o-", color=col, lw=2, ms=7, label=name)
        ax.text(-0.05, v + dv, f"{v:.3f}", va="center", ha="right", fontsize=8.5, color=col)
        ax.text(1.05, t, f"{t:.3f}", va="center", fontsize=8.5, color=col)
    ax.axhline(0.05, color="#bbb", ls=":", lw=1)
    ax.text(0.5, 0.062, "random = 0.05", ha="center", fontsize=8, color="#888")
    ax.set_xticks([0, 1], ["Validation machine (id_04)\nused for every decision", "Test machine (id_06)\nused once"])
    ax.set_xlim(-0.38, 1.35); ax.set_ylabel("macro TPR @ 5 % FPR"); ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.24), frameon=False)
    ax.set_title("The ranking reverses between the validation and the test machine", fontsize=10.5, weight="bold")
    plt.tight_layout(); plt.savefig(FIG / "q3_val_vs_test.png", dpi=180); plt.close()

    # ---- figure: Round 2 with readable labels
    s = pd.read_csv(R / "q3_runs_summary.csv").set_index("exp_id")
    lab = {"E005": "E005\nref", "E008": "E008\nLR 1e-3", "E012": "E012\n3 blocks", "E009": "E009\nbatch 16",
           "E015": "E015\nSGD", "E024": "E024\nSGD+3 blk", "E025": "E025\nSGD+bs16", "E026": "E026\nSGD, no EMA",
           "E027": "E027\nLR1e-3+3 blk"}
    ids = [i for i in lab if i in s.index]
    sub, x = s.loc[ids], np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.bar(x - 0.2, sub["val_TPR@FPR5_pump"], 0.4, label="pump", color="#1f77b4")
    ax.bar(x + 0.2, sub["val_TPR@FPR5_fan"], 0.4, label="fan", color="#ff7f0e")
    ax.plot(x, sub["val_macro_TPR@FPR5"], "ko-", label="macro (primary)")
    ax.plot(x, sub["val_stable_macro_TPR@FPR5"], "k^--", ms=5, lw=.8, label="macro, stable (best ±1 epoch)")
    ax.axvspan(4.5, len(ids) - 0.5, color="#393b79", alpha=.07)
    ax.text(5.5, 0.9, "Round 2 combinations", ha="center", fontsize=9, color="#393b79")
    ax.set_xticks(x, [lab[i] for i in ids], fontsize=8.5); ax.set_ylabel("val TPR @ 5 % FPR"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3, axis="y")
    ax.set_title("Round 2: combining the one-factor winners (validation machine id_04)", weight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "q3_combination.png", dpi=180); plt.close()
    print("\n".join(lines))
    print("\nSaved reports/q3_extra_analysis.md, figures/q3_val_vs_test.png, figures/q3_combination.png")


if __name__ == "__main__":
    main()
