"""
Question 3 – figures and tables from the experiment runs (VALIDATION data only).

Outputs
  reports/q3_runs_summary.csv                  one row per run (all groups)
  reports/figures/q3_learning_curves.png       initial vs final ANN: losses + validation metric
  reports/figures/q3_lr_study.png              learning rate (Q3b)
  reports/figures/q3_bs_study.png              batch size (Q3b)
  reports/figures/q3_depth_study.png           fine-tuning depth (Q3b)
  reports/figures/q3_optimizer_study.png       AdamW vs Adam(L2) vs SGD-momentum (Q3b)
  reports/figures/q3_all_runs.png              every run vs the reference design
  reports/figures/q3_seed_variance.png         final config over 3 seeds

Run any time after experiments:  python -m scripts.q3_plots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.experiment_plan import EXPERIMENTS  # noqa: E402
from src.train import RUNS_DIR  # noqa: E402

FIG = config.FIGURES_DIR
FINAL_JSON = config.ROOT / "experiments" / "final_config.json"
REF = "E003"
GROUP_COLORS = {"initial": "#7f7f7f", "improve": "#1f77b4", "lr": "#ff7f0e", "bs": "#2ca02c", "depth": "#9467bd",
                "optim": "#8c564b", "ablation": "#e377c2", "failed": "#d62728", "final": "#17becf"}


def hist(i):
    p = RUNS_DIR / i / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def summ(i):
    p = RUNS_DIR / i / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def label(i, extra=""):
    s = summ(i)
    return f"{i} {extra}".strip() if s is None else f"{i} {extra} (best {s['val_TPR@FPR5']:.3f})".replace("  ", " ")


def summary_table() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS_DIR.glob("*/summary.json")):
        s = json.loads(d.read_text()); h = hist(s["exp_id"])
        b = h[h.epoch == s["best_epoch"]].iloc[0] if s["best_epoch"] else h.iloc[-1]
        rows.append({"exp_id": s["exp_id"], "group": s["group"], "what": s["what"],
                     "epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"],
                     "val_TPR@FPR5": round(s["val_TPR@FPR5"], 4), "val_ROC_AUC": round(s["val_ROC_AUC"], 4),
                     "val_pAUC_0.1": round(s["val_pAUC_0.1"], 4), "val_TPR@FPR5_pump": round(s["val_TPR@FPR5_pump"], 4),
                     "val_TPR@FPR5_fan": round(s["val_TPR@FPR5_fan"], 4),
                     "train_loss@best": round(b.train_loss, 4), "val_loss@best": round(b.val_loss, 4),
                     "min_val_loss": round(h.val_loss.min(), 4), "train_time_min": s["train_time_min"],
                     "sec_per_epoch": round(h.epoch_time_s.mean(), 1), "trainable_params": s["trainable_params"],
                     "diverged": s["diverged"]})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "q3_runs_summary.csv", index=False)
    return df


def curves(ids, names, title, path, extra_panel=None):
    hs = {i: hist(i) for i in ids if hist(i) is not None}
    if not hs:
        return
    n = 3 + (1 if extra_panel else 0)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8))
    for i, h in hs.items():
        lab = label(i, names.get(i, ""))
        axes[0].plot(h.epoch, h.train_loss, marker=".", label=lab)
        axes[1].plot(h.epoch, h.val_loss, marker=".", label=lab)
        axes[2].plot(h.epoch, h["val_TPR@FPR5"], marker=".", label=lab)
        if extra_panel:
            axes[3].plot(h.epoch, h[extra_panel], marker=".", label=lab)
    axes[0].set_title("Training loss (weighted BCE, with augmentation)")
    axes[1].set_title("Validation loss (full 10-s clips)")
    axes[2].set_title("Validation TPR @ 5 % FPR (primary metric)")
    if extra_panel:
        axes[3].set_title({"grad_norm": "Mean gradient norm (before clipping)", "lr": "Learning rate"}.get(extra_panel, extra_panel))
        if extra_panel == "grad_norm":
            axes[3].set_yscale("log")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    axes[1].set_yscale("log")
    axes[2].legend(fontsize=7.5)
    fig.suptitle(title, fontsize=11.5, weight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def bar_study(df, ids, xlabels, title, path, xname):
    sub = df.set_index("exp_id").reindex(ids).dropna(subset=["val_TPR@FPR5"])
    if sub.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(sub))
    labs = [xlabels[i] for i in sub.index]
    ax[0].bar(x - 0.2, sub["val_TPR@FPR5_pump"], 0.4, label="pump")
    ax[0].bar(x + 0.2, sub["val_TPR@FPR5_fan"], 0.4, label="fan")
    ax[0].plot(x, sub["val_TPR@FPR5"], "ko-", label="all")
    ax[0].set_xticks(x, labs); ax[0].set_xlabel(xname); ax[0].set_ylabel("best val TPR @ 5 % FPR")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x, sub["train_time_min"], color="#999")
    ax[1].set_xticks(x, labs); ax[1].set_xlabel(xname); ax[1].set_ylabel("training time (min)")
    for k, (bep, ep) in enumerate(zip(sub.best_epoch, sub.epochs_run)):
        ax[1].text(k, sub["train_time_min"].iloc[k], f"best ep {int(bep)}/{int(ep)}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=11.5, weight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = summary_table()
    if df.empty:
        raise SystemExit("No runs found in experiments/runs/")
    final_id = json.loads(FINAL_JSON.read_text())["selected_from"] if FINAL_JSON.exists() else REF

    curves(["E001", final_id], {"E001": "initial", final_id: "final"},
           "Learning curves: initial ANN vs final ANN", FIG / "q3_learning_curves.png")
    curves(["E004", "E003", "E005", "E006"], {"E004": "lr 1e-4", "E003": "lr 3e-4", "E005": "lr 1e-3", "E006": "lr 3e-3"},
           "Learning-rate study (AdamW, full fine-tuning)", FIG / "q3_lr_study.png", extra_panel="grad_norm")
    curves(["E007", "E003", "E008"], {"E007": "bs 16", "E003": "bs 32", "E008": "bs 64"},
           "Batch-size study", FIG / "q3_bs_study.png")
    bar_study(df, ["E009", "E010", "E011", "E003"],
              {"E009": "0 (head)", "E010": "3", "E011": "6", "E003": "9 (all)"},
              "Fine-tuning depth: number of unfrozen feature modules", FIG / "q3_depth_study.png", "unfrozen modules")
    curves(["E003", "E012", "E013", "E014"],
           {"E003": "AdamW 3e-4", "E012": "Adam+L2 3e-4", "E013": "SGD-mom 1e-2", "E014": "SGD-mom 3e-3"},
           "Optimiser comparison", FIG / "q3_optimizer_study.png", extra_panel="grad_norm")

    order = [e["id"] for e in EXPERIMENTS if e["id"] in set(df.exp_id)] + \
            sorted(i for i in df.exp_id if i not in {e["id"] for e in EXPERIMENTS})
    d = df.set_index("exp_id").loc[order]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(range(len(d)), d["val_TPR@FPR5"], color=[GROUP_COLORS.get(g, "#333") for g in d.group])
    if REF in d.index:
        ax.axhline(d.loc[REF, "val_TPR@FPR5"], color="k", ls="--", lw=1, label=f"reference {REF}")
    ax.set_xticks(range(len(d)), d.index, rotation=60, fontsize=8)
    ax.set_ylabel("best val TPR @ 5 % FPR")
    for g, c in GROUP_COLORS.items():
        if g in set(d.group):
            ax.bar(0, 0, color=c, label=g)
    ax.legend(fontsize=7.5, ncol=5); ax.grid(alpha=.3, axis="y")
    ax.set_title("All neural-network runs (validation only)", weight="bold")
    plt.tight_layout(); plt.savefig(FIG / "q3_all_runs.png", dpi=180); plt.close()

    seeds = [i for i in [final_id, f"{final_id}_s43", f"{final_id}_s44"] if i in d.index]
    if len(seeds) >= 2:
        v = d.loc[seeds, "val_TPR@FPR5"]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(seeds, v, color="#17becf"); ax.axhline(v.mean(), color="k", ls="--", lw=1)
        ax.set_title(f"Final config, 3 seeds: {v.mean():.3f} ± {v.std(ddof=1):.3f}", fontsize=10)
        ax.set_ylabel("best val TPR @ 5 % FPR"); ax.grid(alpha=.3, axis="y")
        plt.tight_layout(); plt.savefig(FIG / "q3_seed_variance.png", dpi=180); plt.close()

    print(df[["exp_id", "group", "val_TPR@FPR5", "val_ROC_AUC", "val_TPR@FPR5_pump", "val_TPR@FPR5_fan",
              "best_epoch", "epochs_run", "train_time_min"]].to_string(index=False))
    print("\nSaved reports/q3_runs_summary.csv and reports/figures/q3_*.png")


if __name__ == "__main__":
    main()
