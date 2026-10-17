"""
Question 3 – figures and tables from the experiment runs (VALIDATION data only).

Outputs
  reports/q3_runs_summary.csv                  one row per run (all groups)
  reports/figures/q3_learning_curves.png       initial vs final ANN: losses, metric, generalisation gap
  reports/figures/q3_improvement_path.png      E001 -> E005: what each improvement step did
  reports/figures/q3_lr_study.png              learning rate (Q3b)
  reports/figures/q3_bs_study.png              batch size (Q3b)
  reports/figures/q3_depth_study.png           fine-tuning depth (Q3b)
  reports/figures/q3_optimizer_study.png       AdamW vs Adam(L2) vs SGD-momentum (Q3b)
  reports/figures/q3_ema_stability.png         with vs without weight EMA (epoch-to-epoch noise)
  reports/figures/q3_combination.png           Round 2: winners combined vs their parents
  reports/figures/q3_all_runs.png              every run vs the reference design
  reports/figures/q3_seed_variance.png         final config over 3 seeds
  reports/figures/q3_round0_vs_round1.png      archived Round 0 vs Round 1 (if the archive exists)

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
from src.experiment_plan import EXPERIMENTS, REF_ID  # noqa: E402
from src.train import RUNS_DIR  # noqa: E402

FIG = config.FIGURES_DIR
FINAL_JSON = config.ROOT / "experiments" / "final_config.json"
ARCHIVE = config.ROOT / "experiments" / "archive"
M = "val_macro_TPR@FPR5"
GROUP_COLORS = {"initial": "#7f7f7f", "improve": "#1f77b4", "lr": "#ff7f0e", "bs": "#2ca02c", "depth": "#9467bd",
                "optim": "#8c564b", "ablation": "#e377c2", "domain": "#bcbd22", "failed": "#d62728", "combine": "#393b79",
                "final": "#17becf"}


def hist(i):
    p = RUNS_DIR / i / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def summ(i):
    p = RUNS_DIR / i / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def label(i, extra=""):
    s = summ(i)
    return f"{i} {extra}".strip() if s is None else f"{i} {extra} ({s[M]:.3f})".replace("  ", " ")


def summary_table() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS_DIR.glob("*/summary.json")):
        s = json.loads(d.read_text()); h = hist(s["exp_id"])
        ep_noise = float(h[M].diff().abs().mean()) if len(h) > 1 else np.nan
        rows.append({"exp_id": s["exp_id"], "group": s["group"], "what": s["what"],
                     "epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"],
                     M: round(s[M], 4), "val_stable_macro_TPR@FPR5": round(s["val_stable_macro_TPR@FPR5"], 4),
                     "val_macro_pAUC_0.1": round(s["val_macro_pAUC_0.1"], 4),
                     "val_macro_ROC_AUC": round(s["val_macro_ROC_AUC"], 4),
                     "val_TPR@FPR5_pump": round(s["val_TPR@FPR5_pump"], 4),
                     "val_TPR@FPR5_fan": round(s["val_TPR@FPR5_fan"], 4),
                     "val_pooled_TPR@FPR5": round(s["val_pooled_TPR@FPR5"], 4),
                     "train_loss@best": round(s["train_loss@best"], 4), "val_loss@best": round(s["val_loss@best"], 4),
                     "min_val_loss": round(h.val_loss.min(), 4), "epoch_to_epoch_change": round(ep_noise, 4),
                     "train_time_min": s["train_time_min"], "sec_per_epoch": round(h.epoch_time_s.mean(), 1),
                     "trainable_params": s["trainable_params"], "diverged": s["diverged"]})
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
        axes[2].plot(h.epoch, h[M], marker=".", label=lab)
        if extra_panel:
            axes[3].plot(h.epoch, h[extra_panel], marker=".", label=lab)
    axes[0].set_title("Training loss (weighted BCE, augmented/mixed)")
    axes[1].set_title("Validation loss (full 10-s clips, id_04)")
    axes[2].set_title("Validation macro TPR @ 5 % FPR (primary)")
    if extra_panel:
        axes[3].set_title({"grad_norm": "Mean gradient norm (before clipping)", "lr": "Learning rate",
                           "gap": "Generalisation gap (val - train loss)"}.get(extra_panel, extra_panel))
        if extra_panel == "grad_norm":
            axes[3].set_yscale("log")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    axes[1].set_yscale("log")
    axes[2].legend(fontsize=7.5)
    fig.suptitle(title, fontsize=11.5, weight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def bar_study(df, ids, xlabels, title, path, xname):
    sub = df.set_index("exp_id").reindex(ids).dropna(subset=[M])
    if sub.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(sub))
    labs = [xlabels.get(i, i) for i in sub.index]
    ax[0].bar(x - 0.2, sub["val_TPR@FPR5_pump"], 0.4, label="pump")
    ax[0].bar(x + 0.2, sub["val_TPR@FPR5_fan"], 0.4, label="fan")
    ax[0].plot(x, sub[M], "ko-", label="macro (primary)")
    ax[0].plot(x, sub["val_stable_macro_TPR@FPR5"], "k^--", ms=5, lw=.8, label="macro, stable (best ±1 ep)")
    ax[0].set_xticks(x, labs); ax[0].set_xlabel(xname); ax[0].set_ylabel("val TPR @ 5 % FPR")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x, sub["train_time_min"], color="#999")
    ax[1].set_xticks(x, labs); ax[1].set_xlabel(xname); ax[1].set_ylabel("training time (min)")
    for k, (bep, ep) in enumerate(zip(sub.best_epoch, sub.epochs_run)):
        ax[1].text(k, sub["train_time_min"].iloc[k], f"best ep {int(bep)}/{int(ep)}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=11.5, weight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def round0_vs_round1(df):
    logs = sorted(ARCHIVE.glob("round0_*/experiments/experiment_log.csv"))
    if not logs:
        return
    r0 = pd.read_csv(logs[-1])
    if "val_TPR@FPR5_pump" not in r0:
        return
    r0 = r0[r0.exp_id.isin(["E000", "B001", "E001", "E002", "E003"])].copy()
    r0["macro"] = (r0["val_TPR@FPR5_pump"] + r0["val_TPR@FPR5_fan"]) / 2
    r1 = df.set_index("exp_id")
    names = {"E001": "initial ANN", "E002": "full FT", "E003": "+ Q2 aug", "E004": "+ low LR, wd, EMA",
             "E005": "+ mixup (ref)"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = [i for i in names if i in r1.index]
    ax.plot(range(len(xs)), r1.loc[xs, M], "o-", label="Round 1 (this plan)", color="#1f77b4")
    r0m = r0.set_index("exp_id")["macro"]
    x0 = [k for k, i in enumerate(xs) if i in r0m.index]
    ax.plot(x0, [r0m[xs[k]] for k in x0], "s--", label="Round 0 (archived)", color="#999")
    log = pd.read_csv(config.ROOT / "experiments" / "experiment_log.csv").set_index("exp_id")
    if "B001" in log.index:
        ax.axhline(float(log.loc["B001", M]), color="#d62728", ls=":", label="classical baseline B001")
    ax.set_xticks(range(len(xs)), [f"{i}\n{names[i]}" for i in xs], fontsize=8)
    ax.set_ylabel("val macro TPR @ 5 % FPR"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Improvement path: Round 0 vs Round 1 (validation machine id_04)", weight="bold")
    plt.tight_layout(); plt.savefig(FIG / "q3_round0_vs_round1.png", dpi=180); plt.close()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = summary_table()
    if df.empty:
        raise SystemExit("No runs found in experiments/runs/")
    final_id = json.loads(FINAL_JSON.read_text())["selected_from"] if FINAL_JSON.exists() else REF_ID
    for i in df.exp_id:                                     # generalisation gap column for the curves
        h = hist(i); h["gap"] = h.val_loss - h.train_loss; h.to_csv(RUNS_DIR / i / "history.csv", index=False)

    curves(["E001", final_id], {"E001": "initial", final_id: "final"},
           "Learning curves: initial ANN vs final ANN", FIG / "q3_learning_curves.png", extra_panel="gap")
    curves(["E002", "E003", "E004", "E005"],
           {"E002": "full FT", "E003": "+ aug", "E004": "+ low LR/wd/EMA", "E005": "+ mixup"},
           "Improvement path (one change per step)", FIG / "q3_improvement_path.png", extra_panel="gap")
    curves(["E006", "E005", "E007", "E008"],
           {"E006": "lr 3e-5", "E005": "lr 1e-4", "E007": "lr 3e-4", "E008": "lr 1e-3"},
           "Learning-rate study (AdamW, reference E005)", FIG / "q3_lr_study.png", extra_panel="grad_norm")
    curves(["E009", "E005", "E010"], {"E009": "bs 16", "E005": "bs 32", "E010": "bs 64"},
           "Batch-size study", FIG / "q3_bs_study.png")
    bar_study(df, ["E011", "E012", "E013", "E005"],
              {"E011": "0 (head)", "E012": "3", "E013": "6", "E005": "9 (all)"},
              "Fine-tuning depth: number of unfrozen feature modules", FIG / "q3_depth_study.png", "unfrozen modules")
    curves(["E005", "E014", "E015", "E016"],
           {"E005": "AdamW 1e-4", "E014": "Adam+L2 1e-4", "E015": "SGD-mom 1e-2", "E016": "SGD-mom 3e-3"},
           "Optimiser comparison", FIG / "q3_optimizer_study.png", extra_panel="grad_norm")
    curves(["E005", "E017"], {"E005": "with EMA", "E017": "no EMA"},
           "Weight EMA: effect on epoch-to-epoch stability", FIG / "q3_ema_stability.png")

    bar_study(df, ["E005", "E008", "E012", "E009", "E015", "E024", "E025", "E026", "E027"],
              {"E005": "ref", "E008": "lr1e-3", "E012": "3 blk", "E009": "bs16", "E015": "SGD",
               "E024": "SGD+3blk", "E025": "SGD+bs16", "E026": "SGD-EMA", "E027": "lr1e-3+3blk"},
              "Round 2: combining the one-factor winners", FIG / "q3_combination.png", "run")

    order = [e["id"] for e in EXPERIMENTS if e["id"] in set(df.exp_id)] + \
            sorted(i for i in df.exp_id if i not in {e["id"] for e in EXPERIMENTS})
    d = df.set_index("exp_id").loc[order]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(range(len(d)), d[M], color=[GROUP_COLORS.get(g, "#333") for g in d.group])
    ax.scatter(range(len(d)), d["val_stable_macro_TPR@FPR5"], marker="_", s=120, c="k", zorder=3,
               label="stable (best ±1 epoch)")
    if REF_ID in d.index:
        ax.axhline(d.loc[REF_ID, M], color="k", ls="--", lw=1, label=f"reference {REF_ID}")
    log = pd.read_csv(config.ROOT / "experiments" / "experiment_log.csv").set_index("exp_id")
    if "B001" in log.index:
        ax.axhline(float(log.loc["B001", M]), color="#d62728", ls=":", lw=1.2, label="classical baseline B001")
    ax.set_xticks(range(len(d)), d.index, rotation=60, fontsize=8)
    ax.set_ylabel("best val macro TPR @ 5 % FPR")
    for g, c in GROUP_COLORS.items():
        if g in set(d.group):
            ax.bar(0, 0, color=c, label=g)
    ax.legend(fontsize=7, ncol=6); ax.grid(alpha=.3, axis="y")
    ax.set_title("All neural-network runs (validation machine only)", weight="bold")
    plt.tight_layout(); plt.savefig(FIG / "q3_all_runs.png", dpi=180); plt.close()

    seeds = [i for i in [final_id, f"{final_id}_s43", f"{final_id}_s44"] if i in d.index]
    if len(seeds) >= 2:
        v = d.loc[seeds, M]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(seeds, v, color="#17becf"); ax.axhline(v.mean(), color="k", ls="--", lw=1)
        ax.set_title(f"Final config, {len(seeds)} seeds: {v.mean():.3f} ± {v.std(ddof=1):.3f}", fontsize=10)
        ax.set_ylabel("best val macro TPR @ 5 % FPR"); ax.grid(alpha=.3, axis="y")
        plt.tight_layout(); plt.savefig(FIG / "q3_seed_variance.png", dpi=180); plt.close()

    round0_vs_round1(df)
    print(df[["exp_id", "group", M, "val_stable_macro_TPR@FPR5", "val_TPR@FPR5_pump", "val_TPR@FPR5_fan",
              "best_epoch", "epochs_run", "train_time_min"]].to_string(index=False))
    print("\nSaved reports/q3_runs_summary.csv and reports/figures/q3_*.png")


if __name__ == "__main__":
    main()
