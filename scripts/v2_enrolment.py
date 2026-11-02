"""
AcoustiGuard v2 – per-machine ENROLMENT detector (improves the post-hoc "second approach").

    python -m scripts.v2_enrolment embed     # 3-6 min on the GPU: cache ANN embeddings (+ CLAP, already cached)
    python -m scripts.v2_enrolment select    # chooses the configuration on the 6 DEVELOPMENT machines only
    #   -> commit + git tag v2-before-test
    python -m scripts.v2_enrolment test      # ONE run on the test machines id_06 (refuses to run twice)

WHY
  v1 (locked, Q3) learned "sounds like my training machines" and failed on a new machine. The post-hoc study showed
  that comparing a machine with ITS OWN normal sound works much better, but the simple rule (20 consecutive clips,
  threshold = highest score) gave 60-125 false alarms per day. v2 fixes the three weak points:
    1. enrolment: more normal clips (20 / 50 / 100), spread over the machine's normal recordings
       (simulates recording on several days / operating conditions), and confirmed healthy (normal label);
    2. score: distance to the k nearest enrolment clips in a pretrained ANN feature space, standardised per
       machine with leave-one-out enrolment scores, so scores from different feature spaces can be fused;
    3. decision: threshold = mean + c x std of the enrolment scores, and an alarm only when 2 of the last 3 clips
       exceed it (a real fault is steady; a single odd clip is not).

HONESTY RULES (fixed before running, written in this file and committed)
  * Every choice is made on the 6 DEVELOPMENT machines (pump + fan id_00, id_02, id_04). This answers the critique
    that one validation machine per type is too few. Fault labels of development machines are used ONLY to
    choose the configuration, never by the detector itself (it only sees normal enrolment clips).
  * Feature extractors must be label-free for the development machines, so that the development score is fair:
      effnet  = EfficientNet-B0 with ImageNet weights, same 1-channel log-mel input as FaultCNN (no fine-tuning)
      clap    = LAION CLAP audio embedding (Q6, pinned revision)
      fusion  = mean of the two standardised scores
    The fine-tuned FaultCNN embedding is reported only as a reference (it saw the labels of id_00/id_02).
  * Selection rule: among all configurations with mean development false alarms <= 1.0 per machine per day,
    pick the highest mean development recall (2-of-3 rule); ties -> fewer enrolment clips -> simpler features.
    If none meets 1.0/day, pick the lowest false-alarm rate.
  * The test machines id_06 were already used for the locked v1 evaluations, so this is NOT a blind test of a
    brand-new machine. To limit bias, nothing in this file was tuned on id_06, and the test step runs once.
  * The 2-of-3 rule is evaluated by assuming clips are independent (P = 3p^2 - 2p^3). Real normal sound is
    correlated over time, so real false alarms will be higher: the clip-level numbers are reported as well.
  * The locked Q3/Q6 results and the post-hoc study are NOT changed.

Outputs: data/processed/v2_embeddings.npz, reports/v2_selection.md/.csv, experiments/v2_config.json,
         reports/v2_test_results.md/.json, reports/.v2_test_evaluated.json, reports/figures/v2_*.png
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.dataset import MimiiSpecDataset  # noqa: E402
from src.metrics import tpr_at_fpr  # noqa: E402
from src.model import FaultCNN  # noqa: E402

R, FIG, EXP, MODELS = config.REPORTS_DIR, config.FIGURES_DIR, config.ROOT / "experiments", config.ROOT / "models"
EMB = config.PROCESSED_DIR / "v2_embeddings.npz"
CLAP = config.PROCESSED_DIR / "clap_embeddings.npz"
LOCK = R / ".v2_test_evaluated.json"
CFG = EXP / "v2_config.json"

# ---------------- search space (fixed before running) ----------------
FEATURES = ["effnet", "clap", "fusion"]
N_ENROL = [20, 50, 100]
K_GRID = [1, 3, 5]
C_GRID = [2.0, 3.0, 4.0, 5.0]          # threshold = mean + c * std of leave-one-out enrolment scores
SEEDS = [0, 1, 2, 3, 4]                # 5 random enrolment draws per machine (results averaged)
CLIPS_PER_DAY = 144                    # one 10-s clip every 10 minutes
FA_TARGET = 1.0                        # false alarms per machine per day (2-of-3 rule)
DEV_SPLITS, TEST_SPLIT = ("train", "val"), "test"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ---------------- step 1: embeddings ----------------
@torch.no_grad()
def _embed(model, split, clip_norm, dev):
    ds = MimiiSpecDataset(split, augment=False, clip_norm=clip_norm)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    out = []
    for x, _, _ in dl:
        out.append(torch.nn.functional.normalize(model.embed(x.to(dev)).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def step_embed():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imnet = FaultCNN(pretrained=True).to(dev).eval()                       # label-free ImageNet features
    meta = json.loads((MODELS / "final_model.json").read_text())
    ck = torch.load(MODELS / "final_model.pt", map_location=dev, weights_only=False)
    tuned = FaultCNN(pretrained=False); tuned.load_state_dict(ck["state_dict"]); tuned = tuned.to(dev).eval()
    clip_norm = bool(meta["cfg"].get("clip_norm", False))
    clap = np.load(CLAP, allow_pickle=True)
    store = {}
    for sp in ["train", "val", "test"]:
        idx = pd.read_csv(config.FEATURE_DIR / f"{sp}_index.csv")
        store[f"{sp}_path"] = idx.path.astype(str).to_numpy()
        store[f"{sp}_effnet"] = _embed(imnet, sp, False, dev)
        store[f"{sp}_tuned"] = _embed(tuned, sp, clip_norm, dev)
        pmap = dict(zip(clap[f"{sp}_p"].astype(str), clap[f"{sp}_E"]))
        store[f"{sp}_clap"] = np.stack([pmap[p] for p in idx.path.astype(str)]).astype(np.float32)
        print(f"  {sp}: {len(idx)} clips embedded", flush=True)
    np.savez_compressed(EMB, **store)
    print(f"Saved {EMB.relative_to(config.ROOT)}")


# ---------------- scoring ----------------
def load_machines(splits):
    Z = np.load(EMB, allow_pickle=True)
    machines = []
    for sp in splits:
        idx = pd.read_csv(config.FEATURE_DIR / f"{sp}_index.csv")
        assert (idx.path.astype(str).to_numpy() == Z[f"{sp}_path"]).all(), "index and embeddings out of sync"
        for (mt, mid), g in idx.groupby(["machine_type", "machine_id"]):
            ii = g.index.to_numpy()
            machines.append({"machine": f"{mt} {mid}", "type": mt, "split": sp, "y": g.label.to_numpy(),
                             "E": {f: Z[f"{sp}_{f}"][ii] for f in ["effnet", "clap", "tuned"]}})
    return machines


def _knn(Q, B, k, loo=False):
    d = 1.0 - Q @ B.T
    if loo:
        np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].mean(1)


def machine_scores(m, feature, n, k, seed):
    """Standardised scores of the enrolment clips (leave-one-out) and of all other clips of one machine."""
    y = m["y"]
    normal = np.where(y == 0)[0]
    rng = np.random.default_rng(config.SEED + seed)
    enrol = np.sort(rng.choice(normal, size=n, replace=False))
    rest = np.setdiff1d(np.arange(len(y)), enrol)
    parts = ["effnet", "clap"] if feature == "fusion" else [feature]
    se, sr = 0.0, 0.0
    for f in parts:
        E = m["E"][f]
        e_loo, r = _knn(E[enrol], E[enrol], k, loo=True), _knn(E[rest], E[enrol], k)
        mu, sd = e_loo.mean(), e_loo.std() + 1e-9
        se, sr = se + (e_loo - mu) / sd / len(parts), sr + (r - mu) / sd / len(parts)
    if len(parts) > 1:                      # re-standardise the fused enrolment scores
        mu, sd = se.mean(), se.std() + 1e-9
        se, sr = (se - mu) / sd, (sr - mu) / sd
    return se, sr, y[rest]


def operating(sr, yr, c):
    p0 = float((sr[yr == 0] > c).mean()); p1 = float((sr[yr == 1] > c).mean())
    w = lambda p: 3 * p ** 2 - 2 * p ** 3                    # alarm if >= 2 of 3 clips exceed (independence)
    return {"clip_recall": p1, "clip_FA_day": p0 * CLIPS_PER_DAY, "recall_2of3": w(p1), "FA_day_2of3": w(p0) * CLIPS_PER_DAY}


def evaluate_machine(m, feature, n, k, c):
    rows = []
    for s in SEEDS:
        se, sr, yr = machine_scores(m, feature, n, k, s)
        rows.append({"TPR@FPR5": tpr_at_fpr(yr, sr), "AUC": float(roc_auc_score(yr, sr)), **operating(sr, yr, c)})
    d = pd.DataFrame(rows)
    out = d.mean().to_dict()
    out["recall_2of3_sd"] = float(d.recall_2of3.std()); out["FA_day_2of3_sd"] = float(d.FA_day_2of3.std())
    return out


# ---------------- step 2: selection on the 6 development machines ----------------
def step_select():
    if LOCK.exists():
        sys.exit("The v2 test has already been run – selection is frozen.")
    dev_m = load_machines(DEV_SPLITS)
    print(f"Development machines: {[m['machine'] for m in dev_m]}")
    rows = []
    for f, n, k, c in itertools.product(FEATURES, N_ENROL, K_GRID, C_GRID):
        per = [evaluate_machine(m, f, n, k, c) for m in dev_m]
        P = pd.DataFrame(per)
        rows.append({"feature": f, "n_enrol": n, "k": k, "c": c,
                     "dev_TPR@FPR5": P["TPR@FPR5"].mean(), "dev_AUC": P.AUC.mean(),
                     "dev_recall_2of3": P.recall_2of3.mean(), "dev_FA_day_2of3": P.FA_day_2of3.mean(),
                     "dev_worst_FA_day_2of3": P.FA_day_2of3.max(), "dev_min_recall_2of3": P.recall_2of3.min(),
                     "dev_clip_recall": P.clip_recall.mean(), "dev_clip_FA_day": P.clip_FA_day.mean()})
        print(f"  {f:6s} n={n:3d} k={k} c={c:.0f}: recall(2of3) {rows[-1]['dev_recall_2of3']:.3f}  "
              f"FA/day {rows[-1]['dev_FA_day_2of3']:.2f}  TPR@FPR5 {rows[-1]['dev_TPR@FPR5']:.3f}", flush=True)
    T = pd.DataFrame(rows).round(4)
    simple = {"effnet": 0, "clap": 1, "fusion": 2}
    ok = T[T.dev_FA_day_2of3 <= FA_TARGET]
    if len(ok):
        best = ok.assign(_s=ok.feature.map(simple)).sort_values(
            ["dev_recall_2of3", "n_enrol", "_s"], ascending=[False, True, True]).iloc[0]
        rule = f"highest development recall among {len(ok)} configurations with <= {FA_TARGET} false alarm/day"
    else:
        best = T.sort_values("dev_FA_day_2of3").iloc[0]
        rule = "no configuration met the target – lowest false-alarm rate chosen"
    T["selected"] = (T.feature == best.feature) & (T.n_enrol == best.n_enrol) & (T.k == best.k) & (T.c == best.c)
    T.to_csv(R / "v2_selection.csv", index=False)

    chosen = {"feature": best.feature, "n_enrol": int(best.n_enrol), "k": int(best.k), "c": float(best.c),
              "decision": "alarm if >= 2 of the last 3 clips exceed the threshold", "rule": rule,
              "development": {c_: float(best[c_]) for c_ in T.columns if c_.startswith("dev_")},
              "selected_at": datetime.now().isoformat(timespec="seconds"), "git_commit": _git()}
    CFG.write_text(json.dumps(chosen, indent=2))

    # per-machine table of the chosen configuration (development machines)
    per = pd.DataFrame([{"machine": m["machine"], "split": m["split"],
                         **evaluate_machine(m, best.feature, int(best.n_enrol), int(best.k), float(best.c))} for m in dev_m]).round(3)
    per.to_csv(R / "v2_selection_per_machine.csv", index=False)
    top = T.sort_values(["dev_FA_day_2of3" if not len(ok) else "dev_recall_2of3"], ascending=not len(ok)).head(15)
    md = ["# AcoustiGuard v2 – selection on the 6 development machines", "",
          f"Rule (fixed before running): {rule}.", "",
          f"**Selected:** feature = {best.feature}, enrolment clips = {int(best.n_enrol)}, k = {int(best.k)}, "
          f"threshold = mean + {best.c:g} x std, alarm on 2 of 3 clips.", "",
          "## Best configurations", "", top.to_markdown(index=False), "",
          "## Selected configuration per development machine (mean of 5 enrolment draws)", "", per.to_markdown(index=False), "",
          "## Effect of each choice (mean over the other settings)", ""]
    for col in ["feature", "n_enrol", "k", "c"]:
        md += [f"### {col}", "", T.groupby(col)[["dev_TPR@FPR5", "dev_recall_2of3", "dev_FA_day_2of3"]].mean().round(3).to_markdown(), ""]
    (R / "v2_selection.md").write_text("\n".join(md), encoding="utf-8")
    print("\nSELECTED:", json.dumps({k_: chosen[k_] for k_ in ["feature", "n_enrol", "k", "c"]}))
    print(per.to_string(index=False))
    print("\nNext: git add -A; git commit -m \"v2 selection on development machines\"; git tag v2-before-test; git push --tags")


# ---------------- step 3: one-time test ----------------
def step_test():
    if LOCK.exists():
        sys.exit(f"Refusing to run: the v2 test was already evaluated ({json.loads(LOCK.read_text())['time']}).")
    if not CFG.exists():
        sys.exit("Run the select step first.")
    tags = subprocess.run(["git", "tag"], capture_output=True, text=True, cwd=config.ROOT).stdout.split()
    if "v2-before-test" not in tags:
        sys.exit("Commit the selection and create the git tag v2-before-test first.")
    cfg = json.loads(CFG.read_text())
    test_m = load_machines((TEST_SPLIT,))
    rows = []
    for m in test_m:
        rows.append({"machine": m["machine"], "approach": f"v2 selected ({cfg['feature']}, n={cfg['n_enrol']}, k={cfg['k']}, c={cfg['c']:g})",
                     **evaluate_machine(m, cfg["feature"], cfg["n_enrol"], cfg["k"], cfg["c"])})
        # references with the same enrolment protocol (not selectable)
        rows.append({"machine": m["machine"], "approach": "reference: fine-tuned FaultCNN embedding, same protocol",
                     **evaluate_machine(m, "tuned", cfg["n_enrol"], cfg["k"], cfg["c"])})
    T = pd.DataFrame(rows).round(3)
    S = T.groupby("approach", sort=False).mean(numeric_only=True).round(3)
    res = {"config": cfg, "per_machine": T.to_dict("records"), "macro": S.reset_index().to_dict("records"),
           "locked_v1_reference": {"final_ANN_test_macro_TPR@FPR5": 0.032, "CLAP_enrol20_test_macro_TPR@FPR5": 0.622,
                                   "posthoc_FA_day_CLAP_enrol20": 70.8}}
    (R / "v2_test_results.json").write_text(json.dumps(res, indent=2))
    md = ["# AcoustiGuard v2 – one-time evaluation on the test machines id_06", "",
          "Configuration chosen on the 6 development machines and tagged v2-before-test before this run. "
          "id_06 was already used for the locked v1 evaluations, so this is not a blind test of a new machine; "
          "nothing was tuned on it. 2-of-3 numbers assume independent clips.", "",
          "## Mean of pump and fan", "", S.to_markdown(), "", "## Per machine (mean of 5 enrolment draws)", "",
          T.to_markdown(index=False), "",
          "Locked v1 for comparison: final ANN 0.032 TPR@FPR5; post-hoc CLAP enrolment (20 consecutive clips) 0.622 "
          "with 70.8 false alarms per day.", ""]
    (R / "v2_test_results.md").write_text("\n".join(md), encoding="utf-8")
    LOCK.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "git_commit": _git()}))

    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.9))
    sel = T[T.approach.str.startswith("v2")]
    x = np.arange(len(sel))
    axs[0].bar(x - 0.2, sel.clip_recall * 100, 0.4, label="single clip", color="#9ecae1")
    axs[0].bar(x + 0.2, sel.recall_2of3 * 100, 0.4, label="2 of 3 clips", color="#2171b5")
    axs[0].set_xticks(x, sel.machine); axs[0].set_ylabel("faults caught (%)"); axs[0].legend(fontsize=7)
    axs[0].set_title("v2 on the test machines", fontsize=9)
    axs[1].bar(x - 0.2, sel.clip_FA_day, 0.4, color="#fdae6b", label="single clip")
    axs[1].bar(x + 0.2, sel.FA_day_2of3, 0.4, color="#d94801", label="2 of 3 clips")
    axs[1].set_xticks(x, sel.machine); axs[1].set_ylabel("false alarms per day"); axs[1].legend(fontsize=7)
    axs[1].set_title("v1 post-hoc (CLAP, 20 clips): 70.8 per day", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "v2_test.png", dpi=200); plt.close()
    print(S.to_string()); print("\n" + T.to_string(index=False))
    print("\nNext: git add -A; git commit -m \"v2 one-time test on id_06\"; git tag v2-test-done; git push --tags")


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=config.ROOT).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"embed": step_embed, "select": step_select, "test": step_test}.get(
        step, lambda: sys.exit("usage: python -m scripts.v2_enrolment embed|select|test"))()
