"""
POST-HOC STUDY (after the locked Q3 and Q6 test evaluations) – per-machine calibration and normal-only detection.

    python -m scripts.posthoc_calibration          (2-5 min; needs data/processed/logmel128 and clap_embeddings.npz)

Why: the locked Q3 result showed that thresholds set on one machine do not transfer (deployed ANN: 3.2 % recall;
fan threshold from id_04 raised alarms on 99.9 % of healthy id_06 fans). Real installations can record a few
NORMAL clips of every machine when it is installed or serviced. This study asks, for every physical machine:
if the engineer records 20 normal clips first, how well does each approach work afterwards?

Everything is FIXED BEFORE RUNNING (no choice is made on the results):
  * enrolment set  = the first 20 normal clips of each machine (sorted by file name); scored clips = the rest
  * alarm threshold = the highest score among the 20 enrolment clips (leave-one-out for nearest-neighbour
    scores), i.e. no fault label and no other machine is used to set it
  * k = 3 neighbours (chosen on validation in Q6)
  * approaches:
      ANN-score            deployed FaultCNN fault score, threshold from enrolment (vs the original validation threshold)
      ANN-embedding kNN    distance to the 3 nearest enrolment clips in the ANN's 1280-d embedding
      Loudness deviation   |level - median level of the enrolment clips|   (simple, label-free)
      CLAP kNN             distance to the 3 nearest enrolment clips in the CLAP embedding (Q6 enrolment)
  * machines: all 4 pumps and 4 fans for the label-free approaches (loudness, CLAP); for the ANN approaches only
    the validation (id_04) and test (id_06) machines, because the ANN was trained on id_00 and id_02.
The original locked results are NOT changed and NOT re-run. This file writes only reports/posthoc_*.

Outputs: reports/posthoc_calibration.md/.csv, reports/figures/posthoc_calibration.png
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.dataset import MimiiSpecDataset  # noqa: E402
from src.metrics import tpr_at_fpr  # noqa: E402
from src.model import FaultCNN  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

N_ENROL, K = 20, 3
CLIPS_PER_DAY = 144          # one 10-s clip every 10 minutes
R, FIG, MODELS = config.REPORTS_DIR, config.FIGURES_DIR, config.ROOT / "models"
CLAP = config.PROCESSED_DIR / "clap_embeddings.npz"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


@torch.no_grad()
def ann_outputs(split: str, model, clip_norm: bool, device) -> tuple[np.ndarray, np.ndarray]:
    ds = MimiiSpecDataset(split, augment=False, clip_norm=clip_norm)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    s, e = [], []
    for x, m, _ in loader:
        x, m = x.to(device), m.to(device)
        emb = model.embed(x)
        z = torch.cat([emb, torch.nn.functional.one_hot(m, model.n_types).float()], 1)
        s.append(torch.sigmoid(model.head(z).squeeze(1)).float().cpu().numpy())
        e.append(torch.nn.functional.normalize(emb.float(), dim=1).cpu().numpy())
    return np.concatenate(s), np.concatenate(e)


def level_db(split: str) -> np.ndarray:
    X = np.load(config.FEATURE_DIR / f"{split}.npy", mmap_mode="r")
    return np.array([10 * np.log10(np.mean(10 ** (np.asarray(x, np.float64) / 10)) + 1e-12) for x in X])


def knn(E_query, E_bank, k=K, self_in_bank=False):
    d = 1 - E_query @ E_bank.T
    if self_in_bank:
        d = np.where(d < 1e-9, np.inf, d)          # leave-one-out for the enrolment clips themselves
    return np.sort(d, axis=1)[:, :k].mean(1)


def evaluate(scores_enrol, scores_rest, y_rest):
    thr = float(np.max(scores_enrol))
    pred = scores_rest > thr
    n, a = y_rest == 0, y_rest == 1
    fpr = float(pred[n].mean()); tpr = float(pred[a].mean())
    return {"TPR@FPR5": round(tpr_at_fpr(y_rest, scores_rest), 3), "AUC": round(float(roc_auc_score(y_rest, scores_rest)), 3),
            "recall_at_enrol_thr": round(tpr, 3), "FPR_at_enrol_thr": round(fpr, 3),
            "false_alarms_per_day": round(fpr * CLIPS_PER_DAY, 1), "scored": int(len(y_rest)), "faults": int(a.sum())}


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((MODELS / "final_model.json").read_text())
    ck = torch.load(MODELS / "final_model.pt", map_location=dev, weights_only=False)
    model = FaultCNN(pretrained=False); model.load_state_dict(ck["state_dict"]); model = model.to(dev).eval()
    clip_norm = bool(meta["cfg"].get("clip_norm", False))
    val_thr = {k: float(v) for k, v in meta["thresholds_val_fpr5"].items()}

    frames = []
    clap = np.load(CLAP, allow_pickle=True) if CLAP.exists() else None
    for sp in ["train", "val", "test"]:
        idx = pd.read_csv(config.FEATURE_DIR / f"{sp}_index.csv")
        s, e = ann_outputs(sp, model, clip_norm, dev)
        idx["ann_score"], idx["level"] = s, level_db(sp)
        idx["ann_emb"] = list(e)
        if clap is not None:
            pmap = dict(zip(clap[f"{sp}_p"].astype(str), clap[f"{sp}_E"]))
            idx["clap_emb"] = [pmap.get(p) for p in idx.path]
        idx["split"] = sp
        frames.append(idx)
        print(f"  {sp}: {len(idx)} clips scored", flush=True)
    D = pd.concat(frames, ignore_index=True)

    rows = []
    for (mt, mid), g in D.groupby(["machine_type", "machine_id"]):
        g = g.sort_values("path")
        enrol = g[g.label == 0].head(N_ENROL)
        rest = g.drop(enrol.index)
        y = rest.label.to_numpy()
        role = {"train": "train machine", "val": "validation machine", "test": "TEST machine"}[g.split.iloc[0]]
        ann_ok = g.split.iloc[0] != "train"

        def add(name, se, sr, extra=None):
            rows.append({"machine": f"{mt} {mid}", "machine_type": mt, "role": role, "approach": name,
                         **evaluate(se, sr, y), **(extra or {})})

        if ann_ok:
            # original locked behaviour: threshold from the validation machine
            pr = rest.ann_score.to_numpy() >= val_thr[mt]
            rows.append({"machine": f"{mt} {mid}", "machine_type": mt, "role": role,
                         "approach": "ANN score, validation threshold (as deployed)",
                         "TPR@FPR5": round(tpr_at_fpr(y, rest.ann_score.to_numpy()), 3),
                         "AUC": round(float(roc_auc_score(y, rest.ann_score)), 3),
                         "recall_at_enrol_thr": round(float(pr[y == 1].mean()), 3),
                         "FPR_at_enrol_thr": round(float(pr[y == 0].mean()), 3),
                         "false_alarms_per_day": round(float(pr[y == 0].mean()) * CLIPS_PER_DAY, 1),
                         "scored": int(len(y)), "faults": int((y == 1).sum())})
            add("ANN score, per-machine threshold", enrol.ann_score.to_numpy(), rest.ann_score.to_numpy())
            Ee, Er = np.stack(enrol.ann_emb), np.stack(rest.ann_emb)
            add("ANN embedding kNN (enrolment)", knn(Ee, Ee, self_in_bank=True), knn(Er, Ee))
        med = float(np.median(enrol.level))
        add("Loudness deviation (enrolment)", np.abs(enrol.level - med).to_numpy(), np.abs(rest.level - med).to_numpy())
        if clap is not None and g.clap_emb.notna().all():
            Ce, Cr = np.stack(enrol.clap_emb), np.stack(rest.clap_emb)
            add("CLAP kNN (enrolment)", knn(Ce, Ce, self_in_bank=True), knn(Cr, Ce))
    T = pd.DataFrame(rows)
    T.to_csv(R / "posthoc_calibration.csv", index=False)

    # summaries: test machine (macro over pump+fan) and mean over all machines each approach can use
    def summ(df):
        return df.groupby("approach", sort=False).agg(
            machines=("machine", "nunique"), TPR_at_FPR5=("TPR@FPR5", "mean"), AUC=("AUC", "mean"),
            recall=("recall_at_enrol_thr", "mean"), FPR=("FPR_at_enrol_thr", "mean"),
            false_alarms_per_day=("false_alarms_per_day", "mean")).round(3)
    S_test, S_all = summ(T[T.role == "TEST machine"]), summ(T)
    print("\nTEST machines (id_06), mean of pump and fan:\n", S_test.to_string())
    print("\nAll machines each approach can use:\n", S_all.to_string())

    md = ["# Post-hoc study: per-machine calibration and normal-only detection",
          "", "Run AFTER the locked Q3/Q6 test evaluations; rules fixed in scripts/posthoc_calibration.py before running. "
          f"Enrolment = first {N_ENROL} normal clips of each machine; threshold = highest enrolment score; k = {K}; "
          f"false alarms per day assume one clip every 10 minutes ({CLIPS_PER_DAY} clips).", "",
          "## Test machines (id_06), mean of pump and fan", "", S_test.to_markdown(), "",
          "## All machines each approach can be evaluated on (ANN: id_04 + id_06 only)", "", S_all.to_markdown(), "",
          "## Every machine", "", T.to_markdown(index=False), ""]
    (R / "posthoc_calibration.md").write_text("\n".join(md), encoding="utf-8")

    # figure: recall vs false alarms per day at the enrolment threshold, per machine
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    colors = {"ANN score, validation threshold (as deployed)": "#7f8c8d", "ANN score, per-machine threshold": "#2e86c1",
              "ANN embedding kNN (enrolment)": "#16a085", "Loudness deviation (enrolment)": "#f39c12",
              "CLAP kNN (enrolment)": "#c0392b"}
    for ax, mt in zip(axs, ["pump", "fan"]):
        d = T[T.machine_type == mt]
        for ap, c in colors.items():
            x = d[d.approach == ap]
            ax.scatter(x.FPR_at_enrol_thr * 100, x.recall_at_enrol_thr * 100, color=c, s=28, label=ap,
                       marker="s" if "TEST" in "".join(x.role) and False else "o")
            for r in x.itertuples():
                if r.role == "TEST machine":
                    ax.scatter(r.FPR_at_enrol_thr * 100, r.recall_at_enrol_thr * 100, s=90, facecolors="none", edgecolors="k")
        ax.set_xscale("symlog", linthresh=10); ax.set_title(f"{mt}: every machine (circled = test id_06)", fontsize=9)
        ax.set_xlabel("false-alarm rate on normal clips (%)")
    axs[0].set_ylabel("faults caught (%)")
    axs[1].legend(fontsize=6.5, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "posthoc_calibration.png", dpi=200); plt.close()
    print("\nSaved reports/posthoc_calibration.md/.csv and figures/posthoc_calibration.png")


if __name__ == "__main__":
    main()
