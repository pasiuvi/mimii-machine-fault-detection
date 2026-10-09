"""
Step 4: explicit leakage and data-quality checks (brief section 4 + Question 1).

Checks
  1. Group isolation    - no physical machine appears in two splits
  2. Exact duplicates   - identical audio in two splits (hash of decoded samples)
  3. Near-duplicates    - almost identical clips across splits (nearest neighbours
                          on log-mel summaries, then confirmed on full spectrograms)
  4. Prediction-time    - which columns may / may not be used as model input
  5. Shortcut signals   - can trivial cues (loudness, file size) predict the label?
                          (also gives the "trivial benchmark" needed in Question 3a)
  6. Data quality       - unreadable files, wrong sample rate / channels / duration

Output: reports/leakage_report.md  (paste the tables into your report, Question 1)

Run:  python -m src.leakage_checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config
from src.audio_utils import load_audio, log_mel
from src.eda_features import get_summary

lines: list[str] = []
results: dict = {}


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def main() -> None:
    meta = pd.read_csv(config.METADATA_CSV)
    df = pd.read_csv(config.SPLITS_CSV)
    feats = get_summary()
    df["rms_db"] = feats["rms_db"]
    df["audio_hash"] = feats["audio_hash"]
    df = df.merge(meta[["path", "file_bytes"]], on="path", how="left")

    log("# Leakage and data-quality report\n")
    log(f"Clips: {len(df)} | machines (groups): {df.group.nunique()} | SNR: {config.SNR}\n")

    # 1 ---------------------------------------------------------------- groups
    per_group = df.groupby("group")["split"].nunique()
    ok1 = bool((per_group == 1).all())
    results["group_isolation_ok"] = ok1
    log("## 1. Group isolation (machine ID)")
    log(df.groupby("split")["group"].unique().apply(lambda g: ", ".join(sorted(g))).to_frame("machines").to_markdown())
    log(f"\nResult: {'PASS - every machine is in exactly one split' if ok1 else 'FAIL'}\n")

    # 2 ----------------------------------------------------- exact duplicates
    dup = df[df.duplicated("audio_hash", keep=False)]
    cross = dup.groupby("audio_hash")["split"].nunique()
    n_cross = int((cross > 1).sum())
    results["exact_duplicate_groups"] = int(dup["audio_hash"].nunique())
    results["exact_duplicates_across_splits"] = n_cross
    log("## 2. Exact duplicates (hash of decoded audio)")
    log(f"Duplicate sets: {dup['audio_hash'].nunique()} | crossing split boundaries: {n_cross}")
    if len(dup):
        dup[["path", "split", "label_name"]].to_csv(config.REPORTS_DIR / "exact_duplicates.csv", index=False)
        log("Details saved to reports/exact_duplicates.csv")
    log("")

    # 3 ------------------------------------------------------ near duplicates
    X = np.hstack([feats["mel_mean"], feats["mel_std"]])
    X = StandardScaler().fit_transform(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    train_idx = np.where(df.split == "train")[0]
    other_idx = np.where(df.split != "train")[0]
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X[train_idx])

    d_other, j_other = nn.kneighbors(X[other_idx], n_neighbors=1)
    d_train, _ = nn.kneighbors(X[train_idx], n_neighbors=2)   # [:,1] = nearest *other* train clip
    sim_cross = 1 - d_other[:, 0]
    sim_within = 1 - d_train[:, 1]

    # confirm the most similar cross-split pairs on the full spectrogram
    top = np.argsort(-sim_cross)[:20]
    confirmed = []
    for k in top:
        a = df.path.iloc[other_idx[k]]
        b = df.path.iloc[train_idx[j_other[k, 0]]]
        Sa, Sb = log_mel(load_audio(config.ROOT / a)), log_mel(load_audio(config.ROOT / b))
        t = min(Sa.shape[1], Sb.shape[1])
        r = float(np.corrcoef(Sa[:, :t].ravel(), Sb[:, :t].ravel())[0, 1])
        confirmed.append({"val_or_test_clip": a, "nearest_train_clip": b,
                          "summary_similarity": round(float(sim_cross[k]), 5),
                          "full_spectrogram_corr": round(r, 4), "near_duplicate": r > 0.99})
    conf = pd.DataFrame(confirmed)
    conf.to_csv(config.REPORTS_DIR / "near_duplicate_candidates.csv", index=False)
    n_near = int(conf["near_duplicate"].sum())
    results["near_duplicates_confirmed_top20"] = n_near
    log("## 3. Near-duplicates across splits")
    log("Nearest-neighbour cosine similarity on log-mel summaries:")
    log(pd.DataFrame({
        "within train (clip vs other train clip)": pd.Series(sim_within).describe(),
        "val/test clip vs nearest train clip": pd.Series(sim_cross).describe(),
    }).round(4).to_markdown())
    log(f"\nTop-20 most similar cross-split pairs checked on full spectrograms: "
        f"{n_near} with correlation > 0.99 (near-duplicates).")
    log("Interpretation: if val/test clips are clearly LESS similar to train than train clips are "
        "to each other, the group split is doing its job (unseen machines sound different).\n")

    # 4 ------------------------------------------------- prediction-time inputs
    log("## 4. Features available at prediction time")
    log("| Column | Allowed as model input? | Reason |\n|---|---|---|")
    log("| audio waveform (channel 0) | YES | the only thing the engineer uploads / the sensor records |")
    log("| label, label_name | NO | this is the target |")
    log("| path, folder names (normal/abnormal) | NO | folder name IS the label -> direct leakage |")
    log("| machine_id, group | NO | not known for a new machine; would let the model memorise machines |")
    log("| split, snr | NO | bookkeeping only |")
    same_names = set(df[df.label == 0].path.str.rsplit("/", n=1).str[-1]) & \
                 set(df[df.label == 1].path.str.rsplit("/", n=1).str[-1])
    log(f"\nFile names shared by normal and abnormal clips: {len(same_names)} "
        "(file numbering restarts per folder, so the file name alone carries no label - "
        "but the folder path does, so the data loader passes ONLY the waveform to the model).\n")

    # 5 ------------------------------------------------------- shortcut signals
    log("## 5. Shortcut signals / trivial benchmark (fit on train, scored on validation)")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    rows = []
    for col in ["rms_db", "file_bytes"]:
        if tr[col].nunique() < 2 or va.label.nunique() < 2:
            continue
        clf = LogisticRegression().fit(tr[[col]], tr.label)
        auc = roc_auc_score(va.label, clf.predict_proba(va[[col]])[:, 1])
        rows.append({"single feature": col, "validation ROC-AUC": round(auc, 3)})
    short = pd.DataFrame(rows)
    results["shortcut_auc"] = short.to_dict("records")
    log(short.to_markdown(index=False))
    log("\nAUC near 0.5 = no shortcut. Clearly above 0.5 = loudness alone separates classes: "
        "report it, use it as the trivial benchmark in Q3a, and check the CNN beats it.\n")

    # 6 ------------------------------------------------------------- quality
    log("## 6. Data quality")
    q = {
        "unreadable files": int((~meta["readable"]).sum()),
        "sample rates": sorted(meta["sample_rate"].dropna().astype(int).unique().tolist()),
        "channel counts": sorted(meta["channels"].dropna().astype(int).unique().tolist()),
        "durations (s)": sorted(meta["duration_s"].dropna().round(2).unique().tolist())[:10],
        "silent clips (RMS < -60 dB)": int((df.rms_db < -60).sum()),
    }
    results["quality"] = q
    for k, v in q.items():
        log(f"- {k}: {v}")

    (config.REPORTS_DIR / "leakage_report.md").write_text("\n".join(lines), encoding="utf-8")
    (config.REPORTS_DIR / "leakage_report.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nSaved -> reports/leakage_report.md")


if __name__ == "__main__":
    main()
