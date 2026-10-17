"""
Question 4 – build the small reference file the web app needs: app/reference/q4_reference.json

Uses TRAIN and VALIDATION data only (the test machine is never touched):
  * validation scores of the deployed model and of the other seeds of the final design
      -> "false-alarm rate on the validation machine at this score" shown in the app
      -> each seed's own 5 % FPR threshold (for the seed-agreement vote)
  * the level and spectrum-shape range of all development clips (train + validation machines)
      -> out-of-distribution checks in the app
  * the Q3 headline results (copied from reports/q3_comparison.csv) for the limitations panel

Run once after Question 3:   python -m scripts.q4_build_reference
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.dataset import MimiiSpecDataset  # noqa: E402
from src.evaluation import get_device, predict_scores, type_thresholds  # noqa: E402
from src.model import FaultCNN  # noqa: E402

MODELS = config.ROOT / "models"
OUT = config.ROOT / "app" / "reference" / "q4_reference.json"


def load(path: Path, device) -> FaultCNN:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = FaultCNN(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def level_db(S: np.ndarray) -> np.ndarray:
    """Per-clip level: dB of the mean power over all mel bins (same as app/inference.py)."""
    return 10 * np.log10(np.mean(10 ** (S / 10), axis=(1, 2)) + 1e-12)


def main() -> None:
    device = get_device()
    meta = json.loads((MODELS / "final_model.json").read_text())
    final = json.loads((config.ROOT / "experiments" / "final_config.json").read_text())
    base = final["selected_from"]
    seed_runs = [r for r in [base, f"{base}_s43", f"{base}_s44"] if (MODELS / f"{r}.pt").exists()]
    print(f"Deployed: {meta['source_run']} | seeds of the final design: {seed_runs}")

    val = MimiiSpecDataset("val")
    yv, tv = val.labels.astype(int), val.index.machine_type.to_numpy()
    validation, seed_thr = {}, {}
    for run in seed_runs:
        s = predict_scores(load(MODELS / f"{run}.pt", device), val, device)
        seed_thr[run] = {k: float(v) for k, v in type_thresholds(yv, s, tv).items()}
        if run == meta["source_run"]:
            for t in ["pump", "fan"]:
                k = tv == t
                validation[t] = {"normal_scores": np.round(np.sort(s[k & (yv == 0)]), 5).tolist(),
                                 "abnormal_scores": np.round(np.sort(s[k & (yv == 1)]), 5).tolist()}
            ok = all(abs(seed_thr[run][t] - meta["thresholds_val_fpr5"][t]) < 1e-3 for t in ["pump", "fan"])
            print(f"  thresholds recomputed for {run}: {seed_thr[run]}  "
                  f"({'match' if ok else 'DIFFER FROM'} models/final_model.json)")
    if not validation:
        raise SystemExit("The deployed run is not among the seed runs - check models/ and experiments/final_config.json")

    ranges = {}
    for t in ["pump", "fan"]:
        Ss = []
        for split in ["train", "val"]:
            X = np.load(config.FEATURE_DIR / f"{split}.npy", mmap_mode="r")
            idx = pd.read_csv(config.FEATURE_DIR / f"{split}_index.csv")
            Ss.append(np.asarray(X[np.where(idx.machine_type == t)[0]], dtype=np.float32))
        S = np.concatenate(Ss)
        prof = S.mean(axis=(0, 2))
        lv = level_db(S)
        dist = np.std(S.mean(axis=2) - prof[None, :], axis=1)
        ranges[t] = {"n_clips": int(len(S)), "profile_db": np.round(prof, 3).tolist(),
                     "level_p01": float(np.percentile(lv, 1)), "level_p99": float(np.percentile(lv, 99)),
                     "level_min": float(lv.min()), "level_max": float(lv.max()),
                     "shape_p50": float(np.percentile(dist, 50)), "shape_p99": float(np.percentile(dist, 99))}
        print(f"  {t}: {len(S)} development clips | level {ranges[t]['level_p01']:.1f}..{ranges[t]['level_p99']:.1f} dB "
              f"| shape distance p99 {ranges[t]['shape_p99']:.2f}")

    evaluation = {}
    comp = config.REPORTS_DIR / "q3_comparison.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        evaluation = {r.Model: {"test_macro": float(r["Test macro TPR@FPR5"]), "val_macro": float(r["Val macro TPR@FPR5"]),
                                "pump": float(r["Pump TPR@FPR5"]), "fan": float(r["Fan TPR@FPR5"])}
                      for _, r in c.iterrows() if not r.Model.startswith("Random")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "n/a"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"built": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
                               "deployed_run": meta["source_run"], "seed_runs": seed_runs,
                               "seed_thresholds": seed_thr, "validation": validation, "ranges": ranges,
                               "evaluation": evaluation}, indent=1))
    print(f"Saved {OUT.relative_to(config.ROOT)}  ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
