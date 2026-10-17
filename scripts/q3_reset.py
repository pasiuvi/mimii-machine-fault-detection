"""
Question 3 â€“ start Q3 again cleanly, WITHOUT losing the evidence of the earlier round.

What it does
  1. ARCHIVES the small result files of the earlier round into experiments/archive/round0_<date>/
     (experiment_log.csv, baseline_search.csv, final_config.json, every runs/<id>/history.csv +
      summary.json, reports/q3_* and figures/q3_*). These stay in Git: hiding unsuccessful
     runs is not allowed (brief section 6), and Round 0 is the reason for the Round 1 design.
  2. DELETES the large/derived files so nothing old is reused by mistake:
     models/E*.pt, models/SMOKE.pt, models/baseline.joblib, models/trivial.joblib,
     models/final_model.*, data/processed/baseline_features.npz, experiments/runs/,
     the Q3 rows of experiments/experiment_log.csv.
  It does NOT touch: data/raw, the log-mel cache (data/processed/logmel128), norm_stats.npz,
  splits, Q1/Q2 reports, notebooks 01/02.

Safety: refuses to run if the TEST set has already been evaluated (reports/.test_evaluated.json),
because then a "fresh start" would no longer be honest. Run with --dry-run first to see the list.

Run:  python -m scripts.q3_reset --dry-run
      python -m scripts.q3_reset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

ROOT = config.ROOT
EXP, MODELS, R, FIG = ROOT / "experiments", ROOT / "models", config.REPORTS_DIR, config.FIGURES_DIR
LOCK = R / ".test_evaluated.json"
Q3_GROUPS = {"benchmark", "baseline", "initial", "improve", "lr", "bs", "depth", "optim", "ablation", "domain",
             "failed", "final", "smoke", "debug", "final-test"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only show what would happen")
    a = ap.parse_args()
    if LOCK.exists():
        raise SystemExit(f"STOP: the test set was already evaluated ({LOCK.read_text()}).\n"
                         "A fresh start would not be honest any more - ask before doing anything.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    arch = EXP / "archive" / f"round0_{stamp}"
    to_archive = [p for p in [EXP / "experiment_log.csv", EXP / "baseline_search.csv", EXP / "final_config.json"]
                  if p.exists()]
    to_archive += sorted((EXP / "runs").glob("*/history.csv")) + sorted((EXP / "runs").glob("*/summary.json"))
    to_archive += sorted(p for p in R.glob("q3_*") if p.name.startswith("q3_")) + sorted(FIG.glob("q3_*.png"))
    to_delete = sorted(MODELS.glob("E*.pt")) + [p for p in [
        MODELS / "SMOKE.pt", MODELS / "baseline.joblib", MODELS / "trivial.joblib", MODELS / "final_model.pt",
        MODELS / "final_model.json", config.PROCESSED_DIR / "baseline_features.npz"] if p.exists()]
    to_delete += sorted(p for p in R.glob("q3_*") if p.name.startswith("q3_")) + sorted(FIG.glob("q3_*.png"))

    print(f"ARCHIVE -> {arch.relative_to(ROOT)}  ({len(to_archive)} files)")
    for p in to_archive:
        print("   ", p.relative_to(ROOT))
    print(f"DELETE ({len(to_delete)} files + folder experiments/runs/)")
    for p in to_delete:
        print("   ", p.relative_to(ROOT))
    if a.dry_run:
        print("\n--dry-run: nothing changed."); return
    if not to_archive and not to_delete:
        print("Nothing to reset - Q3 is already clean."); return

    for p in to_archive:
        dst = arch / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for p in to_delete:
        p.unlink(missing_ok=True)
    shutil.rmtree(EXP / "runs", ignore_errors=True)

    log = EXP / "experiment_log.csv"
    if log.exists():
        df = pd.read_csv(log)
        keep = df[~df["group"].isin(Q3_GROUPS)] if "group" in df else df.iloc[0:0]
        if len(keep):
            keep.to_csv(log, index=False)
        else:
            log.unlink()
    for p in [EXP / "baseline_search.csv", EXP / "final_config.json"]:
        p.unlink(missing_ok=True)
    (arch / "README.md").write_text(
        f"# Round 0 (archived {stamp})\n\nEarlier Q3 attempt, kept for honesty and as the evidence that "
        "motivated the Round 1 plan (src/experiment_plan.py docstring). Validation data only; the test set "
        "was NOT used.\n", encoding="utf-8")
    print(f"\nDone. Round 0 archived in {arch.relative_to(ROOT)}. Q3 is clean - start with the smoke test.")


if __name__ == "__main__":
    main()
