"""
Re-run the whole of Question 1 on the data that is ALREADY on your computer.
This script NEVER downloads anything.

Steps
  0. check the audio is present in data/raw (6_dB_pump + 6_dB_fan)
  1. build metadata        -> data/processed/metadata.csv
  2. make the fixed split  -> data/processed/splits.csv
  3. leakage checks        -> reports/leakage_report.md
  4. (optional) execute notebooks/01_eda.ipynb and save all outputs + figures

Usage (from the project folder, with .venv active):
    python -m scripts.run_q1                 # steps 0-3 (about 2 minutes)
    python -m scripts.run_q1 --notebook      # steps 0-4 (about 5-10 minutes)
    python -m scripts.run_q1 --fresh --notebook
        --fresh deletes the old cached results first (NOT the audio) so everything is recomputed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

EXPECTED = {"6_dB_pump": 4205, "6_dB_fan": 5550}   # wav files per folder in the official zips


def banner(text: str) -> None:
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def check_data() -> None:
    banner("STEP 0  Checking the audio already on this computer (no download)")
    ok = True
    for folder, expected in EXPECTED.items():
        d = config.RAW_DIR / folder
        n = sum(1 for _ in d.rglob("*.wav")) if d.exists() else 0
        status = "OK" if n == expected else "PROBLEM"
        print(f"  {folder:<10} {n:>5} wav files (expected {expected})  -> {status}")
        ok &= n == expected
    if not ok:
        raise SystemExit(
            "\nThe audio is missing or incomplete in data/raw.\n"
            "Do NOT delete anything. Check the folder names, or ask for help before downloading again.")


def clean_cache() -> None:
    banner("Removing old results (the audio is NOT touched)")
    targets = [config.METADATA_CSV, config.SPLITS_CSV, config.FEATURE_CACHE,
               *config.REPORTS_DIR.glob("*.md"), *config.REPORTS_DIR.glob("*.csv"),
               *config.REPORTS_DIR.glob("*.json"), *config.FIGURES_DIR.glob("*.png")]
    for t in targets:
        if t.exists() and t.name not in {"Q1_results_summary.md"}:
            t.unlink()
            print(f"  deleted {t.relative_to(config.ROOT)}")


def run_notebook() -> None:
    banner("STEP 4  Executing notebooks/01_eda.ipynb (saves outputs and figures)")
    import nbformat
    from nbclient import NotebookClient

    path = config.ROOT / "notebooks" / "01_eda.ipynb"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    nbformat.write(nb, path)
    print(f"  saved with outputs -> {path.relative_to(config.ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", action="store_true", help="also execute the EDA notebook")
    p.add_argument("--fresh", action="store_true", help="delete cached results first (audio is kept)")
    args = p.parse_args()
    t0 = time.time()

    check_data()
    if args.fresh:
        clean_cache()

    from src import build_metadata, make_splits, leakage_checks
    banner("STEP 1  Building metadata")
    build_metadata.main()
    banner("STEP 2  Making the fixed train / validation / test split")
    make_splits.main()
    banner("STEP 3  Leakage and data-quality checks")
    leakage_checks.main()

    if args.notebook:
        run_notebook()

    banner(f"ALL DONE in {(time.time() - t0) / 60:.1f} minutes")
    print("Outputs: data/processed/metadata.csv, splits.csv, reports/leakage_report.md",
          "+ notebooks/01_eda.ipynb and reports/figures/*.png" if args.notebook else "")


if __name__ == "__main__":
    main()
