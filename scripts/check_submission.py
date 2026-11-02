"""
Pre-submission check: does the repository contain everything the brief asks for?

    python -m scripts.check_submission          (add --tests to also run the pytest suites)

Prints OK / MISSING for each item and a short summary. It changes nothing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ITEMS = {
    "Documentation": ["README.md", "requirements.txt", ".gitignore"],
    "Data procedure (Q1)": ["scripts/download_mimii.py", "src/build_metadata.py", "src/make_splits.py",
                            "src/leakage_checks.py", "data/processed/splits.csv", "data/processed/metadata.csv",
                            "reports/leakage_report.md", "notebooks/01_eda.ipynb"],
    "Design (Q2)": ["src/model.py", "src/dataset.py", "src/features.py", "reports/Q2_prediction.md",
                    "notebooks/02_solution_design.ipynb"],
    "Training and evaluation (Q3)": ["src/train.py", "src/experiment_plan.py", "src/metrics.py", "src/evaluation.py",
                                     "scripts/q3_baseline.py", "scripts/q3_run_experiments.py",
                                     "scripts/q3_final_evaluate.py", "scripts/q3_extra_analysis.py",
                                     "experiments/experiment_log.csv", "experiments/final_config.json",
                                     "reports/q3_comparison.md", "reports/.test_evaluated.json",
                                     "reports/Q3_round1_plan.md", "reports/Q3_round2_combination.md",
                                     "notebooks/03_model_development.ipynb"],
    "Web app (Q4)": ["app/main.py", "app/inference.py", "app/static/index.html", "app/reference/q4_reference.json",
                     "tests/test_app.py", "scripts/q4_test_inputs.py", "reports/q4_test_results.md"],
    "XAI (Q5)": ["app/xai.py", "scripts/q5_xai_analysis.py", "tests/test_xai.py", "reports/q5_xai_results.md",
                 "app/reference/q5_xai.json"],
    "Foundation model (Q6)": ["scripts/q6_clap.py", "experiments/q6_config.json", "reports/q6_selection.md",
                              "reports/q6_results.md", "reports/.q6_test_evaluated.json"],
    "Post-hoc study": ["scripts/posthoc_calibration.py", "reports/posthoc_calibration.md"],
    "AcoustiGuard v2 – enrolment detector": ["scripts/v2_enrolment.py", "reports/v2_selection.md",
                                             "experiments/v2_config.json", "reports/v2_test_results.md",
                                             "reports/.v2_test_evaluated.json"],
    "Deployment": ["deploy/Dockerfile", "deploy/prepare_space.py"],
    "Model files (local only, not in Git)": ["models/final_model.pt", "models/final_model.json"],
}
TAGS = ["q3-before-test", "q3-test-done", "q4-web-app", "q5-xai", "q6-before-test", "q6-test-done", "v2-before-test", "v2-test-done"]
README_WORDS = ["install", "download", "split", "train", "uvicorn", "test"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tests", action="store_true"); a = ap.parse_args()
    missing = 0
    for group, files in ITEMS.items():
        print(f"\n{group}")
        for f in files:
            ok = (ROOT / f).exists()
            missing += not ok
            print(f"  {'OK     ' if ok else 'MISSING'} {f}")
    try:
        tags = subprocess.check_output(["git", "tag"], cwd=ROOT, text=True).split()
        print("\nGit tags")
        for t in TAGS:
            ok = t in tags; missing += not ok
            print(f"  {'OK     ' if ok else 'MISSING'} {t}")
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        print(f"  {'OK     ' if not dirty else 'WARNING'} working tree {'clean' if not dirty else 'has uncommitted changes'}")
    except Exception:
        print("\nGit not available – tags not checked")
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore").lower() if (ROOT / "README.md").exists() else ""
    print("\nREADME covers")
    for w in README_WORDS:
        ok = w in readme; missing += not ok
        print(f"  {'OK     ' if ok else 'MISSING'} '{w}'")
    if a.tests:
        print("\nRunning tests ...")
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
        missing += r.returncode != 0
    print(f"\n{'ALL GOOD' if missing == 0 else f'{missing} item(s) to fix'}")


if __name__ == "__main__":
    main()
