"""
Question 3 – run the planned neural-network experiments (validation only).

Examples (run from the project folder, .venv active):
    python -m scripts.q3_run_experiments --smoke                # 1-epoch test, then cleans up
    python -m scripts.q3_run_experiments --list                 # show the plan and what is done
    python -m scripts.q3_run_experiments --group initial improve
    python -m scripts.q3_run_experiments --group lr bs depth
    python -m scripts.q3_run_experiments --group optim ablation domain failed
    python -m scripts.q3_run_experiments --group combine          # Round 2 confirmation
    python -m scripts.q3_run_experiments --ids E006             # one experiment
    python -m scripts.q3_run_experiments --all                  # everything still missing
    python -m scripts.q3_run_experiments --final                # pick best config + 2 extra seeds

Finished experiments are skipped automatically (safe to stop and continue later).
Use --rerun to force a run again, --max-epochs 2 for a quick smoke test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.experiment_plan import EXPERIMENTS, GROUPS, PLAN, REF_ID  # noqa: E402
from src.train import RUNS_DIR, run_experiment  # noqa: E402

FINAL_JSON = config.ROOT / "experiments" / "final_config.json"
FINAL_SEEDS = [43, 44]   # seed 42 is the development run itself


def done(exp_id: str) -> bool:
    return (RUNS_DIR / exp_id / "summary.json").exists()


def show_plan() -> None:
    rows = []
    for e in EXPERIMENTS:
        s = json.loads((RUNS_DIR / e["id"] / "summary.json").read_text()) if done(e["id"]) else {}
        rows.append({"id": e["id"], "group": e["group"], "done": "yes" if s else "",
                     "val_macro_TPR": round(s["val_macro_TPR@FPR5"], 3) if s else "",
                     "stable": round(s["val_stable_macro_TPR@FPR5"], 3) if s else "",
                     "what": e["what"][:70]})
    print(pd.DataFrame(rows).to_string(index=False))


def select_final() -> dict:
    """Pick the final design on VALIDATION only (never test).
    Rule (fixed before Round 1): highest STABLE val macro TPR@FPR5 = mean over the best epoch and
    its two neighbours, so one lucky epoch cannot win; tie-break val macro pAUC(0.1).
    Candidates: every finished, non-diverged run except the initial model."""
    cands = []
    for e in EXPERIMENTS:
        if e["group"] == "initial" or not done(e["id"]):
            continue
        s = json.loads((RUNS_DIR / e["id"] / "summary.json").read_text())
        if not s.get("diverged"):
            cands.append(s)
    if not cands:
        raise SystemExit("No development runs finished yet.")
    best = max(cands, key=lambda s: (s["val_stable_macro_TPR@FPR5"], s["val_macro_pAUC_0.1"]))
    top = sorted(cands, key=lambda s: -s["val_stable_macro_TPR@FPR5"])[:5]
    sel = {"selected_from": best["exp_id"],
           "rule": "max STABLE validation macro TPR@FPR5 (best epoch +-1), tie-break validation macro pAUC(0.1)",
           "n_candidates": len(cands), "val_macro_TPR@FPR5": best["val_macro_TPR@FPR5"],
           "val_stable_macro_TPR@FPR5": best["val_stable_macro_TPR@FPR5"],
           "top5": [{"id": s["exp_id"], "stable": round(s["val_stable_macro_TPR@FPR5"], 4),
                     "best": round(s["val_macro_TPR@FPR5"], 4)} for s in top],
           "cfg": best["cfg"]}
    FINAL_JSON.write_text(json.dumps(sel, indent=2))
    print(f"\nSelected {best['exp_id']} (stable val macro TPR@FPR5 = {best['val_stable_macro_TPR@FPR5']:.3f}, "
          f"best epoch {best['val_macro_TPR@FPR5']:.3f}) out of {len(cands)} runs -> "
          f"{FINAL_JSON.relative_to(config.ROOT)}")
    for s in sel["top5"]:
        print(f"   {s['id']:<6} stable {s['stable']:.3f}  best {s['best']:.3f}")
    return sel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--group", nargs="+", choices=GROUPS)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--all", action="store_true")
    p.add_argument("--final", action="store_true", help="select best config and run 2 extra seeds")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1-epoch test of the whole training loop, then clean up")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke test only)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    a = p.parse_args()

    if a.list:
        show_plan(); return
    if a.smoke:
        import shutil
        from src.train import LOG_CSV, MODELS_DIR
        s = run_experiment("SMOKE", PLAN[REF_ID]["cfg"], what="smoke test", group="smoke", max_epochs=1,
                           num_workers=a.workers)
        shutil.rmtree(RUNS_DIR / "SMOKE", ignore_errors=True)
        (MODELS_DIR / "SMOKE.pt").unlink(missing_ok=True)
        log = pd.read_csv(LOG_CSV); log[log.exp_id != "SMOKE"].to_csv(LOG_CSV, index=False)
        print(f"\nSMOKE TEST OK on {s['device']} - one epoch took about {s['train_time_min']:.1f} min "
              f"-> a full 25-epoch run takes at most about {25 * s['train_time_min']:.0f} min (early stopping "
              "usually ends sooner). Test outputs removed.")
        return

    todo = []
    if a.all:
        todo = [e["id"] for e in EXPERIMENTS]
    if a.group:
        todo += [e["id"] for e in EXPERIMENTS if e["group"] in a.group]
    if a.ids:
        todo += [i for i in a.ids if i in PLAN]
    for exp_id in dict.fromkeys(todo):
        if done(exp_id) and not a.rerun:
            print(f"[skip] {exp_id} already done (use --rerun to repeat)")
            continue
        e = PLAN[exp_id]
        run_experiment(exp_id, e["cfg"], what=e["what"], group=e["group"], hypothesis=e["hypothesis"],
                       max_epochs=a.max_epochs, num_workers=a.workers)

    if a.final:
        sel = select_final()
        for seed in FINAL_SEEDS:
            exp_id = f"{sel['selected_from']}_s{seed}"
            if done(exp_id) and not a.rerun:
                print(f"[skip] {exp_id} already done"); continue
            cfg = {**sel["cfg"], "seed": seed}
            run_experiment(exp_id, cfg, what=f"Final config ({sel['selected_from']}) with seed {seed}",
                           group="final", hypothesis="Measures run-to-run variance of the selected design",
                           max_epochs=a.max_epochs, num_workers=a.workers)

    if not (a.list or todo or a.final):
        p.print_help()


if __name__ == "__main__":
    main()
