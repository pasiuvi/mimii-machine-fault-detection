"""
Question 5 – explanation analysis for the DEPLOYED model (models/final_model.pt).

    python -m scripts.q5_xai_analysis            (about 5-10 min on the laptop GPU)

Nothing is trained and no model is changed. The script only explains predictions of the deployed network.

Rule written BEFORE running (so the choice cannot be tuned to look good):
  * The Grad-CAM layer used in the web app is chosen on the VALIDATION machine (id_04) only:
    the layer with the larger mean faithfulness gap (logit drop when the top 20 % of the map is hidden
    minus the mean drop for 10 random 20 % masks). Tie (< 0.05) -> the standard last conv layer, features.8.
  * Test-machine clips (id_06) are used only to DESCRIBE explanations of predictions that already exist
    (the demo clips and the hardest errors from Question 3). They are never used to choose anything.

Parts
  1. Method comparison on validation clips: Grad-CAM (2 layers), occlusion (model-agnostic), Integrated
     Gradients – faithfulness (deletion test), agreement with occlusion, and measured cost per clip.
  2. Sanity check A – deletion curves: hide 5-50 % of the most highlighted area vs random areas.
  3. Sanity check B – model randomisation (Adebayo et al., 2018): re-initialise the head and the last
     blocks; a useful explanation must change when the learned weights are destroyed.
  4. Where the model looks on average (frequency profile per machine type and label, validation).
  5. Case studies: the 8 demo clips of the app (incl. missed faults and false alarms) + 3 hardest errors.

Outputs: reports/q5_xai_results.md, reports/q5_*.csv, reports/figures/q5_*.png, app/reference/q5_xai.json
"""
from __future__ import annotations

import copy
import warnings
import json
import subprocess
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from app import xai
from app.inference import FaultDetector, decode_and_check, logmel
from app.main import load_examples
from src import config

ROOT = config.ROOT
warnings.filterwarnings("ignore", message=".*constant.*")
FIG = config.FIGURES_DIR
REP = config.REPORTS_DIR
LAYERS = ["features.8", "features.5"]
N_PER_GROUP = 25                 # validation clips per (machine type, label) for the comparison
N_SLOW = 6                       # per group for the slow methods (occlusion, IG) and randomisation
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_clip(det, path, mtype):
    data = (ROOT / path).read_bytes()
    a = decode_and_check(data, path)
    S = logmel(a.y)
    xt, mt = det.model_input(S, mtype)
    return S, xt, mt


def gap_for(model, xt, mt, cell_grid):
    c = xai.deletion_check(model, xt, mt, cell_grid)
    c.pop("mask")
    return c


def cam_cells(model, xt, mt, layer, shape):
    g, logit = xai.gradcam(model, xt, mt, layer)
    if not np.isfinite(g).all() or g.max() <= 0:
        return None, None, logit
    full = xai.upsample(g / g.max(), shape)
    return xai.pool(full), full, logit


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, 1000 * (time.perf_counter() - t)


def deletion_curve(model, xt, mt, cells, fractions=FRACTIONS, n_random=3):
    rows = []
    for f in fractions:
        c = xai.deletion_check(model, xt, mt, cells, top_fraction=f, n_random=n_random)
        rows.append((f, c["logit_drop_top"], c["logit_drop_random"]))
    return rows


def randomise(model, names, seed=0):
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    for name in names:
        mod = xai._module(m, name)
        for p in mod.parameters():
            with torch.no_grad():
                if p.dim() > 1:
                    p.copy_(torch.randn(p.shape, generator=g).to(p.device) * p.std().clamp_min(1e-3))
                else:
                    p.copy_(torch.randn(p.shape, generator=g).to(p.device) * 0.1)
    return m.eval()


def overlay_ax(ax, S, full, title, mask=None):
    secs = S.shape[1] * config.HOP_LENGTH / config.SAMPLE_RATE
    ext = [0, secs, 0, S.shape[0]]
    ax.imshow(S, origin="lower", aspect="auto", cmap="gray", extent=ext,
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    if full is not None:
        ax.imshow(np.ma.masked_less(full / (full.max() + 1e-12), 0.05), origin="lower", aspect="auto",
                  cmap="inferno", alpha=0.6, extent=ext, vmin=0, vmax=1)
    if mask is not None and mask.any():
        ax.contour(np.linspace(0, secs, S.shape[1]), np.arange(S.shape[0]), mask.astype(float), levels=[0.5],
                   colors="#00e5ff", linewidths=0.8)
    ticks = [0, 32, 64, 96, 127]
    ax.set_yticks(ticks, [f"{xai.MEL_HZ[t] / 1000:.1f}k" if xai.MEL_HZ[t] >= 1000 else f"{xai.MEL_HZ[t]:.0f}"
                          for t in ticks])
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def main() -> None:
    torch.manual_seed(0)
    det = FaultDetector()
    model, dev = det.models["deployed"], det.device
    print(f"Deployed model {det.deployed_id} on {dev}")
    sp = pd.read_csv(config.SPLITS_CSV)
    val = sp[sp.split == "val"]
    rng = np.random.default_rng(0)
    picks = []
    for (t, lab), g in val.groupby(["machine_type", "label"]):
        picks.append(g.sample(n=min(N_PER_GROUP, len(g)), random_state=0))
    val = pd.concat(picks).reset_index(drop=True)
    print(f"Validation clips for the comparison: {len(val)}")

    # ---------------- 1 + 2: methods, faithfulness, cost, deletion curves ----------------
    rows, curves, profiles = [], [], []
    slow_idx = set(val.groupby(["machine_type", "label"]).head(N_SLOW).index)
    example_maps = None
    for i, r in val.iterrows():
        S, xt, mt = load_clip(det, r.path, r.machine_type)
        base = {"path": r.path, "machine_type": r.machine_type, "label": int(r.label)}
        for layer in LAYERS:
            (cells, full, logit), ms = timed(lambda: cam_cells(model, xt, mt, layer, S.shape), dev)
            if cells is None:
                rows.append({**base, "method": f"Grad-CAM {layer}", "empty_map": True, "ms": ms})
                continue
            c = gap_for(model, xt, mt, cells)
            rows.append({**base, "method": f"Grad-CAM {layer}", "empty_map": False, "ms": ms, "logit": logit,
                         "drop_top": c["logit_drop_top"], "drop_random": c["logit_drop_random"],
                         "passed": c["passed"], "cells": cells})
            for f, dt, dr in deletion_curve(model, xt, mt, cells):
                curves.append({"method": f"Grad-CAM {layer}", "fraction": f, "drop_top": dt, "drop_random": dr})
            prof = full.mean(axis=1)
            profiles.append({**base, "layer": layer, "profile": prof / (prof.sum() + 1e-12)})
        if i in slow_idx:
            occ, ms = timed(lambda: xai.occlusion_grid(model, xt, mt), dev)
            c = gap_for(model, xt, mt, np.clip(occ, 0, None) + 1e-9 * np.random.default_rng(i).random(occ.shape))
            rows.append({**base, "method": "Occlusion", "empty_map": False, "ms": ms, "drop_top": c["logit_drop_top"],
                         "drop_random": c["logit_drop_random"], "passed": c["passed"], "cells": occ})
            ig, ms = timed(lambda: xai.integrated_gradients(model, xt, mt), dev)
            igc = xai.pool(np.clip(ig, 0, None))
            c = gap_for(model, xt, mt, igc)
            rows.append({**base, "method": "Integrated Gradients", "empty_map": False, "ms": ms,
                         "drop_top": c["logit_drop_top"], "drop_random": c["logit_drop_random"], "passed": c["passed"],
                         "cells": igc})
            if example_maps is None and r.label == 1:
                example_maps = (S, r, {"Occlusion": xai.upsample(np.clip(occ, 0, None), S.shape),
                                       "Integrated Gradients": np.clip(ig, 0, None)}, xt, mt)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(val)} clips")

    R = pd.DataFrame(rows)
    # agreement with occlusion (same clips, same 8x20 grid)
    occ_by = {r.path: r.cells for r in R[R.method == "Occlusion"].itertuples()}
    R["rho_vs_occlusion"] = [spearmanr(np.ravel(r.cells), np.ravel(occ_by[r.path]))[0]
                             if (r.path in occ_by and isinstance(r.cells, np.ndarray) and r.method != "Occlusion")
                             else np.nan for r in R.itertuples()]
    R["gap"] = R.drop_top - R.drop_random
    summ = (R[~R.empty_map.astype(bool)].groupby("method")
            .agg(clips=("gap", "size"), drop_top=("drop_top", "mean"), drop_random=("drop_random", "mean"),
                 gap=("gap", "mean"), pass_rate=("passed", "mean"), rho_vs_occlusion=("rho_vs_occlusion", "mean"),
                 ms_per_clip=("ms", "median")).round(3))
    empty = R.groupby("method").empty_map.mean().round(3)
    summ["empty_maps"] = empty
    summ = summ.reindex([f"Grad-CAM {l}" for l in LAYERS] + ["Occlusion", "Integrated Gradients"])
    print("\n", summ.to_string())
    R.drop(columns=["cells"]).to_csv(REP / "q5_xai_methods_per_clip.csv", index=False)
    summ.to_csv(REP / "q5_xai_methods.csv")

    # pre-defined layer rule: compare the layers on the SAME clips (both maps non-empty)
    ok = R[R.method.str.startswith("Grad-CAM") & ~R.empty_map.astype(bool)]
    both = ok.groupby("path").method.nunique()
    common = both[both == len(LAYERS)].index
    gaps = {l: round(float(ok[(ok.method == f"Grad-CAM {l}") & ok.path.isin(common)].gap.mean()), 3) for l in LAYERS}
    print(f"Clips with a non-empty map for both layers: {len(common)}")
    chosen = LAYERS[1] if gaps[LAYERS[1]] - gaps[LAYERS[0]] >= 0.05 else LAYERS[0]
    print(f"\nLayer rule: gaps {gaps} -> app uses {chosen}")

    C = pd.DataFrame(curves).groupby(["method", "fraction"]).mean().reset_index()
    fig, axs = plt.subplots(1, len(LAYERS), figsize=(8, 2.8), sharey=True)
    for ax, l in zip(axs, LAYERS):
        c = C[C.method == f"Grad-CAM {l}"]
        ax.plot(100 * c.fraction, c.drop_top, "o-", color="#c0392b", label="hide most highlighted area")
        ax.plot(100 * c.fraction, c.drop_random, "s--", color="#7f8c8d", label="hide random area")
        ax.set_title(f"Grad-CAM {l}" + ("  (used in app)" if l == chosen else ""))
        ax.set_xlabel("share of spectrogram hidden (%)")
        ax.axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("mean drop in logit (fault evidence)")
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "q5_deletion_curves.png", dpi=200); plt.close()

    # ---------------- 3: model randomisation ----------------
    cascade = [["head"], ["head", "features.8"], ["head", "features.8", "features.7"],
               ["head", "features.8", "features.7", "features.6"]]
    sub = val.loc[sorted(slow_idx)]
    rand_rows = []
    ex_path = sub.iloc[0].path
    S0, xt0, mt0 = load_clip(det, ex_path, sub.iloc[0].machine_type)
    rand_example = {"S": S0, "orig": cam_cells(model, xt0, mt0, chosen, S0.shape)[1], "maps": []}
    for k, names in enumerate(cascade):
        mr = randomise(model, names, seed=k)
        for r in sub.itertuples():
            S, xt, mt = load_clip(det, r.path, r.machine_type)
            c0, f0, _ = cam_cells(model, xt, mt, chosen, S.shape)
            c1, f1, _ = cam_cells(mr, xt, mt, chosen, S.shape)
            rho = spearmanr(np.ravel(f0), np.ravel(f1))[0] if (f0 is not None and f1 is not None) else np.nan
            rand_rows.append({"randomised": " + ".join(names), "path": r.path, "spearman_vs_original": rho})
            if r.path == ex_path:
                rand_example["maps"].append((" + ".join(names), f1))
    RR = pd.DataFrame(rand_rows)
    rsum = RR.groupby("randomised", sort=False).spearman_vs_original.agg(["mean", "std", "count"]).round(3)
    print("\nRandomisation sanity check (Spearman vs original map):\n", rsum.to_string())
    RR.to_csv(REP / "q5_sanity_randomisation.csv", index=False)

    if rand_example["orig"] is not None and rand_example["maps"]:
        n = 1 + len(rand_example["maps"])
        fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 2.2), sharey=True)
        overlay_ax(axs[0], rand_example["S"], rand_example["orig"], "trained model")
        for ax, (nm, f) in zip(axs[1:], rand_example["maps"]):
            overlay_ax(ax, rand_example["S"], f, "random: " + nm.replace("features.", "f").replace(" + ", "+"))
        plt.tight_layout(); plt.savefig(FIG / "q5_randomisation.png", dpi=200); plt.close()

    # method comparison figure (one validation fault clip)
    if example_maps:
        S, r, maps, xt, mt = example_maps
        n = len(LAYERS) + 2
        fig, axs = plt.subplots(1, n, figsize=(2.4 * n, 2.3), sharey=True)
        for ax, l in zip(axs, LAYERS):
            _, f, _ = cam_cells(model, xt, mt, l, S.shape)
            overlay_ax(ax, S, f, f"Grad-CAM {l}")
        for ax, (nm, f) in zip(axs[len(LAYERS):], maps.items()):
            overlay_ax(ax, S, f, nm)
        fig.suptitle(f"Validation {r.machine_type} fault clip {r.path.split('/')[-1]}", fontsize=9)
        plt.tight_layout(); plt.savefig(FIG / "q5_methods_example.png", dpi=200); plt.close()

    # ---------------- 4: average frequency profile ----------------
    P = pd.DataFrame(profiles)
    P = P[P.layer == chosen]
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    prof_rows = []
    for ax, t in zip(axs, ["pump", "fan"]):
        for lab, col, nm in [(0, "#2e86c1", "normal"), (1, "#c0392b", "abnormal")]:
            x = P[(P.machine_type == t) & (P.label == lab)]
            if len(x):
                m = np.mean(np.stack(x.profile.values), axis=0)
                ax.plot(xai.MEL_HZ, m, color=col, label=f"{nm} (n={len(x)})")
                for lo, hi in xai.BANDS:
                    sel = (xai.MEL_HZ >= lo) & (xai.MEL_HZ <= hi if hi == 8000 else xai.MEL_HZ < hi)
                    prof_rows.append({"machine_type": t, "label": nm, "band": f"{lo}-{hi} Hz",
                                      "share": round(float(m[sel].sum()), 3)})
        q = xai.Q1_FAULT_BANDS[t]
        ax.axvspan(*q["range"], color="#f5b041", alpha=0.25, label="Q1 fault band")
        if "range2" in q:
            ax.axvspan(*q["range2"], color="#f5b041", alpha=0.25)
        ax.set_xscale("symlog", linthresh=100); ax.set_title(t)
        ax.set_xlabel("frequency (Hz)")
    axs[0].set_ylabel("share of Grad-CAM map"); axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "q5_frequency_profile.png", dpi=200); plt.close()
    PR = pd.DataFrame(prof_rows)
    PR.to_csv(REP / "q5_frequency_profile.csv", index=False)

    # ---------------- 5: case studies ----------------
    det.xai_layer = chosen
    cases = [{"source": "demo", "path": e["path"], "machine_type": e["machine_type"], "kind": e["kind"]}
             for e in load_examples()]
    he = REP / "q3_hardest_errors.csv"
    if he.exists():
        h = pd.read_csv(he)
        h = h[h.path.apply(lambda p: (ROOT / p).exists()) & ~h.path.isin([c["path"] for c in cases])]
        for r in h.head(3).itertuples():
            cases.append({"source": "hardest error", "path": r.path, "machine_type": r.machine_type,
                          "kind": "missed fault" if r.error == "FN" else "false alarm"})
    case_rows, panels = [], []
    for c in cases:
        data = (ROOT / c["path"]).read_bytes()
        res = det.explain(data, c["path"], c["machine_type"])
        e = res["explanation"]
        truth = 1 if "/abnormal/" in c["path"] else 0
        correct = int(res["alarm"]) == truth
        top = max(e["bands"], key=lambda b: b["share"])
        ck = e["check"] or {}
        case_rows.append({"source": c["source"], "file": c["path"].split("/")[-1], "machine_type": c["machine_type"],
                          "true_label": "abnormal" if truth else "normal", "q3_result": c["kind"],
                          "score": res["score"], "threshold": res["threshold"],
                          "verdict": res["verdict"], "correct": correct, "ood": res["ood_status"],
                          "seed_votes": f"{res['seed_votes']}/{res['seed_count']}",
                          "top_band": top["band"], "top_band_share": top["share"],
                          "time": "concentrated " + "-".join(map(str, e["time"]["peak_window_s"])) + " s"
                          if e["time"]["concentrated"] else "spread over clip",
                          "matches_q1_band": e["summary"]["matches_q1_band"],
                          "score_top_hidden": ck.get("score_top_hidden"), "score_random_hidden": ck.get("score_random_hidden"),
                          "logit_drop_top": ck.get("logit_drop_top"), "logit_drop_random": ck.get("logit_drop_random"),
                          "check_passed": ck.get("passed"), "path": c["path"]})
        S, xt, mt = load_clip(det, c["path"], c["machine_type"])
        cells, full, _ = cam_cells(det.models["deployed"], xt, mt, chosen, S.shape)
        mask = None
        if cells is not None:
            k = max(1, int(round(0.2 * cells.size)))
            sel = np.argsort(cells.ravel())[::-1][:k]
            mask = xai.cell_masks(cells.shape, [sel], S.shape)[0]
        panels.append((S, full, mask, c, res, correct))
        print(f"  {c['source']:13s} {c['path'].split('/')[-1]} {c['machine_type']:4s} {c['kind']:12s} "
              f"score {res['score']:.3f} top band {top['band']:>11s} check "
              f"{'passed' if ck.get('passed') else 'NOT passed'}")
    CR = pd.DataFrame(case_rows)
    CR.to_csv(REP / "q5_xai_cases.csv", index=False)
    n = len(panels)
    cols = 2
    rows_n = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_n, cols, figsize=(8, 1.9 * rows_n), squeeze=False)
    for ax, (S, full, mask, c, res, correct) in zip(axs.ravel(), panels):
        overlay_ax(ax, S, full, f"{c['machine_type']} {c['path'].split('/')[-1]} – {c['kind']} – score "
                                f"{res['score']:.3f} ({'correct' if correct else 'WRONG'})", mask)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "q5_cases.png", dpi=200); plt.close()

    # ---------------- outputs ----------------
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    cfg = {"layer": chosen, "rule": "larger validation faithfulness gap; tie (<0.05) -> features.8",
           "gaps": gaps, "clips_compared": int(len(common)), "built": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
           "deployed_run": det.deployed_id}
    (ROOT / "app" / "reference" / "q5_xai.json").write_text(json.dumps(cfg, indent=2))
    md = ["# Q5 – explanation analysis (deployed model " + det.deployed_id + ")", "",
          f"Built {cfg['built']} (commit {commit}). Validation machine id_04 for all choices; test clips only described.",
          "", "## 1. Methods compared on validation clips (8 x 20 cell grid, hide 20 %)", "",
          summ.to_markdown(), "",
          "gap = logit drop when the most highlighted 20 % is hidden minus mean drop for 10 random 20 % masks; "
          "pass_rate = share of clips where the highlighted area mattered more than random; "
          "rho_vs_occlusion = Spearman agreement with the occlusion map.", "",
          f"**Layer rule -> app uses {chosen}** (mean gap on the {len(common)} clips where both layers give a map: {gaps}). "
          "An empty map means no region pushed the fault score up (common for clearly normal clips).", "",
          "## 2. Deletion curves", "", C.round(3).to_markdown(index=False), "",
          "## 3. Model-randomisation sanity check (Spearman with the original map)", "", rsum.to_markdown(), "",
          "## 4. Average Grad-CAM frequency profile (validation)", "",
          PR.pivot_table(index="band", columns=["machine_type", "label"], values="share", sort=False).to_markdown(), "",
          "## 5. Case studies (demo clips + hardest errors)", "",
          CR.drop(columns=["path"]).to_markdown(index=False), ""]
    (REP / "q5_xai_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved reports/q5_xai_results.md, q5_*.csv, figures/q5_*.png and app/reference/q5_xai.json")


if __name__ == "__main__":
    main()
