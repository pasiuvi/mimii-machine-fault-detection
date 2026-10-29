"""
Question 6 – a pretrained audio-language foundation model (CLAP) on the SAME task, split and metric.

Model: LAION CLAP, HTSAT audio encoder + RoBERTa text encoder, "laion/clap-htsat-unfused" (Wu et al., 2023),
run LOCALLY with Hugging Face transformers. No audio is sent to any external service; the only external
service is the one-off download of the public weights from the Hugging Face Hub.

Three ways of using the foundation model (all decided on VALIDATION, test used ONCE):
  A. Zero-shot prompting   score = mean cos(audio, "fault" prompts) - mean cos(audio, "normal" prompts)
                           -> 7 prompt sets written BEFORE any run; the set is selected on validation only.
  B. Linear probe          frozen CLAP embedding -> logistic regression per machine type, trained on the
                           TRAIN machines, C selected on validation (same labels the ANN had).
  C. Normal-only kNN       distance to the nearest NORMAL training clips of the same type (no fault labels),
                           k selected on validation.
Extra (6.b opportunity, reported separately, NOT comparable with the main table):
  D. Enrolment             kNN against 20 normal clips recorded on the SAME machine (as an engineer could do
                           after a service). k selected on the validation machine, then applied once to test.

Primary metric = macro TPR@FPR5 (src/metrics.py), per-type threshold chosen on validation, exactly as Q3.

Steps (run in order; commit between them):
  python -m scripts.q6_clap embed      # ~10-20 min GPU: CLAP embeddings for train/val/test (no labels used)
  python -m scripts.q6_clap select     # validation only -> experiments/q6_config.json, reports/q6_selection.md
  git add -A; git commit -m "Q6: CLAP selection on validation"; git tag q6-before-test
  python -m scripts.q6_clap test       # ONE-TIME test (lock file) -> reports/q6_results.md, figures
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.evaluation import bootstrap_ci, metrics_by_type, operating_points, time_it  # noqa: E402
from src.metrics import TYPES, macro_tpr_at_fpr  # noqa: E402

MODEL_ID = "laion/clap-htsat-unfused"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"   # Hub main branch used for the reported run (29 Oct 2026)
SR = 48_000                                   # CLAP expects 48 kHz; MIMII is 16 kHz -> resampled
EMB = config.PROCESSED_DIR / "clap_embeddings.npz"
CFG = config.ROOT / "experiments" / "q6_config.json"
LOCK = config.REPORTS_DIR / ".q6_test_evaluated.json"
R, FIG = config.REPORTS_DIR, config.FIGURES_DIR
N_ENROL = 20
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------------------------ prompts (fixed BEFORE any run)
# {m} = "pump" or "fan". MIMII fault types: pump = contamination, leakage, clogging; fan = unbalance,
# voltage change, clogging (Purohit et al., 2019). Every set has normal AND fault prompts.
PROMPT_SETS = {
    "P1_simple": {"normal": ["the sound of a {m} running normally"],
                  "abnormal": ["the sound of a faulty {m}"]},
    "P2_industrial": {"normal": ["a healthy industrial {m} operating smoothly in a factory"],
                      "abnormal": ["a broken industrial {m} making an abnormal noise in a factory"]},
    "P3_generic": {"normal": ["normal machine sound"], "abnormal": ["abnormal machine sound"]},
    "P4_fault_types": {"normal": ["a {m} operating normally"],
                       "abnormal": {"pump": ["a pump with a leak", "a clogged pump", "a contaminated pump"],
                                    "fan": ["an unbalanced fan", "a clogged fan", "a fan with an unstable voltage"]}},
    "P5_acoustic": {"normal": ["a steady, smooth humming {m}"],
                    "abnormal": ["a {m} making rattling, grinding or squealing noises"]},
    "P6_maintenance": {"normal": ["a {m} that does not need maintenance"],
                       "abnormal": ["a {m} that needs urgent maintenance"]},
    "P7_ensemble": "all",          # average of every prompt above (prompt ensembling, Radford et al., 2021)
}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
K_GRID = [1, 3, 5, 10, 20]


def prompts_for(name: str, mtype: str) -> tuple[list[str], list[str]]:
    if PROMPT_SETS[name] == "all":
        n, a = [], []
        for k, v in PROMPT_SETS.items():
            if v != "all":
                pn, pa = prompts_for(k, mtype)
                n += pn; a += pa
        return n, a
    s = PROMPT_SETS[name]
    ab = s["abnormal"][mtype] if isinstance(s["abnormal"], dict) else s["abnormal"]
    return [p.format(m=mtype) for p in s["normal"]], [p.format(m=mtype) for p in ab]


# ------------------------------------------------------------------ model
def _features(out) -> torch.Tensor:
    """transformers 4.x returns a tensor, 5.x an output object with pooler_output."""
    t = out if torch.is_tensor(out) else getattr(out, "pooler_output", None)
    if t is None:
        t = out[0]
    return F.normalize(t.float(), dim=-1)


class Clap:
    def __init__(self, device: str | None = None):
        from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ClapModel.from_pretrained(MODEL_ID, revision=REVISION).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.revision = getattr(self.model.config, "_commit_hash", None) or REVISION
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_audio_params = sum(p.numel() for n, p in self.model.named_parameters()
                                  if n.startswith(("audio_model", "audio_projection")))

    @staticmethod
    def load(path: str) -> np.ndarray:
        y, sr = sf.read(str(config.ROOT / path), dtype="float32", always_2d=True)
        y = y[:, config.CHANNEL if config.CHANNEL < y.shape[1] else 0]
        return librosa.resample(y, orig_sr=sr, target_sr=SR)

    @torch.no_grad()
    def audio(self, waves: list[np.ndarray]) -> np.ndarray:
        # the "unfused" checkpoint takes ONE 10-s mel input: "rand_trunc" only cuts audio longer than 10 s,
        # and MIMII clips are exactly 10 s, so the input is deterministic
        fusion = bool(getattr(self.model.config.audio_config, "enable_fusion", False))
        x = self.fe(waves, sampling_rate=SR, return_tensors="pt",
                    truncation="fusion" if fusion else "rand_trunc", padding="repeatpad")
        x = {k: v.to(self.device) for k, v in x.items()}
        return _features(self.model.get_audio_features(**x)).cpu().numpy()

    @torch.no_grad()
    def text(self, texts: list[str]) -> np.ndarray:
        x = self.tok(texts, padding=True, return_tensors="pt")
        x = {k: v.to(self.device) for k, v in x.items()}
        return _features(self.model.get_text_features(**x)).cpu().numpy()


# ------------------------------------------------------------------ scoring
def zero_shot(E: np.ndarray, mt: np.ndarray, text_emb: dict, name: str) -> np.ndarray:
    s = np.zeros(len(E))
    for t in TYPES:
        k = mt == t
        n, a = text_emb[(name, t)]
        s[k] = (E[k] @ a.T).mean(1) - (E[k] @ n.T).mean(1)
    return s


def knn_score(E: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    d = 1 - E @ bank.T                       # cosine distance (embeddings are unit length)
    k = min(k, bank.shape[1] if bank.ndim > 1 else 1, len(bank))
    return np.sort(d, axis=1)[:, :k].mean(1)


def knn_normal_only(Etr, ytr, ttr, E, mt, k):
    s = np.zeros(len(E))
    for t in TYPES:
        s[mt == t] = knn_score(E[mt == t], Etr[(ttr == t) & (ytr == 0)], k)
    return s


def probe(Etr, ytr, ttr, E, mt, C):
    s = np.zeros(len(E))
    for t in TYPES:
        sc = StandardScaler().fit(Etr[ttr == t])
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        lr.fit(sc.transform(Etr[ttr == t]), ytr[ttr == t])
        s[mt == t] = lr.decision_function(sc.transform(E[mt == t]))
    return s


def enrolment(E, y, mt, paths, k, n=N_ENROL):
    """Use the first n NORMAL clips (by file name) of each machine as its reference; score the rest."""
    keep = np.ones(len(E), bool)
    s = np.zeros(len(E))
    for t in TYPES:
        idx = np.where(mt == t)[0]
        normal = [i for i in sorted(idx, key=lambda i: paths[i]) if y[i] == 0][:n]
        keep[normal] = False
        rest = [i for i in idx if keep[i]]
        s[rest] = knn_score(E[rest], E[normal], k)
    return s[keep], y[keep], mt[keep]


# ------------------------------------------------------------------ data
def splits() -> pd.DataFrame:
    return pd.read_csv(config.SPLITS_CSV)


def load_emb():
    if not EMB.exists():
        sys.exit("Run 'python -m scripts.q6_clap embed' first.")
    z = np.load(EMB, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    out = {}
    for sp in ["train", "val", "test"]:
        out[sp] = (z[f"{sp}_E"], z[f"{sp}_y"], z[f"{sp}_t"].astype(str), z[f"{sp}_p"].astype(str))
    return out, meta


# ------------------------------------------------------------------ steps
def step_embed(batch: int = 16) -> None:
    clap = Clap()
    print(f"CLAP {MODEL_ID} (revision {clap.revision}) on {clap.device}: "
          f"{clap.n_params / 1e6:.1f} M parameters ({clap.n_audio_params / 1e6:.1f} M audio branch)")
    d = splits()
    arrays, t0 = {}, time.perf_counter()
    for sp in ["train", "val", "test"]:
        x = d[d.split == sp].reset_index(drop=True)
        E = []
        for i in range(0, len(x), batch):
            E.append(clap.audio([clap.load(p) for p in x.path[i:i + batch]]))
            if (i // batch) % 20 == 0:
                print(f"  {sp}: {min(i + batch, len(x))}/{len(x)}", flush=True)
        arrays[f"{sp}_E"] = np.concatenate(E).astype(np.float32)
        arrays[f"{sp}_y"] = x.label.values.astype(int)
        arrays[f"{sp}_t"] = x.machine_type.values.astype(str)
        arrays[f"{sp}_p"] = x.path.values.astype(str)
    meta = {"model": MODEL_ID, "revision": clap.revision, "device": str(clap.device),
            "params_total": clap.n_params, "params_audio": clap.n_audio_params, "sample_rate": SR,
            "embedding_minutes": round((time.perf_counter() - t0) / 60, 2),
            "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "built": datetime.now().isoformat(timespec="seconds")}
    # text embeddings for every prompt set (cached so select/test do not need the model)
    texts = {}
    for name in PROMPT_SETS:
        for t in TYPES:
            n, a = prompts_for(name, t)
            texts[f"{name}|{t}|n"] = clap.text(n)
            texts[f"{name}|{t}|a"] = clap.text(a)
    arrays.update({f"txt::{k}": v for k, v in texts.items()})
    # latency per 10-s clip (load + resample + CLAP audio encoder), batch of 1
    p = d.path.iloc[0]
    meta["latency_ms_device"] = round(time_it(lambda: clap.audio([clap.load(p)]), repeats=10), 1)
    meta["latency_ms_model_only_device"] = round(time_it(lambda w=clap.load(p): clap.audio([w]), repeats=10), 1)
    if clap.device.type == "cuda":
        cpu = Clap("cpu")
        meta["latency_ms_cpu"] = round(time_it(lambda: cpu.audio([cpu.load(p)]), repeats=5, warmup=1), 1)
    else:
        meta["latency_ms_cpu"] = meta["latency_ms_device"]
    meta["machine"] = platform.processor() or platform.machine()
    np.savez_compressed(EMB, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta, indent=2))
    print(f"Saved {EMB}")


def text_emb_from(z) -> dict:
    return {(name, t): (z[f"txt::{name}|{t}|n"], z[f"txt::{name}|{t}|a"]) for name in PROMPT_SETS for t in TYPES}


def step_select() -> None:
    S, meta = load_emb()
    z = np.load(EMB, allow_pickle=True)
    T = text_emb_from(z)
    Etr, ytr, ttr, _ = S["train"]
    Ev, yv, tv, pv = S["val"]
    rows = []
    for name in PROMPT_SETS:
        m = metrics_by_type(yv, zero_shot(Ev, tv, T, name), tv)
        rows.append({"approach": "A zero-shot", "setting": name, "val_macro_TPR@FPR5": m["macro"]["TPR@FPR5"],
                     "val_macro_AUC": m["macro"]["ROC_AUC"], "pump": m["pump"]["TPR@FPR5"], "fan": m["fan"]["TPR@FPR5"]})
    for C in C_GRID:
        m = metrics_by_type(yv, probe(Etr, ytr, ttr, Ev, tv, C), tv)
        rows.append({"approach": "B linear probe", "setting": f"C={C}", "val_macro_TPR@FPR5": m["macro"]["TPR@FPR5"],
                     "val_macro_AUC": m["macro"]["ROC_AUC"], "pump": m["pump"]["TPR@FPR5"], "fan": m["fan"]["TPR@FPR5"]})
    for k in K_GRID:
        m = metrics_by_type(yv, knn_normal_only(Etr, ytr, ttr, Ev, tv, k), tv)
        rows.append({"approach": "C normal-only kNN", "setting": f"k={k}", "val_macro_TPR@FPR5": m["macro"]["TPR@FPR5"],
                     "val_macro_AUC": m["macro"]["ROC_AUC"], "pump": m["pump"]["TPR@FPR5"], "fan": m["fan"]["TPR@FPR5"]})
    for k in K_GRID:
        s, y2, t2 = enrolment(Ev, yv, tv, pv, k)
        m = metrics_by_type(y2, s, t2)
        rows.append({"approach": "D enrolment (extra)", "setting": f"k={k}", "val_macro_TPR@FPR5": m["macro"]["TPR@FPR5"],
                     "val_macro_AUC": m["macro"]["ROC_AUC"], "pump": m["pump"]["TPR@FPR5"], "fan": m["fan"]["TPR@FPR5"]})
    V = pd.DataFrame(rows)
    # rule: highest validation macro TPR@FPR5, tie-break validation macro AUC, then the simpler setting (first)
    chosen = {}
    for ap, g in V.groupby("approach", sort=False):
        best = g.sort_values(["val_macro_TPR@FPR5", "val_macro_AUC"], ascending=False, kind="stable").iloc[0]
        chosen[ap] = {"setting": best.setting, "val_macro_TPR@FPR5": float(best["val_macro_TPR@FPR5"]),
                      "val_macro_AUC": float(best.val_macro_AUC)}
    V["selected"] = [chosen[r.approach]["setting"] == r.setting for r in V.itertuples()]
    print(V.to_string(index=False))
    cfg = {"model": MODEL_ID, "revision": meta["revision"], "rule": "max validation macro TPR@FPR5, tie-break "
           "validation macro ROC-AUC", "chosen": chosen, "prompt_sets": PROMPT_SETS,
           "n_enrol": N_ENROL, "built": datetime.now().isoformat(timespec="seconds")}
    CFG.write_text(json.dumps(cfg, indent=2))
    V.to_csv(R / "q6_selection.csv", index=False)
    md = ["# Q6 – selection on the validation machine (id_04) only", "",
          f"Model {MODEL_ID} (revision {meta['revision']}), transformers {meta['transformers']}. "
          "Test data not used.", "", V.to_markdown(index=False), "",
          "## Prompts (written before any run)", ""]
    for name in PROMPT_SETS:
        for t in TYPES:
            n, a = prompts_for(name, t)
            md.append(f"- **{name} / {t}** – normal: {n} | fault: {a}")
    (R / "q6_selection.md").write_text("\n".join(md), encoding="utf-8")
    # figure: zero-shot prompt selection
    g = V[V.approach == "A zero-shot"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(g.setting, g["val_macro_TPR@FPR5"], color=["#c0392b" if s else "#95a5a6" for s in g.selected])
    ax.axhline(0.05, ls=":", color="k", lw=0.8); ax.text(len(g) - 0.5, 0.055, "chance", fontsize=7, ha="right")
    ax.set_ylabel("validation macro TPR@FPR5"); ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.set_title("CLAP zero-shot: prompt sets on the validation machine (selected in red)", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "q6_prompt_selection.png", dpi=200); plt.close()
    print(f"\nSelected: {json.dumps(chosen, indent=1)}\nSaved experiments/q6_config.json, reports/q6_selection.md")


def setting_value(s: str):
    return s.split("=")[1] if "=" in s else s


def step_test(force: bool = False) -> None:
    if LOCK.exists() and not force:
        sys.exit(f"Test already evaluated on {json.loads(LOCK.read_text())['date']}. Refusing to run again.")
    if not CFG.exists():
        sys.exit("Run the 'select' step first.")
    cfg = json.loads(CFG.read_text())
    S, meta = load_emb()
    T = text_emb_from(np.load(EMB, allow_pickle=True))
    Etr, ytr, ttr, _ = S["train"]; Ev, yv, tv, pv = S["val"]; Et, yt, tt, pt = S["test"]
    ch = cfg["chosen"]
    zs = setting_value(ch["A zero-shot"]["setting"])
    C = float(setting_value(ch["B linear probe"]["setting"]))
    k = int(setting_value(ch["C normal-only kNN"]["setting"]))
    ke = int(setting_value(ch["D enrolment (extra)"]["setting"]))
    models = {
        f"CLAP zero-shot ({zs})": (zero_shot(Ev, tv, T, zs), zero_shot(Et, tt, T, zs), "0 labels, 0 training"),
        f"CLAP linear probe (C={C:g})": (probe(Etr, ytr, ttr, Ev, tv, C), probe(Etr, ytr, ttr, Et, tt, C),
                                          "same labelled train clips as the ANN"),
        f"CLAP normal-only kNN (k={k})": (knn_normal_only(Etr, ytr, ttr, Ev, tv, k),
                                          knn_normal_only(Etr, ytr, ttr, Et, tt, k), "normal train clips only"),
    }
    rows, ops = [], {}
    for name, (sv, st, data) in models.items():
        m = metrics_by_type(yt, st, tt)
        lo, hi = bootstrap_ci(yt, st, tt)
        ops[name] = operating_points(yv, sv, tv, yt, st, tt)
        rows.append({"Model": name, "Test macro TPR@FPR5": m["macro"]["TPR@FPR5"], "95% CI": f"{lo:.3f}–{hi:.3f}",
                     "Pump TPR@FPR5": m["pump"]["TPR@FPR5"], "Fan TPR@FPR5": m["fan"]["TPR@FPR5"],
                     "Test pooled TPR@FPR5": m["all"]["TPR@FPR5"], "Test macro ROC-AUC": m["macro"]["ROC_AUC"],
                     "Test macro pAUC(0.1)": m["macro"]["pAUC_0.1"],
                     "Val macro TPR@FPR5": round(macro_tpr_at_fpr(yv, sv, tv), 4), "Training data": data})
    G = pd.DataFrame(rows)
    q3p = R / "q3_comparison.csv"
    Q3 = pd.read_csv(q3p) if q3p.exists() else pd.DataFrame()
    keep = ["Model", "Test macro TPR@FPR5", "95% CI", "Pump TPR@FPR5", "Fan TPR@FPR5", "Test pooled TPR@FPR5",
            "Test macro ROC-AUC", "Test macro pAUC(0.1)", "Val macro TPR@FPR5"]
    ALL = pd.concat([Q3[[c for c in keep if c in Q3.columns]], G[keep]], ignore_index=True) if len(Q3) else G[keep]
    # enrolment (separate: fewer test clips, uses 20 normal clips of the test machine as reference)
    se, ye, te = enrolment(Et, yt, tt, pt, ke)
    me = metrics_by_type(ye, se, te)
    enrol = {"setting": f"k={ke}, {N_ENROL} normal reference clips per machine", "clips_scored": int(len(ye)),
             "test_macro_TPR@FPR5": me["macro"]["TPR@FPR5"], "pump": me["pump"]["TPR@FPR5"],
             "fan": me["fan"]["TPR@FPR5"], "test_macro_AUC": me["macro"]["ROC_AUC"],
             "val_macro_TPR@FPR5": ch["D enrolment (extra)"]["val_macro_TPR@FPR5"]}
    lat = {"CLAP latency per clip (ms), device": meta["latency_ms_device"], "CLAP latency CPU (ms)": meta["latency_ms_cpu"],
           "CLAP model only (ms), device": meta["latency_ms_model_only_device"], "device": meta["device"],
           "params_total_M": round(meta["params_total"] / 1e6, 1), "params_audio_M": round(meta["params_audio"] / 1e6, 1)}
    out = {"date": datetime.now().isoformat(timespec="seconds"), "config": cfg, "results": rows,
           "operating_points": ops, "enrolment": enrol, "latency": lat, "meta": meta}
    (R / "q6_results.json").write_text(json.dumps(out, indent=2, default=str))
    ALL.to_csv(R / "q6_comparison.csv", index=False)
    md = ["# Q6 – CLAP on the held-out test machine (id_06), evaluated once", "",
          f"{out['date']} · {MODEL_ID} rev {meta['revision']} · selections from experiments/q6_config.json", "",
          "## Comparison with Question 3 (same test clips, same metric, thresholds from validation)", "",
          ALL.to_markdown(index=False), "", "## Operating points (validation threshold applied to test)", ""]
    for name, o in ops.items():
        md.append(f"- **{name}**: " + "; ".join(f"{t}: TPR {v['test_TPR_at_val_threshold']}, FPR "
                                                  f"{v['test_FPR_at_val_threshold']}" for t, v in o.items()))
    md += ["", "## Extra (6.b opportunity): enrolment with 20 normal clips of the same machine", "",
           "Not comparable with the table above: the 20 reference clips per machine are removed from the test set.", "",
           pd.DataFrame([enrol]).to_markdown(index=False), "", "## Cost and latency", "",
           pd.DataFrame([lat]).to_markdown(index=False), ""]
    (R / "q6_results.md").write_text("\n".join(md), encoding="utf-8")
    LOCK.write_text(json.dumps({"date": out["date"], "force": force}))
    # figure
    plot = ALL[~ALL.Model.str.startswith("Random")].copy()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    cols = ["#c0392b" if m.startswith("CLAP") else "#2e86c1" for m in plot.Model]
    ax.barh(plot.Model, plot["Test macro TPR@FPR5"], color=cols)
    ax.axvline(0.05, ls=":", color="k", lw=0.8)
    for i, v in enumerate(plot["Test macro TPR@FPR5"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("test macro TPR@FPR5 (machine id_06)")
    ax.set_title("Same test clips, same metric: Q3 models (blue) and CLAP (red)", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "q6_comparison.png", dpi=200); plt.close()
    print("\n".join(md))
    print(f"\nSaved reports/q6_results.md/.json, q6_comparison.csv, figures/q6_comparison.png; lock {LOCK.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "select", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"embed": step_embed, "select": step_select, "test": lambda: step_test(a.force)}[a.step]()
