"""
AcoustiGuard v2 – per-machine monitoring with ENROLMENT (the recommended mode after the v2 study).

How it works (identical to scripts/v2_enrolment.py; settings read from experiments/v2_config.json):
  1. Enrol: the engineer uploads normal recordings of ONE machine, confirmed healthy (e.g. just after a service),
     ideally spread over several days and operating conditions. Recommended 50 clips (minimum 20).
  2. Each clip -> CLAP audio embedding (pinned revision). Each enrolment clip gets a leave-one-out score
     = mean cosine distance to its k nearest other enrolment clips. Mean and std of these scores are stored.
  3. Check: a new clip is scored the same way against the enrolment bank and standardised: z = (d - mean) / std.
     The clip is "unusual" if z > c. An ALARM is raised only when at least 2 of the last 3 checked clips of this
     machine are unusual (a real fault is steady; one odd clip is not).

Nothing here uses fault labels. The result is a screening signal for an inspection, not a diagnosis.
Enrolled machines are stored in monitor_data/ (not in Git).
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np

from app.inference import InputError, decode_and_check
from src import config

ROOT = config.ROOT
STORE = ROOT / "monitor_data"
V2_CFG = ROOT / "experiments" / "v2_config.json"
CLAP_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
CLAP_SR = 48_000
MIN_ENROL, RECOMMENDED_ENROL, MAX_ENROL = 20, 50, 200
HISTORY = 3                     # decision window: alarm if >= 2 of the last 3 clips are unusual
ALARM_MIN = 2
FAR_Z = 15.0                    # far outside the enrolment spread -> also check the recording set-up


def settings() -> dict:
    c = json.loads(V2_CFG.read_text()) if V2_CFG.exists() else {}
    return {"feature": c.get("feature", "clap"), "k": int(c.get("k", 3)), "c": float(c.get("c", 2.0)),
            "n_recommended": int(c.get("n_enrol", RECOMMENDED_ENROL))}


# ------------------------------------------------------------------ embedder (CLAP, loaded on first use)
class ClapEmbedder:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, ClapModel
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.device_count() > 0 else "cpu"))
        self.model = ClapModel.from_pretrained(CLAP_ID, revision=CLAP_REVISION).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(CLAP_ID, revision=CLAP_REVISION)

    def __call__(self, sigs_16k: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        waves = [librosa.resample(s, orig_sr=config.SAMPLE_RATE, target_sr=CLAP_SR) for s in sigs_16k]
        out = []
        with torch.no_grad():
            for i in range(0, len(waves), 16):
                x = self.fe(waves[i:i + 16], sampling_rate=CLAP_SR, return_tensors="pt",
                            truncation="rand_trunc", padding="repeatpad")
                x = {k: v.to(self.device) for k, v in x.items()}
                f = self.model.get_audio_features(**x)
                f = f if torch.is_tensor(f) else getattr(f, "pooler_output", f[0])
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


_EMB, _LOCK = None, threading.Lock()


def embedder():
    """The CLAP model is large (615 MB), so it is loaded only when monitoring is first used."""
    global _EMB
    with _LOCK:
        if _EMB is None:
            _EMB = ClapEmbedder()
        return _EMB


def set_embedder(fn) -> None:
    """For tests: replace CLAP with a small deterministic function."""
    global _EMB
    _EMB = fn


# ------------------------------------------------------------------ scoring (same maths as the v2 study)
def knn_dist(Q: np.ndarray, B: np.ndarray, k: int, loo: bool = False) -> np.ndarray:
    d = 1.0 - Q @ B.T
    if loo:
        np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].mean(1)


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    if not s or len(s) > 60:
        raise InputError(422, "bad_name", "Give the machine a short name (letters, numbers, - or _), e.g. 'pump-line-3'.")
    return s


def _path(mid: str) -> Path:
    return STORE / f"{mid}.npz"


def _hist_path(mid: str) -> Path:
    return STORE / f"{mid}_history.json"


def list_machines() -> list[dict]:
    STORE.mkdir(exist_ok=True)
    out = []
    for p in sorted(STORE.glob("*.npz")):
        info = json.loads(str(np.load(p, allow_pickle=True)["info"]))
        h = _history(info["id"])
        out.append({**info, "checks": len(h), "recent": h[-HISTORY:]})
    return out


def _load(mid: str) -> tuple[np.ndarray, dict]:
    p = _path(mid)
    if not p.exists():
        raise InputError(404, "unknown_machine", f"Machine '{mid}' is not enrolled.")
    z = np.load(p, allow_pickle=True)
    return z["bank"], json.loads(str(z["info"]))


def _history(mid: str) -> list[dict]:
    p = _hist_path(mid)
    return json.loads(p.read_text()) if p.exists() else []


def enrol(name: str, machine_type: str, files: list[tuple[bytes, str]]) -> dict:
    mid = slug(name)
    if machine_type not in ("pump", "fan"):
        raise InputError(422, "bad_machine_type", "Machine type must be 'pump' or 'fan'.")
    if len(files) > MAX_ENROL:
        raise InputError(422, "too_many_files", f"Please upload at most {MAX_ENROL} clips.")
    sigs, used, rejected = [], [], []
    for data, fname in files:
        try:
            a = decode_and_check(data, fname)
            sigs.append(a.y); used.append(fname)
        except InputError as e:
            rejected.append({"file": fname, "reason": e.message})
    if len(sigs) < MIN_ENROL:
        raise InputError(422, "too_few_clips", f"{len(sigs)} usable normal clips – at least {MIN_ENROL} are needed "
                                               f"(recommended {RECOMMENDED_ENROL}, recorded on several days).")
    s = settings()
    bank = embedder()(sigs).astype(np.float32)
    loo = knn_dist(bank, bank, s["k"], loo=True)
    mu, sd = float(loo.mean()), float(loo.std() + 1e-9)
    info = {"id": mid, "name": name.strip(), "machine_type": machine_type, "n_enrol": len(sigs), "k": s["k"],
            "c": s["c"], "mean": mu, "std": sd, "enrolled_at": datetime.now().isoformat(timespec="seconds"),
            "files": used, "model": f"{CLAP_ID}@{CLAP_REVISION[:7]}"}
    STORE.mkdir(exist_ok=True)
    np.savez(_path(mid), bank=bank, info=json.dumps(info))
    _hist_path(mid).write_text("[]")
    z_loo = (loo - mu) / sd
    warnings = []
    if len(sigs) < s["n_recommended"]:
        warnings.append(f"Only {len(sigs)} clips: with fewer than {s['n_recommended']} the false-alarm rate was clearly "
                        "higher in the v2 study. Add more normal clips from other days if possible.")
    if (z_loo > s["c"]).mean() > 0.10:
        warnings.append("More than 10 % of the enrolment clips look unusual compared with the others. Check that all "
                        "clips are from the same healthy machine and microphone position.")
    return {**info, "rejected": rejected, "warnings": warnings,
            "enrol_unusual_share": round(float((z_loo > s["c"]).mean()), 3)}


def check(name: str, data: bytes, filename: str) -> dict:
    mid = slug(name)
    bank, info = _load(mid)
    a = decode_and_check(data, filename)
    q = embedder()([a.y]).astype(np.float32)
    d = float(knn_dist(q, bank, info["k"])[0])
    z = (d - info["mean"]) / info["std"]
    unusual = bool(z > info["c"])
    hist = _history(mid)
    hist.append({"time": datetime.now().isoformat(timespec="seconds"), "file": filename, "z": round(z, 2),
                 "unusual": unusual})
    _hist_path(mid).write_text(json.dumps(hist[-200:]))
    window = hist[-HISTORY:]
    n_unusual = sum(h["unusual"] for h in window)
    alarm = n_unusual >= ALARM_MIN
    if alarm:
        status, action = "alarm", "Fault-like change: inspect this machine (raise a work order)."
    elif unusual:
        status, action = "watch", "This clip is unusual for this machine. Record again soon – an alarm needs 2 of 3."
    else:
        status, action = "normal", "Sounds like this machine's own normal recordings – keep the planned schedule."
    notes = []
    if z > FAR_Z:
        notes.append("Very far from every enrolment clip. Also check the recording itself (microphone position, "
                     "gain, background noise) before inspecting the machine.")
    if len(window) < HISTORY:
        notes.append(f"Only {len(window)} clip(s) checked since enrolment – the 2-of-3 rule needs 3 clips.")
    return {"machine": info["name"], "machine_id": mid, "machine_type": info["machine_type"], "file": filename,
            "distance": round(d, 4), "z": round(z, 2), "threshold_z": info["c"], "unusual": unusual,
            "window": window, "unusual_in_window": n_unusual, "status": status, "action": action, "notes": notes,
            "checks": [c.__dict__ for c in a.checks]}


def reset_history(name: str) -> dict:
    mid = slug(name); _load(mid)
    _hist_path(mid).write_text("[]")
    return {"machine_id": mid, "history": []}


def delete(name: str) -> dict:
    mid = slug(name)
    for p in (_path(mid), _hist_path(mid)):
        if p.exists():
            p.unlink()
    return {"deleted": mid}


# ------------------------------------------------------------------ demo (uses the project data, if present)
def demo_sets(n: int | None = None, seed: int = 0) -> dict:
    """The same enrolment draw as the v2 test (seed 0) for pump/fan id_06, plus clips that were NOT enrolled."""
    import pandas as pd
    n = n or settings()["n_recommended"]
    p = config.FEATURE_DIR / "test_index.csv"
    if not p.exists():
        return {}
    idx = pd.read_csv(p)
    out = {}
    for (mt, mid), g in idx.groupby(["machine_type", "machine_id"]):
        y = g.label.to_numpy(); paths = g.path.astype(str).to_numpy()
        normal = np.where(y == 0)[0]
        if len(normal) < n + 4 or (y == 1).sum() < 4:
            continue
        rng = np.random.default_rng(config.SEED + seed)
        enrol_i = np.sort(rng.choice(normal, size=n, replace=False))
        rest = np.setdiff1d(np.arange(len(y)), enrol_i)
        if not all((ROOT / paths[i]).exists() for i in enrol_i[:3]):
            continue
        r2 = np.random.default_rng(7)
        nrm = r2.choice(rest[y[rest] == 0], size=4, replace=False)
        abn = r2.choice(rest[y[rest] == 1], size=4, replace=False)
        out[f"{mt}-{mid}-demo"] = {
            "name": f"{mt} {mid} (demo)", "machine_type": mt, "enrol": [paths[i] for i in enrol_i],
            "clips": [{"path": paths[i], "file": Path(paths[i]).name, "true_label": "normal"} for i in nrm] +
                     [{"path": paths[i], "file": Path(paths[i]).name, "true_label": "abnormal"} for i in abn]}
    return out
