"""
Question 4 – web application backend (FastAPI).

    uvicorn app.main:app --port 8000          then open http://127.0.0.1:8000

Endpoints
  GET  /                          the single-page frontend (app/static)
  GET  /api/health                liveness + model loaded?
  GET  /api/model-card            deployed model, thresholds, input rules, Q3 results (for the limitations panel)
  POST /api/predict               multipart form: file=<WAV/FLAC>, machine_type=pump|fan  -> prediction
  GET  /api/examples              demo clips from the project data (normal, abnormal, and known mistakes)
  GET  /api/examples/{id}/audio   stream an example clip
  POST /api/examples/{id}/predict run the model on an example clip (at request time, like an upload)
  POST /api/explain               (Q5) same form as /api/predict -> prediction + Grad-CAM explanation + check
  POST /api/examples/{id}/explain (Q5) explanation for a demo clip

  v2 – per-machine monitoring with enrolment (page /monitor; app/monitor.py):
  GET    /api/monitor/machines             enrolled machines and their last checks
  POST   /api/monitor/enrol                form: name, machine_type, files=<20-200 normal WAV/FLAC clips>
  POST   /api/monitor/{name}/check         form: file -> unusual? + 2-of-3 alarm decision
  POST   /api/monitor/{name}/reset         clear the check history of one machine
  DELETE /api/monitor/{name}               remove an enrolled machine
  GET    /api/monitor/demo                 demo machines from the project data (test machines id_06)
  POST   /api/monitor/demo/{key}/enrol     enrol a demo machine with its 50 normal clips
  POST   /api/monitor/demo/{key}/check/{i} check one demo clip that was NOT used for enrolment

Every prediction is appended to logs/predictions.jsonl (audit trail, not committed to Git).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import monitor
from app.inference import MAX_BYTES, FaultDetector, InputError
from src import config

ROOT = config.ROOT
STATIC = ROOT / "app" / "static"
LOG_DIR = ROOT / "logs"
log = logging.getLogger("acoustiguard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STATE: dict = {}


def load_examples() -> list[dict]:
    """Demo clips: a few correct and a few wrong test-machine predictions from Q3 (if the audio exists locally)."""
    p = config.REPORTS_DIR / "q3_test_predictions.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    d = d[d.path.apply(lambda s: (ROOT / s).exists())]
    if d.empty:
        return []
    picks = []
    for t in ["pump", "fan"]:
        x = d[d.machine_type == t]
        for name, sel in [("normal", (x.label == 0) & (x.error == "ok")), ("abnormal", (x.label == 1) & (x.error == "ok")),
                          ("missed fault", x.error == "FN"), ("false alarm", x.error == "FP")]:
            s = x[sel]
            if len(s):
                r = s.sort_values("score").iloc[len(s) // 2]
                picks.append({"machine_type": t, "true_label": "abnormal" if r.label == 1 else "normal",
                              "kind": name, "path": r.path, "file": Path(r.path).name,
                              "machine": f"{t} {r.machine_id} (test machine)"})
    for i, e in enumerate(picks):
        e["id"] = f"ex{i + 1}"
    return picks


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    STATE["detector"] = FaultDetector()
    STATE["examples"] = load_examples()
    LOG_DIR.mkdir(exist_ok=True)
    log.info("Model loaded in %.1f s on %s (%d models); %d example clips",
             time.perf_counter() - t, STATE["detector"].device, len(STATE["detector"].models), len(STATE["examples"]))
    yield


app = FastAPI(title="AcoustiGuard – acoustic fault screening", version="1.0", lifespan=lifespan)


@app.exception_handler(InputError)
async def input_error_handler(request: Request, exc: InputError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message})


def audit(entry: dict) -> None:
    with open(LOG_DIR / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_prediction(data: bytes, filename: str, machine_type: str, source: str, explain: bool = False) -> dict:
    rid = uuid.uuid4().hex[:12]
    try:
        d = STATE["detector"]
        res = d.explain(data, filename, machine_type) if explain else d.predict(data, filename, machine_type)
    except InputError as e:
        audit({"time": datetime.now().isoformat(timespec="seconds"), "request_id": rid, "source": source,
               "file": filename, "machine_type": machine_type, "rejected": e.code})
        raise
    res["request_id"] = rid
    res["file"] = filename
    audit({"time": datetime.now().isoformat(timespec="seconds"), "request_id": rid, "source": source, "file": filename,
           "machine_type": machine_type, "score": res["score"], "threshold": res["threshold"], "verdict": res["verdict"],
           "ood_status": res["ood_status"], "latency_ms": res["timing_ms"]["total"],
           **({"explained": True, "xai_check_passed": (res["explanation"].get("check") or {}).get("passed")}
              if explain else {})})
    return res


@app.get("/api/health")
def health():
    d = STATE.get("detector")
    return {"status": "ok" if d else "loading", "model_loaded": d is not None,
            "device": str(d.device) if d else None}


@app.get("/api/model-card")
def model_card():
    return STATE["detector"].model_card()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), machine_type: str = Form(...)):
    data = await file.read(MAX_BYTES + 1)
    return run_prediction(data, file.filename or "", machine_type.strip().lower(), "upload")


@app.post("/api/explain")
async def explain(file: UploadFile = File(...), machine_type: str = Form(...)):
    data = await file.read(MAX_BYTES + 1)
    return run_prediction(data, file.filename or "", machine_type.strip().lower(), "upload", explain=True)


@app.get("/api/examples")
def examples():
    return [{k: v for k, v in e.items() if k != "path"} for e in STATE["examples"]]


def _example(ex_id: str) -> dict:
    for e in STATE["examples"]:
        if e["id"] == ex_id:
            return e
    raise HTTPException(404, detail="Example not found")


@app.get("/api/examples/{ex_id}/audio")
def example_audio(ex_id: str):
    e = _example(ex_id)
    return FileResponse(ROOT / e["path"], media_type="audio/wav", filename=e["file"])


@app.post("/api/examples/{ex_id}/predict")
def example_predict(ex_id: str):
    e = _example(ex_id)
    res = run_prediction((ROOT / e["path"]).read_bytes(), e["file"], e["machine_type"], f"example:{ex_id}")
    res["true_label"] = e["true_label"]
    return res


@app.post("/api/examples/{ex_id}/explain")
def example_explain(ex_id: str):
    e = _example(ex_id)
    res = run_prediction((ROOT / e["path"]).read_bytes(), e["file"], e["machine_type"], f"example:{ex_id}", explain=True)
    res["true_label"] = e["true_label"]
    return res


# ------------------------------------------------------------------ v2: per-machine monitoring
def _audit_monitor(entry: dict) -> None:
    audit({"time": datetime.now().isoformat(timespec="seconds"), "request_id": uuid.uuid4().hex[:12],
           "mode": "monitor", **entry})


@app.get("/api/monitor/machines")
def monitor_machines():
    return {"settings": monitor.settings(), "machines": monitor.list_machines()}


@app.post("/api/monitor/enrol")
async def monitor_enrol(name: str = Form(...), machine_type: str = Form(...), files: list[UploadFile] = File(...)):
    data = [(await f.read(MAX_BYTES + 1), f.filename or "") for f in files]
    t = time.perf_counter()
    res = monitor.enrol(name, machine_type.strip().lower(), data)
    res["timing_ms"] = round(1000 * (time.perf_counter() - t))
    _audit_monitor({"action": "enrol", "machine": res["id"], "n_enrol": res["n_enrol"]})
    return res


@app.post("/api/monitor/{name}/check")
async def monitor_check(name: str, file: UploadFile = File(...)):
    t = time.perf_counter()
    res = monitor.check(name, await file.read(MAX_BYTES + 1), file.filename or "")
    res["timing_ms"] = round(1000 * (time.perf_counter() - t))
    _audit_monitor({"action": "check", "machine": res["machine_id"], "file": res["file"], "z": res["z"],
                    "status": res["status"]})
    return res


@app.post("/api/monitor/{name}/reset")
def monitor_reset(name: str):
    return monitor.reset_history(name)


@app.delete("/api/monitor/{name}")
def monitor_delete(name: str):
    return monitor.delete(name)


def _demo(key: str) -> dict:
    d = STATE.setdefault("demo_sets", monitor.demo_sets()).get(key)
    if not d:
        raise HTTPException(404, detail="Demo machine not found (project audio not available)")
    return d


@app.get("/api/monitor/demo")
def monitor_demo():
    sets = STATE.setdefault("demo_sets", monitor.demo_sets())
    return [{"key": k, "name": v["name"], "machine_type": v["machine_type"], "n_enrol": len(v["enrol"]),
             "clips": [{"i": i, "file": c["file"], "true_label": c["true_label"]} for i, c in enumerate(v["clips"])]}
            for k, v in sets.items()]


@app.post("/api/monitor/demo/{key}/enrol")
def monitor_demo_enrol(key: str):
    d = _demo(key)
    t = time.perf_counter()
    res = monitor.enrol(d["name"].replace(" (demo)", "-demo"), d["machine_type"],
                        [((ROOT / p).read_bytes(), Path(p).name) for p in d["enrol"]])
    res["timing_ms"] = round(1000 * (time.perf_counter() - t))
    _audit_monitor({"action": "enrol-demo", "machine": res["id"], "n_enrol": res["n_enrol"]})
    return res


@app.post("/api/monitor/demo/{key}/check/{i}")
def monitor_demo_check(key: str, i: int):
    d = _demo(key)
    if not 0 <= i < len(d["clips"]):
        raise HTTPException(404, detail="Demo clip not found")
    c = d["clips"][i]
    t = time.perf_counter()
    res = monitor.check(d["name"].replace(" (demo)", "-demo"), (ROOT / c["path"]).read_bytes(), c["file"])
    res["timing_ms"] = round(1000 * (time.perf_counter() - t))
    res["true_label"] = c["true_label"]
    _audit_monitor({"action": "check-demo", "machine": res["machine_id"], "file": c["file"], "z": res["z"],
                    "status": res["status"], "true_label": c["true_label"]})
    return res


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/monitor")
def monitor_page():
    return FileResponse(STATIC / "monitor.html")
