"""
Question 4 – test the web application with normal, invalid and out-of-distribution inputs.

Every case is sent through the real HTTP API (FastAPI TestClient = same code path as the browser),
so the trained model runs at request time for each one.

Outputs
  reports/q4_test_results.csv / .md   one row per case: expected vs. actual, score, verdict, input checks
  tests/q4_inputs/*.wav|flac          the generated test files (upload them in the UI for the demo video)

Run:  python -m scripts.q4_test_inputs          (pytest tests/test_app.py runs the same cases as unit tests)
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SR = config.SAMPLE_RATE
INPUT_DIR = config.ROOT / "tests" / "q4_inputs"
OK_STATES = {"in_distribution", "unusual", "out_of_distribution"}


@dataclass
class Case:
    id: str
    category: str          # normal | invalid | out-of-distribution
    description: str
    filename: str
    data: bytes | None
    machine_type: str
    expect_status: int
    expect_ood: tuple = ()          # acceptable ood_status values (empty = any)
    expect_check: str = ""          # name of a check that must be warn/fail/info
    note: str = ""                  # documented known limitation (reported, not asserted)


def wav(sig, sr=SR, fmt="WAV", subtype="PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(sig, dtype=np.float32), -1, 1), sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def real_clips() -> dict:
    """One normal and one abnormal clip per machine type from the VALIDATION machine (id_04)."""
    idx = pd.read_csv(config.FEATURE_DIR / "val_index.csv")
    out = {}
    for t in ["pump", "fan"]:
        for lab, name in [(0, "normal"), (1, "abnormal")]:
            rows = idx[(idx.machine_type == t) & (idx.label == lab)]
            for p in rows.path:
                if (config.ROOT / p).exists():
                    out[(t, name)] = config.ROOT / p
                    break
    return out


def cases() -> list[Case]:
    rng = np.random.default_rng(0)
    clips = real_clips()
    C: list[Case] = []
    # ---------------- normal inputs ----------------
    for (t, name), p in clips.items():
        C.append(Case(f"N-{t}-{name}", "normal", f"Real {name} {t} clip, validation machine id_04 (8-channel, 16 kHz, 10 s)",
                      p.name, p.read_bytes(), t, 200))
    if ("pump", "normal") in clips:
        y, sr = sf.read(clips[("pump", "normal")], dtype="float32", always_2d=True)
        mono = y[:, 0]
        C.append(Case("N-flac", "normal", "Same pump clip saved as FLAC (score must match the WAV)", "pump_normal.flac",
                      wav(mono, sr, "FLAC", "PCM_16"), "pump", 200))
        import librosa
        st = librosa.resample(mono, orig_sr=sr, target_sr=44100)
        C.append(Case("N-44k-stereo", "normal", "Same pump clip at 44.1 kHz stereo (resampled by the app)",
                      "pump_normal_44k_stereo.wav", wav(np.stack([st, st], 1), 44100), "pump", 200,
                      expect_check="Sample rate"))
        C.append(Case("N-short-3s", "normal", "First 3 s of the pump clip (shorter than the 10-s training clips)",
                      "pump_normal_3s.wav", wav(mono[: 3 * sr], sr), "pump", 200, expect_check="Duration"))
    # ---------------- invalid inputs ----------------
    C += [
        Case("I-empty", "invalid", "Empty file", "empty.wav", b"", "pump", 400),
        Case("I-text-as-wav", "invalid", "Text file renamed to .wav", "notes.wav", b"this is not audio\n" * 50, "pump", 422),
        Case("I-mp3-ext", "invalid", "Unsupported extension (.mp3)", "clip.mp3", b"ID3" + bytes(200), "pump", 415),
        Case("I-pdf", "invalid", "PDF document", "report.pdf", b"%PDF-1.4\n" + bytes(500), "fan", 415),
        Case("I-truncated", "invalid", "WAV cut after 30 bytes (corrupted header)", "truncated.wav",
             wav(rng.standard_normal(SR) * 0.05)[:30], "pump", 422),
        Case("I-too-short", "invalid", "0.5-second recording", "short.wav", wav(0.05 * rng.standard_normal(SR // 2)), "pump", 422),
        Case("I-too-long", "invalid", "40-second recording", "long.wav", wav(0.05 * rng.standard_normal(40 * SR)), "fan", 422),
        Case("I-silence", "invalid", "10 s of digital silence", "silence.wav", wav(np.zeros(10 * SR)), "pump", 422),
        Case("I-machine-type", "invalid", "Valid audio but machine type 'valve'", "valve.wav",
             wav(0.05 * rng.standard_normal(10 * SR)), "valve", 422),
        Case("I-too-large", "invalid", "26 MB upload (over the 25 MB limit)", "huge.wav", bytes(26 * 1024 * 1024), "pump", 413),
        Case("I-no-file", "invalid", "Request without a file", "", None, "pump", 422),
    ]
    # ---------------- out-of-distribution inputs ----------------
    t10 = np.arange(10 * SR) / SR
    C += [
        Case("O-white-noise", "out-of-distribution", "White noise (flat spectrum, no machine)", "white_noise.wav",
             wav(0.05 * rng.standard_normal(10 * SR)), "pump", 200, ("unusual", "out_of_distribution")),
        Case("O-tone-1k", "out-of-distribution", "Pure 1 kHz sine tone", "tone_1khz.wav", wav(0.3 * np.sin(2 * np.pi * 1000 * t10)),
             "fan", 200, ("unusual", "out_of_distribution")),
        Case("O-music-chord", "out-of-distribution", "Musical chord (A major, with harmonics)", "chord.wav",
             wav(sum(0.08 / k * np.sin(2 * np.pi * f * k * t10) for f in (220, 277.2, 329.6) for k in (1, 2, 3))),
             "pump", 200, ("unusual", "out_of_distribution")),
    ]
    if ("pump", "normal") in clips:
        rms = 20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-12)
        gain_loud = 10 ** ((-3 - rms) / 20)              # push the level to about -3 dBFS -> heavy clipping
        C += [
            Case("O-too-loud", "out-of-distribution", f"Real pump clip amplified {gain_loud:.0f}x to about -3 dBFS (heavy clipping)",
                 "pump_clipped.wav", wav(np.clip(mono * gain_loud, -1, 1), sr), "pump", 200,
                 ("unusual", "out_of_distribution"), "Clipping"),
            Case("O-too-quiet", "out-of-distribution", "Real pump clip turned down to about -65 dBFS (very quiet, still audible)",
                 "pump_quiet.wav", wav(mono * 10 ** ((-65 - rms) / 20), sr), "pump", 200, ("unusual", "out_of_distribution")),
            Case("I-near-silent", "invalid", "Real pump clip turned down to about -95 dBFS (below the -80 dBFS silence limit)",
                 "pump_near_silent.wav", wav(mono * 10 ** ((-95 - rms) / 20), sr), "pump", 422),
        ]
    if ("fan", "normal") in clips:
        yf, srf = sf.read(clips[("fan", "normal")], dtype="float32", always_2d=True)
        C.append(Case("O-wrong-type", "out-of-distribution", "Real FAN clip submitted as a PUMP", "fan_as_pump.wav",
                      wav(yf[:, 0], srf), "pump", 200,
                      note="Known limitation: pump and fan recordings have similar level and spectrum shape, so the "
                           "simple checks cannot detect a wrong machine type; the user must choose it correctly."))
    return C


def post(client, c: Case):
    form = {"machine_type": c.machine_type}
    if c.data is None:
        return client.post("/api/predict", data=form)
    return client.post("/api/predict", data=form, files={"file": (c.filename, c.data, "application/octet-stream")})


def evaluate(client, c: Case) -> dict:
    r = post(client, c)
    body = r.json()
    row = {"id": c.id, "category": c.category, "description": c.description, "machine_type": c.machine_type,
           "expected_status": c.expect_status, "http_status": r.status_code}
    if r.status_code == 200:
        flagged = [f"{x['name']} ({x['status']})" for x in body["checks"] if x["status"] != "pass"]
        row.update({"verdict": body["verdict"], "score": body["score"], "threshold": body["threshold"],
                    "ood_status": body["ood_status"], "flagged_checks": "; ".join(flagged) or "none",
                    "latency_ms": body["timing_ms"]["total"], "message": body["headline"]})
        ok = r.status_code == c.expect_status and body["ood_status"] in OK_STATES
        if c.expect_ood:
            ok &= body["ood_status"] in c.expect_ood
        if c.expect_check:
            ok &= any(x["name"] == c.expect_check and x["status"] != "pass" for x in body["checks"])
    else:
        row.update({"verdict": "rejected", "message": body.get("message") or str(body.get("detail"))[:120],
                    "error_code": body.get("error", "validation_error")})
        ok = r.status_code == c.expect_status
    row["as_expected"] = bool(ok)
    if c.note:
        row["note"] = c.note
        if r.status_code == 200:
            row["note"] += f" (result: {body['ood_status'].replace('_', ' ')})"
    return row


def main() -> None:
    from fastapi.testclient import TestClient
    from app.main import app
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with TestClient(app) as client:
        for c in cases():
            if c.data and len(c.data) < 5_000_000 and c.filename and not c.id.startswith("N-pump") and not c.id.startswith("N-fan"):
                (INPUT_DIR / f"{c.id}_{c.filename}").write_bytes(c.data)
            rows.append(evaluate(client, c))
            r = rows[-1]
            print(f"{'PASS' if r['as_expected'] else 'FAIL'}  {r['id']:<16} HTTP {r['http_status']}  "
                  f"{r.get('verdict', ''):<10} {r.get('ood_status', r.get('error_code', '')):<20} {r.get('score', '')}")
        # determinism: the same file twice must give the same score
        c = next((x for x in cases() if x.id == "N-pump-normal"), None)
        if c:
            s1, s2 = post(client, c).json()["score"], post(client, c).json()["score"]
            rows.append({"id": "N-repeat", "category": "normal", "description": "Same pump clip sent twice (deterministic?)",
                         "machine_type": "pump", "expected_status": 200, "http_status": 200, "verdict": "-",
                         "score": s1, "message": f"scores {s1} and {s2}", "as_expected": s1 == s2})
            flac = next(r for r in rows if r["id"] == "N-flac")
            wavr = next(r for r in rows if r["id"] == "N-pump-normal")
            rows.append({"id": "N-flac-equal", "category": "normal", "description": "FLAC score equals WAV score",
                         "machine_type": "pump", "expected_status": 200, "http_status": 200, "verdict": "-",
                         "score": flac["score"], "message": f"WAV {wavr['score']} vs FLAC {flac['score']}",
                         "as_expected": abs(flac["score"] - wavr["score"]) < 1e-3})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "q4_test_results.csv", index=False)
    cols = ["id", "category", "description", "http_status", "verdict", "score", "ood_status", "flagged_checks", "as_expected"]
    notes = df[df.get("note", pd.Series(dtype=str)).notna()] if "note" in df else df.iloc[0:0]
    md = ["# Q4 – web application test results\n",
          f"{int(df.as_expected.sum())} of {len(df)} cases behaved as expected.\n",
          df.reindex(columns=cols).fillna("").to_markdown(index=False),
          "\n\nRejected inputs and their messages:\n",
          df[df.verdict == "rejected"][["id", "http_status", "error_code", "message"]].to_markdown(index=False),
          "\n\nDocumented known limitations (reported, not hidden):\n",
          "\n".join(f"- **{r.id}**: {r.note}" for r in notes.itertuples()) or "- none"]
    (config.REPORTS_DIR / "q4_test_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{int(df.as_expected.sum())} of {len(df)} cases as expected -> reports/q4_test_results.md, "
          f"test files in tests/q4_inputs/")


if __name__ == "__main__":
    main()
