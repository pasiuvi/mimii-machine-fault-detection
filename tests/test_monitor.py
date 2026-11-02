"""
v2 – tests for per-machine monitoring with enrolment (app/monitor.py and its API).

CLAP is replaced by a small deterministic embedder (spectral shape of the clip), so the tests are fast and
check the LOGIC: enrolment rules, standardised score, the 2-of-3 alarm rule, input errors and the endpoints.
"""
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import monitor
from app.inference import InputError

SR = 16_000


def wav(freq: float, seed: int, noise: float = 0.05, seconds: float = 10.0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = 0.3 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(t.size)
    b = io.BytesIO(); sf.write(b, y.astype(np.float32), SR, format="WAV"); return b.getvalue()


def fake_embedder(sigs):
    """64-band log spectrum, unit length: different tones -> different embeddings."""
    out = []
    for s in sigs:
        S = np.abs(np.fft.rfft(s[: SR * 2]))[: 8000]
        e = np.log1p(S.reshape(64, -1).mean(1))
        out.append(e / np.linalg.norm(e))
    return np.stack(out).astype(np.float32)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "STORE", tmp_path / "monitor_data")
    monitor.set_embedder(fake_embedder)
    yield
    monitor.set_embedder(None)


def normal_set(n=30):
    return [(wav(500 + (i % 5), seed=i), f"n{i}.wav") for i in range(n)]


def test_enrol_and_normal_clip_is_not_unusual():
    r = monitor.enrol("Pump Line 3", "pump", normal_set())
    assert r["id"] == "pump-line-3" and r["n_enrol"] == 30 and r["std"] > 0
    c = monitor.check("pump-line-3", wav(502, seed=999), "new_normal.wav")
    assert c["status"] == "normal" and c["unusual"] is False


def test_two_of_three_rule():
    monitor.enrol("fan-a", "fan", normal_set())
    first = monitor.check("fan-a", wav(3000, seed=1, noise=0.2), "odd1.wav")
    assert first["unusual"] and first["status"] == "watch"            # one unusual clip is only a watch
    second = monitor.check("fan-a", wav(3100, seed=2, noise=0.2), "odd2.wav")
    assert second["status"] == "alarm" and second["unusual_in_window"] == 2
    monitor.reset_history("fan-a")
    again = monitor.check("fan-a", wav(3000, seed=3, noise=0.2), "odd3.wav")
    assert again["status"] == "watch"                                   # history was cleared


def test_too_few_clips_rejected():
    with pytest.raises(InputError) as e:
        monitor.enrol("p", "pump", normal_set(10))
    assert e.value.code == "too_few_clips"


def test_bad_files_are_skipped_and_reported():
    files = normal_set(22) + [(b"not audio", "broken.wav"), (b"", "empty.wav")]
    r = monitor.enrol("p2", "pump", files)
    assert r["n_enrol"] == 22 and len(r["rejected"]) == 2
    assert any("fewer than" in w for w in r["warnings"])                # below the recommended 50


def test_unknown_machine_and_bad_inputs():
    with pytest.raises(InputError) as e:
        monitor.check("nope", wav(500, 1), "x.wav")
    assert e.value.status == 404
    with pytest.raises(InputError):
        monitor.enrol("", "pump", normal_set())
    with pytest.raises(InputError):
        monitor.enrol("valve-1", "valve", normal_set())


def test_api_enrol_check_list_delete():
    from app.main import app
    with TestClient(app) as client:
        files = [("files", (f, b, "audio/wav")) for b, f in normal_set(25)]
        r = client.post("/api/monitor/enrol", data={"name": "api-pump", "machine_type": "pump"}, files=files)
        assert r.status_code == 200, r.text
        r = client.post("/api/monitor/api-pump/check", files={"file": ("n.wav", wav(501, 77), "audio/wav")})
        assert r.status_code == 200 and r.json()["status"] == "normal"
        assert any(m["id"] == "api-pump" for m in client.get("/api/monitor/machines").json()["machines"])
        r = client.post("/api/monitor/api-pump/check", files={"file": ("bad.txt", b"hello", "text/plain")})
        assert r.status_code == 415
        assert client.delete("/api/monitor/api-pump").json()["deleted"] == "api-pump"
        assert client.get("/monitor").status_code == 200
