"""
Question 5 – tests for the explanation feature (run:  pytest tests/test_xai.py -v).

1. Unit tests on a tiny hand-made network whose output depends ONLY on one known region:
   Grad-CAM must point to that region, and the deletion check must pass. This proves the
   implementation (hooks, gradients, masks) is correct before it is used on the real model.
2. API tests through the real FastAPI app and the deployed model.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app import xai
from app.main import app
from scripts.q4_test_inputs import cases, post


class RegionNet(nn.Module):
    """logit = mean of the input inside one region (frequency rows 64-95, time frames 100-180)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.AvgPool2d(16))
        with torch.no_grad():
            self.features[0].weight.zero_()
            self.features[0].weight[:, 0, 1, 1] = 1.0
            self.features[0].bias.zero_()

    def forward(self, x, mt):
        f = self.features(x)                                   # [B, 4, 8, 19]
        return f[:, :, 4:6, 6:11].mean(dim=(1, 2, 3)) * 10.0


@pytest.fixture
def region_input():
    x = torch.zeros(1, 1, 128, 313)
    x[:, :, 64:96, 96:176] = 1.0                               # the "fault sound"
    x += 0.05 * torch.rand_like(x)
    return x, torch.tensor([0])


def test_gradcam_points_to_the_region(region_input):
    xt, mt = region_input
    grid, logit = xai.gradcam(RegionNet().eval(), xt, mt, layer="features.0")
    full = xai.upsample(grid / grid.max(), (128, 313))
    inside = full[64:96, 96:176].mean()
    outside = np.concatenate([full[:48].ravel(), full[112:].ravel()]).mean()
    assert logit > 0 and inside > 5 * outside


def test_deletion_check_passes_for_true_region_and_fails_for_wrong_one(region_input):
    m, (xt, mt) = RegionNet().eval(), region_input
    right = np.zeros(xai.EVAL_GRID); right[4:6, 6:11] = 1.0
    wrong = np.zeros(xai.EVAL_GRID); wrong[0:2, 0:5] = 1.0
    ok = xai.deletion_check(m, xt, mt, right + 1e-6 * np.random.default_rng(0).random(xai.EVAL_GRID))
    bad = xai.deletion_check(m, xt, mt, wrong + 1e-6 * np.random.default_rng(1).random(xai.EVAL_GRID))
    assert ok["passed"] and ok["logit_drop_top"] > ok["logit_drop_random"]
    assert not bad["passed"]


def test_occlusion_and_ig_find_the_region(region_input):
    m, (xt, mt) = RegionNet().eval(), region_input
    occ = xai.occlusion_grid(m, xt, mt)
    assert np.unravel_index(np.argmax(occ), occ.shape)[0] in (4, 5)
    ig = xai.integrated_gradients(m, xt, mt, steps=8)
    assert ig[64:96, 96:176].sum() > 0.8 * ig.clip(0).sum()


# ---------------- API ----------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _case(cid):
    return next(c for c in cases() if c.id == cid)


@pytest.mark.parametrize("cid", ["N-pump-normal", "N-fan-abnormal"])
def test_explain_endpoint(client, cid):
    case = _case(cid)
    r = client.post("/api/explain", files={"file": (case.filename, case.data)}, data={"machine_type": case.machine_type})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    e = body["explanation"]
    assert e["method"] == "Grad-CAM" and e["overlay_png"].startswith("data:image/png;base64,")
    assert abs(sum(b["share"] for b in e["bands"]) - 1) < 0.02 or e["empty"]
    assert e["summary"]["lines"]
    if not e["empty"]:
        c = e["check"]
        assert {"score_top_hidden", "score_random_hidden", "passed", "logit_drop_top"} <= set(c)
        assert abs(c["score"] - body["score"]) < 1e-3          # explanation is for the same prediction
    # same score as the plain prediction endpoint (the explanation does not change the prediction)
    p = post(client, case).json()
    assert abs(p["score"] - body["score"]) < 1e-4


def test_explain_rejects_invalid_input(client):
    case = _case("I-text-as-wav")
    r = client.post("/api/explain", files={"file": (case.filename, case.data)}, data={"machine_type": "pump"})
    assert r.status_code == 422 and r.json()["error"] == "unreadable_audio"


def test_explain_is_deterministic(client):
    case = _case("N-fan-normal")
    a, b = [client.post("/api/explain", files={"file": (case.filename, case.data)},
                        data={"machine_type": case.machine_type}).json() for _ in range(2)]
    assert a["request_id"] != b["request_id"]
    assert a["explanation"]["bands"] == b["explanation"]["bands"]
    assert a["explanation"]["check"] == b["explanation"]["check"]


def test_example_explain(client):
    ex = client.get("/api/examples").json()
    if not ex:
        pytest.skip("no demo clips on this computer")
    r = client.post(f"/api/examples/{ex[0]['id']}/explain")
    assert r.status_code == 200 and "explanation" in r.json() and "true_label" in r.json()
