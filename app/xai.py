"""
Question 5 – explanations for the fault score (Grad-CAM) and a live sanity check.

    log-mel -> FaultCNN -> logit
                  ^ activations A_k of a convolutional layer (default: the last one, features.8)
    Grad-CAM (Selvaraju et al., 2017):  w_k = mean over (freq, time) of d logit / d A_k
                                        map = ReLU( sum_k w_k * A_k )   -> upsampled to the spectrogram

The map shows which time-frequency regions pushed the FAULT score UP for this recording.
It is computed from the logit (not the sigmoid), because the sigmoid is saturated near 1 for pumps.

Every explanation comes with a faithfulness check that runs at request time:
    hide the most highlighted 20 % of the map (set to the band's average level, 0 in model-input space)
    vs. hide 20 % chosen at random (10 draws, fixed seed) -> how much does the fault score drop?
If the highlighted region really matters to the model, hiding it should lower the score MORE than hiding
random regions. This does not prove that the region is the true cause; it only shows the model depends on it.
"""
from __future__ import annotations

import base64
import io
import time

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import config

TOP_FRACTION = 0.20          # share of the map hidden in the faithfulness check
N_RANDOM = 10                # random masks to compare against
DEFAULT_LAYER = "features.8"
EVAL_GRID = (8, 20)          # cells used for the faithfulness check: 16 mel bands x ~0.5 s each

# Human-readable frequency bands for the engineer (Hz)
BANDS = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
# Where faults showed up in the training data (Question 1, Figure 3) – used only as context, never as proof
Q1_FAULT_BANDS = {
    "pump": {"range": (431, 910), "text": "pump faults in the training data raised energy most at 431–910 Hz"},
    "fan": {"range": (0, 50), "range2": (479, 2400),
            "text": "fan faults in the training data added energy below 50 Hz and removed energy at 0.5–2.4 kHz"},
}
MEL_HZ = librosa.mel_frequencies(n_mels=config.MODEL_N_MELS, fmax=config.SAMPLE_RATE / 2)


def _module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    m = model
    for part in name.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def gradcam(model: torch.nn.Module, xt: torch.Tensor, mt: torch.Tensor, layer: str = DEFAULT_LAYER):
    """Return (cam grid [h, w] as numpy, logit). xt: [1, 1, 128, T]."""
    store = {}

    def hook(_m, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = _module(model, layer).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logit = model(xt, mt)
            logit.sum().backward()
    finally:
        h.remove()
    act, grad = store["act"].detach()[0], store["grad"].detach()[0]          # [C, h, w]
    weights = grad.mean(dim=(1, 2))                                           # [C]
    cam = F.relu((weights[:, None, None] * act).sum(0))                       # [h, w]
    model.zero_grad(set_to_none=True)
    return cam.float().cpu().numpy(), float(logit.detach().float().cpu()[0])


def upsample(grid: np.ndarray, shape: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(grid))[None, None].float()
    kw = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(t, size=shape, mode=mode, **kw)[0, 0].numpy()


def pool(full: np.ndarray, grid_shape: tuple[int, int] = EVAL_GRID) -> np.ndarray:
    """Average a full-resolution map [128, T] into the evaluation grid (same cells for every method)."""
    t = torch.from_numpy(np.ascontiguousarray(full))[None, None].float()
    return F.adaptive_avg_pool2d(t, grid_shape)[0, 0].numpy()


def occlusion_grid(model, xt, mt, grid_shape=EVAL_GRID, batch=40) -> np.ndarray:
    """Model-agnostic occlusion: hide each cell in turn and record the logit drop (Zeiler & Fergus, 2014)."""
    shape = tuple(xt.shape[-2:])
    n = grid_shape[0] * grid_shape[1]
    masks = torch.from_numpy(cell_masks(grid_shape, [np.array([i]) for i in range(n)], shape)).to(xt.device)
    with torch.no_grad():
        base = float(model(xt, mt).float()[0])
        drops = []
        for i in range(0, n, batch):
            mb = masks[i:i + batch]
            b = xt.repeat(len(mb), 1, 1, 1)
            b[:, 0][mb] = 0.0
            drops.append(base - model(b, mt.repeat(len(mb))).float().cpu().numpy())
    return np.concatenate(drops).reshape(grid_shape)


def integrated_gradients(model, xt, mt, steps=32, batch=16) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al., 2017) from a band-average baseline (0) – full resolution."""
    alphas = torch.linspace(1 / steps, 1, steps, device=xt.device)
    total = torch.zeros_like(xt)
    with torch.enable_grad():
        for i in range(0, steps, batch):
            a = alphas[i:i + batch].view(-1, 1, 1, 1)
            inp = (a * xt).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            model(inp, mt.repeat(len(a))).sum().backward()
            total += inp.grad.sum(0, keepdim=True)
    model.zero_grad(set_to_none=True)
    return (xt * total / steps)[0, 0].detach().float().cpu().numpy()


def cell_masks(grid_shape: tuple[int, int], cells: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Boolean masks [n, 128, T] that cover the chosen grid cells (nearest-neighbour upsampling)."""
    out = []
    for sel in cells:
        g = np.zeros(grid_shape, dtype=np.float32)
        g.flat[sel] = 1.0
        out.append(upsample(g, shape, "nearest") > 0.5)
    return np.stack(out)


def deletion_check(model, xt, mt, grid: np.ndarray, top_fraction=TOP_FRACTION, n_random=N_RANDOM, seed=0) -> dict:
    """Hide the top-ranked grid cells vs. random cells of the same number; compare the score drop."""
    shape = tuple(xt.shape[-2:])
    n_cells = grid.size
    k = max(1, int(round(top_fraction * n_cells)))
    order = np.argsort(grid.ravel())[::-1]
    rng = np.random.default_rng(seed)
    cells = [order[:k]] + [rng.choice(n_cells, size=k, replace=False) for _ in range(n_random)]
    masks = torch.from_numpy(cell_masks(grid.shape, cells, shape)).to(xt.device)      # [1+n, 128, T]
    batch = xt.repeat(len(cells) + 1, 1, 1, 1)
    batch[1:, 0][masks] = 0.0                                                         # band average level
    with torch.no_grad():
        logits = model(batch, mt.repeat(len(cells) + 1)).float().cpu().numpy()
    base, top, rand = logits[0], logits[1], logits[2:]
    sig = lambda v: 1 / (1 + np.exp(-v))
    return {
        "hidden_share": round(k / n_cells, 3),
        "score": round(float(sig(base)), 4),
        "score_top_hidden": round(float(sig(top)), 4),
        "score_random_hidden": round(float(np.mean(sig(rand))), 4),
        "logit": round(float(base), 3), "logit_top_hidden": round(float(top), 3),
        "logit_random_hidden": round(float(np.mean(rand)), 3),
        "logit_drop_top": round(float(base - top), 3),
        "logit_drop_random": round(float(np.mean(base - rand)), 3),
        "logit_drop_random_sd": round(float(np.std(base - rand)), 3),
        "passed": bool((base - top) > np.mean(base - rand)),
        "mask": masks[0].cpu().numpy(),
    }


def band_shares(cam_full: np.ndarray) -> list[dict]:
    prof = cam_full.mean(axis=1)
    tot = prof.sum() + 1e-12
    out = []
    for lo, hi in BANDS:
        sel = (MEL_HZ >= lo) & ((MEL_HZ < hi) if hi < 8000 else (MEL_HZ <= hi))
        out.append({"band": f"{lo}–{hi} Hz" if hi < 1000 else f"{lo / 1000:g}–{hi / 1000:g} kHz",
                    "lo": lo, "hi": hi, "share": round(float(prof[sel].sum() / tot), 3)})
    return out


def time_summary(cam_full: np.ndarray) -> dict:
    secs = cam_full.shape[1] * config.HOP_LENGTH / config.SAMPLE_RATE
    prof = cam_full.mean(axis=0)
    prof = prof / (prof.sum() + 1e-12)
    win = max(1, int(round(len(prof) * 2.0 / secs)))                 # 2-second window
    conv = np.convolve(prof, np.ones(win), mode="valid")
    i = int(np.argmax(conv))
    start = i * secs / len(prof)
    return {"peak_window_s": [round(start, 1), round(start + 2.0, 1)], "peak_window_share": round(float(conv[i]), 3),
            "concentrated": bool(conv[i] > 0.4)}


def summary_text(mtype: str, bands: list[dict], tsum: dict, check: dict, verdict: str) -> dict:
    top = sorted(bands, key=lambda b: -b["share"])[:2]
    where = f"{top[0]['band']} ({100 * top[0]['share']:.0f} %)"
    if top[1]["share"] > 0.2:
        where += f" and {top[1]['band']} ({100 * top[1]['share']:.0f} %)"
    when = (f"mostly between {tsum['peak_window_s'][0]:.0f} and {tsum['peak_window_s'][1]:.0f} s"
            if tsum["concentrated"] else "spread across the whole recording")
    q1 = Q1_FAULT_BANDS[mtype]
    lo, hi = top[0]["lo"], top[0]["hi"]
    overlaps = lambda r: lo < r[1] and hi > r[0]
    in_q1 = overlaps(q1["range"]) or ("range2" in q1 and overlaps(q1["range2"]))
    lines = [f"The sounds that pushed the fault score up are mainly at {where}, {when}."]
    lines.append(("This overlaps the band where " if in_q1 else "This is outside the band where ") + q1["text"] + ".")
    if check["passed"]:
        lines.append(f"Check passed: hiding the highlighted area lowered the fault evidence more than hiding random areas "
                     f"(evidence drop {check['logit_drop_top']:.2f} vs {check['logit_drop_random']:.2f}).")
    else:
        lines.append(f"Check NOT passed: hiding the highlighted area did not lower the fault evidence more than hiding random "
                     f"areas (evidence drop {check['logit_drop_top']:.2f} vs {check['logit_drop_random']:.2f}). "
                     "Treat this map with extra caution.")
    if verdict == "unreliable":
        lines.append("The input is outside what the model knows, so this explanation is not meaningful.")
    return {"lines": lines, "matches_q1_band": in_q1}


def overlay_png(S: np.ndarray, cam_full: np.ndarray, mask: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=110)
    fig.patch.set_alpha(0)
    secs = S.shape[1] * config.HOP_LENGTH / config.SAMPLE_RATE
    ext = [0, secs, 0, S.shape[0]]
    ax.imshow(S, origin="lower", aspect="auto", cmap="gray", extent=ext,
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    c = cam_full / (cam_full.max() + 1e-12)
    ax.imshow(np.ma.masked_less(c, 0.05), origin="lower", aspect="auto", cmap="inferno", alpha=0.6, extent=ext,
              vmin=0, vmax=1)
    if mask.any():
        ax.contour(np.linspace(0, secs, S.shape[1]), np.arange(S.shape[0]), mask.astype(float), levels=[0.5],
                   colors="#22d3ee", linewidths=1.0)
    ticks = [0, 32, 64, 96, 127]
    ax.set_yticks(ticks, [f"{MEL_HZ[t] / 1000:.1f}k" if MEL_HZ[t] >= 1000 else f"{MEL_HZ[t]:.0f}" for t in ticks])
    ax.set_xlabel("time (s)", color="#9aa4b2", fontsize=8); ax.set_ylabel("frequency (Hz)", color="#9aa4b2", fontsize=8)
    ax.tick_params(colors="#9aa4b2", labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def explain(detector, xt: torch.Tensor, mt: torch.Tensor, S: np.ndarray, mtype: str, verdict: str,
            layer: str = DEFAULT_LAYER, with_image: bool = True) -> dict:
    """Grad-CAM + faithfulness check for the DEPLOYED model. Thread-safe through the detector lock."""
    model = detector.models["deployed"]
    t0 = time.perf_counter()
    with detector.lock:
        grid, logit = gradcam(model, xt, mt, layer)
        t1 = time.perf_counter()
        empty = not np.isfinite(grid).all() or float(grid.max()) <= 0.0
        if not empty:
            grid = grid / float(grid.max())                      # 0..1, relative importance within this clip
        cam_full = upsample(grid, S.shape) if not empty else np.zeros_like(S)
        check = deletion_check(model, xt, mt, pool(cam_full)) if not empty else None
    t2 = time.perf_counter()
    bands, tsum = band_shares(cam_full), time_summary(cam_full)
    out = {"method": "Grad-CAM", "layer": layer, "grid": list(grid.shape),
           "timing_ms": {"gradcam": round((t1 - t0) * 1000, 1), "check": round((t2 - t1) * 1000, 1)}}
    if empty:
        out.update({"empty": True, "bands": bands, "time": tsum, "check": None,
                    "summary": {"lines": ["No region pushed the fault score up for this recording, so there is nothing "
                                          "to highlight. The score comes from the absence of fault-like patterns."],
                                "matches_q1_band": False},
                    "overlay_png": overlay_png(S, cam_full, np.zeros_like(S, bool)) if with_image else None})
        return out
    mask = check.pop("mask")
    out.update({"empty": False, "bands": bands, "time": tsum, "check": check,
                "summary": summary_text(mtype, bands, tsum, check, verdict),
                "overlay_png": overlay_png(S, cam_full, mask) if with_image else None})
    if with_image is False:
        out["cam_grid"] = grid.tolist()
    return out
