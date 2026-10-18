"""
Question 4 – inference engine used by the web application.

Everything the API returns is computed HERE, at request time, by the trained network:
    uploaded bytes
      -> decode + input-quality checks          (invalid input is rejected with a clear message)
      -> channel 0, 16 kHz, 128-band log-mel     (exactly the training pipeline, src/features.py)
      -> out-of-distribution checks              (level and spectral shape vs. development data)
      -> train-only normalisation -> FaultCNN    (deployed model + the 2 other seeds of the final design)
      -> fault score, per-type alarm threshold, validation false-alarm rate, seed agreement

Nothing is pre-computed per input. The only pre-computed file is app/reference/q4_reference.json
(validation score distribution and development-data ranges), built by scripts/q4_build_reference.py
from TRAIN and VALIDATION data only.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.special import expit

from src import config
from src.model import FaultCNN

ROOT = config.ROOT
MODELS = ROOT / "models"
REFERENCE = ROOT / "app" / "reference" / "q4_reference.json"

# ---- input rules (shown to the user in the UI as well) ----
ALLOWED_EXT = {".wav", ".flac"}
MAX_BYTES = 25 * 1024 * 1024          # 25 MB
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
EXPECTED_SECONDS = 10.0               # the model was trained and evaluated on 10-s clips
SILENCE_DB = -80.0                    # RMS below this = no usable signal
CLIP_LEVEL, CLIP_FRACTION = 0.999, 0.001


class InputError(Exception):
    """Raised for inputs that cannot be analysed. Carries an HTTP status and a code for the UI."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Check:
    name: str
    status: str            # "pass" | "info" | "warn" | "fail"
    detail: str


@dataclass
class Audio:
    y: np.ndarray
    original_sr: int
    channels: int
    seconds: float
    checks: list[Check] = field(default_factory=list)


# ------------------------------------------------------------------ decoding + quality checks
def decode_and_check(data: bytes, filename: str) -> Audio:
    ext = Path(filename or "").suffix.lower()
    if not data:
        raise InputError(400, "empty_file", "The uploaded file is empty.")
    if ext not in ALLOWED_EXT:
        raise InputError(415, "unsupported_type",
                         f"File type '{ext or 'none'}' is not supported. Please upload a WAV or FLAC recording.")
    if len(data) > MAX_BYTES:
        raise InputError(413, "file_too_large", f"File is {len(data) / 1e6:.1f} MB; the limit is 25 MB.")
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise InputError(422, "unreadable_audio",
                         "The file could not be decoded as audio. It may be corrupted or not really a WAV/FLAC file.")
    if y.size == 0 or sr <= 0:
        raise InputError(422, "no_samples", "The file contains no audio samples.")
    if not np.all(np.isfinite(y)):
        raise InputError(422, "invalid_samples", "The audio contains invalid (NaN or infinite) sample values.")

    channels, seconds = y.shape[1], y.shape[0] / sr
    checks: list[Check] = []
    if seconds < MIN_SECONDS:
        raise InputError(422, "too_short", f"The recording is {seconds:.2f} s long; at least {MIN_SECONDS:.0f} s is needed.")
    if seconds > MAX_SECONDS:
        raise InputError(422, "too_long", f"The recording is {seconds:.1f} s long; the maximum is {MAX_SECONDS:.0f} s. "
                                          "Please upload a 10-second clip.")
    if abs(seconds - EXPECTED_SECONDS) <= 0.5:
        checks.append(Check("Duration", "pass", f"{seconds:.1f} s (the model was trained on 10-s clips)"))
    else:
        checks.append(Check("Duration", "warn", f"{seconds:.1f} s – the model was trained and evaluated on 10-s clips; "
                                                "results for other lengths are less reliable"))

    ch = config.CHANNEL if config.CHANNEL < channels else 0
    sig = y[:, ch]
    if channels == 1:
        checks.append(Check("Channels", "pass", "mono recording"))
    else:
        checks.append(Check("Channels", "info", f"{channels} channels – channel {ch} is analysed (same as training)"))

    if sr != config.SAMPLE_RATE:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        checks.append(Check("Sample rate", "info", f"{sr} Hz resampled to {config.SAMPLE_RATE} Hz"))
    else:
        checks.append(Check("Sample rate", "pass", f"{sr} Hz"))

    rms = float(20 * np.log10(np.sqrt(np.mean(sig ** 2)) + 1e-12))
    if rms < SILENCE_DB:
        raise InputError(422, "silent", f"The recording is silent (level {rms:.0f} dBFS). Check the microphone.")
    clipped = float(np.mean(np.abs(sig) >= CLIP_LEVEL))
    if clipped > CLIP_FRACTION:
        checks.append(Check("Clipping", "warn", f"{100 * clipped:.2f} % of samples are clipped – the microphone gain is "
                                                "too high, which distorts the spectrum"))
    else:
        checks.append(Check("Clipping", "pass", "no clipping"))
    return Audio(sig.astype(np.float32), sr, channels, seconds, checks)


def logmel(sig: np.ndarray) -> np.ndarray:
    """Same features as training (src/features.py): 128 mel bands, dB. Clips of ~10 s get exactly 313 frames."""
    m = librosa.feature.melspectrogram(y=sig, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
                                       hop_length=config.HOP_LENGTH, n_mels=config.MODEL_N_MELS)
    S = librosa.power_to_db(m, ref=1.0).astype(np.float32)
    if abs(S.shape[1] - config.CLIP_FRAMES) <= 16:
        S = librosa.util.fix_length(S, size=config.CLIP_FRAMES, axis=1)
    return S


def level_db(S: np.ndarray) -> float:
    """Overall level of a log-mel spectrogram (dB of the mean power) – same definition as the reference file."""
    return float(10 * np.log10(np.mean(10 ** (S / 10)) + 1e-12))


def shape_distance(S: np.ndarray, ref_profile: np.ndarray) -> float:
    """How different the average spectrum shape is from the development data (level difference removed)."""
    diff = S.mean(axis=1) - ref_profile
    return float(np.std(diff))


# ------------------------------------------------------------------ the detector
class FaultDetector:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta = json.loads((MODELS / "final_model.json").read_text())
        self.deployed_id = meta["source_run"]
        self.thresholds = {k: float(v) for k, v in meta["thresholds_val_fpr5"].items()}
        self.cfg = meta["cfg"]
        self.clip_norm = bool(self.cfg.get("clip_norm", False))   # same preprocessing as in training
        s = np.load(config.NORM_STATS)
        self.mean, self.std = s["mean"].astype(np.float32), s["std"].astype(np.float32)
        self.ref = json.loads(REFERENCE.read_text())
        self.models = {"deployed": self._load(MODELS / "final_model.pt")}
        for sid in self.ref["seed_runs"]:
            p = MODELS / f"{sid}.pt"
            if sid != self.deployed_id and p.exists():
                self.models[sid] = self._load(p)
        self.lock = threading.Lock()
        self.ref_profiles = {t: np.array(v["profile_db"], dtype=np.float32) for t, v in self.ref["ranges"].items()}
        xcfg = ROOT / "app" / "reference" / "q5_xai.json"          # written by scripts/q5_xai_analysis.py
        self.xai_layer = json.loads(xcfg.read_text()).get("layer", "features.8") if xcfg.exists() else "features.8"
        self.warm_up()

    def warm_up(self) -> None:
        """Run one dummy prediction so the first real request is not slowed down by one-off library start-up."""
        rng = np.random.default_rng(0)
        self.explain(wav_bytes(0.05 * rng.standard_normal(int(EXPECTED_SECONDS * config.SAMPLE_RATE))), "warmup.wav", "pump")

    def _load(self, path: Path) -> FaultCNN:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        m = FaultCNN(pretrained=False)
        m.load_state_dict(ck["state_dict"])
        return m.to(self.device).eval()

    # ---------- out-of-distribution checks ----------
    def ood_checks(self, S: np.ndarray, mtype: str) -> tuple[str, list[Check], dict]:
        r = self.ref["ranges"][mtype]
        lvl, dist = level_db(S), shape_distance(S, self.ref_profiles[mtype])
        checks, severity = [], 0
        lo, hi, mn, mx = r["level_p01"], r["level_p99"], r["level_min"], r["level_max"]
        if lo <= lvl <= hi:
            checks.append(Check("Loudness vs. development data", "pass", f"{lvl:.1f} dB (usual range {lo:.1f} to {hi:.1f} dB)"))
        elif mn - 6 <= lvl <= mx + 6:
            severity = max(severity, 1)
            checks.append(Check("Loudness vs. development data", "warn", f"{lvl:.1f} dB is unusual for a {mtype} "
                                f"(usual range {lo:.1f} to {hi:.1f} dB)"))
        else:
            severity = 2
            checks.append(Check("Loudness vs. development data", "fail", f"{lvl:.1f} dB is far outside anything seen "
                                f"for a {mtype} ({mn:.1f} to {mx:.1f} dB)"))
        p99 = r["shape_p99"]
        if dist <= p99:
            checks.append(Check("Spectrum shape vs. development data", "pass", f"distance {dist:.2f} (99 % of development "
                                f"clips are below {p99:.2f})"))
        elif dist <= 2 * p99:
            severity = max(severity, 1)
            checks.append(Check("Spectrum shape vs. development data", "warn", f"distance {dist:.2f} – the sound is unlike "
                                f"most {mtype} recordings the model has seen (99 % below {p99:.2f})"))
        else:
            severity = 2
            checks.append(Check("Spectrum shape vs. development data", "fail", f"distance {dist:.2f} – this does not sound "
                                f"like a {mtype} recording the model knows (99 % below {p99:.2f})"))
        status = ["in_distribution", "unusual", "out_of_distribution"][severity]
        return status, checks, {"level_db": round(lvl, 2), "shape_distance": round(dist, 3)}

    # ---------- model ----------
    def model_input(self, S: np.ndarray, mtype: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-mel -> the exact tensor the network sees (train-only normalisation, optional clip_norm)."""
        x = (S - self.mean[:, None]) / self.std[:, None]
        if self.clip_norm:
            x = x - x.mean(axis=1, keepdims=True)
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        mt = torch.tensor([config.MACHINE_TYPE_INDEX[mtype]], device=self.device)
        return xt, mt

    @torch.no_grad()
    def _scores(self, S: np.ndarray, mtype: str) -> dict[str, float]:
        xt, mt = self.model_input(S, mtype)
        out = {}
        with self.lock:
            for name, m in self.models.items():
                out[name] = float(expit(m(xt, mt).float().cpu().numpy()[0]))
        return out

    def false_alarm_rate(self, score: float, mtype: str) -> float:
        """Share of NORMAL validation clips (machine id_04) that score at least this high."""
        normal = np.asarray(self.ref["validation"][mtype]["normal_scores"])
        return float(np.mean(normal >= score))

    # ---------- public ----------
    def predict(self, data: bytes, filename: str, mtype: str) -> dict:
        return self._predict(data, filename, mtype)[0]

    def explain(self, data: bytes, filename: str, mtype: str) -> dict:
        """Question 5: prediction + Grad-CAM explanation + live faithfulness check, all at request time."""
        from app import xai
        res, S = self._predict(data, filename, mtype)
        xt, mt = self.model_input(S, mtype)
        res["explanation"] = xai.explain(self, xt, mt, S, mtype, res["verdict"], layer=self.xai_layer)
        return res

    def _predict(self, data: bytes, filename: str, mtype: str) -> tuple[dict, np.ndarray]:
        if mtype not in config.MACHINE_TYPE_INDEX:
            raise InputError(422, "bad_machine_type", "Machine type must be 'pump' or 'fan'.")
        t0 = time.perf_counter()
        audio = decode_and_check(data, filename)
        S = logmel(audio.y)
        ood, ood_checks, ood_values = self.ood_checks(S, mtype)
        t1 = time.perf_counter()
        scores = self._scores(S, mtype)
        t2 = time.perf_counter()

        score, thr = scores["deployed"], self.thresholds[mtype]
        alarm = score >= thr
        seed_thr = self.ref["seed_thresholds"]
        run_id = {n: (self.deployed_id if n == "deployed" else n) for n in scores}
        votes = [scores[n] >= seed_thr[run_id[n]][mtype] for n in scores if run_id[n] in seed_thr]
        all_scores = list(scores.values())
        far = self.false_alarm_rate(score, mtype)

        if ood == "out_of_distribution":
            verdict, headline = "unreliable", "Result not reliable – the input is outside what the model knows"
        elif alarm:
            verdict, headline = "alarm", "Fault-like sound – inspect this machine"
        else:
            verdict, headline = "no_alarm", "No alarm – keep the planned maintenance schedule"

        return ({
            "verdict": verdict, "headline": headline, "machine_type": mtype,
            "score": round(score, 4), "threshold": round(thr, 4), "alarm": bool(alarm),
            "margin": round(score - thr, 4),
            "validation_false_alarm_rate": round(far, 4),
            "seed_scores": {k if k != "deployed" else self.deployed_id: round(v, 4) for k, v in scores.items()},
            "seed_spread": round(max(all_scores) - min(all_scores), 4),
            "seed_votes": int(sum(votes)), "seed_count": len(votes),
            "ood_status": ood, "ood_values": ood_values,
            "checks": [c.__dict__ for c in audio.checks + ood_checks],
            "audio": {"seconds": round(audio.seconds, 2), "original_sample_rate": audio.original_sr,
                      "channels": audio.channels},
            "spectrogram_png": spectrogram_png(S),
            "timing_ms": {"preprocess": round((t1 - t0) * 1000, 1), "model": round((t2 - t1) * 1000, 1),
                          "total": round((t2 - t0) * 1000, 1)},
            "model": {"deployed_run": self.deployed_id, "device": str(self.device), "n_models": len(scores)},
        }, S)

    def model_card(self) -> dict:
        return {"deployed_run": self.deployed_id, "thresholds": self.thresholds, "device": str(self.device),
                "seed_runs": list(self.ref["seed_runs"]), "input_rules": {
                    "formats": sorted(ALLOWED_EXT), "max_mb": MAX_BYTES // (1024 * 1024),
                    "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS, "expected_seconds": EXPECTED_SECONDS},
                "evaluation": self.ref.get("evaluation", {}), "built": self.ref.get("built"),
                "explanation_method": {"name": "Grad-CAM", "layer": self.xai_layer}}


def wav_bytes(sig: np.ndarray, sr: int = config.SAMPLE_RATE) -> bytes:
    """Encode a float signal as 16-bit WAV bytes (used for warm-up and tests)."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(sig, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def spectrogram_png(S: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=110)
    fig.patch.set_alpha(0)
    secs = S.shape[1] * config.HOP_LENGTH / config.SAMPLE_RATE
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, secs, 0, S.shape[0]],
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    hz = librosa.mel_frequencies(n_mels=config.MODEL_N_MELS, fmax=config.SAMPLE_RATE / 2)
    ticks = [0, 32, 64, 96, 127]
    ax.set_yticks(ticks, [f"{hz[t] / 1000:.1f}k" if hz[t] >= 1000 else f"{hz[t]:.0f}" for t in ticks])
    ax.set_xlabel("time (s)", color="#9aa4b2", fontsize=8); ax.set_ylabel("frequency (Hz)", color="#9aa4b2", fontsize=8)
    ax.tick_params(colors="#9aa4b2", labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
