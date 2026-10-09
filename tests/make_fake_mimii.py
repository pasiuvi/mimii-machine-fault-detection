"""
Create a tiny FAKE dataset with the same folder structure as MIMII, so you can
test the whole pipeline in 1 minute before the real 18 GB download finishes.

    python -m tests.make_fake_mimii        # writes data/raw/6_dB_pump, data/raw/6_dB_fan
    (delete data/raw/* and data/processed/* before using the real data)
"""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

rng = np.random.default_rng(0)
SR, SECONDS, CH = 16_000, 2, 8


def clip(base_hz: float, abnormal: bool) -> np.ndarray:
    t = np.arange(SR * SECONDS) / SR
    y = 0.1 * np.sin(2 * np.pi * base_hz * t) + 0.05 * rng.standard_normal(t.size)
    if abnormal:  # add a short high-frequency "knock"
        s = rng.integers(0, t.size - 1600)
        y[s:s + 1600] += 0.3 * np.sin(2 * np.pi * 3000 * t[:1600])
    return np.tile(y[:, None], (1, CH)).astype("float32")


for m_i, machine in enumerate(["pump", "fan"]):
    for k, mid in enumerate(["id_00", "id_02", "id_04", "id_06"]):
        base = 100 + 40 * k + 200 * m_i  # each machine sounds a bit different
        for label, n in [("normal", 30), ("abnormal", 8)]:
            d = config.RAW_DIR / f"6_dB_{machine}" / machine / mid / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                sf.write(d / f"{i:08d}.wav", clip(base, label == "abnormal"), SR)
print("Fake MIMII-like data written to data/raw/")
