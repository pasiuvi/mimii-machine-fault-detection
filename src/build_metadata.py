"""
Step 2: scan the extracted wav files and build one metadata table.

Output: data/processed/metadata.csv  - one row per audio clip with:
    path, snr, machine_type, machine_id, group, label, label_name,
    sample_rate, channels, n_frames, duration_s, file_bytes, readable

Run:  python -m src.build_metadata
"""
from __future__ import annotations

import re

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import config

ID_RE = re.compile(r"^id_\d+$")


def parse(path) -> dict | None:
    """Read labels from the folder structure .../<machine>/id_XX/<normal|abnormal>/file.wav"""
    parts = path.relative_to(config.RAW_DIR).parts
    try:
        i = next(k for k, p in enumerate(parts) if ID_RE.match(p))
    except StopIteration:
        return None
    label_name = parts[i + 1]
    if label_name not in ("normal", "abnormal"):
        return None
    snr_folder = parts[0]                      # e.g. "6_dB_pump"
    snr = snr_folder.rsplit("_", 1)[0]         # -> "6_dB"
    machine_type = parts[i - 1] if i >= 1 else snr_folder.rsplit("_", 1)[1]
    return {
        "path": path.relative_to(config.ROOT).as_posix(),
        "snr": snr,
        "machine_type": machine_type,
        "machine_id": parts[i],
        "group": f"{machine_type}_{parts[i]}",   # the physical machine = the leakage group
        "label": int(label_name == "abnormal"),  # 1 = abnormal (fault), 0 = normal
        "label_name": label_name,
    }


def main() -> None:
    wavs = sorted(config.RAW_DIR.rglob("*.wav"))
    if not wavs:
        raise SystemExit("No wav files in data/raw. Run:  python -m scripts.download_mimii")

    rows = []
    for w in tqdm(wavs, desc="Reading audio headers", unit="file"):
        row = parse(w)
        if row is None:
            continue
        try:
            info = sf.info(str(w))
            row.update(sample_rate=info.samplerate, channels=info.channels,
                       n_frames=info.frames, duration_s=round(info.duration, 3), readable=True)
        except Exception:  # corrupted / unreadable file -> keep a record so EDA can report it
            row.update(sample_rate=None, channels=None, n_frames=None, duration_s=None, readable=False)
        row["file_bytes"] = w.stat().st_size
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["machine_type"].isin(config.MACHINE_TYPES) & (df["snr"] == config.SNR)]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.METADATA_CSV, index=False)

    print(f"\nSaved {len(df)} rows -> {config.METADATA_CSV.relative_to(config.ROOT)}")
    print(pd.crosstab([df.machine_type, df.machine_id], df.label_name, margins=True))
    print("\nNext step:  python -m src.make_splits")


if __name__ == "__main__":
    main()
