"""
Create the small, representative dataset sample required for submission
(brief section 8: dataset > 500 MB -> submit a download script + a sample).

Takes N clips per (machine, machine_id, label) using the fixed seed, keeps only
microphone channel 0 (what the model uses), and writes them to data/sample/
with the same folder structure plus a sample_splits.csv.

Run:  python -m scripts.make_sample --per-group 20
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.audio_utils import load_audio  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=20, help="clips per machine-id x label")
    args = p.parse_args()

    df = pd.read_csv(config.SPLITS_CSV)
    sample = df.groupby(["group", "label"]).sample(frac=1, random_state=config.SEED) \
               .groupby(["group", "label"]).head(args.per_group)

    if config.SAMPLE_DIR.exists():
        shutil.rmtree(config.SAMPLE_DIR)
    new_paths = []
    for rel in sample["path"]:
        src = config.ROOT / rel
        dst = config.SAMPLE_DIR / src.relative_to(config.RAW_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, load_audio(src), config.SAMPLE_RATE, subtype="PCM_16")
        new_paths.append(dst.relative_to(config.ROOT).as_posix())
    sample = sample.assign(path=new_paths)
    sample.to_csv(config.SAMPLE_DIR / "sample_splits.csv", index=False)

    mb = sum(f.stat().st_size for f in config.SAMPLE_DIR.rglob("*.wav")) / 1e6
    print(f"Sample: {len(sample)} clips, {mb:.0f} MB -> {config.SAMPLE_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
