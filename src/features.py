"""
Question 2 – build the model input once and cache it.

For every clip in splits.csv:
    waveform (channel 0, 16 kHz, 10 s)
      -> log-mel spectrogram, 128 mel bands x 313 frames, in dB  (float16 to save disk)
and stores one array per split:
    data/processed/logmel128/train.npy   (5058, 128, 313)
    data/processed/logmel128/val.npy     (2183, 128, 313)
    data/processed/logmel128/test.npy    (2514, 128, 313)
    data/processed/logmel128/<split>_index.csv   (row order = array order)

Then it computes the normalisation statistics (mean and std of every mel band)
from the TRAINING split only, and saves them to data/processed/norm_stats.npz.
Validation and test are later normalised with these same numbers (no leakage).

Run:  python -m src.features            (about 2-5 minutes, ~0.8 GB on disk)
"""
from __future__ import annotations

import time

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.audio_utils import load_audio


def logmel128(path) -> np.ndarray:
    """Log-mel spectrogram used as CNN input: shape (128, 313), dB, float16."""
    y = load_audio(config.ROOT / path)
    m = librosa.feature.melspectrogram(y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
                                       hop_length=config.HOP_LENGTH, n_mels=config.MODEL_N_MELS)
    S = librosa.power_to_db(m, ref=1.0)
    S = librosa.util.fix_length(S, size=config.CLIP_FRAMES, axis=1)   # guarantees 313 frames
    return S.astype(np.float16)


def build_split(split: str, df: pd.DataFrame) -> np.ndarray:
    rows = df[df.split == split].reset_index(drop=True)
    print(f"  {split:<5}: {len(rows)} clips ...", flush=True)
    arrs = Parallel(n_jobs=-1)(delayed(logmel128)(p) for p in rows.path)
    X = np.stack(arrs)
    np.save(config.FEATURE_DIR / f"{split}.npy", X)
    rows[["path", "label", "machine_type", "machine_id", "group"]].to_csv(
        config.FEATURE_DIR / f"{split}_index.csv", index=False)
    return X


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(config.SPLITS_CSV)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing 128-band log-mel spectrograms (channel 0):")
    X_train = None
    for split in ["train", "val", "test"]:
        X = build_split(split, df)
        if split == "train":
            X_train = X

    # normalisation statistics from TRAIN only (rule from Question 1, Table 4)
    Xt = X_train.astype(np.float32)
    mean = Xt.mean(axis=(0, 2))                  # one value per mel band
    std = Xt.std(axis=(0, 2)) + 1e-6
    np.savez(config.NORM_STATS, mean=mean, std=std, n_train_clips=len(Xt))
    print(f"\nNormalisation stats (train only, {len(Xt)} clips) -> "
          f"{config.NORM_STATS.relative_to(config.ROOT)}")
    print(f"  band mean range {mean.min():.1f} .. {mean.max():.1f} dB, "
          f"std range {std.min():.1f} .. {std.max():.1f} dB")
    size_mb = sum(f.stat().st_size for f in config.FEATURE_DIR.glob("*.npy")) / 1e6
    print(f"Cache size: {size_mb:.0f} MB | time {(time.time() - t0) / 60:.1f} min")
    print("Next step:  python -m scripts.q2_architecture_report")


if __name__ == "__main__":
    main()
