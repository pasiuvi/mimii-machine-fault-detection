"""
Compute (once) a compact summary of every clip and cache it to disk.

For each clip we store:
  * mean and std of the log-mel spectrogram over time (64 + 64 numbers)
  * overall loudness (RMS in dB)
  * a hash of the decoded audio samples (for exact-duplicate detection)

Used by the EDA notebook and src/leakage_checks.py. Computing it takes a few
minutes the first time; afterwards it loads instantly from
data/processed/eda_logmel_summary.npz.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.audio_utils import load_audio, log_mel, rms_db


def _summarise(rel_path: str):
    y = load_audio(config.ROOT / rel_path)
    S = log_mel(y)
    audio_hash = hashlib.md5(np.round(y, 5).tobytes()).hexdigest()
    return S.mean(axis=1), S.std(axis=1), rms_db(y), audio_hash


def get_summary(force: bool = False, n_jobs: int = -1) -> dict:
    """Return dict with keys: path, mel_mean, mel_std, rms_db, audio_hash (aligned with splits.csv)."""
    splits = pd.read_csv(config.SPLITS_CSV)
    if config.FEATURE_CACHE.exists() and not force:
        cache = dict(np.load(config.FEATURE_CACHE, allow_pickle=True))
        if list(cache["path"]) == splits["path"].tolist():
            return cache
        print("Cache is out of date (splits changed) - recomputing.")

    print(f"Computing log-mel summaries for {len(splits)} clips (one-off, a few minutes)...")
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_summarise)(p) for p in splits["path"])
    cache = {
        "path": splits["path"].to_numpy(),
        "mel_mean": np.stack([o[0] for o in out]).astype("float32"),
        "mel_std": np.stack([o[1] for o in out]).astype("float32"),
        "rms_db": np.array([o[2] for o in out], dtype="float32"),
        "audio_hash": np.array([o[3] for o in out]),
    }
    np.savez_compressed(config.FEATURE_CACHE, **cache)
    return cache
