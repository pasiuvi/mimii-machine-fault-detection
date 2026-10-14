"""
PyTorch Dataset that feeds cached log-mel spectrograms to the CNN.

Training mode (augment=True), in this order:
  1. random 4-second crop (126 of 313 frames)      - Finding 2: faults are steady, so any
                                                     4-s window still contains the fault
  2. random gain, a constant shift of +-6 dB       - Finding 3: loudness is an unreliable cue
  3. per-band normalisation with TRAIN statistics  - same numbers for train/val/test
  4. SpecAugment: 2 frequency masks + 2 time masks - regularisation against over-fitting
Evaluation mode (augment=False): the full 10-s clip, normalised, nothing random.

Each item: (spectrogram [1, 128, T], machine_type_index, label as float)
Only audio-derived input + the machine TYPE (pump/fan) are given to the model;
machine ID, file path and folder name are never used (leakage rule, Q1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


def load_norm_stats():
    s = np.load(config.NORM_STATS)
    return s["mean"].astype(np.float32), s["std"].astype(np.float32)


class MimiiSpecDataset(Dataset):
    def __init__(self, split: str, augment: bool = False, seed: int = config.SEED):
        self.X = np.load(config.FEATURE_DIR / f"{split}.npy", mmap_mode="r")
        self.index = pd.read_csv(config.FEATURE_DIR / f"{split}_index.csv")
        assert len(self.X) == len(self.index), "cache and index are out of sync - rerun src.features"
        self.labels = self.index["label"].to_numpy(np.float32)
        self.mtypes = self.index["machine_type"].map(config.MACHINE_TYPE_INDEX).to_numpy(np.int64)
        self.mean, self.std = load_norm_stats()
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def _spec_augment(self, x: np.ndarray) -> np.ndarray:
        p = config.SPEC_AUGMENT
        n_mels, n_frames = x.shape
        for _ in range(p["freq_masks"]):
            w = self.rng.integers(0, p["freq_width"] + 1)
            f0 = self.rng.integers(0, n_mels - w + 1)
            x[f0:f0 + w, :] = 0.0                      # 0 = band mean after normalisation
        for _ in range(p["time_masks"]):
            w = self.rng.integers(0, p["time_width"] + 1)
            t0 = self.rng.integers(0, n_frames - w + 1)
            x[:, t0:t0 + w] = 0.0
        return x

    def __getitem__(self, i: int):
        x = np.asarray(self.X[i], dtype=np.float32)            # (128, 313) dB
        if self.augment:
            t0 = self.rng.integers(0, x.shape[1] - config.CROP_FRAMES + 1)
            x = x[:, t0:t0 + config.CROP_FRAMES].copy()
            x += self.rng.uniform(-config.GAIN_DB, config.GAIN_DB)
        x = (x - self.mean[:, None]) / self.std[:, None]
        if self.augment:
            x = self._spec_augment(x)
        return (torch.from_numpy(x).unsqueeze(0),
                torch.tensor(self.mtypes[i]),
                torch.tensor(self.labels[i]))

    def pos_weight(self) -> float:
        """Negative/positive ratio, used to weight the loss for the rare abnormal class."""
        pos = self.labels.sum()
        return float((len(self.labels) - pos) / pos)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible random stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
