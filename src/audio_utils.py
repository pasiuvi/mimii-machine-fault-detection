"""Small, reusable audio helpers used by EDA, training and the web app."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

from src import config


def load_audio(path, channel: int = config.CHANNEL, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load one microphone channel as float32 in [-1, 1].

    Works for both the original 8-channel MIMII files and mono files
    (e.g. the submission sample or a file uploaded to the web app).
    """
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    ch = channel if channel < y.shape[1] else 0
    y = y[:, ch]
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def log_mel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, time_frames), in decibels."""
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    )
    return librosa.power_to_db(m, ref=1.0)


def rms_db(y: np.ndarray) -> float:
    """Overall loudness of a clip in dB (used in EDA and as a trivial benchmark)."""
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))
