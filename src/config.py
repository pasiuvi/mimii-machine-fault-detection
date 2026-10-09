"""
Central configuration for the whole project.

Every script imports from here, so a decision (seed, split, sample rate...)
is made ONCE and used everywhere. Change values here, never inside scripts.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the code works on any computer)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # extracted MIMII .wav files (NOT on GitHub)
PROCESSED_DIR = DATA_DIR / "processed"  # metadata.csv, splits.csv, cached features
SAMPLE_DIR = DATA_DIR / "sample"      # small sample for the coursework submission
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

METADATA_CSV = PROCESSED_DIR / "metadata.csv"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
FEATURE_CACHE = PROCESSED_DIR / "eda_logmel_summary.npz"

# ----------------------------------------------------------------------------
# Dataset scope (approved: MIMII, Zenodo DOI 10.5281/zenodo.3384388, CC BY-SA 4.0)
# ----------------------------------------------------------------------------
ZENODO_RECORD = "3384388"
SNR = "6_dB"                     # one noise level for training/testing
MACHINE_TYPES = ["pump", "fan"]  # start with these two; add more only if time allows

# ----------------------------------------------------------------------------
# Audio settings (MIMII = 16 kHz, 8-channel microphone array, 10-second clips)
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CHANNEL = 0        # we use microphone channel 0 only (a single-mic deployment)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# THE single train / validation / test split  (brief section 6: "Single split")
#
# Split by MACHINE ID (group split), never by random file.
# Why: all clips from one physical machine sound alike. A random split would put
# clips of the same machine in train AND test, so the model could "recognise the
# machine" instead of "recognise the fault" -> leakage (brief section 4).
# Holding out a whole machine also matches the real use case: the model is
# installed on a machine it has never heard before.
#
# This assignment was fixed BEFORE looking at any model result. Do not change it
# after seeing test scores (brief section 12).
# ----------------------------------------------------------------------------
SPLIT_BY_MACHINE_ID = {
    "id_00": "train",
    "id_02": "train",
    "id_04": "val",
    "id_06": "test",
}
