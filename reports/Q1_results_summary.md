# Question 1 – Results summary (verified numbers for the report)

All numbers below were checked against the outputs of `notebooks/01_eda.ipynb` and `reports/leakage_report.md` (MIMII, pump + fan, 6 dB SNR).

## 1. Dataset
| Item | Value |
|---|---|
| Source | MIMII (Hitachi, Ltd.), Zenodo record 3384388, DOI 10.5281/zenodo.3384388 |
| Licence | CC BY-SA 4.0 |
| Citation | Purohit, H. et al. (2019) *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*. Proc. DCASE 2019 Workshop. arXiv:1909.09347 |
| Subset used | pump + fan, machine IDs 00/02/04/06, 6 dB SNR |
| Size | **9,755 clips**, 27.1 hours of audio, 25.0 GB |
| Format | 16 kHz, 8 channels (channel 0 used), 10 s per clip |
| Target | normal (0) / abnormal (1); **1,931 abnormal (19.8 %)** |
| Download check | both zip files matched the official Zenodo MD5 checksums |

### Clips per machine
| Machine | Split | Clips | Abnormal | % abnormal |
|---|---|---|---|---|
| fan id_00 | train | 1,418 | 407 | 28.7 |
| fan id_02 | train | 1,375 | 359 | 26.1 |
| fan id_04 | val | 1,381 | 348 | 25.2 |
| fan id_06 | test | 1,376 | 361 | 26.2 |
| pump id_00 | train | 1,149 | 143 | 12.4 |
| pump id_02 | train | 1,116 | 111 | 9.9 |
| pump id_04 | val | 802 | 100 | 12.5 |
| pump id_06 | test | 1,138 | 102 | 9.0 |

### Per split
| Split | Clips | % abnormal |
|---|---|---|
| train | 5,058 | 20.2 |
| val | 2,183 | 20.5 |
| test | 2,514 | 18.4 |

## 2. Data quality
| Check | Result |
|---|---|
| Unreadable files | 0 |
| Sample rates | 16,000 Hz only |
| Channels | 8 only |
| Durations | 10.0 s only |
| Silent clips (< −60 dB) | 0 |
| Mic channel correlation (one clip) | 0.68 – 0.93 → channels are largely redundant; channel 0 used, which also matches a single-sensor deployment |

## 3. Leakage checks
| Check | Result |
|---|---|
| Group isolation (machine ID) | **PASS** – every machine is in exactly one split |
| Exact duplicates (audio hash) | 0 duplicate sets; 0 across splits |
| Near-duplicates | 0 of the top-20 most similar cross-split pairs have spectrogram correlation > 0.99 |
| Similarity to nearest training clip | within train median **0.97**; val/test → train median **0.59** |
| Prediction-time inputs | waveform only; path, folder name, machine ID and split are excluded from model input |
| Shortcut / trivial benchmark | loudness-only logistic regression: **validation ROC-AUC 0.796** |

## 4. Key EDA numbers
| Analysis | Result |
|---|---|
| Machine vs fault signal (5-fold CV, same features) | machine ID accuracy **0.983** (chance 0.17) vs fault ROC-AUC **0.776** |
| Pump spectral difference (abnormal − normal, train) | largest at **431–910 Hz, +4.9 to +5.4 dB** |
| Fan spectral difference (train) | **+2.6 dB at 0–48 Hz**; **−2.6 to −2.8 dB at 479 Hz, 1.9 kHz, 2.4 kHz** |
| Peak-to-median frame deviation (normal / abnormal) | pump **2.21 / 1.56**; fan **1.47 / 1.44** |
| Loudness direction (abnormal vs normal) | louder: pump_id_00, pump_id_02, fan_id_00, fan_id_04 · quieter: fan_id_02, pump_id_04 |

## 5. Decisions taken from Q1
| Decision | Because of |
|---|---|
| Split by machine ID (train 00+02, val 04, test 06) | Finding 1 + leakage rule |
| Pretrained audio CNN on log-mel spectrograms + noise augmentation | Findings 1 and 2 |
| Expect the classical baseline to be competitive (stated in Q2b) | Finding 2 |
| Primary metric: recall at 5 % false-alarm rate; ROC-AUC and pAUC(0.1) secondary | Finding 3 + FP/FN costs |
| Class-weighted loss; threshold chosen on validation only | Finding 3 |
| Results reported per machine type | Finding 3 |
| Trivial benchmark = loudness only (val AUC 0.796) | Leakage check 5 |

## 6. Figures to use in the report
| File (reports/figures/) | Use it to show |
|---|---|
| q1_class_balance.png | imbalance per split and machine type |
| q1_examples_waveform_logmel.png | what the model's input looks like (normal vs abnormal) |
| q1_mean_spectrum_by_class.png | Finding 2: where in frequency the faults show |
| q1_temporal_deviation.png | Finding 2: faults are steady, not spikes |
| q1_machine_vs_fault_variation.png | Finding 1 (PCA) + Finding 3 (loudness by machine) |
| q1_channel_correlation.png | why only channel 0 is used (appendix is fine) |
