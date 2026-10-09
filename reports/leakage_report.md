# Leakage and data-quality report

Clips: 9755 | machines (groups): 8 | SNR: 6_dB

## 1. Group isolation (machine ID)
| split   | machines                                     |
|:--------|:---------------------------------------------|
| test    | fan_id_06, pump_id_06                        |
| train   | fan_id_00, fan_id_02, pump_id_00, pump_id_02 |
| val     | fan_id_04, pump_id_04                        |

Result: PASS - every machine is in exactly one split

## 2. Exact duplicates (hash of decoded audio)
Duplicate sets: 0 | crossing split boundaries: 0

## 3. Near-duplicates across splits
Nearest-neighbour cosine similarity on log-mel summaries:
|       |   within train (clip vs other train clip) |   val/test clip vs nearest train clip |
|:------|------------------------------------------:|--------------------------------------:|
| count |                                 5058      |                             4697      |
| mean  |                                    0.9518 |                                0.5943 |
| std   |                                    0.0483 |                                0.1499 |
| min   |                                    0.5471 |                                0.2094 |
| 25%   |                                    0.9424 |                                0.4841 |
| 50%   |                                    0.9671 |                                0.5885 |
| 75%   |                                    0.9788 |                                0.697  |
| max   |                                    0.9955 |                                0.9732 |

Top-20 most similar cross-split pairs checked on full spectrograms: 0 with correlation > 0.99 (near-duplicates).
Interpretation: if val/test clips are clearly LESS similar to train than train clips are to each other, the group split is doing its job (unseen machines sound different).

## 4. Features available at prediction time
| Column | Allowed as model input? | Reason |
|---|---|---|
| audio waveform (channel 0) | YES | the only thing the engineer uploads / the sensor records |
| label, label_name | NO | this is the target |
| path, folder names (normal/abnormal) | NO | folder name IS the label -> direct leakage |
| machine_id, group | NO | not known for a new machine; would let the model memorise machines |
| split, snr | NO | bookkeeping only |

File names shared by normal and abnormal clips: 407 (file numbering restarts per folder, so the file name alone carries no label - but the folder path does, so the data loader passes ONLY the waveform to the model).

## 5. Shortcut signals / trivial benchmark (fit on train, scored on validation)
| single feature   |   validation ROC-AUC |
|:-----------------|---------------------:|
| rms_db           |                0.796 |

AUC near 0.5 = no shortcut. Clearly above 0.5 = loudness alone separates classes: report it, use it as the trivial benchmark in Q3a, and check the CNN beats it.

## 6. Data quality
- unreadable files: 0
- sample rates: [16000]
- channel counts: [8]
- durations (s): [10.0]
- silent clips (RMS < -60 dB): 0